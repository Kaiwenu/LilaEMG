"""
Visualize Stage 2 training data: EMG **inputs** (what the EMG head sees) and **labels**
(teacher latents ``z_star = encoder(hand_state, hand_velocity, language)`` from a Stage 1 checkpoint).

Uses the same stacking, train split, and optional z-score normalization as ``train_teleop.py``.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import joblib
import numpy as np
import torch

from train_teleop import (
    PCA_NAME,
    NormStats,
    build_model,
    compute_norm_stats,
    discover_session_dirs,
    group_sessions_by_gesture,
    load_model_checkpoint,
    load_stacked_sessions,
    row_indices_for_stems,
    split_train_val_test,
)

ROOT = Path(__file__).resolve().parent
DEFAULT_PREPROCESSED = ROOT / "preprocessed_sessions"
NORM_FIELDS = ("emg_window", "hand_state", "hand_velocity")


def gesture_from_session_stem(stem: str) -> str:
    """``grasp_1`` -> ``grasp``; ``index_pick_2`` -> ``index_pick``."""
    m = re.match(r"^(.+)_(\d+)$", stem)
    return m.group(1) if m else stem


def apply_norm(
    arrays: dict[str, np.ndarray],
    idx: np.ndarray,
    norm: dict[str, NormStats],
    field: str,
) -> np.ndarray:
    x = np.asarray(arrays[field][idx], dtype=np.float32)
    st = norm[field]
    m = st.mean.numpy()
    s = st.std.numpy()
    return (x - m) / s


def main() -> None:
    p = argparse.ArgumentParser(description="Plot Stage 2 EMG inputs and teacher latent labels.")
    p.add_argument("--preprocessed-dir", type=Path, default=DEFAULT_PREPROCESSED)
    p.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Stage 1 checkpoint (.pt) with trained encoder (for z_star labels).",
    )
    p.add_argument(
        "--emg-input",
        type=str,
        choices=("window", "features"),
        default="features",
        help="Must match training (default: features).",
    )
    p.add_argument(
        "--split",
        type=str,
        choices=("train", "val"),
        default="train",
        help="Which split to sample from (Stage 2 trains on train).",
    )
    p.add_argument("--n-scatter", type=int, default=2000, help="Points for latent scatter (subsampled).")
    p.add_argument("--n-heatmap", type=int, default=6, help="Number of EMG windows/features to show as heatmaps.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--normalize", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("-o", "--output", type=Path, default=ROOT / "checkpoints" / "stage2_data_viz.png")
    p.add_argument("--latent-dim", type=int, default=2)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--emg-layers", type=int, default=2)
    p.add_argument("--no-language", action="store_true")
    args = p.parse_args()

    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise SystemExit("Install matplotlib: pip install matplotlib") from e

    pre = args.preprocessed_dir.resolve()
    pca_path = pre / PCA_NAME
    if not pca_path.is_file():
        raise SystemExit(f"Missing {pca_path}")

    pca = joblib.load(pca_path)
    synergy_dim = int(pca.n_components_)

    session_dirs = discover_session_dirs(pre)
    emg_name = "emg_features.npy" if args.emg_input == "features" else "emg_window.npy"
    arrays, stem_per_row = load_stacked_sessions(session_dirs, emg_npy_name=emg_name)

    grouped = group_sessions_by_gesture(session_dirs)
    train_stems, val_stems, _ = split_train_val_test(grouped)
    stems_use = train_stems if args.split == "train" else val_stems
    idx_pool = row_indices_for_stems(stem_per_row, stems_use)
    if idx_pool.shape[0] == 0:
        raise SystemExit(f"No samples in split {args.split!r}")

    rng = np.random.default_rng(args.seed)
    emg_dim = int(arrays["emg_window"].shape[1])
    lang_dim_data = int(arrays["language_embedding"].shape[1])

    norm: dict[str, NormStats] | None = None
    train_idx = row_indices_for_stems(stem_per_row, train_stems)
    if args.normalize:
        raw = compute_norm_stats(arrays, train_idx, NORM_FIELDS)
        norm = raw

    # Subsample for scatter
    n_sc = min(args.n_scatter, idx_pool.shape[0])
    sub_idx = rng.choice(idx_pool, size=n_sc, replace=False)

    # Tensors (normalized like training)
    if norm is not None:
        emg_np = apply_norm(arrays, sub_idx, norm, "emg_window")
        s_np = apply_norm(arrays, sub_idx, norm, "hand_state")
        v_np = apply_norm(arrays, sub_idx, norm, "hand_velocity")
    else:
        emg_np = np.asarray(arrays["emg_window"][sub_idx], dtype=np.float32)
        s_np = np.asarray(arrays["hand_state"][sub_idx], dtype=np.float32)
        v_np = np.asarray(arrays["hand_velocity"][sub_idx], dtype=np.float32)

    lang_np = np.asarray(arrays["language_embedding"][sub_idx], dtype=np.float32)

    device = torch.device(args.device)
    ns = argparse.Namespace(
        latent_dim=args.latent_dim,
        emg_dim=emg_dim,
        language_dim=lang_dim_data,
        hidden_dim=args.hidden_dim,
        emg_layers=args.emg_layers,
        no_language=args.no_language,
    )
    model = build_model(ns, synergy_dim).to(device)
    load_model_checkpoint(args.checkpoint.resolve(), model)
    model.eval()

    with torch.no_grad():
        z_star = model.encode_action(
            torch.from_numpy(s_np).to(device),
            torch.from_numpy(v_np).to(device),
            torch.from_numpy(lang_np).to(device),
        )
    z_np = z_star.cpu().numpy()
    if z_np.shape[1] < 2:
        raise SystemExit("Need latent_dim >= 2 for 2D scatter.")

    # Gestures for coloring
    stems_sub = stem_per_row[sub_idx]
    gestures = np.array([gesture_from_session_stem(str(s)) for s in stems_sub])
    uniq_g = sorted(np.unique(gestures))
    gcolors = plt.cm.tab10(np.linspace(0, 0.9, max(len(uniq_g), 1)))

    # --- Figure ---
    fig = plt.figure(figsize=(12, 5.5))

    # 1) Latent scatter (labels)
    ax0 = fig.add_subplot(1, 2, 1)
    for i, g in enumerate(uniq_g):
        m = gestures == g
        ax0.scatter(
            z_np[m, 0],
            z_np[m, 1],
            s=4,
            alpha=0.35,
            label=g,
            color=gcolors[i],
        )
    ax0.set_title("Stage 2 labels: teacher latent $z^\\star$ = encoder(s, v, lang)")
    ax0.set_xlabel("latent dim 0")
    ax0.set_ylabel("latent dim 1")
    ax0.legend(loc="best", fontsize=7, ncol=2)
    ax0.grid(True, alpha=0.3)
    ax0.set_aspect("equal", adjustable="box")

    # 2) EMG inputs: heatmaps
    ax1 = fig.add_subplot(1, 2, 2)
    heat_n = min(args.n_heatmap, sub_idx.shape[0])
    pick = rng.choice(np.arange(sub_idx.shape[0]), size=heat_n, replace=False)

    def _tile_four(stack: list[np.ndarray]) -> np.ndarray:
        if not stack:
            raise ValueError("empty stack")
        if len(stack) == 1:
            return stack[0]
        if len(stack) == 2:
            return np.hstack([stack[0], stack[1]])
        if len(stack) == 3:
            stack = stack[:2]
            return np.hstack([stack[0], stack[1]])
        return np.block([[stack[0], stack[1]], [stack[2], stack[3]]])

    d_emg = emg_np.shape[1]
    if d_emg == 320 or (args.emg_input == "window" and d_emg == 320):
        stack = [emg_np[j].reshape(8, 40) for j in pick[: min(4, heat_n)]]
        if not stack:
            ax1.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax1.transAxes)
        else:
            grid = _tile_four(stack)
            im = ax1.imshow(grid, aspect="auto", cmap="coolwarm", interpolation="nearest")
            plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
            ax1.set_title("Stage 2 input: EMG windows (≤4 samples, 8×40 tiled)")
            ax1.set_xlabel("time (within window)")
            ax1.set_ylabel("channel blocks")
    elif d_emg == 48:
        stack = [emg_np[j].reshape(8, 6) for j in pick[: min(4, heat_n)]]
        grid = _tile_four(stack)
        im = ax1.imshow(grid, aspect="auto", cmap="viridis", interpolation="nearest")
        plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
        ax1.set_title("Stage 2 input: EMG features (≤4 samples, 8×6 tiled)")
        ax1.set_xlabel("feature index (6 per channel)")
        ax1.set_ylabel("channel")
    else:
        # Generic: first min(40) dims as strip
        grid = emg_np[pick[:8]]
        im = ax1.imshow(grid, aspect="auto", cmap="magma", interpolation="nearest")
        plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
        ax1.set_title(f"Stage 2 input: EMG vector (first {grid.shape[0]} samples × D={emg_dim})")
        ax1.set_xlabel("dimension")

    fig.suptitle(
        f"Stage 2 data ({args.split} split, n={n_sc})  |  norm={args.normalize}  |  ckpt={args.checkpoint.name}",
        fontsize=10,
    )
    fig.tight_layout()
    out = args.output.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
