"""
Train LILA-style EMG teleop model (see teleop_model.LilaTeleopModel).

Stages (from proposal):
  1 — Train FiLM action encoder + FiLM decoder on vision labels
      (hand_state, hand_velocity, language_embedding). Loss: velocity MSE.
      Encoder matches ``lila/src/models/film.py`` (language-conditioned latent).
  2 — Freeze encoder & decoder; train EMG net to match encoder latent (teacher uses language).
  3 — Freeze encoder only; fine-tune EMG + decoder toward velocity MSE (optional).

Data: ``preprocessed_sessions/`` from the preprocessing pipeline (see ``preprocessing/run_pipeline.py``;
core steps include ``preprocessing_kinematics.py`` after ``semg_windowing.py`` and ``language_preprocessing.py``). One subfolder per
recording (``{gesture}_{session}/``) with ``hand_state.npy``, ``hand_velocity.npy``,
``language.npy``, plus either ``emg_window.npy`` (raw windows) or ``emg_features.npy``
(from ``preprocessing/extract_emg_features.py``), and root ``pca_joint_angles.joblib``.

Stage 2/3 EMG input is selected by ``--emg-input`` (default: features). Stage 1 does not
use EMG in the loss; stacked EMG tensors are still loaded for optional normalization.

Split: per gesture, sessions sorted by numeric session id — first 4 → train, next → val,
next → test; any sessions beyond the 6th are added to train. If a gesture has fewer than
6 sessions, the split degrades (see ``split_train_val_test``).

Use ``--only-gestures grasp`` (comma-separated) to keep only those gestures' sessions before
splitting — e.g. six ``grasp_*`` sessions yield 4 train / 1 val / 1 test on that gesture alone
(overfit / single-task experiments). PCA in ``preprocessed_sessions/`` is still the global fit.

Use ``--single-session grasp_1`` to load **only** that folder and set **train = val = test** to
that session (same windows for fit, validation metrics, and test — intentional overfit / sanity).

Checkpoints and logs are written under ``<out-dir>/stage{N}[_tag]/run_<UTC-timestamp>/`` so
each training run keeps its own ``metrics.csv``, ``training_loss.png`` (train/val curves),
``best.pt``, and epoch checkpoints without overwriting previous runs.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
import time
import types
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

from teleop_model import LilaTeleopModel

ROOT = Path(__file__).resolve().parent
DEFAULT_PREPROCESSED_DIR = ROOT / "preprocessed_sessions"
PCA_NAME = "pca_joint_angles.joblib"

# Must match ``GESTURES`` in ``preprocessing/language_preprocessing.py`` (used for ordering and parsing).
GESTURES: tuple[str, ...] = (
    "grasp",
    "index_pick",
    "press",
    "single_finger",
    "spray",
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_session_stem(stem: str) -> tuple[str, int]:
    m = re.match(r"^(.+)_(\d+)$", stem)
    if not m:
        raise ValueError(f"expected session folder name {{gesture}}_{{id}}, got {stem!r}")
    g, sid = m.group(1), int(m.group(2))
    if g not in GESTURES:
        raise ValueError(f"unknown gesture {g!r} in {stem!r}; expected one of {GESTURES}")
    return g, sid


def session_sort_key(path: Path) -> tuple[int, int]:
    g, sid = parse_session_stem(path.name)
    return (GESTURES.index(g), sid)


def discover_session_dirs(sessions_root: Path) -> list[Path]:
    return sorted(
        [p for p in sessions_root.iterdir() if p.is_dir()],
        key=session_sort_key,
    )


def group_sessions_by_gesture(session_dirs: list[Path]) -> dict[str, list[tuple[int, Path]]]:
    gmap: dict[str, list[tuple[int, Path]]] = {}
    for p in session_dirs:
        g, sid = parse_session_stem(p.name)
        gmap.setdefault(g, []).append((sid, p))
    return gmap


def filter_session_dirs_by_gestures(session_dirs: list[Path], gestures: set[str]) -> list[Path]:
    """Keep only folders whose stem parses to a gesture in ``gestures``; preserve sort order."""
    out = [p for p in session_dirs if parse_session_stem(p.name)[0] in gestures]
    return out


def split_train_val_test(
    grouped: dict[str, list[tuple[int, Path]]],
) -> tuple[set[str], set[str], set[str]]:
    """
    Per gesture: sort by session id; assign 4 train, 1 val, 1 test when n >= 6;
    further sessions go to train. Smaller counts use a shorter train/val/(test) tail.
    """
    train_stems: set[str] = set()
    val_stems: set[str] = set()
    test_stems: set[str] = set()
    for g in sorted(grouped.keys()):
        items = sorted(grouped[g], key=lambda x: x[0])
        stems = [p.name for _, p in items]
        n = len(stems)
        if n >= 6:
            train_stems.update(stems[:4])
            val_stems.add(stems[4])
            test_stems.add(stems[5])
            train_stems.update(stems[6:])
        elif n == 5:
            train_stems.update(stems[:4])
            val_stems.add(stems[4])
        elif n == 4:
            train_stems.update(stems[:3])
            val_stems.add(stems[3])
        elif n == 3:
            train_stems.update(stems[:2])
            val_stems.add(stems[2])
        elif n == 2:
            train_stems.add(stems[0])
            val_stems.add(stems[1])
        elif n == 1:
            train_stems.add(stems[0])
        else:
            pass
        if n < 6:
            print(
                f"Warning: gesture {g!r} has {n} session(s); fewer than 6 — "
                f"using reduced train/val/test split (see code).",
                file=sys.stderr,
            )
    return train_stems, val_stems, test_stems


def load_stacked_sessions(
    session_dirs: list[Path],
    *,
    emg_npy_name: str,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Load all session npy files; stack rows; return arrays + stem per row (object array).

    ``emg_npy_name`` is ``emg_features.npy``.
    Stacked EMG is always stored under key ``emg_window`` for the dataset / norm field name.
    """
    emg_chunks: list[np.ndarray] = []
    hs_chunks: list[np.ndarray] = []
    hv_chunks: list[np.ndarray] = []
    lang_chunks: list[np.ndarray] = []
    stems_per_row: list[str] = []

    for p in session_dirs:
        emg_p = p / emg_npy_name
        hs_p = p / "hand_state.npy"
        hv_p = p / "hand_velocity.npy"
        lang_p = p / "language.npy"
        for req in (emg_p, hs_p, hv_p, lang_p):
            if not req.is_file():
                raise FileNotFoundError(f"Missing {req}")
        emg = np.load(emg_p)
        hs = np.load(hs_p)
        hv = np.load(hv_p)
        lang1d = np.load(lang_p)
        n = emg.shape[0]
        if hs.shape[0] != n or hv.shape[0] != n:
            raise ValueError(f"{p.name}: length mismatch emg {n}, hand_state {hs.shape[0]}, hand_velocity {hv.shape[0]}")
        if emg.ndim == 3:
            emg = emg.reshape(n, -1).astype(np.float32, copy=False)
        else:
            emg = emg.astype(np.float32, copy=False)
        hs = hs.astype(np.float32, copy=False)
        hv = hv.astype(np.float32, copy=False)
        lang1d = lang1d.astype(np.float32, copy=False).reshape(-1)
        lang_rep = np.broadcast_to(lang1d[np.newaxis, :], (n, lang1d.shape[0])).copy()
        emg_chunks.append(emg)
        hs_chunks.append(hs)
        hv_chunks.append(hv)
        lang_chunks.append(lang_rep)
        stems_per_row.extend([p.name] * n)

    arrays = {
        "emg_window": np.concatenate(emg_chunks, axis=0),
        "hand_state": np.concatenate(hs_chunks, axis=0),
        "hand_velocity": np.concatenate(hv_chunks, axis=0),
        "language_embedding": np.concatenate(lang_chunks, axis=0),
    }
    return arrays, np.array(stems_per_row, dtype=object)


def row_indices_for_stems(stem_per_row: np.ndarray, stems: set[str]) -> np.ndarray:
    if not stems:
        return np.array([], dtype=np.int64)
    mask = np.isin(stem_per_row, list(stems))
    return np.nonzero(mask)[0].astype(np.int64)


class NormStats:
    def __init__(self, mean: torch.Tensor, std: torch.Tensor, names: tuple[str, ...]):
        self.mean = mean
        self.std = std
        self.names = names

    def to(self, device: torch.device) -> NormStats:
        return NormStats(self.mean.to(device), self.std.to(device), self.names)

    def state_dict(self) -> dict:
        return {"mean": self.mean.cpu(), "std": self.std.cpu(), "names": self.names}

    @classmethod
    def from_state_dict(cls, d: dict) -> NormStats:
        return cls(d["mean"], d["std"], tuple(d["names"]))


def compute_norm_stats(
    z: dict[str, np.ndarray],
    train_idx: np.ndarray,
    fields: tuple[str, ...],
) -> dict[str, NormStats]:
    out: dict[str, NormStats] = {}
    for name in fields:
        arr = np.asarray(z[name][train_idx], dtype=np.float64)
        mean = torch.from_numpy(arr.mean(axis=0)).float()
        std = torch.from_numpy(arr.std(axis=0)).float()
        std = torch.clamp(std, min=1e-6)
        out[name] = NormStats(mean, std, (name,))
    return out


class TeleopNpzDataset(Dataset):
    """Index into in-memory arrays built from ``preprocessed_sessions/*/`` (or any stacked dict with the same keys)."""

    def __init__(
        self,
        arrays: dict[str, np.ndarray],
        indices: np.ndarray | None,
        norm: dict[str, NormStats] | None,
        device_for_norm: str = "cpu",
    ):
        self.arrays = arrays
        self.indices = indices if indices is not None else np.arange(arrays["hand_state"].shape[0])
        self.norm = norm
        self.dev = torch.device(device_for_norm)

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    def _norm_field(self, name: str, x: torch.Tensor) -> torch.Tensor:
        if self.norm is None or name not in self.norm:
            return x
        st = self.norm[name]
        m = st.mean.to(self.dev)
        s = st.std.to(self.dev)
        return (x.to(self.dev) - m) / s

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        idx = int(self.indices[i])
        emg = torch.from_numpy(self.arrays["emg_window"][idx].astype(np.float32, copy=False))
        s = torch.from_numpy(self.arrays["hand_state"][idx].astype(np.float32, copy=False))
        v = torch.from_numpy(self.arrays["hand_velocity"][idx].astype(np.float32, copy=False))
        lang = torch.from_numpy(self.arrays["language_embedding"][idx].astype(np.float32, copy=False))
        emg = self._norm_field("emg_window", emg)
        s = self._norm_field("hand_state", s)
        v = self._norm_field("hand_velocity", v)
        return {"emg_window": emg, "hand_state": s, "hand_velocity": v, "language_embedding": lang}


def collate(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    keys = batch[0].keys()
    return {k: torch.stack([b[k] for b in batch], dim=0) for k in keys}


def save_checkpoint(
    path: Path,
    model: LilaTeleopModel,
    optimizer: torch.optim.Optimizer | None,
    stage: int,
    epoch: int,
    meta_train: dict,
    norm_state: dict | None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "stage": stage,
        "epoch": epoch,
        # JSON-safe only: avoids pathlib / Namespace pickles that break across Python versions.
        "meta_train": _json_safe(meta_train),
        "norm": norm_state,
    }
    torch.save(payload, path)


def _install_pathlib_local_shim() -> None:
    """
    Checkpoints saved on Python 3.12+ may reference ``pathlib._local`` (not present on 3.11).
    Map those names onto the stdlib ``pathlib`` module so ``torch.load`` can unpickle.
    """
    if sys.version_info >= (3, 12):
        return
    if "pathlib._local" in sys.modules:
        return
    import pathlib as pl

    stub = types.ModuleType("pathlib._local")
    for name in (
        "PurePath",
        "PurePosixPath",
        "PureWindowsPath",
        "Path",
        "PosixPath",
        "WindowsPath",
    ):
        if hasattr(pl, name):
            setattr(stub, name, getattr(pl, name))
    sys.modules["pathlib._local"] = stub


def load_model_checkpoint(path: Path, model: LilaTeleopModel) -> dict:
    _install_pathlib_local_shim()
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"], strict=True)
    return ckpt


def load_model_checkpoint_matching_shapes(path: Path, model: LilaTeleopModel) -> dict:
    """
    Load checkpoint tensors only when shapes match the current model (strict per-tensor).

    Use when resuming stage 2/3 from stage 1: encoder/decoder load; ``emg.*`` is skipped
    if ``emg_dim`` changed (e.g. raw windows vs feature vector).
    """
    _install_pathlib_local_shim()
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    src = ckpt["model"]
    dst = model.state_dict()
    to_load: dict[str, torch.Tensor] = {}
    skipped: list[str] = []
    for k, v in src.items():
        if k not in dst:
            skipped.append(f"{k} (not in model)")
            continue
        if dst[k].shape != v.shape:
            skipped.append(f"{k}  checkpoint{tuple(v.shape)}  model{tuple(dst[k].shape)}")
            continue
        to_load[k] = v
    not_in_ckpt = [k for k in dst if k not in src]
    model.load_state_dict(to_load, strict=False)
    if skipped:
        print(
            "Checkpoint load: skipped (shape or key mismatch):\n  "
            + "\n  ".join(skipped[:12])
            + (f"\n  ... and {len(skipped) - 12} more" if len(skipped) > 12 else ""),
            flush=True,
        )
    if not_in_ckpt:
        print(
            f"Checkpoint load: {len(not_in_ckpt)} model key(s) absent from checkpoint (random init).",
            flush=True,
        )
    return ckpt


def _json_safe(obj: object) -> object:
    """Recursively make objects JSON-serializable (Path -> str, nested dicts/lists)."""
    if isinstance(obj, Path):
        return str(obj.resolve())
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(x) for x in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


def write_run_meta(
    path: Path,
    *,
    args: argparse.Namespace,
    device: torch.device,
    synergy_dim: int,
    train_n: int,
    val_n: int,
    test_n: int = 0,
    train_sessions: list[str] | None = None,
    val_sessions: list[str] | None = None,
    test_sessions: list[str] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "stage": args.stage,
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "synergy_dim": synergy_dim,
        "train_samples": train_n,
        "val_samples": val_n,
        "test_samples": test_n,
        "train_sessions": train_sessions or [],
        "val_sessions": val_sessions or [],
        "test_sessions": test_sessions or [],
        "epochs_planned": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "split_policy": "per_gesture_4_train_1_val_1_test_by_session_id",
        "seed": args.seed,
        "normalize": args.normalize,
        "no_language": args.no_language,
        "preprocessed_dir": str(args.preprocessed_dir.resolve()),
        "out_dir": str(args.out_dir.resolve()),
        "run_dir": str(path.parent.resolve()),
        "full_args": _json_safe(vars(args)),
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def append_metrics_row(
    path: Path,
    *,
    epoch: int,
    train_loss: float,
    val_loss: float,
    lr: float,
    best_val: float,
    wall_sec: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.is_file()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(
                ["epoch", "train_loss", "val_loss", "lr", "best_val", "wall_sec"]
            )
        w.writerow(
            [
                epoch,
                f"{train_loss:.8f}",
                f"{val_loss:.8f}",
                f"{lr:.10g}",
                f"{best_val:.8f}",
                f"{wall_sec:.3f}",
            ]
        )


def save_training_loss_plot(
    metrics_csv: Path,
    out_png: Path,
    *,
    stage: int,
    tag: str = "",
) -> bool:
    """
    Read ``metrics.csv`` and write a train/val loss vs epoch PNG. Returns True if written.
    """
    if not metrics_csv.is_file():
        return False
    epochs: list[int] = []
    train_loss: list[float] = []
    val_loss: list[float] = []
    with metrics_csv.open(newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            epochs.append(int(row["epoch"]))
            train_loss.append(float(row["train_loss"]))
            val_loss.append(float(row["val_loss"]))
    if not epochs:
        return False
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print(
            "matplotlib not installed; skipping training_loss.png. Install: pip install matplotlib",
            file=sys.stderr,
        )
        return False

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs, train_loss, color="#1565c0", label="train", linewidth=1.8)
    ax.plot(epochs, val_loss, color="#c62828", linestyle="--", label="val", linewidth=1.8, alpha=0.9)
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss (MSE)")
    title = f"Stage {stage} — train vs validation loss"
    if tag:
        title += f" ({tag})"
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


def run_epoch_stage1(
    model: LilaTeleopModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    train: bool,
) -> float:
    if train:
        model.encoder.train()
        model.decoder.train()
    else:
        model.encoder.eval()
        model.decoder.eval()
    total = 0.0
    n = 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch in loader:
            s = batch["hand_state"].to(device)
            v = batch["hand_velocity"].to(device)
            lang = batch["language_embedding"].to(device)
            if train:
                optimizer.zero_grad(set_to_none=True)
            z = model.encode_action(s, v, lang)
            v_hat = model.decode(s, z, lang)
            loss = F.mse_loss(v_hat, v)
            if train:
                loss.backward()
                optimizer.step()
            total += float(loss.item()) * s.shape[0]
            n += s.shape[0]
    return total / max(n, 1)


def run_epoch_stage2(
    model: LilaTeleopModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    train: bool,
) -> float:
    model.encoder.eval()
    model.decoder.eval()
    model.emg.train(train)
    total = 0.0
    n = 0
    for batch in loader:
        s = batch["hand_state"].to(device)
        v = batch["hand_velocity"].to(device)
        lang = batch["language_embedding"].to(device)
        emg = batch["emg_window"].to(device)
        with torch.no_grad():
            z_star = model.encode_action(s, v, lang)
        if train:
            optimizer.zero_grad(set_to_none=True)
            z_emg = model.emg_to_latent(emg)
            loss = F.mse_loss(z_emg, z_star)
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                z_emg = model.emg_to_latent(emg)
                loss = F.mse_loss(z_emg, z_star)
        total += float(loss.item()) * s.shape[0]
        n += s.shape[0]
    return total / max(n, 1)


def run_epoch_stage3(
    model: LilaTeleopModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    train: bool,
) -> float:
    model.encoder.eval()
    if train:
        model.emg.train()
        model.decoder.train()
    else:
        model.emg.eval()
        model.decoder.eval()
    total = 0.0
    n = 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch in loader:
            s = batch["hand_state"].to(device)
            v = batch["hand_velocity"].to(device)
            emg = batch["emg_window"].to(device)
            lang = batch["language_embedding"].to(device)
            if train:
                optimizer.zero_grad(set_to_none=True)
            z = model.emg_to_latent(emg)
            v_hat = model.decode(s, z, lang)
            loss = F.mse_loss(v_hat, v)
            if train:
                loss.backward()
                optimizer.step()
            total += float(loss.item()) * s.shape[0]
            n += s.shape[0]
    return total / max(n, 1)


def freeze_module(m: nn.Module, frozen: bool) -> None:
    for p in m.parameters():
        p.requires_grad_(not frozen)


def build_model(args: argparse.Namespace, synergy_dim: int) -> LilaTeleopModel:
    use_lang = not args.no_language
    return LilaTeleopModel(
        synergy_dim=synergy_dim,
        latent_dim=args.latent_dim,
        emg_dim=args.emg_dim,
        language_dim=args.language_dim,
        hidden_dim=args.hidden_dim,
        emg_hidden_layers=args.emg_layers,
        encoder_use_language=use_lang,
        decoder_use_language=use_lang,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train LILA-style EMG teleop model.")
    parser.add_argument("--preprocessed-dir", type=Path, default=DEFAULT_PREPROCESSED_DIR)
    parser.add_argument("--stage", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument("--resume", type=Path, default=None, help="Checkpoint to load (required after stage 1).")
    parser.add_argument(
        "--emg-input",
        type=str,
        choices=("window", "features"),
        default="features",
        help="Stack emg_window.npy (320-D raw) or emg_features.npy (48-D from preprocessing/extract_emg_features.py).",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--normalize", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "checkpoints")
    parser.add_argument("--tag", type=str, default="", help="Subfolder under out-dir.")

    parser.add_argument("--latent-dim", type=int, default=2)
    parser.add_argument(
        "--emg-dim",
        type=int,
        default=48,
        help="EMG MLP input size (default 48 for features; overridden by loaded data width).",
    )
    parser.add_argument("--language-dim", type=int, default=768)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument(
        "--encoder-layers",
        type=int,
        default=2,
        help="Ignored (Lila FiLM encoder uses fixed depth); kept for CLI compatibility.",
    )
    parser.add_argument("--emg-layers", type=int, default=2)
    parser.add_argument(
        "--decoder-film-layers",
        type=int,
        default=2,
        help="Ignored (Lila FiLM decoder uses a single FiLM block); kept for CLI compatibility.",
    )
    parser.add_argument(
        "--no-language",
        action="store_true",
        help="Disable language FiLM on both encoder and decoder (baseline).",
    )
    parser.add_argument(
        "--only-gestures",
        type=str,
        default="",
        help=(
            "Comma-separated gesture names to include (subset of known gestures), e.g. `grasp` "
            "or `grasp,index_pick`. Other session folders are dropped before per-gesture split. "
            "Six sessions of one gesture → 4 train / 1 val / 1 test on that gesture."
        ),
    )
    parser.add_argument(
        "--single-session",
        type=str,
        default="",
        help=(
            "Session folder stem under preprocessed_sessions, e.g. grasp_1. Only this session "
            "is loaded; train, val, and test splits all use it (same data — overfit / debug). "
            "Ignores per-gesture split; apply after --only-gestures if both are set."
        ),
    )

    args = parser.parse_args()
    set_seed(args.seed)

    pre = args.preprocessed_dir.resolve()
    pca_path = pre / PCA_NAME
    if not pca_path.is_file():
        print(
            f"Missing {pca_path}; run the full stack: python preprocessing/run_pipeline.py "
            f"(or semg_windowing → language_preprocessing → preprocessing_kinematics).",
            file=sys.stderr,
        )
        sys.exit(1)

    pca = joblib.load(pca_path)
    synergy_dim = int(pca.n_components_)

    session_dirs = discover_session_dirs(pre)
    if not session_dirs:
        print(f"No session subfolders in {pre}", file=sys.stderr)
        sys.exit(1)

    only_list = [x.strip() for x in str(args.only_gestures).split(",") if x.strip()]
    if only_list:
        unknown = [g for g in only_list if g not in GESTURES]
        if unknown:
            print(
                f"Unknown gesture(s) in --only-gestures: {unknown}. "
                f"Use names from: {', '.join(GESTURES)}",
                file=sys.stderr,
            )
            sys.exit(1)
        allow = set(only_list)
        before_n = len(session_dirs)
        session_dirs = filter_session_dirs_by_gestures(session_dirs, allow)
        if not session_dirs:
            print(f"No sessions left after --only-gestures filter {sorted(allow)!r}.", file=sys.stderr)
            sys.exit(1)
        print(
            f"Filtered {before_n} → {len(session_dirs)} session(s) (--only-gestures {sorted(allow)!r}): "
            f"{[p.name for p in session_dirs]}",
            flush=True,
        )

    single = str(args.single_session).strip()
    if single:
        parse_session_stem(single)
        names = {p.name for p in session_dirs}
        if single not in names:
            print(
                f"--single-session {single!r} not found among session folders. "
                f"Available: {sorted(names)}",
                file=sys.stderr,
            )
            sys.exit(1)
        session_dirs = [p for p in session_dirs if p.name == single]
        train_stems = val_stems = test_stems = {single}
        print(
            f"Single-session overfit: train = val = test = {single!r} (only this folder is loaded).",
            flush=True,
        )
    else:
        grouped = group_sessions_by_gesture(session_dirs)
        train_stems, val_stems, test_stems = split_train_val_test(grouped)

    emg_npy_name = "emg_features.npy" if args.emg_input == "features" else "emg_window.npy"
    print(f"Loading session .npy files and stacking (EMG: {emg_npy_name})...", flush=True)
    arrays, stem_per_row = load_stacked_sessions(session_dirs, emg_npy_name=emg_npy_name)

    emg_flat_dim = int(arrays["emg_window"].shape[1])
    args.emg_dim = emg_flat_dim
    lang_dim_data = int(arrays["language_embedding"].shape[1])
    print(f"EMG input dimension (data): {emg_flat_dim}", flush=True)
    if args.language_dim != lang_dim_data:
        print(
            f"Warning: --language-dim {args.language_dim} != data language width {lang_dim_data}",
            file=sys.stderr,
        )

    train_idx = row_indices_for_stems(stem_per_row, train_stems)
    val_idx = row_indices_for_stems(stem_per_row, val_stems)
    test_idx = row_indices_for_stems(stem_per_row, test_stems)

    if val_idx.shape[0] == 0:
        print("Warning: validation set is empty.", file=sys.stderr)
    if train_idx.shape[0] == 0:
        print("No training samples; check session split and data.", file=sys.stderr)
        sys.exit(1)

    norm_fields = ("emg_window", "hand_state", "hand_velocity")
    norm_stats: dict[str, NormStats] | None = None
    if args.normalize:
        raw_stats = compute_norm_stats(arrays, train_idx, norm_fields)
        norm_stats = raw_stats

    train_ds = TeleopNpzDataset(arrays, train_idx, norm_stats, device_for_norm="cpu")
    val_ds = TeleopNpzDataset(arrays, val_idx, norm_stats, device_for_norm="cpu")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate,
        pin_memory=args.device.startswith("cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate,
        pin_memory=args.device.startswith("cuda"),
    )

    test_loader: DataLoader | None = None
    if test_idx.shape[0] > 0:
        test_ds = TeleopNpzDataset(arrays, test_idx, norm_stats, device_for_norm="cpu")
        test_loader = DataLoader(
            test_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate,
            pin_memory=args.device.startswith("cuda"),
        )

    device = torch.device(args.device)
    model = build_model(args, synergy_dim).to(device)

    if args.stage >= 2:
        if args.resume is None or not args.resume.is_file():
            print("Stage 2/3 requires --resume PATH to a checkpoint from the previous stage.", file=sys.stderr)
            sys.exit(1)
        ckpt_prev = load_model_checkpoint_matching_shapes(args.resume, model)
        print(
            f"Loaded weights from {args.resume} "
            f"(saved stage {ckpt_prev.get('stage')}, epoch {ckpt_prev.get('epoch')})"
        )
    elif args.resume is not None and args.resume.is_file():
        ckpt_prev = load_model_checkpoint(args.resume, model)
        print(f"Warm start from {args.resume} (epoch {ckpt_prev.get('epoch')})")

    if args.stage == 2:
        freeze_module(model.encoder, True)
        freeze_module(model.decoder, True)
        freeze_module(model.emg, False)
        params = list(model.emg.parameters())
    elif args.stage == 3:
        freeze_module(model.encoder, True)
        freeze_module(model.decoder, False)
        freeze_module(model.emg, False)
        params = list(model.emg.parameters()) + list(model.decoder.parameters())
    else:
        freeze_module(model.encoder, False)
        freeze_module(model.decoder, False)
        freeze_module(model.emg, False)
        params = list(model.encoder.parameters()) + list(model.decoder.parameters())

    optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)

    stage_base = args.out_dir / (f"stage{args.stage}" + (f"_{args.tag}" if args.tag else ""))
    stage_base.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    run_dir = stage_base / f"run_{stamp}"
    run_dir.mkdir(parents=False, exist_ok=False)

    write_run_meta(
        run_dir / "run_meta.json",
        args=args,
        device=device,
        synergy_dim=synergy_dim,
        train_n=int(train_idx.shape[0]),
        val_n=int(val_idx.shape[0]),
        test_n=int(test_idx.shape[0]),
        train_sessions=sorted(train_stems),
        val_sessions=sorted(val_stems),
        test_sessions=sorted(test_stems),
    )
    print(f"Run directory (logs + checkpoints): {run_dir}", flush=True)
    print(f"Logging metrics to {run_dir / 'metrics.csv'}", flush=True)

    meta_train = {
        "synergy_dim": synergy_dim,
        "stage": args.stage,
        "preprocessed_dir": str(pre),
        "train_samples": int(train_idx.shape[0]),
        "val_samples": int(val_idx.shape[0]),
        "test_samples": int(test_idx.shape[0]),
        "train_sessions": sorted(train_stems),
        "val_sessions": sorted(val_stems),
        "test_sessions": sorted(test_stems),
        "normalize": args.normalize,
        "no_language": args.no_language,
        "args": vars(args),
    }
    norm_state = {k: v.state_dict() for k, v in norm_stats.items()} if norm_stats else None

    best_val = float("inf")
    t0 = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        if args.stage == 1:
            tr = run_epoch_stage1(model, train_loader, optimizer, device, train=True)
            va = run_epoch_stage1(model, val_loader, optimizer, device, train=False)
        elif args.stage == 2:
            tr = run_epoch_stage2(model, train_loader, optimizer, device, train=True)
            va = run_epoch_stage2(model, val_loader, optimizer, device, train=False)
        else:
            tr = run_epoch_stage3(model, train_loader, optimizer, device, train=True)
            va = run_epoch_stage3(model, val_loader, optimizer, device, train=False)

        print(f"epoch {epoch:03d}  train {tr:.6f}  val {va:.6f}")

        ckpt_path = run_dir / f"epoch_{epoch:03d}.pt"
        save_checkpoint(ckpt_path, model, optimizer, args.stage, epoch, meta_train, norm_state)

        if va < best_val:
            best_val = va
            save_checkpoint(run_dir / "best.pt", model, optimizer, args.stage, epoch, meta_train, norm_state)

        lr_now = optimizer.param_groups[0]["lr"]
        wall_sec = time.perf_counter() - t0
        append_metrics_row(
            run_dir / "metrics.csv",
            epoch=epoch,
            train_loss=tr,
            val_loss=va,
            lr=lr_now,
            best_val=best_val,
            wall_sec=wall_sec,
        )

    dt = time.perf_counter() - t0

    test_loss: float | None = None
    best_path = run_dir / "best.pt"
    if test_loader is not None and best_path.is_file():
        load_model_checkpoint(best_path, model)
        if args.stage == 1:
            test_loss = run_epoch_stage1(model, test_loader, optimizer, device, train=False)
        elif args.stage == 2:
            test_loss = run_epoch_stage2(model, test_loader, optimizer, device, train=False)
        else:
            test_loss = run_epoch_stage3(model, test_loader, optimizer, device, train=False)
        print(f"Test loss (best val checkpoint): {test_loss:.6f}", flush=True)
    elif test_loader is None:
        print("Test set empty — skipping test evaluation.", flush=True)

    summary = {
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "stage": args.stage,
        "epochs_run": args.epochs,
        "best_val_loss": best_val,
        "test_loss": test_loss,
        "test_samples": int(test_idx.shape[0]),
        "wall_sec_total": round(dt, 3),
        "run_dir": str(run_dir.resolve()),
    }
    with (run_dir / "run_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")

    plot_path = run_dir / "training_loss.png"
    if save_training_loss_plot(
        run_dir / "metrics.csv",
        plot_path,
        stage=args.stage,
        tag=args.tag or "",
    ):
        print(f"Wrote loss plot {plot_path}", flush=True)

    test_msg = f"  test {test_loss:.6f}" if test_loss is not None else ""
    print(
        f"Done stage {args.stage} in {dt:.1f}s. Best val loss {best_val:.6f}.{test_msg} Artifacts in {run_dir}"
    )


if __name__ == "__main__":
    main()
