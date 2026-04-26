#!/usr/bin/env python3
"""
Plot train/val loss vs epoch for stages 1–3 from metrics.csv written by train_teleop.py.

Looks for ``metrics.csv`` in each stage folder, either:

  - legacy: ``{checkpoints_dir}/stageN/metrics.csv``
  - current: ``{checkpoints_dir}/stageN/run_<timestamp>/metrics.csv`` (uses the **newest** run)

Use ``--tag`` if you trained with ``--tag NAME`` (expects ``stage1_TAG/``, ...).
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def resolve_stage_metrics_csv(stage_dir: Path) -> Path | None:
    """Prefer legacy ``stage_dir/metrics.csv``; else newest ``stage_dir/run_*/metrics.csv``."""
    legacy = stage_dir / "metrics.csv"
    if legacy.is_file():
        return legacy
    candidates = list(stage_dir.glob("run_*/metrics.csv"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def load_metrics_csv(path: Path) -> tuple[list[int], list[float], list[float]]:
    epochs: list[int] = []
    train_loss: list[float] = []
    val_loss: list[float] = []
    if not path.is_file():
        return epochs, train_loss, val_loss
    with path.open(newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            epochs.append(int(row["epoch"]))
            train_loss.append(float(row["train_loss"]))
            val_loss.append(float(row["val_loss"]))
    return epochs, train_loss, val_loss


def main() -> None:
    p = argparse.ArgumentParser(description="Plot training loss curves from metrics.csv")
    p.add_argument(
        "--checkpoints-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "checkpoints",
    )
    p.add_argument("--tag", type=str, default="", help="Same --tag used in train_teleop.py")
    p.add_argument("-o", "--output", type=Path, default=None, help="PNG path (default: checkpoints/training_loss.png)")
    args = p.parse_args()

    base = args.checkpoints_dir
    suffix = f"_{args.tag}" if args.tag else ""

    stages = []
    for s in (1, 2, 3):
        d = base / f"stage{s}{suffix}"
        csv_path = resolve_stage_metrics_csv(d)
        if csv_path is None:
            ep, tr, va = [], [], []
        else:
            ep, tr, va = load_metrics_csv(csv_path)
        stages.append((s, csv_path, ep, tr, va))

    any_data = any(ep for _, _, ep, _, _ in stages)
    if not any_data:
        print("No metrics.csv found. Train with train_teleop.py (it writes metrics.csv each epoch).")
        for s, path, ep, _, _ in stages:
            d = base / f"stage{s}{suffix}"
            hint = path if path is not None else f"{d}/metrics.csv or {d}/run_*/metrics.csv"
            print(f"  expected: {hint} ({'missing' if path is None or not path.is_file() else 'empty'})")
        raise SystemExit(1)

    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        print("Install matplotlib: pip install matplotlib")
        raise SystemExit(1) from e

    out = args.output or (base / "training_loss.png")
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
    colors = ("#2e7d32", "#1565c0", "#c62828")
    for ax, (s, _, ep, tr, va), c in zip(axes, stages, colors):
        if not ep:
            ax.set_title(f"Stage {s} (no metrics.csv)")
            ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
            continue
        ax.plot(ep, tr, color=c, alpha=0.85, label="train", linewidth=1.6)
        ax.plot(ep, va, color=c, alpha=0.55, linestyle="--", label="val", linewidth=1.6)
        ax.set_xlabel("epoch")
        ax.set_title(f"Stage {s}")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)

    axes[0].set_ylabel("loss (MSE)")
    fig.suptitle("LilaEMG teleop training — loss per stage", fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Wrote {out}")

    # Combined val loss vs global epoch (stages concatenated)
    combined_path = base / "training_loss_val_combined.png"
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    offset = 0
    boundaries: list[float] = []
    for (s, _, ep, tr, va), c in zip(stages, colors):
        if not ep:
            continue
        gx = [offset + e for e in ep]
        ax2.plot(gx, va, color=c, label=f"stage {s} val", linewidth=1.8)
        offset += max(ep)
        boundaries.append(offset + 0.5)
    for b in boundaries[:-1]:
        ax2.axvline(b, color="gray", linestyle=":", linewidth=1, alpha=0.7)
    ax2.set_xlabel("global epoch (stage1, then stage2, then stage3)")
    ax2.set_ylabel("val loss (MSE)")
    ax2.set_title("Validation loss across stages")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper right")
    fig2.tight_layout()
    fig2.savefig(combined_path, dpi=150, bbox_inches="tight")
    print(f"Wrote {combined_path}")


if __name__ == "__main__":
    main()
