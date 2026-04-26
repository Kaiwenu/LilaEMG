"""
Z-score normalize ``semg_filtered.npy`` per session (channel-wise).

**Option A (common):** mean and standard deviation are computed from **training sessions only**
(same split policy as ``train_teleop.split_train_val_test``), then applied to every session:

    x_norm = (x - mean) / std

with ``std`` clamped to at least ``1e-6`` per channel.

Writes:

  - ``<sessions-dir>/<stem>/semg_filtered_norm.npy`` — float32, same shape as input
  - ``--stats-out`` (default ``<sessions-dir>/semg_zscore_train_stats.npz``) — ``mean``, ``std`` (each shape (8,))
  - ``--meta-out`` (default ``<sessions-dir>/semg_zscore_train_stats.json``) — train/val/test stems and paths
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from train_teleop import (  # noqa: E402
    discover_session_dirs,
    group_sessions_by_gesture,
    split_train_val_test,
)

STD_FLOOR = 1e-6


def main() -> None:
    p = argparse.ArgumentParser(description="Z-score semg_filtered.npy using train-split stats.")
    p.add_argument(
        "--sessions-dir",
        type=Path,
        default=PROJECT_ROOT / "sessions",
        help="Folder with one subfolder per recording (default: <repo>/sessions)",
    )
    p.add_argument(
        "--in-name",
        type=str,
        default="semg_filtered.npy",
        help="Input filename in each session folder (default: semg_filtered.npy)",
    )
    p.add_argument(
        "--out-name",
        type=str,
        default="semg_filtered_norm.npy",
        help="Output filename per session (default: semg_filtered_norm.npy)",
    )
    p.add_argument(
        "--stats-out",
        type=Path,
        default=None,
        help="Where to save mean/std npz (default: <sessions-dir>/semg_zscore_train_stats.npz)",
    )
    p.add_argument(
        "--meta-out",
        type=Path,
        default=None,
        help="JSON with train/val/test stems (default: <sessions-dir>/semg_zscore_train_stats.json)",
    )
    args = p.parse_args()

    sessions_root = args.sessions_dir.resolve()
    if not sessions_root.is_dir():
        raise SystemExit(f"Not a directory: {sessions_root}")

    session_dirs = discover_session_dirs(sessions_root)
    if not session_dirs:
        raise SystemExit(f"No session subfolders in {sessions_root}")

    grouped = group_sessions_by_gesture(session_dirs)
    train_stems, val_stems, test_stems = split_train_val_test(grouped)

    train_paths = [p for p in session_dirs if p.name in train_stems]
    if not train_paths:
        raise SystemExit("No training sessions in split; cannot compute stats.")

    chunks: list[np.ndarray] = []
    for d in train_paths:
        src = d / args.in_name
        if not src.is_file():
            raise FileNotFoundError(f"Missing {src} (required for train stats)")
        x = np.asarray(np.load(src), dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != 8:
            raise ValueError(f"{src}: expected (T, 8), got {x.shape}")
        chunks.append(x)

    X = np.vstack(chunks)
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std = np.maximum(std, STD_FLOOR)

    stats_path = args.stats_out
    if stats_path is None:
        stats_path = sessions_root / "semg_zscore_train_stats.npz"
    else:
        stats_path = stats_path.resolve()
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        stats_path,
        mean=mean.astype(np.float32),
        std=std.astype(np.float32),
    )
    print(
        f"Pooled train sEMG: N={X.shape[0]} time steps, 8 channels  "
        f"(from {len(train_paths)} sessions). Wrote {stats_path}"
    )

    meta_path = args.meta_out
    if meta_path is None:
        meta_path = sessions_root / "semg_zscore_train_stats.json"
    else:
        meta_path = meta_path.resolve()
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "train_stems": sorted(train_stems),
                "val_stems": sorted(val_stems),
                "test_stems": sorted(test_stems),
                "in_name": args.in_name,
                "out_name": args.out_name,
                "stats_npz": str(stats_path),
            },
            f,
            indent=2,
        )
        f.write("\n")
    print(f"Wrote split meta {meta_path}")

    mean_f = mean.astype(np.float32)
    std_f = std.astype(np.float32)

    for d in session_dirs:
        src = d / args.in_name
        if not src.is_file():
            print(f"Skip {d.name}: missing {args.in_name}", file=sys.stderr)
            continue
        x = np.asarray(np.load(src), dtype=np.float32)
        if x.ndim != 2 or x.shape[1] != 8:
            raise ValueError(f"{src}: expected (T, 8), got {x.shape}")
        out = (x - mean_f) / std_f
        out_path = d / args.out_name
        np.save(out_path, out.astype(np.float32, copy=False))
        split = (
            "train"
            if d.name in train_stems
            else "val"
            if d.name in val_stems
            else "test"
            if d.name in test_stems
            else "?"
        )
        print(f"{d.name}  [{split}]  {src.name} -> {out_path.name}  {x.shape}")


if __name__ == "__main__":
    main()
