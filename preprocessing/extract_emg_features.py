"""
Extract time-domain features from preprocessed EMG windows.

Reads ``preprocessed_sessions/<session>/emg_window.npy`` — float32, shape (N, 40, 8)
— and writes ``emg_features.npy`` in the same folder.

Per window and per channel (8 channels), over the 40 time samples:
  - MAV — mean absolute value
  - RMS — root mean square
  - WL — waveform length (sum of absolute successive differences)
  - ZC — zero crossings (sign change between consecutive samples)
  - SSC — slope sign changes
  - VAR — variance

Output array: float32, shape (N, 48) — 6 features × 8 channels (channel-major).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent

FEATURE_NAMES: tuple[str, ...] = ("mav", "rms", "wl", "zc", "ssc", "var")
N_FEATURES = len(FEATURE_NAMES)


def time_domain_features(emg: np.ndarray) -> np.ndarray:
    """
    ``emg``: (N, T, C) — windows, time, channels.

    Returns float32 array of shape (N, C * N_FEATURES), features grouped by channel
    (all 6 features for ch0, then ch1, ...).
    """
    if emg.ndim != 3:
        raise ValueError(f"expected (N, T, C), got shape {emg.shape}")
    n, t, c = emg.shape
    if t < 3:
        raise ValueError(f"need T>=3 for SSC, got T={t}")

    x = emg.astype(np.float64, copy=False)

    mav = np.mean(np.abs(x), axis=1)
    rms = np.sqrt(np.mean(x**2, axis=1))
    wl = np.sum(np.abs(np.diff(x, axis=1)), axis=1)
    var = np.var(x, axis=1)

    a, b = x[:, :-1, :], x[:, 1:, :]
    zc = np.sum((a * b) < 0, axis=1).astype(np.float64)

    xm = x[:, 1:-1, :]
    left = xm - x[:, :-2, :]
    right = xm - x[:, 2:, :]
    ssc = np.sum((left * right) < 0, axis=1).astype(np.float64)

    # (N, C, 6) -> (N, C*6)
    stack = np.stack((mav, rms, wl, zc, ssc, var), axis=-1)
    out = stack.reshape(n, c * N_FEATURES)
    return out.astype(np.float32)


def process_session(session_dir: Path, dry_run: bool) -> tuple[Path, tuple[int, ...] | None]:
    emg_path = session_dir / "emg_window.npy"
    if not emg_path.is_file():
        return emg_path, None

    emg = np.load(emg_path, mmap_mode="r")
    if emg.ndim != 3:
        raise ValueError(f"{emg_path}: expected 3D (N,T,C), got {emg.shape}")
    if emg.shape[0] == 0:
        feat = np.empty((0, emg.shape[2] * N_FEATURES), dtype=np.float32)
    else:
        emg_full = np.asarray(emg, dtype=np.float32)
        feat = time_domain_features(emg_full)

    out_path = session_dir / "emg_features.npy"
    if not dry_run:
        np.save(out_path, feat)
    return out_path, feat.shape


def main() -> None:
    p = argparse.ArgumentParser(
        description="Extract per-window EMG features from preprocessed_sessions/*/emg_window.npy"
    )
    p.add_argument(
        "--preprocessed-dir",
        type=Path,
        default=PROJECT_ROOT / "preprocessed_sessions",
        help="Root folder (default: <repo>/preprocessed_sessions)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print shapes only; do not write emg_features.npy",
    )
    args = p.parse_args()
    root = args.preprocessed_dir.resolve()
    if not root.is_dir():
        raise SystemExit(f"Not a directory: {root}")

    # Session folders only: subdirs that contain emg_window.npy
    sessions = sorted(
        d for d in root.iterdir() if d.is_dir() and (d / "emg_window.npy").is_file()
    )
    if not sessions:
        raise SystemExit(f"No session folders with emg_window.npy under {root}")

    print(
        f"Features ({', '.join(FEATURE_NAMES)}): {N_FEATURES} per channel → "
        f"{8 * N_FEATURES} floats per window (channel-major)."
    )
    for session_dir in sessions:
        out_path, shape = process_session(session_dir, args.dry_run)
        if shape is None:
            continue
        action = "would write" if args.dry_run else "wrote"
        print(f"{session_dir.name}:  emg_window.npy → {shape}  {action}  {out_path.name}")


if __name__ == "__main__":
    main()
