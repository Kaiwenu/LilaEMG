"""
Export raw HDF5 timeseries to NumPy files per recording.

Reads each ``data/<name>.hdf5`` from ``emg2pose/timeseries`` (fields ``time``, ``joint_angles``, ``emg``).

Writes under ``sessions/<name>/``:

  - ``semg.npy``   — float32, shape (T, 8)
  - ``joint_angles.npy`` — float32, shape (T, 20)
  - ``time.npy``   — float64, shape (T,) — wall-clock style timestamps (seconds), as stored in HDF5
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def export_one(hdf5_path: Path, out_root: Path) -> None:
    stem = hdf5_path.stem
    session_dir = out_root / stem
    session_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(hdf5_path, "r") as f:
        ts = f["emg2pose/timeseries"]
        t = np.asarray(ts["time"][:], dtype=np.float64)
        ja = np.asarray(ts["joint_angles"][:], dtype=np.float32)
        emg = np.asarray(ts["emg"][:], dtype=np.float32)

    if t.shape[0] != emg.shape[0] or t.shape[0] != ja.shape[0]:
        raise ValueError(
            f"{hdf5_path.name}: length mismatch time={t.shape[0]} emg={emg.shape[0]} joint_angles={ja.shape[0]}"
        )

    np.save(session_dir / "time.npy", t)
    np.save(session_dir / "joint_angles.npy", ja)
    np.save(session_dir / "semg.npy", emg)
    print(
        f"{hdf5_path.name}  ->  {session_dir}/"
        f"time{t.shape}  semg{emg.shape}  joint_angles{ja.shape}"
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Export EMG, joint angles, and time to npy per session.")
    p.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "data",
        help="Folder containing *.hdf5 (default: <repo>/data)",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "sessions",
        help="Root folder for per-session subdirs (default: <repo>/sessions)",
    )
    args = p.parse_args()
    data_dir = args.data_dir.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = sorted(data_dir.glob("*.hdf5"))
    if not paths:
        raise SystemExit(f"No *.hdf5 files in {data_dir}")

    for path in paths:
        export_one(path, out_dir)


if __name__ == "__main__":
    main()
