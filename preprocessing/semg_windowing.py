"""
Sliding-window sEMG only (no joints, PCA, or language).

For each ``data/<name>.hdf5`` stem, loads ``sessions/<name>/semg_filtered_norm.npy`` (T×8),
optionally checks T matches joint-angle length in the HDF5, applies the same windowing as
``preprocessing_kinematics.py`` (see ``window_utils``), and writes:

  ``<out-dir>/<name>/emg_windows_full.npy`` — float32, (N, 40, 8) with
  ``N = floor((T − 40) / 6) + 1`` (0 rows if ``T < 40``).

Run **before** ``preprocessing_kinematics.py``, which reads this file, trims the last window
to align with velocity, and writes final ``emg_window.npy`` plus hand state / velocity / language.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np

from window_utils import WINDOW_SIZE, STRIDE, window_time_series

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
SESSIONS_DIR = PROJECT_ROOT / "sessions"
SEMG_INPUT_NPY_NAME = "semg_filtered_norm.npy"
OUT_DIR = PROJECT_ROOT / "preprocessed_sessions"
EMG_WINDOWS_FULL_NAME = "emg_windows_full.npy"


def load_semg_session(stem: str, sessions_dir: Path, semg_name: str) -> np.ndarray:
    semg_path = sessions_dir / stem / semg_name
    if not semg_path.is_file():
        raise FileNotFoundError(
            f"Missing {semg_path}. Prepare sessions first (from repo root):\n"
            f"  python preprocessing/export_sessions_npy.py\n"
            f"  python preprocessing/filter_semg_sessions.py\n"
            f"  python preprocessing/normalize_semg_sessions.py"
        )
    emg = np.asarray(np.load(semg_path), dtype=np.float32)
    if emg.ndim != 2 or emg.shape[1] != 8:
        raise ValueError(f"{semg_path}: expected (T, 8), got {emg.shape}")
    return emg


def joint_angle_length(hdf5_path: Path) -> int:
    with h5py.File(hdf5_path, "r") as f:
        ja = f["emg2pose/timeseries"]["joint_angles"]
        return int(ja.shape[0])


def main() -> None:
    p = argparse.ArgumentParser(description="Window normalized sEMG into emg_windows_full.npy per session.")
    p.add_argument("--data-dir", type=Path, default=DATA_DIR, help="HDF5 recordings (default: <repo>/data)")
    p.add_argument(
        "--sessions-dir",
        type=Path,
        default=SESSIONS_DIR,
        help="Session folders with semg_filtered_norm.npy (default: <repo>/sessions)",
    )
    p.add_argument(
        "--semg-name",
        type=str,
        default=SEMG_INPUT_NPY_NAME,
        help="Input sEMG filename inside each session folder",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=OUT_DIR,
        help="Where to write <stem>/emg_windows_full.npy (default: <repo>/preprocessed_sessions)",
    )
    p.add_argument(
        "--no-hdf5-length-check",
        action="store_true",
        help="Do not verify sEMG T equals joint_angles length in each HDF5",
    )
    args = p.parse_args()

    data_dir = args.data_dir.resolve()
    sessions_dir = args.sessions_dir.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = sorted(data_dir.glob("*.hdf5"))
    if not paths:
        raise SystemExit(f"No *.hdf5 files in {data_dir}")

    window, stride = WINDOW_SIZE, STRIDE

    for path in paths:
        stem = path.stem
        emg = load_semg_session(stem, sessions_dir, args.semg_name)
        if not args.no_hdf5_length_check:
            t_ja = joint_angle_length(path)
            if emg.shape[0] != t_ja:
                raise ValueError(
                    f"{path.name}: sEMG T={emg.shape[0]} vs joint_angles T={t_ja} "
                    f"({sessions_dir / stem / args.semg_name})"
                )
        emg_w = window_time_series(emg, window, stride)
        session_dir = out_dir / stem
        session_dir.mkdir(parents=True, exist_ok=True)
        out_path = session_dir / EMG_WINDOWS_FULL_NAME
        np.save(out_path, emg_w)
        print(
            f"{path.name}  ->  {out_path.name}  T={emg.shape[0]}  "
            f"emg_windows_full{emg_w.shape}  window={window} stride={stride}"
        )


if __name__ == "__main__":
    main()
