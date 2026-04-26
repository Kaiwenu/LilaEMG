"""
Extract only the sEMG array from each HDF5 under ``data/`` and save as ``.npy``.

Reads:  ``<name>.hdf5`` → dataset ``emg2pose/timeseries``, field ``emg``  (shape T×8).

Writes: ``<out_dir>/<name>_emg.npy``  (float32 array, no other processing).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    p = argparse.ArgumentParser(description="Extract EMG from HDF5 to NumPy files.")
    p.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "data",
        help="Folder containing *.hdf5 (default: <repo>/data)",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "emg_npy",
        help="Where to write <stem>_emg.npy (default: <repo>/emg_npy)",
    )
    args = p.parse_args()
    data_dir = args.data_dir.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = sorted(data_dir.glob("*.hdf5"))
    if not paths:
        raise SystemExit(f"No *.hdf5 files in {data_dir}")

    for path in paths:
        with h5py.File(path, "r") as f:
            emg = np.asarray(f["emg2pose/timeseries"]["emg"][:], dtype=np.float32)
        out_path = out_dir / f"{path.stem}_emg.npy"
        np.save(out_path, emg)
        print(f"{path.name}  ->  {out_path}  shape={emg.shape} dtype={emg.dtype}")


if __name__ == "__main__":
    main()
