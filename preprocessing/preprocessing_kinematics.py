"""
Joint angles, PCA hand state, velocity, and per-session language vectors (from table).

Expects **sEMG windows** from ``semg_windowing.py``:

  ``<out-dir>/<stem>/emg_windows_full.npy`` — (N, 40, 8)

Expects **language embedding table** from ``language_preprocessing.py``:

  ``<out-dir>/language_embedding_table.npy`` — (len(GESTURES), D), rows in ``GESTURES`` order.

For each ``data/<name>.hdf5``, loads joint angles (T×20), applies the **same** sliding window
(``window_utils.WINDOW_SIZE``, ``STRIDE``), pools joints per window, fits **global** PCA on
all ``ja_mean`` rows, computes PCA-space velocity, aligns lengths by dropping the last EMG
window and last PCA row, and writes:

  **Per session** under ``<out-dir>/<stem>/``:
  - ``emg_window.npy`` — (N−1, 40, 8)
  - ``hand_state.npy``, ``hand_velocity.npy``, ``language.npy`` (one row from the table)

  **Top-level** under ``<out-dir>/``:
  - ``pca_joint_angles.joblib``

Run **after** ``python preprocessing/semg_windowing.py`` and
``python preprocessing/language_preprocessing.py``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import joblib
import numpy as np
from sklearn.decomposition import PCA

from language_preprocessing import GESTURES, LANG_TABLE_NAME, parse_recording_name
from window_utils import WINDOW_SIZE, STRIDE, window_time_series

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
OUT_DIR = PROJECT_ROOT / "preprocessed_sessions"

JOINT_DIM = 20
PCA_VARIANCE_TARGET = 0.95
VELOCITY_DT_MS = 30.0
VELOCITY_DT_S = VELOCITY_DT_MS / 1000.0

EMG_WINDOWS_FULL_NAME = "emg_windows_full.npy"


def load_joint_angles(hdf5_path: Path) -> np.ndarray:
    with h5py.File(hdf5_path, "r") as f:
        return np.asarray(f["emg2pose/timeseries"]["joint_angles"][:], dtype=np.float32)


def mean_per_window_channels(ja_w: np.ndarray) -> np.ndarray:
    """``ja_w``: (N, window, C) -> (N, C), mean over time within each window per channel."""
    return ja_w.mean(axis=1).astype(np.float32)


def pca_velocity(ja_pca: np.ndarray, dt_s: float) -> np.ndarray:
    """``(ja_pca[i] - ja_pca[i-1]) / dt_s`` for i ≥ 1 → shape (N−1, K). Empty if N < 2."""
    if ja_pca.shape[0] < 2:
        k = ja_pca.shape[1]
        return np.empty((0, k), dtype=np.float32)
    d = ja_pca[1:] - ja_pca[:-1]
    return (d / np.float32(dt_s)).astype(np.float32)


def load_emg_windows_full(session_dir: Path) -> np.ndarray:
    path = session_dir / EMG_WINDOWS_FULL_NAME
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path}. Run first (from repo root):\n"
            f"  python preprocessing/semg_windowing.py"
        )
    emg_w = np.asarray(np.load(path), dtype=np.float32)
    if emg_w.ndim != 3 or emg_w.shape[1:] != (WINDOW_SIZE, 8):
        raise ValueError(f"{path}: expected (N, {WINDOW_SIZE}, 8), got {emg_w.shape}")
    return emg_w


def load_language_table(out_dir: Path) -> np.ndarray:
    table_path = out_dir / LANG_TABLE_NAME
    if not table_path.is_file():
        raise FileNotFoundError(
            f"Missing {table_path}. Run first (from repo root):\n"
            f"  python preprocessing/language_preprocessing.py"
        )
    lang_emb_table = np.asarray(np.load(table_path), dtype=np.float32)
    if lang_emb_table.ndim != 2 or lang_emb_table.shape[0] != len(GESTURES):
        raise ValueError(
            f"{table_path}: expected ({len(GESTURES)}, D) language table, got {lang_emb_table.shape}"
        )
    return lang_emb_table


def main() -> None:
    p = argparse.ArgumentParser(description="PCA, velocity, language from joints + emg_windows_full + language table")
    p.add_argument("--data-dir", type=Path, default=DATA_DIR, help="HDF5 recordings (default: <repo>/data)")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=OUT_DIR,
        help="preprocessed_sessions root (default: <repo>/preprocessed_sessions)",
    )
    args = p.parse_args()

    data_dir = args.data_dir.resolve()
    out_dir = args.out_dir.resolve()
    window, stride = WINDOW_SIZE, STRIDE
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = sorted(data_dir.glob("*.hdf5"))
    if not paths:
        raise SystemExit(f"No *.hdf5 files in {data_dir}")

    lang_emb_table = load_language_table(out_dir)
    gesture_index = {g: i for i, g in enumerate(GESTURES)}

    sessions: list[tuple[Path, int, np.ndarray, np.ndarray]] = []
    ja_for_pca: list[np.ndarray] = []

    for path in paths:
        stem = path.stem
        session_dir = out_dir / stem
        emg_w = load_emg_windows_full(session_dir)

        joint_angles = load_joint_angles(path)
        if joint_angles.shape[1] != JOINT_DIM:
            raise ValueError(
                f"{path.name}: expected joint_angles with {JOINT_DIM} channels, got {joint_angles.shape[1]}"
            )
        t_len = joint_angles.shape[0]
        ja_w = window_time_series(joint_angles, window, stride)
        if ja_w.shape[0] != emg_w.shape[0]:
            raise ValueError(
                f"{path.name}: joint windows N={ja_w.shape[0]} vs "
                f"{EMG_WINDOWS_FULL_NAME} N={emg_w.shape[0]} (check T and window_utils match semg_windowing)"
            )
        ja_mean = mean_per_window_channels(ja_w)
        sessions.append((path, t_len, emg_w, ja_mean))
        if ja_mean.shape[0] > 0:
            ja_for_pca.append(ja_mean)

    if not ja_for_pca:
        raise SystemExit("No joint-angle windows across sessions; cannot fit PCA.")

    X = np.vstack(ja_for_pca)
    pca = PCA(n_components=PCA_VARIANCE_TARGET, svd_solver="full")
    pca.fit(X)
    k = int(pca.n_components_)
    ev = float(pca.explained_variance_ratio_.sum())
    print(
        f"PCA on ja_mean: n_samples={X.shape[0]}, n_components={k}, "
        f"cumulative explained variance={ev:.4f} (target>={PCA_VARIANCE_TARGET})"
    )

    pca_path = out_dir / "pca_joint_angles.joblib"
    joblib.dump(pca, pca_path)
    print(f"Wrote {pca_path}")

    dt_s = VELOCITY_DT_S
    for path, t_len, emg_w, ja_mean in sessions:
        gesture, _session_num = parse_recording_name(path.stem)
        lang_vec = lang_emb_table[gesture_index[gesture]].copy()

        if ja_mean.shape[0] == 0:
            ja_pca = np.empty((0, k), dtype=np.float32)
        else:
            ja_pca = pca.transform(ja_mean).astype(np.float32)
        ja_vel = pca_velocity(ja_pca, dt_s)
        emg_w = emg_w[:-1]
        ja_pca = ja_pca[:-1]
        session_dir = out_dir / path.stem
        session_dir.mkdir(parents=True, exist_ok=True)
        np.save(session_dir / "emg_window.npy", emg_w)
        np.save(session_dir / "hand_state.npy", ja_pca)
        np.save(session_dir / "hand_velocity.npy", ja_vel)
        np.save(session_dir / "language.npy", lang_vec)
        print(
            f"{path.name}  ->  {session_dir.name}/"
            f"emg_window.npy, hand_state.npy, hand_velocity.npy, language.npy  "
            f"T={t_len}  N={emg_w.shape[0]}  (aligned)  window={window} stride={stride}  "
            f"emg_window{emg_w.shape}  hand_state{ja_pca.shape}  hand_velocity{ja_vel.shape}  "
            f"language{lang_vec.shape}  gesture={gesture!r}  dt={VELOCITY_DT_MS:g} ms"
        )


if __name__ == "__main__":
    main()
