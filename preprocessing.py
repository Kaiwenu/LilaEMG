"""
Load LilaEMG HDF5 recordings from ./data, build synergy (PCA) space, and
emit (emg_window, hand_state, hand_velocity, language) samples.

Each file is ``{gesture}_{session}.hdf5`` with group ``emg2pose``:
  - dataset ``timeseries``: compound columns ``time``, ``joint_angles`` (20,),
    ``emg`` (8,) at ``sample_rate`` Hz (200 in current captures).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterator

import joblib
import numpy as np
from sklearn.decomposition import PCA

# ---------------------------------------------------------------------------
# Paths & gesture → natural-language labels (edit phrases to match your task)
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).resolve().parent / "data"
EMG2POSE_GROUP = "emg2pose"
TIMESERIES_DATASET = "timeseries"

GESTURE_LANGUAGE: dict[str, str] = {
    "grasp": "grasp the cup",
    "press": "press the button",
    "spray": "spray",
    "index_pick": "pick with the index finger",
    "single_finger": "single finger movement",
}

# EMG context window: 200 ms at 200 Hz → 40 samples; shape (8, 40) → 320-D flat
EMG_WINDOW_SAMPLES = 40

# Saved artifacts (see ``save_preprocessed_bundle``)
DEFAULT_PREPROCESSED_DIR = Path(__file__).resolve().parent / "preprocessed"
PCA_FILENAME = "pca.joblib"
NPZ_FILENAME = "preprocessed.npz"
META_FILENAME = "preprocessed_meta.json"


def list_hdf5_recordings(data_dir: Path = DATA_DIR) -> list[Path]:
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    paths = sorted(data_dir.glob("*.hdf5"))
    if not paths:
        raise FileNotFoundError(f"No .hdf5 files in {data_dir}")
    return paths


def parse_gesture_session(path: Path) -> tuple[str, int]:
    m = re.match(r"^(.+)_(\d+)\.hdf5$", path.name, re.IGNORECASE)
    if not m:
        raise ValueError(f"Expected name like grasp_1.hdf5, got {path.name!r}")
    return m.group(1).lower(), int(m.group(2))


def load_recording(path: Path) -> dict[str, Any]:
    import h5py

    gesture, session_num = parse_gesture_session(path)
    if gesture not in GESTURE_LANGUAGE:
        raise KeyError(f"No language label for gesture {gesture!r}; add it to GESTURE_LANGUAGE")

    with h5py.File(path, "r") as f:
        g = f[EMG2POSE_GROUP]
        ts = g[TIMESERIES_DATASET]
        block = ts[:]
        sample_rate = float(np.asarray(g.attrs["sample_rate"]).reshape(()))

    joint_angles = np.asarray(block["joint_angles"], dtype=np.float64)
    emg = np.asarray(block["emg"], dtype=np.float64)
    times = np.asarray(block["time"], dtype=np.float64)

    return {
        "path": path,
        "gesture": gesture,
        "session": session_num,
        "language": GESTURE_LANGUAGE[gesture],
        "sample_rate": sample_rate,
        "joint_angles": joint_angles,
        "emg": emg,
        "time": times,
    }


def iter_recordings(paths: list[Path] | None = None) -> Iterator[dict[str, Any]]:
    for p in paths or list_hdf5_recordings():
        yield load_recording(p)


def moving_average_states(states: np.ndarray, window: int) -> np.ndarray:
    """Smooth (T, K) synergy trajectories along time (reduces jitter before diff)."""
    states = np.asarray(states, dtype=np.float64)
    if window <= 1:
        return states.copy()
    pad = (window - 1) // 2
    padded = np.pad(states, ((pad, pad), (0, 0)), mode="edge")
    kernel = np.ones(window, dtype=np.float64) / window
    out = np.empty_like(states)
    for j in range(states.shape[1]):
        out[:, j] = np.convolve(padded[:, j], kernel, mode="valid")
    return out


def collect_joint_angles_for_pca(paths: list[Path] | None = None) -> np.ndarray:
    rows: list[np.ndarray] = []
    for rec in iter_recordings(paths):
        rows.append(rec["joint_angles"])
    return np.vstack(rows)


def fit_synergy_pca(
    joint_angles_all: np.ndarray,
    variance_ratio: float = 0.95,
) -> PCA:
    pca = PCA(n_components=variance_ratio, svd_solver="full")
    pca.fit(joint_angles_all)
    return pca


def build_samples_for_recording(
    rec: dict[str, Any],
    pca: PCA,
    ma_window: int = 5,
    emg_window_samples: int = EMG_WINDOW_SAMPLES,
) -> list[dict[str, Any]]:
    ja = rec["joint_angles"]
    emg = rec["emg"]
    sr = rec["sample_rate"]
    dt = 1.0 / sr

    synergy = pca.transform(ja)
    synergy = moving_average_states(synergy, ma_window)

    n = synergy.shape[0]
    samples: list[dict[str, Any]] = []
    for t in range(n - 1):
        if t < emg_window_samples:
            continue
        s_t = synergy[t]
        s_next = synergy[t + 1]
        delta_q = (s_next - s_t) / dt
        emg_window = emg[t - emg_window_samples : t].T
        if emg_window.shape != (8, emg_window_samples):
            continue
        samples.append(
            {
                "emg_window": emg_window.reshape(-1),
                "hand_state": s_t.astype(np.float32),
                "hand_velocity": delta_q.astype(np.float32),
                "language": rec["language"],
                "gesture": rec["gesture"],
                "session": rec["session"],
                "time_index": t,
            }
        )
    return samples


def build_all_samples(
    pca: PCA,
    paths: list[Path] | None = None,
    ma_window: int = 5,
) -> list[dict[str, Any]]:
    all_samples: list[dict[str, Any]] = []
    for rec in iter_recordings(paths):
        all_samples.extend(build_samples_for_recording(rec, pca, ma_window=ma_window))
    return all_samples


def add_language_embeddings(
    samples: list[dict[str, Any]],
) -> None:
    """In-place: add ``language_embedding`` (768,) per sample using DistilRoBERTa."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("distilroberta-base")
    model = AutoModel.from_pretrained("distilroberta-base")
    model.eval()

    phrases = list({s["language"] for s in samples})
    text_to_emb: dict[str, np.ndarray] = {}
    with torch.no_grad():
        for phrase in phrases:
            tokens = tokenizer(phrase, return_tensors="pt")
            hidden = model(**tokens).last_hidden_state
            emb = hidden.mean(dim=1).squeeze(0).cpu().numpy()
            text_to_emb[phrase] = emb.astype(np.float32)

    for s in samples:
        s["language_embedding"] = text_to_emb[s["language"]]


def save_preprocessed_bundle(
    pca: PCA,
    samples: list[dict[str, Any]],
    out_dir: Path = DEFAULT_PREPROCESSED_DIR,
) -> None:
    """
    Write ``pca.joblib``, ``preprocessed.npz`` (stacked float32 arrays), and
    ``preprocessed_meta.json`` (gesture / language id → string tables).
    """
    if not samples:
        raise ValueError("No samples to save.")
    if not all("language_embedding" in s for s in samples):
        raise ValueError(
            "Every sample must have 'language_embedding'. "
            "Install torch and transformers, or run add_language_embeddings first."
        )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gestures = sorted({str(s["gesture"]) for s in samples})
    languages = sorted({str(s["language"]) for s in samples})
    gesture_to_id = {g: i for i, g in enumerate(gestures)}
    language_to_id = {l: i for i, l in enumerate(languages)}

    n = len(samples)
    emg_window = np.empty((n, 320), dtype=np.float32)
    hand_state = np.empty((n, pca.n_components_), dtype=np.float32)
    hand_velocity = np.empty((n, pca.n_components_), dtype=np.float32)
    language_embedding = np.empty((n, 768), dtype=np.float32)
    gesture_id = np.empty(n, dtype=np.int32)
    language_id = np.empty(n, dtype=np.int32)
    session = np.empty(n, dtype=np.int32)
    time_index = np.empty(n, dtype=np.int32)

    for i, s in enumerate(samples):
        ew = np.asarray(s["emg_window"], dtype=np.float32).reshape(-1)
        if ew.shape != (320,):
            raise ValueError(f"Sample {i}: expected emg_window length 320, got {ew.shape}")
        emg_window[i] = ew

        hs = np.asarray(s["hand_state"], dtype=np.float32).reshape(-1)
        hv = np.asarray(s["hand_velocity"], dtype=np.float32).reshape(-1)
        if hs.shape[0] != pca.n_components_ or hv.shape[0] != pca.n_components_:
            raise ValueError(
                f"Sample {i}: hand_state/hand_velocity dim {hs.shape[0]} != PCA K {pca.n_components_}"
            )
        hand_state[i] = hs
        hand_velocity[i] = hv

        le = np.asarray(s["language_embedding"], dtype=np.float32).reshape(-1)
        if le.shape != (768,):
            raise ValueError(f"Sample {i}: expected language_embedding length 768, got {le.shape}")
        language_embedding[i] = le

        gesture_id[i] = gesture_to_id[str(s["gesture"])]
        language_id[i] = language_to_id[str(s["language"])]
        session[i] = int(s["session"])
        time_index[i] = int(s["time_index"])

    joblib.dump(pca, out_dir / PCA_FILENAME)
    np.savez_compressed(
        out_dir / NPZ_FILENAME,
        emg_window=emg_window,
        hand_state=hand_state,
        hand_velocity=hand_velocity,
        language_embedding=language_embedding,
        gesture_id=gesture_id,
        language_id=language_id,
        session=session,
        time_index=time_index,
    )

    meta = {
        "n_samples": n,
        "n_synergy_components": int(pca.n_components_),
        "emg_window_length": 320,
        "language_embedding_dim": 768,
        "gestures": gestures,
        "languages": languages,
        "note": "gesture_id[i] indexes meta['gestures']; language_id[i] indexes meta['languages'].",
    }
    with (out_dir / META_FILENAME).open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Wrote {out_dir / PCA_FILENAME}")
    print(f"Wrote {out_dir / NPZ_FILENAME}")
    print(f"Wrote {out_dir / META_FILENAME}")


def main() -> None:
    paths = list_hdf5_recordings()
    print(f"Found {len(paths)} recordings under {DATA_DIR}")

    joint_all = collect_joint_angles_for_pca(paths)
    print(f"Joint-angle matrix for PCA: {joint_all.shape}")

    pca = fit_synergy_pca(joint_all)
    k = pca.n_components_
    explained = float(np.sum(pca.explained_variance_ratio_))
    print(f"PCA synergy dim K = {k}, cumulative explained variance = {explained:.4f}")

    samples = build_all_samples(pca, paths=paths)
    print(f"Built {len(samples)} training samples (after EMG window warmup)")

    try:
        add_language_embeddings(samples)
        print("Attached DistilRoBERTa language embeddings (768-D).")
    except ImportError:
        print("Skipping language embeddings: install torch and transformers for step 3.")

    if samples and all("language_embedding" in s for s in samples):
        save_preprocessed_bundle(pca, samples, DEFAULT_PREPROCESSED_DIR)
    elif samples:
        print(
            "Skipping save: add torch + transformers to write "
            f"{DEFAULT_PREPROCESSED_DIR / NPZ_FILENAME} and {DEFAULT_PREPROCESSED_DIR / PCA_FILENAME}."
        )

    if samples:
        s0 = samples[0]
        print(
            "Example sample:",
            {k: (v.shape if hasattr(v, "shape") else type(v).__name__) for k, v in s0.items()},
        )


if __name__ == "__main__":
    main()
