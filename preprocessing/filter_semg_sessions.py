"""
Filter sEMG per session (``sessions/<name>/semg.npy`` → ``semg_filtered.npy``).

Default pipeline (edit ``DEFAULT_PIPELINE``):

1. **highpass** — attenuate slow drift / motion artefacts below ``cutoff`` Hz (Butterworth SOS).
2. **bandpass** — keep EMG band ``[low, high]`` Hz (Butterworth SOS).
3. **notch** — remove mains interference at ``cutoff`` Hz with given ``bandwidth`` (IIR notch, Q = f0 / bandwidth).

Uses zero-phase filtering (``sosfiltfilt`` / ``filtfilt``) along time (axis 0), independently per channel.

Sampling rate defaults to **200 Hz** (project EMG rate). If bandpass ``high`` is above Nyquist
(``fs/2``), it is **clamped** with a warning — e.g. ``[20, 450]`` requires a much higher ``--fs``; at 200 Hz the
upper band is limited to ~98 Hz.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Any

import numpy as np
from scipy import signal

PROJECT_ROOT = Path(__file__).resolve().parent.parent

_bandpass_clamp_warned = False

# Example-style configs (applied in list order).
DEFAULT_PIPELINE: list[dict[str, Any]] = [
    {"name": "highpass", "cutoff": 20, "order": 2},
    {"name": "bandpass", "cutoff": [20, 450], "order": 4},
    {"name": "notch", "cutoff": 60, "bandwidth": 3},
]


def _clamp_band_edges(lo: float, hi: float, fs: float) -> tuple[float, float]:
    global _bandpass_clamp_warned
    nyq = 0.5 * fs
    max_usable = 0.49 * fs
    if lo <= 0 or lo >= nyq:
        raise ValueError(f"Invalid bandpass low {lo} Hz for fs={fs} (Nyquist {nyq} Hz).")
    if hi > max_usable:
        if not _bandpass_clamp_warned:
            warnings.warn(
                f"Bandpass high {hi} Hz > usable {max_usable:.2f} Hz for fs={fs} Hz; clamping.",
                UserWarning,
                stacklevel=2,
            )
            _bandpass_clamp_warned = True
        hi = max_usable
    if hi <= lo:
        raise ValueError(f"Invalid bandpass [{lo}, {hi}] after clamping.")
    return lo, hi


def _sos_highpass(cut_hz: float, order: int, fs: float) -> np.ndarray:
    return signal.butter(order, cut_hz, btype="highpass", fs=fs, output="sos")


def _sos_bandpass(lo: float, hi: float, order: int, fs: float) -> np.ndarray:
    lo, hi = _clamp_band_edges(lo, hi, fs)
    return signal.butter(order, [lo, hi], btype="bandpass", fs=fs, output="sos")


def _sos_notch(f0: float, bandwidth: float, fs: float) -> np.ndarray:
    if f0 <= 0 or f0 >= 0.5 * fs:
        raise ValueError(f"Notch frequency {f0} Hz invalid for fs={fs} Hz.")
    if bandwidth <= 0:
        raise ValueError("Notch bandwidth must be positive.")
    Q = f0 / bandwidth
    b, a = signal.iirnotch(w0=f0, Q=Q, fs=fs)
    return signal.tf2sos(b, a)


def build_sos_chain(pipeline: list[dict[str, Any]], fs: float) -> list[np.ndarray]:
    """Return list of SOS arrays, each suitable for ``sosfiltfilt``."""
    stages: list[np.ndarray] = []
    for d in pipeline:
        name = d["name"]
        if name == "highpass":
            stages.append(_sos_highpass(float(d["cutoff"]), int(d["order"]), fs))
        elif name == "bandpass":
            lo, hi = float(d["cutoff"][0]), float(d["cutoff"][1])
            stages.append(_sos_bandpass(lo, hi, int(d["order"]), fs))
        elif name == "notch":
            stages.append(
                _sos_notch(float(d["cutoff"]), float(d["bandwidth"]), fs)
            )
        else:
            raise ValueError(f"Unknown filter name: {name!r}")
    return stages


def filter_semg(x: np.ndarray, sos_chain: list[np.ndarray], axis: int = 0) -> np.ndarray:
    """
    ``x``: (T, C) EMG samples. Applies each SOS stage with zero-phase filtering.
    """
    y = np.asarray(x, dtype=np.float64)
    for sos in sos_chain:
        y = signal.sosfiltfilt(sos, y, axis=axis)
    return y.astype(np.float32, copy=False)


def process_session(
    session_dir: Path,
    *,
    out_name: str,
    fs: float,
    pipeline: list[dict[str, Any]],
) -> None:
    semg_path = session_dir / "semg.npy"
    if not semg_path.is_file():
        raise FileNotFoundError(f"Missing {semg_path}")

    x = np.load(semg_path)
    if x.ndim != 2:
        raise ValueError(f"{semg_path}: expected 2D (T, C), got {x.shape}")

    sos_chain = build_sos_chain(pipeline, fs)
    y = filter_semg(x, sos_chain, axis=0)
    out_path = session_dir / out_name
    np.save(out_path, y)
    print(f"{session_dir.name}:  semg{x.shape}  ->  {out_path.name}  pipeline={len(sos_chain)} stages")


def main() -> None:
    p = argparse.ArgumentParser(description="Butterworth / notch filtering for sessions/*/semg.npy")
    p.add_argument(
        "--sessions-dir",
        type=Path,
        default=PROJECT_ROOT / "sessions",
        help="Folder with one subfolder per recording containing semg.npy (default: <repo>/sessions)",
    )
    p.add_argument(
        "--fs",
        type=float,
        default=200.0,
        help="EMG sampling rate in Hz (default: 200, matches project preprocessing)",
    )
    p.add_argument(
        "--out-name",
        type=str,
        default="semg_filtered.npy",
        help="Output filename inside each session folder (default: semg_filtered.npy)",
    )
    args = p.parse_args()
    root = args.sessions_dir.resolve()
    if not root.is_dir():
        raise SystemExit(f"Not a directory: {root}")

    dirs = sorted([d for d in root.iterdir() if d.is_dir()])
    if not dirs:
        raise SystemExit(f"No session subfolders in {root}")

    for d in dirs:
        process_session(
            d,
            out_name=args.out_name,
            fs=args.fs,
            pipeline=DEFAULT_PIPELINE,
        )


if __name__ == "__main__":
    main()
