"""Shared sliding-window parameters and helpers for sEMG and joint-angle pipelines."""

from __future__ import annotations

import numpy as np

WINDOW_SIZE = 40
STRIDE = 6


def window_time_series(x: np.ndarray, window: int = WINDOW_SIZE, stride: int = STRIDE) -> np.ndarray:
    """
    ``x``: (T, C) -> (N, window, C) with starts 0, stride, 2*stride, ...
    """
    t, c = x.shape
    if t < window:
        return np.empty((0, window, c), dtype=x.dtype)
    n = (t - window) // stride + 1
    out = np.empty((n, window, c), dtype=x.dtype)
    for i in range(n):
        s = i * stride
        out[i] = x[s : s + window]
    return out
