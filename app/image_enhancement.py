"""Raster-to-RGB normalization used for display and export."""
from __future__ import annotations

import numpy as np


def to_uint8_rgb(arr: np.ndarray) -> np.ndarray:
    """Normalize any raster array to uint8 RGB for display/export."""
    if arr is None:
        raise ValueError("Image array is None")
    data = np.asarray(arr)
    if data.ndim == 2:
        data = data[:, :, None]
    if data.ndim != 3:
        raise ValueError(f"Unsupported raster dimensions: {data.shape}")
    if data.shape[2] == 1:
        data = np.repeat(data, 3, axis=2)
    if data.shape[2] > 3:
        data = data[:, :, :3]
    if data.dtype == np.uint8:
        return data.copy()
    out = np.zeros(data.shape[:2] + (3,), dtype=np.float32)
    for c in range(3):
        band = data[:, :, c].astype(np.float32, copy=False)
        finite = np.isfinite(band)
        if not finite.any():
            continue
        valid = band[finite]
        lo, hi = np.nanpercentile(valid, (2, 98))
        if hi <= lo:
            lo, hi = np.nanmin(valid), np.nanmax(valid)
        if hi > lo:
            out[:, :, c] = (band - lo) / (hi - lo) * 255.0
    out = np.nan_to_num(out, nan=0.0, posinf=255.0, neginf=0.0)
    return np.clip(out, 0, 255).astype(np.uint8)
