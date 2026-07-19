"""Tiling for running object detection on large rasters.

The active raster is already fully loaded in memory by ``app.raster_loader``
(consistent with how the rest of Jagad View handles rasters), so tiling here
slices/pads in-memory numpy arrays rather than doing windowed disk I/O.

This is the same tiling scheme as ``afe/tiling.py`` (overlapping windows,
clipped/padded at raster edges); TCS does not need AFE's ``TileMerger``
because detections are boxes to be merged via cross-tile NMS
(``tcs/postprocess.py``), not per-pixel probabilities to be averaged.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass(frozen=True)
class TileWindow:
    col_off: int
    row_off: int
    width: int
    height: int


def compute_tile_windows(width: int, height: int, tile_size: int, overlap: int) -> List[TileWindow]:
    """Compute overlapping tile windows covering the full width/height.

    Tiles are clipped at the raster edges (no padding at this stage); padding
    to a full ``tile_size`` square happens in :func:`extract_tile`.
    """
    if tile_size <= 0:
        raise ValueError("tile_size harus > 0")
    overlap = max(0, min(overlap, tile_size - 1))
    stride = tile_size - overlap

    windows: List[TileWindow] = []
    row_offsets = list(range(0, max(height - tile_size, 0) + 1, stride)) or [0]
    if row_offsets[-1] + tile_size < height:
        row_offsets.append(max(height - tile_size, 0))
    col_offsets = list(range(0, max(width - tile_size, 0) + 1, stride)) or [0]
    if col_offsets[-1] + tile_size < width:
        col_offsets.append(max(width - tile_size, 0))

    for row_off in row_offsets:
        tile_h = min(tile_size, height - row_off)
        for col_off in col_offsets:
            tile_w = min(tile_size, width - col_off)
            windows.append(TileWindow(col_off=col_off, row_off=row_off, width=tile_w, height=tile_h))
    return windows


def extract_tile(array: np.ndarray, window: TileWindow, tile_size: int) -> np.ndarray:
    """Return a ``tile_size`` x ``tile_size`` (x bands) tile, zero-padded at edges."""
    row0, row1 = window.row_off, window.row_off + window.height
    col0, col1 = window.col_off, window.col_off + window.width
    chunk = array[row0:row1, col0:col1, ...]
    if window.height == tile_size and window.width == tile_size:
        return chunk
    pad_shape = (tile_size, tile_size) + chunk.shape[2:]
    padded = np.zeros(pad_shape, dtype=chunk.dtype)
    padded[: window.height, : window.width, ...] = chunk
    return padded
