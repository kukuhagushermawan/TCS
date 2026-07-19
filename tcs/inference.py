"""Object-detection inference backends for TCS.

``TreeDetector`` is the swap point: the UI/worker never depends on a specific
framework, only on ``detect_tile``. Two backends are provided:

- ``PlaceholderTreeDetector``: a dependency-free heuristic so the full TCS
  pipeline (tiling -> detect -> NMS -> points -> layer) is runnable and
  demoable before a trained model exists.
- ``YoloTreeDetector``: loads a real Ultralytics YOLO (v5/v8) model from a
  ``.pt`` weight file. Swap in another detector (Faster R-CNN, etc.) by
  adding another backend class with the same ``detect_tile`` signature.
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .config import DEFAULT_CROWN_RADIUS_PX, DEFAULT_CROWN_SPACING_PX

try:
    from rasterio.features import shapes as rasterio_shapes
except Exception:  # pragma: no cover
    rasterio_shapes = None

from shapely.geometry import shape as shapely_shape


@dataclass
class Detection:
    """A single detected object in tile-pixel space (0,0 = tile top-left)."""

    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float


class TreeDetector(ABC):
    """Common interface every TCS object-detection backend must implement."""

    @abstractmethod
    def detect_tile(self, tile_rgb: np.ndarray) -> List[Detection]:
        """Return detected bounding boxes for one tile.

        tile_rgb: uint8 array, shape (tile_size, tile_size, 3).
        returns: list of Detection, coordinates in tile-pixel space.
        """
        raise NotImplementedError


def _excess_green_index(tile_rgb: np.ndarray) -> np.ndarray:
    """Standard vegetation index: ExG = 2G - R - B, clipped to non-negative."""
    arr = tile_rgb.astype(np.float32)
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    return np.clip(2.0 * g - r - b, 0.0, None)


def _otsu_threshold(values: np.ndarray, bins: int = 256) -> float:
    """Automatic vegetation/background split point (Otsu's method, numpy-only).

    Used instead of a single fixed cutoff because different tiles/scenes have
    very different lighting and vegetation density, so no single fixed
    threshold on normalized ExG works well across an entire orthophoto.
    """
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0
    hist, bin_edges = np.histogram(finite, bins=bins)
    hist = hist.astype(np.float64)
    total = hist.sum()
    if total <= 0:
        return 0.0
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    weight_bg = np.cumsum(hist)
    weight_fg = total - weight_bg
    sum_all = float(np.sum(hist * bin_centers))
    sum_bg = np.cumsum(hist * bin_centers)
    with np.errstate(divide="ignore", invalid="ignore"):
        mean_bg = np.where(weight_bg > 0, sum_bg / weight_bg, 0.0)
        mean_fg = np.where(weight_fg > 0, (sum_all - sum_bg) / weight_fg, 0.0)
    variance_between = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
    best = int(np.argmax(variance_between))
    return float(bin_centers[best])


class PlaceholderTreeDetector(TreeDetector):
    """Dependency-free stand-in used when no trained weight file is provided.

    Oil-palm canopies are roughly evenly spaced, bright-green, near-circular
    blobs seen from above. This backend scores "greenness" per pixel with the
    Excess Green Index, separates vegetation from background with a per-tile
    Otsu threshold, then labels connected vegetation regions
    (``rasterio.features.shapes``, the same polygonize call AFE uses for mask
    -> vector) so that *one* blob produces *one* point, regardless of how much
    internal leaf/shadow texture it has. A fixed per-pixel brightness-peak
    search was tried first and badly over-counted on real imagery: every
    bright leaf cluster or specular highlight inside a single canopy registered
    as its own peak. Blobs much larger than one expected canopy (likely several
    touching trees) fall back to peak-picking restricted to that blob only, so
    merged canopies still yield roughly one point per tree. This exercises the
    full TCS pipeline end-to-end (tiling -> NMS -> georeferenced points ->
    counts) well before a real trained weight file is available. Replace with
    YoloTreeDetector once one is.
    """

    def __init__(self, crown_radius_px: float = DEFAULT_CROWN_RADIUS_PX, crown_spacing_px: float = DEFAULT_CROWN_SPACING_PX) -> None:
        self.crown_radius_px = float(crown_radius_px)
        self.crown_spacing_px = max(1, int(round(crown_spacing_px)))
        # Tracked across all tiles of a run so the worker can warn the user
        # when "Radius kanopi" is set much smaller than the canopies actually
        # visible in the raster (the single biggest cause of over-counting).
        self.normal_blob_count = 0
        self.merged_blob_count = 0

    def detect_tile(self, tile_rgb: np.ndarray) -> List[Detection]:
        if rasterio_shapes is None:
            raise RuntimeError("rasterio belum tersedia untuk model Placeholder TCS.")

        exg = _excess_green_index(tile_rgb)
        max_val = float(exg.max())
        if max_val <= 1e-6:
            return []
        normalized = exg / max_val

        vegetated = normalized[normalized > 0]
        threshold = max(0.12, _otsu_threshold(vegetated) if vegetated.size else 0.12)
        binary = (normalized >= threshold).astype(np.uint8)
        if not binary.any():
            return []

        height, width = normalized.shape
        single_tree_area = math.pi * (self.crown_radius_px ** 2)
        min_area_px = max(4.0, single_tree_area * 0.15)
        merged_blob_area = single_tree_area * 3.0
        tile_area = float(height * width)

        detections: List[Detection] = []
        for geom, value in rasterio_shapes(binary, mask=binary.astype(bool), connectivity=8):
            if value != 1:
                continue
            poly = shapely_shape(geom)
            area = poly.area
            if area < min_area_px or area > tile_area * 0.6:
                continue
            minx, miny, maxx, maxy = poly.bounds
            if area > merged_blob_area:
                self.merged_blob_count += 1
                detections.extend(
                    self._split_merged_blob(normalized, binary, minx, miny, maxx, maxy, area, single_tree_area)
                )
                continue
            self.normal_blob_count += 1
            centroid = poly.centroid
            cx, cy = centroid.x, centroid.y
            detections.append(
                Detection(
                    x1=cx - self.crown_radius_px,
                    y1=cy - self.crown_radius_px,
                    x2=cx + self.crown_radius_px,
                    y2=cy + self.crown_radius_px,
                    confidence=self._blob_confidence(normalized, minx, miny, maxx, maxy),
                )
            )
        return detections

    def _blob_confidence(self, normalized: np.ndarray, minx: float, miny: float, maxx: float, maxy: float) -> float:
        x0, x1 = max(0, int(minx)), max(1, int(math.ceil(maxx)))
        y0, y1 = max(0, int(miny)), max(1, int(math.ceil(maxy)))
        region = normalized[y0:y1, x0:x1]
        if region.size == 0:
            return 0.5
        return float(np.clip(region.mean(), 0.05, 1.0))

    def _split_merged_blob(
        self,
        normalized: np.ndarray,
        binary: np.ndarray,
        minx: float,
        miny: float,
        maxx: float,
        maxy: float,
        area: float,
        single_tree_area: float,
    ) -> List[Detection]:
        """Peak-pick within one oversized blob only (likely several touching trees).

        The number of peaks taken is capped at the blob's area divided by one
        expected canopy area, not "however many pixels clear the score
        threshold" - otherwise leaf/shadow texture noise inside the blob (many
        local brightness maxima within what is geometrically a single or
        few-tree region) still produces far more points than the blob's size
        could plausibly contain.
        """
        x0, x1 = max(0, int(minx)), min(normalized.shape[1], int(math.ceil(maxx)))
        y0, y1 = max(0, int(miny)), min(normalized.shape[0], int(math.ceil(maxy)))
        region = normalized[y0:y1, x0:x1].copy()
        mask = binary[y0:y1, x0:x1] > 0
        region[~mask] = -1.0

        min_dist = self.crown_spacing_px
        min_score = 0.12
        rh, rw = region.shape
        expected_count = max(2, int(round(area / max(1.0, single_tree_area))))

        detections: List[Detection] = []
        for _ in range(expected_count):
            peak_idx = int(np.argmax(region))
            peak_val = float(region.flat[peak_idx])
            if peak_val < min_score:
                break
            py, px = np.unravel_index(peak_idx, region.shape)
            cy, cx = y0 + py, x0 + px
            detections.append(
                Detection(
                    x1=cx - self.crown_radius_px,
                    y1=cy - self.crown_radius_px,
                    x2=cx + self.crown_radius_px,
                    y2=cy + self.crown_radius_px,
                    confidence=min(1.0, peak_val),
                )
            )
            ry0, ry1 = max(0, py - min_dist), min(rh, py + min_dist + 1)
            rx0, rx1 = max(0, px - min_dist), min(rw, px + min_dist + 1)
            region[ry0:ry1, rx0:rx1] = -1.0
        return detections


class YoloTreeDetector(TreeDetector):
    """Ultralytics YOLO (v5/v8) inference backend. Requires a trained .pt weight file.

    ``imgsz`` must match the resolution the weight file was actually trained
    at (not the tile's raw pixel size) - the canopy-to-frame scale the model
    learned depends on how much the training images were downscaled to reach
    that imgsz, and inference needs to reproduce the same downscaling ratio.
    TCS's bundled model (see training/train.py) was trained on whole source
    images (not pre-tiled) resized to imgsz=1280, so the "Tile size" panel
    setting should stay large enough to cover the whole raster when using
    that model, letting this fixed imgsz do the necessary downscaling itself.
    """

    def __init__(self, weights_path: str, imgsz: int = 1280) -> None:
        try:
            from ultralytics import YOLO
        except Exception as exc:  # pragma: no cover
            import traceback
            try:
                import tempfile
                from pathlib import Path

                Path(tempfile.gettempdir(), "tcs_yolo_import_error.log").write_text(
                    traceback.format_exc(), encoding="utf-8"
                )
            except Exception:
                pass
            raise RuntimeError("Model YOLO butuh 'pip install ultralytics torch'.") from exc

        try:
            self.model = YOLO(weights_path)
        except Exception as exc:
            raise RuntimeError(f"Gagal memuat model YOLO dari '{weights_path}': {exc}") from exc
        self.imgsz = imgsz

    def detect_tile(self, tile_rgb: np.ndarray) -> List[Detection]:
        results = self.model.predict(source=tile_rgb, imgsz=self.imgsz, verbose=False)
        detections: List[Detection] = []
        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            for (x1, y1, x2, y2), conf in zip(xyxy, confs):
                detections.append(Detection(float(x1), float(y1), float(x2), float(y2), float(conf)))
        return detections


def build_detector(
    weights_path: Optional[str] = None,
    crown_radius_px: float = DEFAULT_CROWN_RADIUS_PX,
    crown_spacing_px: float = DEFAULT_CROWN_SPACING_PX,
) -> TreeDetector:
    """Select a backend: YOLO when a weight file is given, placeholder otherwise."""
    if weights_path:
        return YoloTreeDetector(weights_path)
    return PlaceholderTreeDetector(crown_radius_px=crown_radius_px, crown_spacing_px=crown_spacing_px)
