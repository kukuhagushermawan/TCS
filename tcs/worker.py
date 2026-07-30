"""Background TCS pipeline: tiling -> detect -> NMS -> points.

Runs on a QThread so heavy inference never blocks the viewer's UI thread.
Emits progress/log messages the panel renders in its progress bar and log,
and reports errors as readable messages instead of raising into the UI.
"""
from __future__ import annotations

import time
from typing import Any, List, Optional

import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

from .inference import Detection, OnnxTreeDetector, PlaceholderTreeDetector, build_detector
from .postprocess import detections_to_point_features, nms
from .tiling import compute_tile_windows, extract_tile


class TCSWorker(QThread):
    progress = pyqtSignal(int, str)
    log = pyqtSignal(str)
    finished_ok = pyqtSignal(list)
    failed = pyqtSignal(str)

    def __init__(
        self,
        raster_rgb: np.ndarray,
        transform: Any,
        source_crs: Any,
        weights_path: Optional[str],
        confidence: float,
        iou: float,
        tile_size: int,
        overlap: int,
        crown_radius_px: float,
        crown_spacing_px: float,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.raster_rgb = raster_rgb
        self.transform = transform
        self.source_crs = source_crs
        self.weights_path = weights_path
        self.confidence = confidence
        self.iou = iou
        self.tile_size = tile_size
        self.overlap = overlap
        self.crown_radius_px = crown_radius_px
        self.crown_spacing_px = crown_spacing_px

    def run(self) -> None:  # noqa: D102 - QThread entry point
        try:
            features = self._run_pipeline()
            self.finished_ok.emit(features)
        except Exception as exc:
            self.failed.emit(str(exc).strip() or repr(exc))

    def _run_pipeline(self) -> List[dict]:
        # Allow processing even if transform is None (e.g. for non-georeferenced images like JPG/PNG)

        t0 = time.time()
        self.log.emit("Mulai TCS (Tree Counting Sawit)...")
        try:
            detector = build_detector(
                weights_path=self.weights_path,
                confidence=self.confidence,
                iou=self.iou,
                crown_radius_px=self.crown_radius_px,
                crown_spacing_px=self.crown_spacing_px,
            )
        except Exception as exc:
            raise RuntimeError(f"Gagal memuat model TCS: {exc}") from exc

        height, width = self.raster_rgb.shape[:2]
        windows = compute_tile_windows(width, height, self.tile_size, self.overlap)
        t1 = time.time()
        self.log.emit(f"Tiling: {len(windows)} tile disiapkan ({t1 - t0:.2f}s)")
        if isinstance(detector, OnnxTreeDetector) and len(windows) > 1 and self.tile_size < max(width, height):
            self.log.emit("Peringatan: naikkan Tile size >= ukuran piksel raster agar skala kanopi cocok.")

        raw_detections: List[Detection] = []
        total = len(windows) or 1
        for i, window in enumerate(windows):
            tile = extract_tile(self.raster_rgb, window, self.tile_size)
            for det in detector.detect_tile(tile):
                x1, y1 = det.x1 + window.col_off, det.y1 + window.row_off
                x2, y2 = det.x2 + window.col_off, det.y2 + window.row_off
                # Clip to the raster's real pixel bounds. extract_tile() pads
                # a non-square raster out to a square tile_size (see
                # tiling.py), so a detection can land partly or fully inside
                # that zero-padded margin; clipping (and dropping boxes left
                # with no real area) keeps every surviving detection's
                # coordinates inside the actual image before they are
                # converted to world coordinates.
                x1, x2 = max(0.0, min(x1, width)), max(0.0, min(x2, width))
                y1, y2 = max(0.0, min(y1, height)), max(0.0, min(y2, height))
                if x2 - x1 <= 0 or y2 - y1 <= 0:
                    continue
                raw_detections.append(Detection(x1=x1, y1=y1, x2=x2, y2=y2, confidence=det.confidence))
            self.progress.emit(int((i + 1) / total * 60), f"Inference tile {i + 1}/{total}")
        t2 = time.time()
        self.log.emit(f"Inference: {t2 - t1:.2f}s, {len(raw_detections)} deteksi mentah (sebelum filter/NMS)")
        if isinstance(detector, PlaceholderTreeDetector):
            total_blobs = detector.normal_blob_count + detector.merged_blob_count
            if total_blobs > 0 and detector.merged_blob_count / total_blobs > 0.3:
                self.log.emit("Peringatan: banyak pohon berdempetan, jumlah bisa kurang akurat.")

        self.progress.emit(75, "Filter confidence + NMS lintas-tile...")
        filtered = [d for d in raw_detections if d.confidence >= self.confidence]
        survivors = nms(filtered, self.iou)
        t3 = time.time()
        self.log.emit(f"NMS: {t3 - t2:.2f}s, {len(survivors)} pohon setelah filter confidence + NMS")

        self.progress.emit(90, "Konversi ke titik geografis...")
        features = detections_to_point_features(survivors, self.transform)
        t4 = time.time()
        self.log.emit(f"Konversi ke titik: {t4 - t3:.2f}s, total {t4 - t0:.2f}s, {len(features)} pohon terdeteksi")
        self.progress.emit(100, "Selesai")
        return features
