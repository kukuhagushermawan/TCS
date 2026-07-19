"""Background TCS pipeline: tiling -> detect -> NMS -> points -> block counts.

Runs on a QThread so heavy inference never blocks the viewer's UI thread.
Emits progress/log messages the panel renders in its progress bar and log,
and reports errors as readable messages instead of raising into the UI.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

from .inference import Detection, PlaceholderTreeDetector, YoloTreeDetector, build_detector
from .postprocess import assign_blocks, detections_to_box_features, nms
from .tiling import compute_tile_windows, extract_tile


class TCSWorker(QThread):
    progress = pyqtSignal(int, str)
    log = pyqtSignal(str)
    finished_ok = pyqtSignal(list, dict)
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
        boundary_layer: Optional[Any] = None,
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
        self.boundary_layer = boundary_layer

    def run(self) -> None:  # noqa: D102 - QThread entry point
        try:
            features, block_counts = self._run_pipeline()
            self.finished_ok.emit(features, block_counts)
        except Exception as exc:
            self.failed.emit(str(exc).strip() or repr(exc))

    def _run_pipeline(self) -> tuple[List[dict], Dict[str, int]]:
        if self.transform is None:
            raise RuntimeError("Raster harus sudah ter-georeferensi (mis. GeoTIFF dengan CRS).")

        t0 = time.time()
        self.log.emit("Mulai TCS (Tree Counting Sawit)...")
        try:
            detector = build_detector(
                weights_path=self.weights_path,
                crown_radius_px=self.crown_radius_px,
                crown_spacing_px=self.crown_spacing_px,
            )
        except Exception as exc:
            raise RuntimeError(f"Gagal memuat model TCS: {exc}") from exc

        height, width = self.raster_rgb.shape[:2]
        windows = compute_tile_windows(width, height, self.tile_size, self.overlap)
        t1 = time.time()
        self.log.emit(f"Tiling: {len(windows)} tile disiapkan ({t1 - t0:.2f}s)")
        if isinstance(detector, YoloTreeDetector) and len(windows) > 1 and self.tile_size < max(width, height):
            self.log.emit("Peringatan: naikkan Tile size >= ukuran piksel raster agar skala kanopi cocok.")

        raw_detections: List[Detection] = []
        total = len(windows) or 1
        for i, window in enumerate(windows):
            tile = extract_tile(self.raster_rgb, window, self.tile_size)
            for det in detector.detect_tile(tile):
                raw_detections.append(
                    Detection(
                        x1=det.x1 + window.col_off,
                        y1=det.y1 + window.row_off,
                        x2=det.x2 + window.col_off,
                        y2=det.y2 + window.row_off,
                        confidence=det.confidence,
                    )
                )
            self.progress.emit(int((i + 1) / total * 60), f"Inference tile {i + 1}/{total}")
        t2 = time.time()
        self.log.emit(f"Inference: {t2 - t1:.2f}s, {len(raw_detections)} deteksi mentah (sebelum filter/NMS)")
        if isinstance(detector, PlaceholderTreeDetector):
            total_blobs = detector.normal_blob_count + detector.merged_blob_count
            if total_blobs > 0 and detector.merged_blob_count / total_blobs > 0.3:
                self.log.emit("Peringatan: perbesar 'Radius kanopi' (Advanced) agar pohon tidak terhitung ganda.")

        self.progress.emit(75, "Filter confidence + NMS lintas-tile...")
        filtered = [d for d in raw_detections if d.confidence >= self.confidence]
        survivors = nms(filtered, self.iou)
        t3 = time.time()
        self.log.emit(f"NMS: {t3 - t2:.2f}s, {len(survivors)} pohon setelah filter confidence + NMS")

        self.progress.emit(90, "Konversi ke poligon geografis...")
        features = detections_to_box_features(survivors, self.transform)
        block_counts: Dict[str, int] = {}
        if self.boundary_layer is not None:
            features, block_counts = assign_blocks(features, self.source_crs, self.boundary_layer)
        t4 = time.time()
        self.log.emit(f"Konversi ke poligon: {t4 - t3:.2f}s, total {t4 - t0:.2f}s, {len(features)} pohon terdeteksi")
        self.progress.emit(100, "Selesai")
        return features, block_counts
