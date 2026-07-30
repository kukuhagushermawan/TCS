"""Background worker for TCS's optional "Cari Lahan Kosong" (find empty
planting land) feature.

Runs on a QThread, same pattern as TCSWorker: row detection and per-row gap
scanning can take a moment for a plantation with many trees and must never
freeze the viewer's UI thread.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from PyQt6.QtCore import QThread, pyqtSignal

from .config import GAP_TOLERANCE_RATIO
from .gap_analysis import (
    boundary_polygon_from_layer,
    cluster_rows,
    convex_hull_polygon,
    estimate_row_direction,
    estimate_spacing,
    filter_empty_slots,
    filter_within_polygon,
    filter_within_raster_margin,
    find_missing_in_rows,
    points_from_features,
    points_to_features,
    pooled_along_row_spacing,
    rotate_points,
)


class GapAnalysisWorker(QThread):
    progress = pyqtSignal(int, str)
    finished_ok = pyqtSignal(list, float, str)  # features, spacing_used, boundary_source
    failed = pyqtSignal(str)

    def __init__(
        self,
        tree_features: List[Dict[str, Any]],
        tree_crs: Any,
        boundary_layer: Optional[Any] = None,
        raster_bounds: Optional[Tuple[float, float, float, float]] = None,
        tolerance_ratio: float = GAP_TOLERANCE_RATIO,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.tree_features = tree_features
        self.tree_crs = tree_crs
        self.boundary_layer = boundary_layer
        self.raster_bounds = raster_bounds
        self.tolerance_ratio = tolerance_ratio

    def run(self) -> None:  # noqa: D102 - QThread entry point
        try:
            features, spacing, boundary_source = self._run_pipeline()
            self.finished_ok.emit(features, spacing, boundary_source)
        except Exception as exc:
            self.failed.emit(str(exc).strip() or repr(exc))

    def _run_pipeline(self) -> Tuple[List[dict], float, str]:
        self.progress.emit(10, "Menghitung arah barisan pohon...")
        tree_points = points_from_features(self.tree_features)
        if tree_points.shape[0] < 4:
            raise RuntimeError("Tidak cukup titik pohon untuk mendeteksi pola barisan (minimal 4 pohon).")

        row_angle = estimate_row_direction(tree_points)
        if row_angle is None:
            raise RuntimeError("Tidak ditemukan pola barisan yang jelas dari titik pohon terdeteksi.")

        bootstrap_spacing = estimate_spacing(tree_points)
        if not bootstrap_spacing or bootstrap_spacing <= 0:
            raise RuntimeError("Tidak cukup titik pohon untuk memperkirakan jarak tanam.")

        self.progress.emit(30, "Mengelompokkan pohon ke dalam barisan...")
        rotated = rotate_points(tree_points, -row_angle)
        rows = cluster_rows(rotated, row_gap_threshold=bootstrap_spacing * 0.6)
        rows = [r for r in rows if r.shape[0] >= 2]
        if len(rows) < 2:
            raise RuntimeError("Tidak ditemukan cukup barisan pohon untuk mendeteksi lahan kosong.")

        row_spacing = pooled_along_row_spacing(rows)
        if not row_spacing or row_spacing <= 0:
            raise RuntimeError("Tidak cukup data jarak antar pohon dalam barisan.")

        self.progress.emit(50, "Mencari posisi pohon yang hilang dalam barisan...")
        missing_rotated = find_missing_in_rows(rows, row_spacing, self.tolerance_ratio)
        missing_world = rotate_points(missing_rotated, row_angle)

        self.progress.emit(70, "Menyaring kandidat lahan kosong...")
        missing_world = filter_empty_slots(missing_world, tree_points, row_spacing * self.tolerance_ratio)

        boundary_source = "convex hull dari titik pohon"
        boundary = None
        if self.boundary_layer is not None:
            boundary = boundary_polygon_from_layer(self.boundary_layer, self.tree_crs)
            if boundary is not None:
                boundary_source = f"boundary '{self.boundary_layer.name}'"
        if boundary is None:
            # Small buffer: a genuinely missing tree at the very end of a row
            # can sit right at (or just past) a hull drawn tight around the
            # existing trees themselves.
            boundary = convex_hull_polygon(tree_points).buffer(row_spacing * 0.15)
        missing_world = filter_within_polygon(missing_world, boundary)

        if self.raster_bounds is not None:
            missing_world = filter_within_raster_margin(missing_world, self.raster_bounds, row_spacing * 0.5)

        self.progress.emit(100, "Selesai")
        return points_to_features(missing_world), row_spacing, boundary_source
