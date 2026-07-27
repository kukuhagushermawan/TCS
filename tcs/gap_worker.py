"""Background worker for TCS's optional "Cari Lahan Kosong" (find empty
planting land) feature.

Runs on a QThread, same pattern as TCSWorker: spacing estimation and grid
generation over a large boundary can take a moment for bigger plantations
and must never freeze the viewer's UI thread.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from PyQt6.QtCore import QThread, pyqtSignal

from .config import GAP_TOLERANCE_RATIO
from .gap_analysis import (
    boundary_polygon_from_layer,
    convex_hull_polygon,
    estimate_spacing,
    filter_empty_slots,
    generate_triangular_grid,
    points_from_features,
    points_to_features,
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
        tolerance_ratio: float = GAP_TOLERANCE_RATIO,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.tree_features = tree_features
        self.tree_crs = tree_crs
        self.boundary_layer = boundary_layer
        self.tolerance_ratio = tolerance_ratio

    def run(self) -> None:  # noqa: D102 - QThread entry point
        try:
            features, spacing, boundary_source = self._run_pipeline()
            self.finished_ok.emit(features, spacing, boundary_source)
        except Exception as exc:
            self.failed.emit(str(exc).strip() or repr(exc))

    def _run_pipeline(self) -> Tuple[List[dict], float, str]:
        self.progress.emit(10, "Menghitung estimasi jarak tanam...")
        tree_points = points_from_features(self.tree_features)
        spacing = estimate_spacing(tree_points)
        if spacing is None or spacing <= 0:
            raise RuntimeError("Tidak cukup titik pohon untuk memperkirakan jarak tanam (minimal 2 pohon).")

        self.progress.emit(35, "Menentukan area lahan...")
        boundary = None
        boundary_source = "convex hull dari titik pohon"
        if self.boundary_layer is not None:
            boundary = boundary_polygon_from_layer(self.boundary_layer, self.tree_crs)
            if boundary is not None:
                boundary_source = f"boundary '{self.boundary_layer.name}'"
        if boundary is None:
            boundary = convex_hull_polygon(tree_points)

        self.progress.emit(60, "Membuat grid titik tanam ideal...")
        grid_points = generate_triangular_grid(boundary, spacing)

        self.progress.emit(85, "Menyaring titik yang sudah ada pohonnya...")
        empty_points = filter_empty_slots(grid_points, tree_points, spacing * self.tolerance_ratio)

        self.progress.emit(100, "Selesai")
        return points_to_features(empty_points), spacing, boundary_source
