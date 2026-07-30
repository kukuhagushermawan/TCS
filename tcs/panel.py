"""TCS popup: the only UI surface the TCS module adds to the viewer.

A standalone popup dialog (opened from the TCS menu/toolbar button), not a
persistent side panel - it can be closed entirely without taking up any
space in the main window. Kept deliberately simple for non-technical users:
pick a raster, adjust the confidence threshold, click Run Counting. AOI
always covers the full raster, and NMS/tiling/placeholder-detector tuning use
fixed defaults from ``tcs/config.py`` - there is no Advanced Settings dialog
to configure them. Accuracy metrics are intentionally not surfaced in this
UI - they belong in the training report, not in front of an end user running
a count.

The result is always shown as a point overlay on top of the raster it was
counted from (one point per detected tree) - a plain counting tool, nothing
more. Talks to the host window only through the small helper methods on
``app.ui_main.TCSMainWindow`` (``overlay_tcs_result_on_source``,
``remove_tcs_overlay``, ``add_tcs_result_layer``, ``remove_layer_by_id``), so
this popup can be removed/disabled without touching the core viewer.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSlider,
    QVBoxLayout,
)

from app.coordinate_tools import is_identity_transform
from app.layer_manager import Layer
from app.resources import resource_path

from .config import (
    DEFAULT_CONFIDENCE,
    DEFAULT_CROWN_RADIUS_PX,
    DEFAULT_CROWN_SPACING_PX,
    DEFAULT_IOU,
    DEFAULT_TILE_OVERLAP,
    DEFAULT_TILE_SIZE,
)
from .model_registry import discover_models
from .postprocess import feature_bounds, save_features
from .worker import TCSWorker


class TCSPanel(QDialog):
    def __init__(self, main_window, parent=None) -> None:
        super().__init__(parent)
        self.main_window = main_window
        self.setWindowTitle("TCS - Tree Counting Sawit")
        self.setModal(False)
        # Default size fits the simplified content (no Advanced Settings
        # button, no per-block table) without leftover empty space. No
        # maximum is set, so the window stays freely resizable smaller
        # (down to a sane floor) or larger by dragging its edges.
        self.resize(400, 400)
        self.setMinimumSize(360, 340)
        self._worker: Optional[TCSWorker] = None
        self._pending_layer: Optional[Layer] = None
        self._pending_source_layer: Optional[Layer] = None
        self._pending_features: Optional[list] = None
        # Results normally land straight on their source raster; this is only
        # False when that raster's window had already been closed, so Discard
        # knows which teardown to use.
        self._result_in_overlay = False
        self._build_ui()
        self._apply_style()

    # ---------- UI ----------
    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        input_group = QGroupBox("Input")
        input_layout = QVBoxLayout(input_group)

        raster_row = QHBoxLayout()
        self.raster_combo = QComboBox()
        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self.refresh_rasters)
        raster_row.addWidget(QLabel("Raster:"))
        raster_row.addWidget(self.raster_combo, 1)
        raster_row.addWidget(refresh_btn)
        input_layout.addLayout(raster_row)

        # Model deteksi bawaan dipakai otomatis (tidak ada pemilihan/unggah
        # model), jadi user langsung mengatur tingkat keyakinan: ambang
        # seberapa yakin model bahwa sebuah objek adalah pohon sawit sebelum
        # ikut dihitung.
        input_layout.addWidget(QLabel("Tingkat keyakinan (seberapa yakin objek adalah pohon sawit):"))
        conf_row = QHBoxLayout()
        self.confidence_slider = QSlider(Qt.Orientation.Horizontal)
        self.confidence_slider.setRange(1, 99)
        self.confidence_slider.setValue(int(DEFAULT_CONFIDENCE * 100))
        self.confidence_value_label = QLabel(f"{DEFAULT_CONFIDENCE:.2f}")
        self.confidence_slider.valueChanged.connect(
            lambda v: self.confidence_value_label.setText(f"{v / 100:.2f}")
        )
        conf_row.addWidget(self.confidence_slider, 1)
        conf_row.addWidget(self.confidence_value_label)
        input_layout.addLayout(conf_row)
        conf_hint = QLabel("Makin tinggi makin ketat: hanya objek yang sangat diyakini sawit yang dihitung.")
        conf_hint.setObjectName("StatusLabel")
        conf_hint.setWordWrap(True)
        input_layout.addWidget(conf_hint)

        layout.addWidget(input_group)

        action_row = QHBoxLayout()
        self.run_btn = QPushButton("Run Counting")
        self.run_btn.setObjectName("PrimaryButton")
        self.run_btn.clicked.connect(self.run_counting)
        action_row.addWidget(self.run_btn, 1)
        layout.addLayout(action_row)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        layout.addWidget(self.progress_bar)

        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        self.status_label.setObjectName("StatusLabel")
        layout.addWidget(self.status_label)

        result_group = QGroupBox("Hasil")
        result_layout = QVBoxLayout(result_group)

        self.total_label = QLabel("Jumlah Pohon: -")
        self.total_label.setObjectName("TotalLabel")
        result_layout.addWidget(self.total_label)

        result_row = QHBoxLayout()
        self.save_btn = QPushButton("Simpan sebagai Layer Permanen...")
        self.save_btn.clicked.connect(self._save_result)
        self.discard_btn = QPushButton("Discard")
        self.discard_btn.clicked.connect(self._discard_result)
        result_row.addWidget(self.save_btn)
        result_row.addWidget(self.discard_btn)
        result_layout.addLayout(result_row)

        self._set_result_controls_visible(False)
        layout.addWidget(result_group)

        layout.addStretch(1)
        self.refresh_rasters()

    def showEvent(self, event) -> None:  # noqa: N802 - Qt method
        super().showEvent(event)
        self.refresh_rasters()

    def _apply_style(self) -> None:
        self.setStyleSheet(
            """
            QDialog { background: #F7F8FA; }
            QGroupBox {
                font-weight: 600; color: #374151; border: 1px solid #D8DCE3;
                border-radius: 6px; margin-top: 10px; padding-top: 10px;
            }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; }
            QLabel#TotalLabel { font-weight: 700; font-size: 14pt; color: #111827; }
            QLabel#StatLabel { color: #374151; font-size: 10pt; }
            QLabel#StatusLabel { color: #6B7280; font-size: 8.5pt; }
            QPushButton { padding: 6px 10px; border: 1px solid #C9CED6; border-radius: 4px; background: #FFFFFF; }
            QPushButton:hover { background: #EEF2F6; }
            QPushButton#PrimaryButton { background: #2563EB; color: white; font-weight: 600; border: none; }
            QPushButton#PrimaryButton:hover { background: #1D4ED8; }
            QComboBox, QLineEdit { padding: 3px; border: 1px solid #C9CED6; border-radius: 4px; background: #FFFFFF; }
            """
        )

    # ---------- Raster ----------
    def refresh_rasters(self) -> None:
        current_id = self.raster_combo.currentData()
        self.raster_combo.clear()
        for layer in self.main_window.layer_manager.raster_layers():
            self.raster_combo.addItem(layer.name, layer.id)
        if current_id:
            idx = self.raster_combo.findData(current_id)
            if idx >= 0:
                self.raster_combo.setCurrentIndex(idx)

    @staticmethod
    def _in_pixel_space(layer: Optional[Layer]) -> bool:
        """True when a raster's coordinates are raw image pixels rather than
        real-world ground positions."""
        if layer is None:
            return True
        return is_identity_transform(getattr(layer, "transform", None))

    # ---------- Run ----------
    def _georeference_caveat(self, layer: Layer) -> Optional[str]:
        """One-sentence warning when this raster has no real georeference, or
        None when it does.

        Counting itself is pure geometry and stays accurate in pixel space; what
        a missing georeference costs is that the result's coordinates cannot be
        used outside this app (no real-world position for GIS/field use).
        """
        if self._in_pixel_space(layer):
            return (
                "Raster tanpa georeferensi: hasil tetap akurat, tapi koordinat "
                "titiknya tidak bisa dipakai di GIS/lapangan."
            )
        return None

    def run_counting(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            return

        layer_id = self.raster_combo.currentData()
        layer = self.main_window.layer_manager.get(layer_id) if layer_id else None
        if layer is None:
            QMessageBox.warning(self, "TCS", "Pilih raster yang sudah dimuat di viewer terlebih dahulu.")
            return
        image = layer.display_image()
        if image is None:
            QMessageBox.warning(self, "TCS", "Raster terpilih tidak memiliki data citra yang bisa diproses.")
            return
        height, width = image.shape[:2]

        # AOI is always the full raster - no bounding-box/block-boundary
        # picker anymore. Tile size auto-fits to cover the whole raster in a
        # single tile (>= its largest dimension) so the detector always sees
        # canopy at the same scale it was trained at, matching what the old
        # Advanced Settings dialog used to compute automatically.
        raster_rgb = np.ascontiguousarray(image)
        if raster_rgb.ndim == 2:
            raster_rgb = np.repeat(raster_rgb[:, :, None], 3, axis=2)
        raster_rgb = raster_rgb[:, :, :3].astype(np.uint8)
        tile_size = max(width, height, DEFAULT_TILE_SIZE)

        # Selalu pakai model deteksi bawaan (models/*.onnx) - tidak ada
        # pemilihan atau unggah model dari user. Placeholder hanya dipakai
        # jika tidak ada file model sama sekali, agar pipeline tetap bisa
        # dijalankan.
        discovered = discover_models()
        weights_path = str(discovered[0].path) if discovered else None

        self._pending_source_layer = layer
        self.status_label.setText("")
        self.total_label.setText("Jumlah Pohon: -")
        self._set_running(True)

        self._worker = TCSWorker(
            raster_rgb=raster_rgb,
            transform=layer.transform,
            source_crs=layer.crs,
            weights_path=weights_path,
            confidence=self.confidence_slider.value() / 100.0,
            iou=DEFAULT_IOU,
            tile_size=tile_size,
            overlap=DEFAULT_TILE_OVERLAP,
            crown_radius_px=DEFAULT_CROWN_RADIUS_PX,
            crown_spacing_px=DEFAULT_CROWN_SPACING_PX,
        )
        self._worker.progress.connect(self._on_progress)
        self._worker.log.connect(self._on_log)
        self._worker.finished_ok.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)
        self._worker.start()

    def _set_running(self, running: bool) -> None:
        self.run_btn.setEnabled(not running)
        self.run_btn.setText("Memproses..." if running else "Run Counting")
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("%p%")

    def _on_progress(self, percent: int, message: str) -> None:
        self.progress_bar.setValue(percent)
        self.progress_bar.setFormat(f"{percent}% - {message}")

    def _on_log(self, message: str) -> None:
        self.status_label.setText(message)

    def _on_finished(self, features: list) -> None:
        self._set_running(False)
        if not features:
            QMessageBox.information(
                self, "TCS", "Tidak ada pohon terdeteksi pada tingkat keyakinan ini."
            )
            return
        source_layer = self._pending_source_layer
        timestamp = datetime.now().strftime("%H%M%S")

        # Plain point vector output (one point per detected tree) instead of
        # boxes burned into the raster, so the result stays ordinary vector data
        # that exports like any other layer.
        preview_layer = Layer(
            name=f"TCS_hasil_{timestamp} - {len(features)} pohon",
            layer_type="vector",
            path=None,
            features=features,
            crs=source_layer.crs if source_layer else None,
            bounds=feature_bounds(features),
            source_driver="TCS",
            metadata={"tcs_total": len(features), "tcs_style": "tree"},
        )
        self._pending_layer = preview_layer
        self._pending_features = features

        # Shown straight on top of the raster it was counted from, so the
        # points can be read against the imagery itself. Falling back to a
        # standalone window only if that raster's window was closed mid-run,
        # so a result is never left displayed nowhere.
        self._result_in_overlay = bool(
            source_layer is not None
            and self.main_window.overlay_tcs_result_on_source(preview_layer, source_layer)
        )
        if not self._result_in_overlay:
            self.main_window.add_tcs_result_layer(preview_layer)

        self._set_result_controls_visible(True)
        self.total_label.setText(f"Jumlah Pohon: {len(features)}")
        self.status_label.setText(
            "Selesai, hasil ditampilkan di atas raster."
            if self._result_in_overlay
            else "Selesai, hasil dibuka di jendela sendiri karena jendela raster tertutup."
        )

    def _on_failed(self, message: str) -> None:
        self._set_running(False)
        QMessageBox.critical(self, "TCS Error", message)

    # ---------- Save / discard ----------
    def _set_result_controls_visible(self, visible: bool) -> None:
        self.save_btn.setVisible(visible)
        self.discard_btn.setVisible(visible)

    def _save_result(self) -> None:
        if self._pending_layer is None:
            return
        # Warned before writing, not after: a .geojson/.shp of pixel coordinates
        # looks like ordinary spatial data but lands near the origin in any GIS,
        # so the user should know that before the file exists.
        if self._in_pixel_space(self._pending_source_layer):
            answer = QMessageBox.question(
                self,
                "TCS - Koordinat piksel",
                "Titik yang disimpan berkoordinat piksel, jadi posisinya tidak akan "
                "cocok bila dibuka di QGIS/ArcGIS.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
        default = str(resource_path("output", "tcs", "tcs_hasil.geojson"))
        path, _ = QFileDialog.getSaveFileName(
            self, "Simpan hasil TCS", default, "GeoJSON (*.geojson);;Shapefile (*.shp)"
        )
        if not path:
            return
        try:
            save_features(path, self._pending_features or [], self._pending_layer.crs)
        except Exception as exc:
            QMessageBox.critical(self, "TCS Error", f"Gagal menyimpan hasil TCS: {exc}")
            return
        self._pending_layer.path = path
        self._pending_layer.name = Path(path).name
        self._pending_layer = None
        self._pending_features = None
        self._set_result_controls_visible(False)
        self.status_label.setText(f"Tersimpan: {path}")

    def _discard_result(self) -> None:
        if self._pending_layer is None:
            return
        # Which teardown is correct depends on how the result is currently
        # shown: remove_layer_by_id closes the whole window holding the layer,
        # which would close the user's raster window if the result were
        # overlaid on it.
        if self._result_in_overlay:
            self.main_window.remove_tcs_overlay(self._pending_layer.id)
        else:
            self.main_window.remove_layer_by_id(self._pending_layer.id)
        self._pending_layer = None
        self._pending_features = None
        self._result_in_overlay = False
        self._set_result_controls_visible(False)
        self.status_label.setText("Hasil TCS dibuang.")
        self.total_label.setText("Jumlah Pohon: -")
