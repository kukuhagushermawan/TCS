"""TCS popup: the only UI surface the TCS module adds to the viewer.

A standalone popup dialog (opened from the TCS menu/toolbar button), not a
persistent side panel - it can be closed entirely without taking up any
space in the main window. Kept deliberately simple for non-technical users:
pick a raster, pick a model, click Run Counting. Every technical knob (AOI
mode, confidence/IoU thresholds, tiling, placeholder-detector tuning) lives
in the separate Advanced Settings dialog (tcs/advanced_settings.py), opened
on demand. Accuracy metrics are intentionally not surfaced in this UI - they
belong in the training report (see training/train_tcs.py), not in front of
an end user running a count.

Talks to the host window only through the small helper methods
``app.ui_main.TCSMainWindow.add_tcs_result_layer`` and
``remove_layer_by_id``, so this popup can be removed/disabled without
touching the core viewer.
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
    QHeaderView,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSlider,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

try:
    from rasterio.transform import Affine
except Exception:  # pragma: no cover
    Affine = None

from app.layer_manager import Layer
from app.resources import resource_path

from .advanced_settings import AdvancedSettingsDialog
from .config import DEFAULT_CONFIDENCE
from .model_registry import discover_models
from .postprocess import feature_bounds, save_features
from .worker import TCSWorker


class TCSPanel(QDialog):
    def __init__(self, main_window, parent=None) -> None:
        super().__init__(parent)
        self.main_window = main_window
        self.setWindowTitle("TCS - Tree Counting Sawit")
        self.setModal(False)
        self.resize(420, 480)
        self._worker: Optional[TCSWorker] = None
        self._pending_layer: Optional[Layer] = None
        self._pending_source_layer: Optional[Layer] = None
        self._pending_features: Optional[list] = None
        self.advanced = AdvancedSettingsDialog(main_window, parent=self)
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
        self.raster_combo.currentIndexChanged.connect(self._on_raster_changed)
        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self.refresh_rasters)
        raster_row.addWidget(QLabel("Raster:"))
        raster_row.addWidget(self.raster_combo, 1)
        raster_row.addWidget(refresh_btn)
        input_layout.addLayout(raster_row)

        # Model YOLO bawaan dipakai otomatis (tidak ada pemilihan/unggah model),
        # jadi user langsung mengatur tingkat keyakinan: ambang seberapa yakin
        # model bahwa sebuah objek adalah pohon sawit sebelum ikut dihitung.
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
        advanced_btn = QPushButton("Advanced Settings...")
        advanced_btn.clicked.connect(self._open_advanced_settings)
        self.run_btn = QPushButton("Run Counting")
        self.run_btn.setObjectName("PrimaryButton")
        self.run_btn.clicked.connect(self.run_counting)
        action_row.addWidget(advanced_btn)
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

        self.block_table = QTableWidget(0, 2)
        self.block_table.setHorizontalHeaderLabels(["Blok", "Jumlah Pohon"])
        self.block_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.block_table.verticalHeader().setVisible(False)
        self.block_table.setMaximumHeight(160)
        result_layout.addWidget(self.block_table)
        self.block_table.setVisible(False)

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
        self.advanced.refresh_blocks()

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

    def _on_raster_changed(self, _index: int) -> None:
        layer_id = self.raster_combo.currentData()
        layer = self.main_window.layer_manager.get(layer_id) if layer_id else None
        if layer is not None:
            self.advanced.auto_fit_tile_size(layer.width, layer.height)

    def _open_advanced_settings(self) -> None:
        self.advanced.refresh_blocks()
        self.advanced.show()
        self.advanced.raise_()
        self.advanced.activateWindow()

    # ---------- Run ----------
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
        transform = layer.transform
        boundary_layer = None
        aoi_mode = self.advanced.get_aoi_mode()

        if aoi_mode == "bbox" and self.advanced.get_aoi_rect():
            col0, row0, col1, row1 = self.advanced.get_aoi_rect()
            col0, col1 = sorted((max(0, min(col0, width)), max(0, min(col1, width))))
            row0, row1 = sorted((max(0, min(row0, height)), max(0, min(row1, height))))
            if col1 - col0 < 2 or row1 - row0 < 2:
                QMessageBox.warning(self, "TCS", "AOI terlalu kecil atau di luar batas raster.")
                return
            raster_rgb = np.ascontiguousarray(image[row0:row1, col0:col1])
            sub_transform = transform * Affine.translation(col0, row0) if (transform is not None and Affine is not None) else transform
        elif aoi_mode == "block":
            block_id = self.advanced.get_block_layer_id()
            boundary_layer = self.main_window.layer_manager.get(block_id) if block_id else None
            if boundary_layer is None:
                QMessageBox.warning(self, "TCS", "Pilih layer boundary blok (vector polygon) di Advanced Settings terlebih dahulu.")
                return
            raster_rgb = np.ascontiguousarray(image)
            sub_transform = transform
        else:
            raster_rgb = np.ascontiguousarray(image)
            sub_transform = transform

        if raster_rgb.ndim == 2:
            raster_rgb = np.repeat(raster_rgb[:, :, None], 3, axis=2)
        raster_rgb = raster_rgb[:, :, :3].astype(np.uint8)

        # Selalu pakai model YOLO bawaan (models/*.pt) - tidak ada pemilihan
        # atau unggah model dari user. Placeholder hanya dipakai jika tidak ada
        # file model sama sekali, agar pipeline tetap bisa dijalankan.
        discovered = discover_models()
        weights_path = str(discovered[0].path) if discovered else None

        self._pending_source_layer = layer
        self.status_label.setText("")
        self.block_table.setRowCount(0)
        self.block_table.setVisible(False)
        self.total_label.setText("Jumlah Pohon: -")
        self._set_running(True)

        self._worker = TCSWorker(
            raster_rgb=raster_rgb,
            transform=sub_transform,
            source_crs=layer.crs,
            weights_path=weights_path,
            confidence=self.confidence_slider.value() / 100.0,
            iou=self.advanced.get_iou(),
            tile_size=self.advanced.get_tile_size(),
            overlap=self.advanced.get_overlap(),
            crown_radius_px=self.advanced.get_crown_radius(),
            crown_spacing_px=self.advanced.get_crown_spacing(),
            boundary_layer=boundary_layer,
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

    def _on_finished(self, features: list, block_counts: dict) -> None:
        self._set_running(False)
        if not features:
            QMessageBox.information(self, "TCS", "Tidak ada pohon terdeteksi pada AOI/threshold saat ini.")
            return
        source_layer = self._pending_source_layer
        timestamp = datetime.now().strftime("%H%M%S")

        # Plain polygon/vector output (one bounding-box polygon per detected
        # tree) instead of boxes burned into the raster - opens as its own new
        # vector layer/document, same as any other vector file in this viewer.
        preview_layer = Layer(
            name=f"TCS_hasil_{timestamp} - {len(features)} pohon (belum disimpan)",
            layer_type="vector",
            path=None,
            features=features,
            crs=source_layer.crs if source_layer else None,
            bounds=feature_bounds(features),
            source_driver="TCS",
            metadata={"tcs_total": len(features)},
        )
        self._pending_layer = preview_layer
        self._pending_features = features
        self.main_window.add_tcs_result_layer(preview_layer)
        self._set_result_controls_visible(True)
        self.total_label.setText(f"Jumlah Pohon: {len(features)}")
        if block_counts:
            self.block_table.setRowCount(len(block_counts))
            for row, (name, count) in enumerate(sorted(block_counts.items())):
                self.block_table.setItem(row, 0, QTableWidgetItem(str(name)))
                self.block_table.setItem(row, 1, QTableWidgetItem(str(count)))
            self.block_table.setVisible(True)
        self.status_label.setText(f"Selesai: {len(features)} pohon ditampilkan sebagai layer baru.")

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
        self.main_window.remove_layer_by_id(self._pending_layer.id)
        self._pending_layer = None
        self._pending_features = None
        self._set_result_controls_visible(False)
        self.status_label.setText("Hasil TCS dibuang (discard).")
        self.total_label.setText("Jumlah Pohon: -")
        self.block_table.setRowCount(0)
        self.block_table.setVisible(False)
