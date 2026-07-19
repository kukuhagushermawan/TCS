"""TCS "Advanced Settings" dialog: AOI mode, confidence/IoU thresholds, and
tiling/placeholder-detector parameters.

Split out of the main TCS dock panel so the everyday flow (pick raster, pick
model, Run Counting) stays simple for non-technical users - these knobs only
matter when something needs tuning, so they live behind one button instead of
being shown unconditionally. The dialog is non-modal so users can still
interact with the raster canvas (e.g. drawing an AOI box) while it's open.
"""
from __future__ import annotations

from typing import Optional

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from .config import (
    DEFAULT_CROWN_RADIUS_PX,
    DEFAULT_CROWN_SPACING_PX,
    DEFAULT_IOU,
    DEFAULT_TILE_OVERLAP,
    DEFAULT_TILE_SIZE,
)


class AdvancedSettingsDialog(QDialog):
    def __init__(self, main_window, parent=None) -> None:
        super().__init__(parent)
        self.main_window = main_window
        self._aoi_rect: Optional[tuple] = None
        self.setWindowTitle("TCS - Advanced Settings")
        self.setModal(False)
        self.resize(420, 560)
        self._build_ui()

    # ---------- UI ----------
    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        aoi_group = QGroupBox("Area of Interest (AOI)")
        aoi_layout = QVBoxLayout(aoi_group)
        self.radio_full = QRadioButton("Full Extent")
        self.radio_full.setChecked(True)
        self.radio_bbox = QRadioButton("Bounding box (gambar di canvas)")
        self.radio_block = QRadioButton("Ikuti boundary blok kebun (layer polygon)")
        self.radio_full.toggled.connect(self._on_aoi_mode_changed)
        self.radio_bbox.toggled.connect(self._on_aoi_mode_changed)
        self.radio_block.toggled.connect(self._on_aoi_mode_changed)
        aoi_layout.addWidget(self.radio_full)
        aoi_layout.addWidget(self.radio_bbox)
        aoi_btn_row = QHBoxLayout()
        self.draw_aoi_btn = QPushButton("Gambar AOI di Canvas Aktif")
        self.draw_aoi_btn.clicked.connect(self._start_draw_aoi)
        self.draw_aoi_btn.setEnabled(False)
        clear_aoi_btn = QPushButton("Reset AOI")
        clear_aoi_btn.clicked.connect(self._clear_aoi)
        aoi_btn_row.addWidget(self.draw_aoi_btn)
        aoi_btn_row.addWidget(clear_aoi_btn)
        aoi_layout.addLayout(aoi_btn_row)
        aoi_layout.addWidget(self.radio_block)
        self.block_combo = QComboBox()
        self.block_combo.setEnabled(False)
        aoi_layout.addWidget(self.block_combo)
        self.aoi_status_label = QLabel("AOI: Full extent raster")
        self.aoi_status_label.setWordWrap(True)
        aoi_layout.addWidget(self.aoi_status_label)
        layout.addWidget(aoi_group)

        # Ambang keyakinan (confidence) diatur langsung di panel utama, jadi
        # dialog ini hanya memuat pengaturan lanjutan seperti NMS/IoU di bawah.
        iou_row = QHBoxLayout()
        self.iou_slider = QSlider(Qt.Orientation.Horizontal)
        self.iou_slider.setRange(1, 99)
        self.iou_slider.setValue(int(DEFAULT_IOU * 100))
        self.iou_label = QLabel(f"{DEFAULT_IOU:.2f}")
        self.iou_slider.valueChanged.connect(lambda v: self.iou_label.setText(f"{v / 100:.2f}"))
        iou_row.addWidget(QLabel("IoU threshold (NMS):"))
        iou_row.addWidget(self.iou_slider, 1)
        iou_row.addWidget(self.iou_label)
        layout.addLayout(iou_row)
        nms_hint = QLabel("NMS menghapus kotak deteksi yang saling bertumpuk agar satu pohon tidak terhitung dua kali.")
        nms_hint.setWordWrap(True)
        nms_hint.setStyleSheet("color: #6B7280; font-size: 8.5pt;")
        layout.addWidget(nms_hint)

        adv_group = QGroupBox("Tiling & Placeholder")
        adv_layout = QGridLayout(adv_group)
        self.tile_size_spin = QSpinBox()
        self.tile_size_spin.setRange(128, 20000)
        self.tile_size_spin.setSingleStep(64)
        self.tile_size_spin.setValue(DEFAULT_TILE_SIZE)
        self.tile_size_spin.setToolTip(
            "Untuk model YOLO bawaan TCS (dilatih pada seluruh gambar yang diperkecil ke 1280px, "
            "bukan potongan/tile beresolusi asli): set Tile size >= lebar/tinggi piksel raster "
            "(agar praktis tidak ada tiling nyata) supaya skala kanopi saat inferensi cocok dengan "
            "skala saat training. Tile size kecil akan membuat model salah mengenali skala kanopi. "
            "Sudah disesuaikan otomatis saat raster dipilih di panel utama."
        )
        self.overlap_spin = QSpinBox()
        self.overlap_spin.setRange(0, 4096)
        self.overlap_spin.setValue(DEFAULT_TILE_OVERLAP)
        self.crown_radius_spin = QDoubleSpinBox()
        self.crown_radius_spin.setRange(2, 500)
        self.crown_radius_spin.setValue(DEFAULT_CROWN_RADIUS_PX)
        self.crown_radius_spin.setSuffix(" px")
        self.crown_radius_spin.setToolTip(
            "Hanya dipakai oleh model Placeholder. Set mendekati radius piksel kanopi sawit "
            "yang terlihat di citra (perbesar/zoom raster untuk mengukurnya). Jika hasil deteksi "
            "jauh lebih banyak dari jumlah pohon sebenarnya, perbesar nilai ini."
        )
        self.crown_spacing_spin = QDoubleSpinBox()
        self.crown_spacing_spin.setRange(4, 800)
        self.crown_spacing_spin.setValue(DEFAULT_CROWN_SPACING_PX)
        self.crown_spacing_spin.setSuffix(" px")
        self.crown_spacing_spin.setToolTip(
            "Hanya dipakai oleh model Placeholder. Jarak minimum antar pusat pohon (px); juga "
            "dipakai untuk memisahkan kanopi yang saling menempel."
        )
        adv_layout.addWidget(QLabel("Tile size:"), 0, 0)
        adv_layout.addWidget(self.tile_size_spin, 0, 1)
        adv_layout.addWidget(QLabel("Overlap:"), 1, 0)
        adv_layout.addWidget(self.overlap_spin, 1, 1)
        adv_layout.addWidget(QLabel("Radius kanopi (placeholder):"), 2, 0)
        adv_layout.addWidget(self.crown_radius_spin, 2, 1)
        adv_layout.addWidget(QLabel("Jarak antar pohon (placeholder):"), 3, 0)
        adv_layout.addWidget(self.crown_spacing_spin, 3, 1)
        layout.addWidget(adv_group)

        layout.addStretch(1)

        button_row = QHBoxLayout()
        reset_btn = QPushButton("Reset ke Default")
        reset_btn.clicked.connect(self.reset_to_defaults)
        close_btn = QPushButton("Tutup")
        close_btn.clicked.connect(self.close)
        button_row.addWidget(reset_btn)
        button_row.addStretch(1)
        button_row.addWidget(close_btn)
        layout.addLayout(button_row)

        self.refresh_blocks()

    # ---------- AOI ----------
    def refresh_blocks(self) -> None:
        current_id = self.block_combo.currentData()
        self.block_combo.clear()
        for layer in self.main_window.layer_manager.layers:
            if layer.layer_type == "vector":
                self.block_combo.addItem(layer.name, layer.id)
        if current_id:
            idx = self.block_combo.findData(current_id)
            if idx >= 0:
                self.block_combo.setCurrentIndex(idx)

    def _on_aoi_mode_changed(self, checked: bool) -> None:
        if not checked:
            return
        self.draw_aoi_btn.setEnabled(self.radio_bbox.isChecked())
        self.block_combo.setEnabled(self.radio_block.isChecked())
        if self.radio_full.isChecked():
            self._clear_aoi()
        elif self.radio_block.isChecked():
            self.refresh_blocks()
            self.aoi_status_label.setText("AOI: mengikuti extent layer boundary blok terpilih")

    def _start_draw_aoi(self) -> None:
        canvas = self.main_window.active_canvas
        if canvas is None:
            QMessageBox.warning(self, "TCS", "Buka raster dan aktifkan window-nya dulu sebelum menggambar AOI.")
            return
        try:
            canvas.aoiSelected.disconnect(self._on_aoi_selected)
        except Exception:
            pass
        canvas.aoiSelected.connect(self._on_aoi_selected)
        canvas.set_aoi_box_mode(True)
        self.aoi_status_label.setText("Gambar kotak AOI langsung di canvas raster...")

    def _on_aoi_selected(self, rect) -> None:
        col0, row0, col1, row1 = rect
        self._aoi_rect = (col0, row0, col1, row1)
        self.radio_bbox.setChecked(True)
        self.aoi_status_label.setText(f"AOI: kolom {col0}-{col1}, baris {row0}-{row1}")

    def _clear_aoi(self) -> None:
        self._aoi_rect = None
        canvas = self.main_window.active_canvas
        if canvas is not None:
            canvas.clear_aoi()
        self.aoi_status_label.setText("AOI: Full extent raster")

    # ---------- External hooks ----------
    def auto_fit_tile_size(self, width: Optional[int], height: Optional[int]) -> None:
        """Keep Tile size >= raster size so canopy scale at inference matches
        the bundled model's training scale (see tcs/inference.py YoloTreeDetector)."""
        if not width or not height:
            return
        needed = max(width, height)
        if needed > self.tile_size_spin.maximum():
            self.tile_size_spin.setMaximum(needed)
        if self.tile_size_spin.value() < needed:
            self.tile_size_spin.setValue(needed)

    def reset_to_defaults(self) -> None:
        self.radio_full.setChecked(True)
        self.iou_slider.setValue(int(DEFAULT_IOU * 100))
        self.tile_size_spin.setValue(DEFAULT_TILE_SIZE)
        self.overlap_spin.setValue(DEFAULT_TILE_OVERLAP)
        self.crown_radius_spin.setValue(DEFAULT_CROWN_RADIUS_PX)
        self.crown_spacing_spin.setValue(DEFAULT_CROWN_SPACING_PX)

    # ---------- Getters used by TCSPanel.run_counting ----------
    def get_iou(self) -> float:
        return self.iou_slider.value() / 100.0

    def get_tile_size(self) -> int:
        return self.tile_size_spin.value()

    def get_overlap(self) -> int:
        return self.overlap_spin.value()

    def get_crown_radius(self) -> float:
        return self.crown_radius_spin.value()

    def get_crown_spacing(self) -> float:
        return self.crown_spacing_spin.value()

    def get_aoi_mode(self) -> str:
        if self.radio_bbox.isChecked():
            return "bbox"
        if self.radio_block.isChecked():
            return "block"
        return "full"

    def get_aoi_rect(self) -> Optional[tuple]:
        return self._aoi_rect

    def get_block_layer_id(self) -> Optional[str]:
        return self.block_combo.currentData()
