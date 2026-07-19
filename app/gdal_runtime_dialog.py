"""GDAL/ECW optional-runtime acquisition dialog for Terra View."""
from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import QUrl
from PyQt6.QtGui import QDesktopServices
from PyQt6.QtWidgets import QDialog, QFileDialog, QHBoxLayout, QLabel, QMessageBox, QPushButton, QVBoxLayout, QWidget

from . import format_converter
from .resources import resource_path

GDAL_MISSING_MESSAGE = (
    "GDAL ECW runtime belum tersedia.\n"
    "Untuk membuka file ECW, install OSGeo4W + gdal-ecw atau pilih folder runtime GDAL "
    "yang sudah memiliki ECW driver."
)


class GdalRuntimeDialog(QDialog):
    """Lets the user point Terra View at a GDAL/ECW runtime, or read the install guide."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("GDAL ECW Runtime")
        self.setMinimumWidth(480)
        self.ready = False

        layout = QVBoxLayout(self)
        message = QLabel(GDAL_MISSING_MESSAGE)
        message.setWordWrap(True)
        layout.addWidget(message)

        buttons = QHBoxLayout()
        btn_select = QPushButton("Select GDAL Runtime Folder")
        btn_select.clicked.connect(self._select_folder)
        btn_guide = QPushButton("Open Installation Guide")
        btn_guide.clicked.connect(self._open_guide)
        btn_cancel = QPushButton("Cancel")
        btn_cancel.clicked.connect(self.reject)
        buttons.addWidget(btn_select)
        buttons.addWidget(btn_guide)
        buttons.addStretch()
        buttons.addWidget(btn_cancel)
        layout.addLayout(buttons)

    def _select_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select GDAL Runtime Folder")
        if not folder:
            return
        folder_path = Path(folder)
        exe = folder_path / "bin" / "gdal_translate.exe"
        if not exe.exists():
            exe = folder_path / "gdal_translate.exe"
        if not exe.exists():
            QMessageBox.warning(
                self,
                "Folder tidak valid",
                f"gdal_translate.exe tidak ditemukan di:\n{folder_path}\n\n"
                "Pilih folder root GDAL runtime (berisi bin\\gdal_translate.exe).",
            )
            return
        format_converter.register_gdal_runtime_folder(folder_path)
        self.ready = True
        self.accept()

    def _open_guide(self) -> None:
        guide = resource_path("vendor", "osgeo4w", "README_ECW_RUNTIME.txt")
        if guide.exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(guide)))
        else:
            QMessageBox.information(
                self,
                "Installation Guide",
                "Install OSGeo4W (https://trac.osgeo.org/osgeo4w/) dan pilih package "
                "gdal + gdal-ecw, atau install QGIS yang menyertakan GDAL.\n\n"
                "Setelah terinstall, gunakan tombol Select GDAL Runtime Folder untuk "
                "menunjuk ke folder instalasinya.",
            )
