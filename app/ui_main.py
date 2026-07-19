"""TCS main UI.

A lightweight ER Viewer inspired raster/vector viewer. The interface is intentionally
simple: no login, no startup workflow, no side layer panel, and no permanent right
metadata panel. Metadata appears on demand from the View menu. The TCS (Tree Counting
Sawit) module is added as an Analysis menu + toolbar group and its own popup dialog,
for automatic oil-palm tree detection/counting, without touching the inherited viewer
behavior.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional
import re

from PyQt6.QtCore import Qt, QSize, QEvent, QTimer, QThread, pyqtSignal
from PyQt6.QtGui import QAction, QIcon, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QAbstractItemView,
    QDialog,
    QFileDialog,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QMdiArea,
    QGridLayout,
    QInputDialog,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPushButton,
    QStatusBar,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QToolBar,
    QToolButton,
    QVBoxLayout,
    QWidget,
    QFrame,
)

from .export_tools import export_enhancement
from .format_converter import convert_vector
from .layer_manager import Layer, LayerManager
from .map_canvas import MapCanvas
from .promo_dialog import TerramitraPromotionDialog
from .raster_loader import load_raster
from .resources import resource_path
from .vector_loader import load_vector

from tcs.panel import TCSPanel

try:
    from pyproj import CRS as PyProjCRS
except Exception:  # pragma: no cover
    PyProjCRS = None


class _FileLoadWorker(QThread):
    """Runs a blocking loader (rasterio/pyshp/etc.) off the GUI thread.

    Open Raster/Open Vector used to call load_raster()/load_vector() directly
    on the main thread, so Qt never got a chance to repaint or respond to
    Windows during a slow read - large files made the window flash "Not
    Responding" even though nothing was actually stuck. The loader function
    passed in must be pure I/O/CPU work with no Qt widget calls, since it runs
    on a background thread where creating/touching widgets is not safe.
    """

    finished_ok = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, load_fn, parent=None) -> None:
        super().__init__(parent)
        self._load_fn = load_fn

    def run(self) -> None:  # noqa: N802 - Qt method
        try:
            result = self._load_fn()
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc).strip() or repr(exc))
        else:
            self.finished_ok.emit(result)


class CopyableInfoTable(QTableWidget):
    """A copy-friendly metadata table for Ctrl+C."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectItems)
        self.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)

    def keyPressEvent(self, event) -> None:  # noqa: N802 - Qt method
        if event.key() == Qt.Key.Key_C and event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            self.copy_selection_to_clipboard()
            return
        super().keyPressEvent(event)

    def copy_selection_to_clipboard(self) -> None:
        indexes = self.selectedIndexes()
        if not indexes:
            rows = range(self.rowCount())
            cols = range(self.columnCount())
        else:
            rows = sorted(set(i.row() for i in indexes))
            cols = sorted(set(i.column() for i in indexes))
        lines = []
        for r in rows:
            vals = []
            for c in cols:
                item = self.item(r, c)
                vals.append(item.text() if item else "")
            lines.append("\t".join(vals))
        QApplication.clipboard().setText("\n".join(lines))


class TCSMainWindow(QMainWindow):
    """Simple ER-Viewer style desktop window for raster and vector data, plus TCS."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("TCS - Tree Counting Sawit")
        icon_path = resource_path("assets", "logos", "tcs_icon.ico")
        if icon_path.exists():
            self.setWindowIcon(QIcon(str(icon_path)))
        self.resize(1360, 820)

        self.layer_manager = LayerManager()
        self.enhancement_log = "Belum ada enhancement."
        self.window_link_enabled = False
        self.extra_windows: list[object] = []
        self.active_canvas: Optional[MapCanvas] = None
        self._popup_canvases: list[MapCanvas] = []
        # Each popup canvas carries its own layer_manager (canvas.layer_manager),
        # which is the single source of truth for which layer(s) that document
        # window shows - including overlaid vector+raster pairs opened onto the
        # same window. When a popup is closed, every layer still listed in its
        # canvas.layer_manager is removed from the main registry below so
        # Metadata/Export only ever see files that are still open.
        self._dialog_canvases: Dict[object, list[MapCanvas]] = {}
        self.toolbar_buttons: Dict[str, QPushButton] = {}
        self._active_load_worker: Optional[_FileLoadWorker] = None

        self._build_ui()
        self._build_menus()
        self._build_toolbar()
        self._build_statusbar()
        self._build_tcs_dialog()
        self._connect_signals()
        self._set_style()
        self._show_terramitra_promo_if_needed()

    # ---------- UI ----------
    def _build_ui(self) -> None:
        # Keep one internal canvas for shared logic, but show documents in an MDI workspace.
        # This mirrors ER Viewer behavior: the main menu/toolbar/status bar stays visible,
        # while each raster/vector has its own document frame with minimize/maximize/close.
        # No duplicate toolbar/status is created inside the document window.
        self.canvas = MapCanvas(self.layer_manager)
        self.active_canvas = self.canvas

        self.mdi_area = QMdiArea()
        self.mdi_area.setObjectName("MdiWorkspace")
        self.mdi_area.setViewMode(QMdiArea.ViewMode.SubWindowView)
        self.mdi_area.subWindowActivated.connect(self._on_mdi_subwindow_activated)
        self.setCentralWidget(self.mdi_area)

    def _build_menus(self) -> None:
        menubar = self.menuBar()

        file_menu = menubar.addMenu("File")
        self._add_menu_action(file_menu, "Open Raster...", self.open_raster, "Ctrl+O")
        self._add_menu_action(file_menu, "Open Vector...", self.open_vector, "Ctrl+Shift+O")
        file_menu.addSeparator()
        export_menu = file_menu.addMenu("Export")
        self._add_menu_action(export_menu, "Export Raster...", self.export_raster_from_open_layers)
        self._add_menu_action(export_menu, "Export Vector...", self.export_vector_from_open_layers)
        file_menu.addSeparator()
        self._add_menu_action(file_menu, "Exit", self.close, "Alt+F4")

        view_menu = menubar.addMenu("View")
        self.act_toolbar = self._add_menu_action(view_menu, "Toolbar", self.toggle_toolbar)
        self.act_toolbar.setCheckable(True)
        self.act_toolbar.setChecked(True)
        self.act_statusbar = self._add_menu_action(view_menu, "Status Bar", self.toggle_statusbar)
        self.act_statusbar.setCheckable(True)
        self.act_statusbar.setChecked(True)
        view_menu.addSeparator()
        self._add_menu_action(view_menu, "Reset / Data Extents", self.reset_view)
        self._add_menu_action(view_menu, "Zoom In", self.zoom_in, "+")
        self._add_menu_action(view_menu, "Zoom Out", self.zoom_out, "-")
        self.act_zoom_box = self._add_menu_action(view_menu, "Zoom Box", self.toggle_zoom_box)
        self.act_zoom_box.setCheckable(True)
        view_menu.addSeparator()
        self.act_window_link = self._add_menu_action(view_menu, "Window Link", self.toggle_window_link)
        self.act_window_link.setCheckable(True)
        view_menu.addSeparator()
        self._add_menu_action(view_menu, "Metadata / Properties", self.show_metadata_dialog, "Alt+Enter")

        analysis_menu = menubar.addMenu("Analysis")
        self._add_menu_action(analysis_menu, "TCS Panel", self.open_tcs_panel)
        analysis_menu.addSeparator()
        self._add_menu_action(analysis_menu, "Clear All", self.clear_all)

    def _add_menu_action(self, menu: QMenu, text: str, handler, shortcut: str | None = None) -> QAction:
        action = QAction(text, self)
        if shortcut:
            action.setShortcut(shortcut)
        action.triggered.connect(handler)
        menu.addAction(action)
        return action

    def _build_toolbar(self) -> None:
        self.toolbar = QToolBar("TCS Toolbar")
        self.toolbar.setMovable(False)
        self.toolbar.setFloatable(False)
        self.toolbar.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
        self.toolbar.setIconSize(QSize(26, 26))
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, self.toolbar)

        # ER/GIS viewer style: clear groups with separators.
        self._add_toolbar_group("File", [
            ("Open Raster", self.open_raster, "open raster.png"),
            ("Open Vector", self.open_vector, "open vector.png"),
            ("Export / Save As", self._show_export_menu_from_toolbar, "export.png"),
        ])
        self.toolbar.addSeparator()
        self._add_toolbar_group("View", [
            ("Reset / Data Extents", self.reset_view, "reset.png"),
            ("Zoom Box", self.toggle_zoom_box, "zoom box.png"),
            ("Window Link", self.toggle_window_link, "window link.png"),
            ("Metadata / Properties", self.show_metadata_dialog, "metadata.png"),
        ])
        self.toolbar.addSeparator()
        self._add_toolbar_group("Analysis", [
            ("TCS Panel", self.open_tcs_panel, "tcs_tool.png"),
            ("Clear All", self.clear_all, "clear.png"),
        ])

    def _add_toolbar_group(self, title: str, actions: list[tuple[str, object, str]]) -> None:
        group = QFrame()
        group.setObjectName("ToolbarGroup")
        box = QVBoxLayout(group)
        box.setContentsMargins(6, 3, 6, 2)
        box.setSpacing(2)
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(4)
        for text, handler, icon_name in actions:
            btn = QPushButton()
            btn.setObjectName("ToolbarIconButton")
            btn.setToolTip(text)
            btn.setStatusTip(text)
            btn.setFixedSize(42, 36)
            if text in {"Zoom Box", "Window Link"}:
                btn.setCheckable(True)
                self.toolbar_buttons[text] = btn
            icon_path = resource_path("assets", "icons", icon_name)
            if icon_path.exists():
                btn.setIcon(QIcon(str(icon_path)))
                btn.setIconSize(QSize(25, 25))
            else:
                btn.setText(text[:2])
            def _wrap(_checked=False, h=handler, label=text):
                self.statusBar().showMessage(label, 2500)
                h()
            btn.clicked.connect(_wrap)
            row.addWidget(btn)
        label = QLabel(title)
        label.setObjectName("ToolbarGroupLabel")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        box.addLayout(row)
        box.addWidget(label)
        self.toolbar.addWidget(group)

    def _add_toolbar_action(self, text: str, handler, icon_name: str | None = None) -> QAction:
        action = QAction(text, self)
        action.setToolTip(text)
        action.setStatusTip(text)
        if icon_name:
            icon_path = resource_path("assets", "icons", icon_name)
            if icon_path.exists():
                action.setIcon(QIcon(str(icon_path)))
        action.triggered.connect(handler)
        self.toolbar.addAction(action)
        return action

    def _show_export_menu_from_toolbar(self) -> None:
        menu = QMenu(self)
        menu.addAction("Export Raster...", self.export_raster_from_open_layers)
        menu.addAction("Export Vector...", self.export_vector_from_open_layers)
        sender = self.sender()
        if isinstance(sender, QPushButton):
            menu.exec(sender.mapToGlobal(sender.rect().bottomLeft()))
        else:
            menu.exec(self.toolbar.mapToGlobal(self.toolbar.rect().bottomLeft()))

    def _build_statusbar(self) -> None:
        status = QStatusBar()
        self.setStatusBar(status)
        self.lbl_ready = QLabel("Ready")
        self.lbl_cursor = QLabel("Lat/Lon: —")
        status.addWidget(self.lbl_ready, 1)
        status.addPermanentWidget(self.lbl_cursor)

    def _connect_signals(self) -> None:
        self.canvas.cursorInfoChanged.connect(self.update_cursor_status)
        self.canvas.viewChanged.connect(lambda: self._on_canvas_view_changed(self.canvas))

    def _show_terramitra_promo_if_needed(self) -> None:
        """Show the Terramitra Citra Persada promo popup after the main window is visible.

        Uses a zero-delay singleShot + non-modal show() so it never blocks or
        slows down application startup. Appears every time the app opens.
        """

        def _open_popup() -> None:
            self._promo_dialog = TerramitraPromotionDialog(self)
            self._promo_dialog.show()

        QTimer.singleShot(0, _open_popup)

    # ---------- TCS (Tree Counting Sawit) ----------
    def _build_tcs_dialog(self) -> None:
        """TCS lives in its own popup dialog, separate from the core viewer.

        The popup can be closed entirely without touching raster/vector viewing,
        and there is no persistent side panel taking up main-window space.
        """
        self.tcs_panel = TCSPanel(self)

    def open_tcs_panel(self) -> None:
        self.tcs_panel.refresh_rasters()
        self.tcs_panel.show()
        self.tcs_panel.raise_()
        self.tcs_panel.activateWindow()

    def add_tcs_result_layer(self, layer: Layer, source_layer_id: Optional[str] = None) -> None:
        """Show a TCS counting result (a vector layer of detected-tree boxes) as
        its own document window, exactly like opening any other vector file."""
        self._add_layer(layer)

    def remove_layer_by_id(self, layer_id: str) -> None:
        """Close the document window that holds ``layer_id`` (used when a TCS
        preview is discarded). Closing the window removes its layers via
        ``_on_popup_dialog_closed``; falls back to a direct removal otherwise."""
        for dialog, canvases in list(self._dialog_canvases.items()):
            if any(any(lyr.id == layer_id for lyr in c.layer_manager.layers) for c in canvases):
                dialog.close()
                return
        self.layer_manager.remove_layer(layer_id)

    def eventFilter(self, obj, event):  # noqa: N802 - Qt method
        if event.type() == QEvent.Type.WindowActivate and obj in self._dialog_canvases:
            canvases = self._dialog_canvases.get(obj, [])
            if canvases:
                self.active_canvas = canvases[-1]
                active_layer = self.active_canvas.layer_manager.active_layer()
                if active_layer:
                    self.layer_manager.set_active(active_layer.id)
                    self.update_layer_status()
        elif event.type() == QEvent.Type.Close and obj in self._dialog_canvases:
            # A viewer window is a real document window. Once closed, its layer must
            # disappear from Metadata/Export and no stale references remain.
            self._on_popup_dialog_closed(obj)
        return super().eventFilter(obj, event)

    def _on_mdi_subwindow_activated(self, subwindow) -> None:
        if subwindow is None:
            return
        canvases = self._dialog_canvases.get(subwindow, [])
        if canvases:
            self.active_canvas = canvases[-1]
            active_layer = self.active_canvas.layer_manager.active_layer()
            if active_layer:
                self.layer_manager.set_active(active_layer.id)
                self.update_layer_status()
            self._sync_zoom_box_ui()

    def _sync_zoom_box_ui(self) -> None:
        """Keep the Zoom Box toggle in sync with whichever document window is active.

        Zoom Box mode lives on each MapCanvas individually, so switching to a
        different raster/vector document window (or back to the main canvas)
        used to leave the toolbar/menu checkbox showing the previous window's
        state - the button could read "on" while the newly focused canvas was
        still in plain pan mode (or vice versa), making Zoom Box look broken
        or unresponsive on some windows.
        """
        canvas = self._current_canvas()
        enabled = canvas.zoom_box_mode
        if hasattr(self, "act_zoom_box"):
            self.act_zoom_box.setChecked(enabled)
        if "Zoom Box" in self.toolbar_buttons:
            self.toolbar_buttons["Zoom Box"].setChecked(enabled)

    def _set_style(self) -> None:
        self.setStyleSheet("""
        QMainWindow, QWidget { background: #F7F8FA; color: #111827; font-family: Segoe UI, Arial; font-size: 10pt; }
        QMdiArea#MdiWorkspace { background: #FFFFFF; border: none; }
        QMenuBar { background: #F2F3F5; border-bottom: 1px solid #C9CED6; padding: 2px; }
        QMenuBar::item { padding: 5px 10px; background: transparent; }
        QMenuBar::item:selected { background: #E1E5EB; border-radius: 4px; }
        QMenu { background: white; border: 1px solid #BFC5CE; padding: 4px; }
        QMenu::item { padding: 6px 28px 6px 24px; }
        QMenu::item:selected { background: #E8EEF7; color: #111827; }
        QToolBar { background: #E7EBF0; border-bottom: 1px solid #BFC7D2; spacing: 8px; padding: 4px; }
        QFrame#ToolbarGroup { background: #EEF2F6; border-right: 1px solid #BFC7D2; border-left: 1px solid #F9FAFB; padding-left: 4px; padding-right: 4px; }
        QLabel#ToolbarGroupLabel { color: #475569; font-size: 8.5pt; padding-top: 0px; }
        QPushButton#ToolbarIconButton { background: #FDFDFD; border: 1px solid #B7C1CE; border-radius: 3px; padding: 2px; min-width: 34px; min-height: 30px; }
        QPushButton#ToolbarIconButton:hover { background: #E8EEF7; }
        QPushButton#ToolbarIconButton:pressed { background: #D8DEE8; }
        QPushButton#ToolbarIconButton:checked { background: #D9EAFE; border: 2px solid #2563EB; }
        QToolButton, QPushButton { background: #FDFDFD; border: 1px solid #BFC7D2; border-radius: 3px; padding: 4px; color: #111827; }
        QGraphicsView { background: #FFFFFF; border: 1px solid #C9CED6; }
        QStatusBar { background: #F2F3F5; border-top: 1px solid #C9CED6; }
        QTableWidget, QTextEdit { background: #FFFFFF; border: 1px solid #C9CED6; }
        QLabel#DialogTitle { font-size: 16px; font-weight: 700; }
        QToolButton#CanvasOverlayButton {
            background: rgba(253, 253, 253, 0.92); border: 1px solid #BFC7D2; border-radius: 5px;
        }
        QToolButton#CanvasOverlayButton:hover { background: #E8EEF7; border-color: #2563EB; }
        QToolButton#CanvasOverlayButton:pressed { background: #D8DEE8; }
        """)

    # ---------- Opening data ----------
    def open_raster(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Raster",
            str(resource_path("assets", "sample_data")),
            "Raster (*.tif *.tiff *.jpg *.jpeg *.png *.img *.ecw);;"
            "DEM / Elevation (*.tif *.tiff *.asc *.hgt *.bil *.dt0 *.dt1 *.dt2);;"
            "All Files (*)",
        )
        if not path:
            return
        self._start_file_load(
            load_fn=lambda: load_raster(path),
            message="Membuka raster...",
            on_failed=lambda detail: self._on_raster_load_failed(path, detail),
        )

    def _on_raster_load_failed(self, path: str, detail: str) -> None:
        """Handle a failed background raster load.

        raster_loader raises a RuntimeError mentioning "GDAL ECW driver" only
        when the file is .ecw and no usable GDAL/ECW runtime was found - that
        one case gets a chance to fix the runtime and retry. GdalRuntimeDialog
        is a real Qt dialog, so it (and the retry) must run on the GUI thread,
        which is exactly the thread this callback already runs on.
        """
        if "GDAL ECW driver" not in detail:
            self._show_error_box(detail)
            return
        from .gdal_runtime_dialog import GdalRuntimeDialog

        dialog = GdalRuntimeDialog(self)
        dialog.exec()
        if not dialog.ready:
            self._show_error_box(detail)
            return
        self._safe_run(lambda: self._add_layer(load_raster(path)), "Membuka raster...")

    def open_vector(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Vector",
            str(resource_path("assets", "sample_data")),
            "Vector (*.shp *.kml *.kmz *.geojson *.dxf);;All Files (*)",
        )
        if not path:
            return
        self._start_file_load(
            load_fn=lambda: load_vector(path),
            message="Membuka vector...",
            on_failed=self._show_error_box,
        )

    def _start_file_load(self, load_fn, message: str, on_failed, on_success=None) -> None:
        """Run load_fn() (pure I/O, no Qt calls) on a background thread.

        Keeps the window responsive (no Windows "Not Responding") while a
        large raster/vector file is being read, instead of blocking the GUI
        thread for the whole read like a direct call would. on_success
        defaults to opening the layer as its own new window (_add_layer); the
        per-raster overlay button passes a different callback that overlays
        the loaded vector onto one specific canvas instead.
        """
        on_success = on_success or self._add_layer
        self.statusBar().showMessage(message)
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        worker = _FileLoadWorker(load_fn, self)
        self._active_load_worker = worker

        def _finish() -> None:
            QApplication.restoreOverrideCursor()
            self._active_load_worker = None

        def _on_ok(layer) -> None:
            _finish()
            on_success(layer)

        def _on_fail(detail: str) -> None:
            _finish()
            on_failed(detail)

        worker.finished_ok.connect(_on_ok)
        worker.failed.connect(_on_fail)
        worker.start()

    def _add_layer(self, layer: Layer) -> None:
        self.layer_manager.add_layer(layer)
        self.layer_manager.set_active(layer.id)

        # Opening a file always opens its own separate window. Overlaying a
        # vector onto a raster is a deliberate, explicit action from that
        # raster window's own overlay button - never automatic here.
        self._open_layer_popup(layer)
        self.statusBar().showMessage(f"Dibuka: {layer.name}", 4500)
        self.update_layer_status()

    # ---------- Overlay (vector+raster with matching georeference) ----------
    def _crs_match(self, crs_a, crs_b) -> bool:
        """True only when both sides have a real, equivalent CRS/georeference.

        A missing CRS on either side never counts as a match - overlay and
        Window Link both require an explicit, identical georeference.

        Plain pyproj CRS equality (`==`) is too strict for this: a Shapefile's
        .prj is commonly written as ESRI-style WKT with no EPSG authority tag
        (e.g. GEOGCS["GCS_WGS_1984", ...]), which pyproj treats as a different
        CRS object than the EPSG:4326 a GeoTIFF reports - even though both
        describe the exact same coordinate system. Comparing by EPSG code
        first (falling back to axis-order-insensitive .equals()) matches how
        a person would judge "same georeference" instead of failing on that
        kind of harmless WKT formatting difference.
        """
        if crs_a is None or crs_b is None:
            return False
        if PyProjCRS is None:
            return str(crs_a) == str(crs_b)
        try:
            parsed_a = PyProjCRS.from_user_input(crs_a)
            parsed_b = PyProjCRS.from_user_input(crs_b)
        except Exception:
            return str(crs_a) == str(crs_b)
        epsg_a, epsg_b = parsed_a.to_epsg(), parsed_b.to_epsg()
        if epsg_a is not None or epsg_b is not None:
            return epsg_a is not None and epsg_a == epsg_b
        return parsed_a.equals(parsed_b, ignore_axis_order=True)

    def _update_subwindow_title(self, canvas: MapCanvas) -> None:
        names = [lyr.name for lyr in canvas.layer_manager.layers]
        if not names:
            return
        title = " + ".join(names)
        for dialog, canvases in self._dialog_canvases.items():
            if canvas in canvases:
                try:
                    dialog.setWindowTitle(title)
                except Exception:
                    pass
                break

    def _open_layer_popup(self, layer: Layer) -> None:
        """Open raster/vector as an ER Viewer style MDI document.

        The document frame only contains the map canvas. Tools, menu icons, and
        Lat/Lon status are provided by the main window, so maximized documents do
        not duplicate toolbars or status bars.
        """
        manager = LayerManager()
        manager.add_layer(layer)
        manager.set_active(layer.id)

        canvas = MapCanvas(manager)
        canvas.cursorInfoChanged.connect(self.update_cursor_status)
        canvas.viewChanged.connect(lambda c=canvas: self._on_canvas_view_changed(c))
        canvas.overlayVectorRequested.connect(lambda c=canvas: self._pick_vector_for_overlay(c))
        canvas.removeOverlayRequested.connect(lambda c=canvas: self._remove_overlay_from_canvas(c))
        canvas.redraw()
        canvas.reset_view()

        # Create the MDI document with all title-bar controls enabled from
        # construction time. Passing flags to addSubWindow is more reliable on
        # Windows than changing the flags after the QMdiSubWindow is created;
        # it keeps the native Minimize, Maximize/Restore, and Close (X) buttons
        # visible and functional.
        window_flags = (
            Qt.WindowType.SubWindow
            | Qt.WindowType.WindowTitleHint
            | Qt.WindowType.WindowSystemMenuHint
            | Qt.WindowType.WindowMinMaxButtonsHint
            | Qt.WindowType.WindowCloseButtonHint
        )
        sub = self.mdi_area.addSubWindow(canvas, window_flags)
        sub.setWindowTitle(layer.name)
        icon_path = resource_path("assets", "logos", "tcs_icon.ico")
        if icon_path.exists():
            sub.setWindowIcon(QIcon(str(icon_path)))
        sub.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        sub.resize(760, 560)
        sub.installEventFilter(self)

        self.extra_windows.append(sub)
        self._popup_canvases.append(canvas)
        self._dialog_canvases[sub] = [canvas]
        self.active_canvas = canvas
        self.layer_manager.set_active(layer.id)
        # Opening a file always opens its own separate window. A raster window
        # gets its own small overlay button (top-left corner, see
        # MapCanvas.overlay_button) to add a matching vector directly onto it -
        # that is the only way an overlay is created now.
        sub.show()

    def _on_popup_dialog_closed(self, dialog) -> None:
        canvases = self._dialog_canvases.pop(dialog, [])
        if dialog in self.extra_windows:
            self.extra_windows.remove(dialog)
        for canvas in canvases:
            if canvas in self._popup_canvases:
                self._popup_canvases.remove(canvas)
            for layer in list(canvas.layer_manager.layers):
                self.layer_manager.remove_layer(layer.id)
        self.active_canvas = self._popup_canvases[-1] if self._popup_canvases else self.canvas
        active_layer = self.active_canvas.layer_manager.active_layer() if self.active_canvas else None
        self.layer_manager.set_active(active_layer.id if active_layer else None)
        self.update_layer_status()

    # ---------- View / zoom ----------
    def _current_canvas(self) -> MapCanvas:
        return self.active_canvas if self.active_canvas is not None else self.canvas

    def _current_canvas_layer(self) -> Optional[Layer]:
        """Return the layer that belongs to the active viewer document, if any."""
        canvas = self._current_canvas()
        active = canvas.layer_manager.active_layer()
        if active:
            return active
        return self.layer_manager.active_layer()

    def reset_view(self) -> None:
        """Reset the active view and restore the raster to its original image.

        In Terra View, Reset behaves like ER Viewer Data Extents plus a safe
        processing reset: if an enhancement has been applied to the active
        raster, the displayed image is returned to the original raster.
        """
        canvas = self._current_canvas()
        canvas.set_zoom_box_mode(False)
        self._sync_zoom_box_ui()

        layer = self._current_canvas_layer()
        restored_original = False
        if layer and layer.layer_type in {"raster", "dem"} and layer.enhanced_image is not None:
            layer.enhanced_image = None
            restored_original = True
            if not any(lyr.enhanced_image is not None for lyr in self.layer_manager.layers):
                self.enhancement_log = "Belum ada enhancement."

        canvas.redraw()
        canvas.reset_view()
        self.update_layer_status()
        if restored_original:
            self.statusBar().showMessage("Citra dikembalikan ke tampilan asli.", 4500)
        else:
            self.statusBar().showMessage("Tampilan direset.", 3000)

    def zoom_in(self) -> None:
        canvas = self._current_canvas()
        if hasattr(canvas, "_auto_fit"):
            canvas._auto_fit = False
        canvas.scale(1.25, 1.25)
        canvas.viewChanged.emit()

    def zoom_out(self) -> None:
        canvas = self._current_canvas()
        if hasattr(canvas, "_auto_fit"):
            canvas._auto_fit = False
        canvas.scale(0.8, 0.8)
        canvas.viewChanged.emit()

    def toggle_zoom_box(self) -> None:
        canvas = self._current_canvas()
        enabled = not canvas.zoom_box_mode
        canvas.set_zoom_box_mode(enabled)
        self._sync_zoom_box_ui()
        if enabled:
            self.statusBar().showMessage("Zoom Box aktif: tarik kotak untuk memperbesar.", 5000)
        else:
            self.statusBar().showMessage("Zoom Box nonaktif.", 2500)

    def _pick_vector_for_overlay(self, canvas: MapCanvas) -> None:
        """Handle the raster window's top-left overlay button: pick a vector
        file and, once loaded, overlay it onto this specific raster window if
        its georeference matches."""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Overlay Vector",
            str(resource_path("assets", "sample_data")),
            "Vector (*.shp *.kml *.kmz *.geojson *.dxf);;All Files (*)",
        )
        if not path:
            return
        self._start_file_load(
            load_fn=lambda: load_vector(path),
            message="Membuka vector untuk overlay...",
            on_failed=self._show_error_box,
            on_success=lambda layer: self._overlay_vector_into_raster_canvas(canvas, layer),
        )

    def _overlay_vector_into_raster_canvas(self, canvas: MapCanvas, layer: Layer) -> None:
        """Add a freshly-loaded vector layer onto `canvas`, but only when its
        georeference matches the raster already shown there - overlaying two
        different real-world locations on top of each other would be
        meaningless, so that case is refused with an explanation instead."""
        base = canvas.layer_manager.active_base_raster()
        if base is None or not self._crs_match(layer.crs, base.crs):
            self._warn("Georeference vector tidak sesuai, cek kembali vector yang Anda upload.")
            return
        self.layer_manager.add_layer(layer)
        canvas.layer_manager.add_layer(layer)
        canvas.layer_manager.set_active(layer.id)
        self.layer_manager.set_active(layer.id)
        canvas.redraw()
        self._update_subwindow_title(canvas)
        self.active_canvas = canvas
        self.statusBar().showMessage(f'Vector "{layer.name}" di-overlay pada raster ini.', 4500)
        self.update_layer_status()

    def _remove_overlay_from_canvas(self, canvas: MapCanvas) -> None:
        """Remove every vector layer currently overlaid on `canvas`, leaving
        just its raster base - the counterpart to the overlay button."""
        vector_layers = [lyr for lyr in canvas.layer_manager.layers if lyr.layer_type == "vector"]
        if not vector_layers:
            return
        for layer in vector_layers:
            canvas.layer_manager.remove_layer(layer.id)
            self.layer_manager.remove_layer(layer.id)
        canvas.redraw()
        self._update_subwindow_title(canvas)
        self.update_layer_status()
        self.statusBar().showMessage("Overlay dihapus.", 3000)

    def _set_window_link_ui(self, enabled: bool) -> None:
        self.window_link_enabled = enabled
        self.act_window_link.setChecked(enabled)
        if "Window Link" in self.toolbar_buttons:
            self.toolbar_buttons["Window Link"].setChecked(enabled)

    def toggle_window_link(self) -> None:
        """Window Link keeps pan/zoom in sync across every open window whose
        georeference matches - it links any number of windows (not just two),
        and never merges/overlays them; overlaying a vector onto a raster is a
        separate action from that raster window's own overlay button.
        """
        self._set_window_link_ui(not self.window_link_enabled)
        if self.window_link_enabled:
            self.statusBar().showMessage("Window Link aktif: jendela dengan georeference sama akan sinkron.", 4500)
            if self.active_canvas is not None:
                self._on_canvas_view_changed(self.active_canvas)
        else:
            self.statusBar().showMessage("Window Link nonaktif.", 3000)

    def _on_canvas_view_changed(self, source: MapCanvas) -> None:
        """Window Link handler: re-apply the visible real-world area from
        `source` to every other open window whose georeference matches, so
        panning/zooming one window moves every linked window together -
        regardless of how many windows share that georeference.

        blockSignals() on each target while updating it stops that target's
        own viewChanged (fired by the fitInView inside set_visible_world_rect)
        from cascading into another round of syncing.
        """
        if not self.window_link_enabled:
            return
        result = source.visible_world_rect()
        if result is None:
            return
        world_rect, crs = result
        if crs is None:
            return
        for other in list(self._popup_canvases):
            if other is source:
                continue
            if not self._crs_match(crs, self._canvas_crs(other)):
                continue
            other.blockSignals(True)
            try:
                other.set_visible_world_rect(world_rect, crs)
            finally:
                other.blockSignals(False)

    def _canvas_crs(self, canvas: MapCanvas):
        """Representative georeference for a document window (its raster base
        if it has one, otherwise its active vector layer's CRS)."""
        base = canvas.layer_manager.active_base_raster()
        if base is not None:
            return base.crs
        active = canvas.layer_manager.active_layer()
        return active.crs if active else None

    def toggle_toolbar(self) -> None:
        self.toolbar.setVisible(not self.toolbar.isVisible())
        self.act_toolbar.setChecked(self.toolbar.isVisible())

    def toggle_statusbar(self) -> None:
        self.statusBar().setVisible(not self.statusBar().isVisible())
        self.act_statusbar.setChecked(self.statusBar().isVisible())

    # ---------- Analysis helpers ----------
    def _selected_or_base_raster(self) -> Optional[Layer]:
        active = self.layer_manager.active_layer()
        if active and active.layer_type in {"raster", "dem"}:
            return active
        return self.layer_manager.active_base_raster()

    def clear_all(self) -> None:
        self.layer_manager.clear()
        self.canvas.redraw()
        for dlg in list(self.extra_windows):
            try:
                dlg.close()
            except Exception:
                pass
        self.extra_windows.clear()
        self._popup_canvases.clear()
        self._dialog_canvases.clear()
        self.active_canvas = self.canvas
        self.enhancement_log = "Belum ada enhancement."
        self.update_layer_status()
        self.statusBar().showMessage("Semua layer dibersihkan.", 4000)

    # ---------- Export / convert ----------
    def _choose_layer(self, layers, title: str):
        if not layers:
            return None
        if len(layers) == 1:
            return layers[0]
        items = [layer.name for layer in layers]
        item, ok = QInputDialog.getItem(self, title, "Pilih layer yang sedang dibuka:", items, 0, False)
        if not ok or not item:
            return None
        return next((layer for layer in layers if layer.name == item), None)

    def export_raster_from_open_layers(self) -> None:
        raster_layers = self.layer_manager.raster_layers()
        layer = self._choose_layer(raster_layers, "Export Raster")
        if not layer:
            self._warn("Belum ada raster yang dibuka.")
            return
        default = str(resource_path("output", "exports", f"{Path(layer.name).stem}_export.tif"))
        path, _ = QFileDialog.getSaveFileName(self, "Export Raster", default, "GeoTIFF (*.tif);;PNG (*.png);;JPEG (*.jpg)")
        if path:
            self._safe_run(lambda: QMessageBox.information(self, "Export selesai", f"Raster tersimpan:\n{export_enhancement(path, layer)}"), "Export raster...")

    def export_vector_from_open_layers(self) -> None:
        vector_layers = [lyr for lyr in self.layer_manager.layers if lyr.layer_type == "vector"]
        layer = self._choose_layer(vector_layers, "Export Vector")
        if not layer:
            self._warn("Belum ada vector yang dibuka.")
            return
        default = str(resource_path("output", "exports", f"{Path(layer.name).stem}.geojson"))
        path, _ = QFileDialog.getSaveFileName(self, "Export Vector", default, "GeoJSON (*.geojson);;KML (*.kml);;Shapefile (*.shp)")
        if path:
            self._safe_run(lambda: QMessageBox.information(self, "Export selesai", f"Vector tersimpan:\n{convert_vector(layer.path, path)}"), "Export vector...")

    # ---------- Metadata and click status ----------
    def show_metadata_dialog(self) -> None:
        dialog = QDialog(self)
        dialog.setWindowTitle("Metadata / Properties - TCS")
        dialog.resize(480, 420)
        layout = QVBoxLayout(dialog)
        title = QLabel("Metadata / Properties")
        title.setObjectName("DialogTitle")
        layout.addWidget(title)
        table = CopyableInfoTable(0, 2)
        table.setHorizontalHeaderLabels(["Item", "Nilai"])
        table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        table.verticalHeader().setVisible(False)
        table.setWordWrap(True)
        rows = self._metadata_rows()
        table.setRowCount(len(rows))
        for r, (k, v) in enumerate(rows):
            item_key = QTableWidgetItem(str(k))
            item_val = QTableWidgetItem(self._clean_value(v))
            if k == "Statistik":
                item_key.setFlags(item_key.flags() & ~Qt.ItemFlag.ItemIsSelectable)
                item_val.setFlags(item_val.flags() & ~Qt.ItemFlag.ItemIsSelectable)
                item_key.setTextAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
                item_val.setTextAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
                font = item_key.font(); font.setBold(True)
                item_key.setFont(font); item_val.setFont(font)
                item_val.setText("")
            table.setItem(r, 0, item_key)
            table.setItem(r, 1, item_val)
        table.resizeRowsToContents()
        layout.addWidget(table)
        row = QHBoxLayout()
        copy_btn = QPushButton("Copy All")
        copy_btn.clicked.connect(lambda: QApplication.clipboard().setText("\n".join(f"{k}: {self._clean_value(v)}" for k, v in rows)))
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dialog.accept)
        row.addWidget(copy_btn)
        row.addStretch()
        row.addWidget(close_btn)
        layout.addLayout(row)
        dialog.exec()

    def copy_metadata(self) -> None:
        rows = self._metadata_rows()
        QApplication.clipboard().setText("\n".join(f"{k}: {self._clean_value(v)}" for k, v in rows))
        self.statusBar().showMessage("Metadata disalin ke clipboard.", 3000)

    def _metadata_rows(self):
        if not self.layer_manager.layers:
            return [("Status", "Tidak ada file yang masih terbuka")]

        rows = []
        target_layers = self._metadata_target_layers()
        for idx, layer in enumerate(target_layers, start=1):
            rows.extend(self._layer_rows(layer, idx))

        if any(layer.layer_type in {"raster", "dem"} for layer in target_layers) and self.enhancement_log != "Belum ada enhancement.":
            rows.append(("Enhancement", self.enhancement_log))
        return rows

    def _metadata_target_layers(self):
        if self.active_canvas is not None and self.active_canvas is not self.canvas:
            # Show every layer in the active document window - for an overlaid
            # vector+raster pair this lists both, not just whichever is active.
            layers = list(self.active_canvas.layer_manager.layers)
            if layers:
                return layers
        active = self.layer_manager.active_layer()
        if active:
            return [active]
        return list(self.layer_manager.layers)

    def _layer_rows(self, layer: Layer, idx: int):
        path = Path(layer.path) if layer.path else None
        rows = [
            (f"Layer {idx}", layer.name),
            ("Jenis", "Raster" if layer.layer_type in {"raster", "dem"} else "Vector"),
            ("Format", path.suffix.lower() if path else "Tidak tersedia"),
            ("Lokasi", str(path) if path else "Tidak tersedia"),
            ("CRS", self._format_crs(layer.crs)),
        ]
        if layer.layer_type in {"raster", "dem"}:
            rows.append(("Ukuran raster", f"{layer.width} x {layer.height} px" if layer.width and layer.height else "Tidak tersedia"))
            band_count = layer.profile.get("count") if layer.profile else None
            if band_count is None:
                band_count = 1 if layer.dem_data is not None else 3
            rows.append(("Jumlah band", str(band_count)))
            pixel_size = self._format_pixel_size(layer.transform)
            if pixel_size:
                rows.append(("Pixel size / resolusi", pixel_size))
            if layer.dem_data is not None:
                # Single-band numeric raster (DEM/elevation): report real
                # elevation statistics, not the stretched 0-255 preview image.
                rows.append(("Elevasi", ""))
                rows.extend(self._statistic_rows(layer.dem_data, labels=("Min elevation", "Max elevation", "Mean elevation", "Median elevation", "Std Dev elevation")))
            elif layer.display_image() is not None:
                rows.append(("Statistik", ""))
                rows.extend(self._statistic_rows(layer.display_image()))
        elif layer.layer_type == "vector":
            rows.append(("Driver", layer.source_driver or "Tidak tersedia"))
        return rows

    def _format_pixel_size(self, transform) -> str:
        if transform is None:
            return ""
        try:
            px = abs(transform.a)
            py = abs(transform.e)
            return f"{px:.6f} x {py:.6f}"
        except Exception:
            return ""

    def _statistic_rows(self, image, labels=("Min", "Max", "Mean", "Median", "Std Dev")):
        import numpy as _np
        arr = _np.asarray(image)
        rows = []
        finite = _np.isfinite(arr)
        if not finite.any():
            return [("Statistik", "Tidak tersedia (semua NoData/NaN)")]
        try:
            min_lbl, max_lbl, mean_lbl, median_lbl, std_lbl = labels
            rows.append((min_lbl, f"{float(_np.nanmin(arr)):.2f}"))
            rows.append((max_lbl, f"{float(_np.nanmax(arr)):.2f}"))
            rows.append((mean_lbl, f"{float(_np.nanmean(arr)):.2f}"))
            rows.append((median_lbl, f"{float(_np.nanmedian(arr)):.2f}"))
            rows.append((std_lbl, f"{float(_np.nanstd(arr)):.2f}"))
            if arr.ndim == 3 and arr.shape[2] >= 3:
                rows.append(("R mean", f"{float(_np.nanmean(arr[:, :, 0])):.2f}"))
                rows.append(("G mean", f"{float(_np.nanmean(arr[:, :, 1])):.2f}"))
                rows.append(("B mean", f"{float(_np.nanmean(arr[:, :, 2])):.2f}"))
        except Exception:
            rows.append(("Statistik", "Tidak tersedia"))
        return rows

    def update_cursor_status(self, info: Dict[str, object]) -> None:
        lat = info.get("latitude")
        lon = info.get("longitude")
        if lat is not None and lon is not None:
            self.lbl_cursor.setText(f"Lat/Lon: {float(lat):.6f}, {float(lon):.6f}")
        else:
            self.lbl_cursor.setText("Lat/Lon: —")

    def update_layer_status(self) -> None:
        # Status bar intentionally shows only live latitude/longitude.
        # CRS, RGB, and resolution remain available in Metadata/Properties.
        if not self.layer_manager.active_layer():
            self.lbl_cursor.setText("Lat/Lon: —")

    # ---------- helpers ----------
    def _safe_run(self, func, message: str = "Memproses...") -> None:
        try:
            self.statusBar().showMessage(message)
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            QApplication.processEvents()
            func()
        except Exception as exc:
            self._show_error_box(str(exc).strip() or repr(exc))
        finally:
            QApplication.restoreOverrideCursor()

    def _show_error_box(self, detail: str) -> None:
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Critical)
        box.setWindowTitle("TCS Error")
        box.setText(detail)
        box.setMinimumWidth(680)
        box.exec()

    def _warn(self, text: str) -> None:
        QMessageBox.warning(self, "TCS", text)

    def _format_crs(self, crs) -> str:
        if not crs:
            return "Tidak tersedia"
        text = str(crs)
        if PyProjCRS is not None:
            try:
                c = PyProjCRS.from_user_input(crs)
                epsg = c.to_epsg()
                name = c.name or "CRS"
                return f"EPSG:{epsg} - {name}" if epsg else name
            except Exception:
                pass
        m = re.search(r'AUTHORITY\["EPSG","(\d+)"\]', text)
        if m:
            return f"EPSG:{m.group(1)}"
        if len(text) > 80:
            return text[:77] + "..."
        return text

    def _format_bounds(self, bounds) -> str:
        if not bounds:
            return "Tidak tersedia"
        try:
            minx, miny, maxx, maxy = [float(v) for v in bounds]
            return f"X {minx:.3f} – {maxx:.3f}; Y {miny:.3f} – {maxy:.3f}"
        except Exception:
            return str(bounds)

    def _clean_value(self, value) -> str:
        if value is None or value == "-":
            return "Tidak tersedia"
        return str(value)

def create_main_window() -> TCSMainWindow:
    return TCSMainWindow()
