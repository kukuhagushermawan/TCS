"""Map canvas for Terra View."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from PyQt6.QtCore import QPointF, QRectF, QSize, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QIcon, QImage, QPainter, QPainterPath, QPen, QPixmap, QBrush
from PyQt6.QtWidgets import (
    QGraphicsEllipseItem,
    QGraphicsItemGroup,
    QGraphicsPathItem,
    QGraphicsPixmapItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsView,
    QToolButton,
)

from .coordinate_tools import build_click_info, pixel_to_world, to_latlon, world_to_pixel
from .layer_manager import Layer, LayerManager
from .resources import resource_path
from .vector_loader import iter_geometry_coords, iter_geometry_parts

# All vector outlines are drawn in one plain colour - there is no per-feature
# classification anymore, just a single, clearly visible red.
VECTOR_COLOR = "#FF0000"

try:
    from pyproj import CRS, Transformer
except Exception:  # pragma: no cover
    CRS = None
    Transformer = None


class MapCanvas(QGraphicsView):
    """2D canvas with zoom, pan, and raster/vector overlay."""

    cursorInfoChanged = pyqtSignal(dict)
    # Emitted whenever the user pans or zooms this window (wheel, drag, zoom
    # box). Window Link listens to this on every open window and, when
    # enabled, re-applies the same real-world visible area to every other
    # window with a matching georeference - blockSignals() on the receiving
    # end (see ui_main._on_canvas_view_changed) stops this from cascading.
    viewChanged = pyqtSignal()
    # Emitted when the top-left overlay button (shown on raster windows only)
    # is clicked, asking the main window to pick a vector file and overlay it
    # onto this specific canvas if its georeference matches.
    overlayVectorRequested = pyqtSignal()
    # Emitted by the button next to it, asking the main window to remove
    # whatever vector is currently overlaid on this canvas.
    removeOverlayRequested = pyqtSignal()

    def __init__(self, layer_manager: LayerManager, parent=None) -> None:
        super().__init__(parent)
        self.layer_manager = layer_manager
        self.scene_obj = QGraphicsScene(self)
        self.setScene(self.scene_obj)
        self.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setMouseTracking(True)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)

        self._vector_bounds: Optional[Tuple[float, float, float, float]] = None
        self._vector_scene_size = (1000.0, 700.0)
        self.zoom_box_mode = False
        self._zoom_box_start: Optional[QPointF] = None
        self._zoom_box_item: Optional[QGraphicsRectItem] = None
        self._auto_fit = True
        # Cache of pyproj Transformers keyed by (source_crs, dst_crs). Building a
        # Transformer is expensive (milliseconds each); without this cache the
        # old code built one per *coordinate point*, which froze the GUI for
        # seconds when a large vector was overlaid onto a raster via Window Link.
        self._transformer_cache: Dict[Tuple[str, str], Any] = {}
        # Single shared pen for every vector outline drawn on this canvas.
        self._vector_pen = QPen(QColor(VECTOR_COLOR), 2)

        # Small floating button in the top-left corner, shown only while this
        # window has an active raster base (see redraw()). Lets the user pick
        # a vector file and overlay it directly onto this raster, provided its
        # georeference matches - the only way to create an overlay now.
        self.overlay_button = QToolButton(self)
        self.overlay_button.setObjectName("CanvasOverlayButton")
        self.overlay_button.setToolTip("Tambahkan Vector Overlay")
        self.overlay_button.setFixedSize(30, 30)
        self.overlay_button.move(8, 8)
        self.overlay_button.setCursor(Qt.CursorShape.PointingHandCursor)
        icon_path = resource_path("assets", "icons", "open vector.png")
        if icon_path.exists():
            self.overlay_button.setIcon(QIcon(str(icon_path)))
            self.overlay_button.setIconSize(QSize(18, 18))
        else:
            self.overlay_button.setText("V")
        self.overlay_button.clicked.connect(self.overlayVectorRequested.emit)

        # Sits to the right of overlay_button, only shown once this raster
        # window actually has a vector overlaid on it (see redraw()) - removes
        # that overlay, going back to the plain raster.
        self.remove_overlay_button = QToolButton(self)
        self.remove_overlay_button.setObjectName("CanvasOverlayButton")
        self.remove_overlay_button.setToolTip("Hapus Overlay")
        self.remove_overlay_button.setFixedSize(30, 30)
        self.remove_overlay_button.move(8 + 30 + 6, 8)
        self.remove_overlay_button.setCursor(Qt.CursorShape.PointingHandCursor)
        remove_icon_path = resource_path("assets", "icons", "hapus_overlay.png")
        if remove_icon_path.exists():
            self.remove_overlay_button.setIcon(QIcon(str(remove_icon_path)))
            self.remove_overlay_button.setIconSize(QSize(18, 18))
        else:
            self.remove_overlay_button.setText("X")
        self.remove_overlay_button.clicked.connect(self.removeOverlayRequested.emit)
        self.overlay_button.hide()
        self.remove_overlay_button.hide()
        self.overlay_button.raise_()
        self.remove_overlay_button.raise_()

    def set_zoom_box_mode(self, enabled: bool) -> None:
        self.zoom_box_mode = enabled
        if enabled:
            self.setDragMode(QGraphicsView.DragMode.NoDrag)
            self.setCursor(Qt.CursorShape.CrossCursor)
        else:
            self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
            self.unsetCursor()
            self._clear_zoom_box_preview()

    def _clear_zoom_box_preview(self) -> None:
        if self._zoom_box_item is not None:
            try:
                self.scene_obj.removeItem(self._zoom_box_item)
            except Exception:
                pass
        self._zoom_box_item = None
        self._zoom_box_start = None

    def redraw(self) -> None:
        self.scene_obj.clear()
        active = self.layer_manager.active_layer()
        base = self.layer_manager.active_base_raster()
        # Overlay button only makes sense on a window that has a raster to
        # overlay a vector onto. Remove-overlay only makes sense once a
        # vector is actually overlaid on that raster.
        has_vector_overlay = base is not None and any(
            lyr.layer_type == "vector" for lyr in self.layer_manager.layers
        )
        self.overlay_button.setVisible(base is not None)
        self.remove_overlay_button.setVisible(has_vector_overlay)

        # Vector-only mode: only when this canvas has no raster/DEM base at all.
        # Checking only "is the active layer a vector" used to be enough back
        # when a canvas only ever held one layer, but an overlaid raster+vector
        # canvas can easily have a vector as its *active* layer (e.g. right
        # after it was merged in) while still needing its raster base drawn -
        # using active_layer_type alone would wrongly hide the raster and fall
        # back to the small synthetic vector coordinate space instead.
        if base is None and active and active.visible and active.layer_type == "vector":
            # Draw every visible vector layer (not just the active one) over a
            # shared coordinate space, so overlaying two or more vector windows
            # via Window Link shows all of them instead of hiding the rest.
            self._vector_bounds = self._visible_vector_bounds() or active.bounds
            self.setSceneRect(QRectF(0, 0, self._vector_scene_size[0], self._vector_scene_size[1]))
            if self._vector_bounds:
                for layer in self.layer_manager.visible_layers():
                    if layer.layer_type == "vector":
                        if layer.metadata.get("tcs_style"):
                            group = self._draw_tcs_style_layer_no_base(layer, self._vector_bounds)
                        else:
                            group = self._draw_vector_layer_no_base(layer, self._vector_bounds)
                        group.setOpacity(layer.opacity)
                        group.setZValue(10)
                        self.scene_obj.addItem(group)
            self.viewport().update()
            return

        if base:
            self._vector_bounds = None
            img = base.display_image()
            if img is not None:
                pix = numpy_rgb_to_qpixmap(img)
                item = QGraphicsPixmapItem(pix)
                item.setOpacity(base.opacity)
                item.setZValue(0)
                self.scene_obj.addItem(item)

            # Raster mode: draw visible vector layers as overlays. TCS result
            # layers (tagged via layer.metadata["tcs_style"]) get a distinct
            # marker+label rendering instead of the generic single-colour
            # outline used by every other vector file.
            for layer in self.layer_manager.visible_layers():
                if layer.layer_type == "vector":
                    if layer.metadata.get("tcs_style"):
                        group = self._draw_tcs_style_layer(layer, base)
                    else:
                        group = self._draw_vector_layer(layer, base)
                    group.setOpacity(layer.opacity)
                    group.setZValue(10)
                    self.scene_obj.addItem(group)

            if base.width and base.height:
                self.setSceneRect(QRectF(0, 0, float(base.width), float(base.height)))
        else:
            self._vector_bounds = self._visible_vector_bounds()
            self.setSceneRect(QRectF(0, 0, self._vector_scene_size[0], self._vector_scene_size[1]))
            if self._vector_bounds:
                for layer in self.layer_manager.visible_layers():
                    if layer.layer_type == "vector":
                        if layer.metadata.get("tcs_style"):
                            group = self._draw_tcs_style_layer_no_base(layer, self._vector_bounds)
                        else:
                            group = self._draw_vector_layer_no_base(layer, self._vector_bounds)
                        group.setOpacity(layer.opacity)
                        group.setZValue(10)
                        self.scene_obj.addItem(group)

        self.viewport().update()

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt method
        super().resizeEvent(event)
        if getattr(self, "_auto_fit", False) and not self.sceneRect().isNull():
            self.fitInView(self.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def reset_view(self) -> None:
        self._auto_fit = True
        self.fitInView(self.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def scrollContentsBy(self, dx: int, dy: int) -> None:  # noqa: N802 - Qt method
        super().scrollContentsBy(dx, dy)
        self.viewChanged.emit()

    def wheelEvent(self, event) -> None:  # noqa: N802 - Qt method
        self._auto_fit = False
        factor = 1.25 if event.angleDelta().y() > 0 else 0.8
        self.scale(factor, factor)
        self.viewChanged.emit()

    def mousePressEvent(self, event) -> None:  # noqa: N802 - Qt method
        if self.zoom_box_mode and event.button() == Qt.MouseButton.LeftButton:
            # Clear old preview first, then store the new start point.
            # Previous code cleared after assigning and accidentally reset the start point,
            # making Zoom Box feel non-functional.
            self._clear_zoom_box_preview()
            self._zoom_box_start = self.mapToScene(event.pos())
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802 - Qt method
        if not self.zoom_box_mode:
            super().mouseMoveEvent(event)
        scene_pos = self.mapToScene(event.pos())
        info = self._scene_info(scene_pos)
        self.cursorInfoChanged.emit(info)
        if self.zoom_box_mode and self._zoom_box_start is not None:
            if self._zoom_box_item is not None:
                try:
                    self.scene_obj.removeItem(self._zoom_box_item)
                except Exception:
                    pass
            rect = QRectF(self._zoom_box_start, scene_pos).normalized()
            self._zoom_box_item = QGraphicsRectItem(rect)
            self._zoom_box_item.setPen(QPen(QColor("#FACC15"), 3, Qt.PenStyle.SolidLine))
            self._zoom_box_item.setBrush(QBrush(QColor(250, 204, 21, 35)))
            self._zoom_box_item.setZValue(250)
            self.scene_obj.addItem(self._zoom_box_item)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802 - Qt method
        if self.zoom_box_mode and event.button() == Qt.MouseButton.LeftButton:
            end_pos = self.mapToScene(event.pos())
            if self._zoom_box_start is not None:
                rect = QRectF(self._zoom_box_start, end_pos).normalized()
                if rect.width() > 5 and rect.height() > 5:
                    self._auto_fit = False
                    self.fitInView(rect, Qt.AspectRatioMode.KeepAspectRatio)
                    self.viewChanged.emit()
            self._clear_zoom_box_preview()
            event.accept()
            return

        super().mouseReleaseEvent(event)

    # ---------- Window Link (pan/zoom sync across windows) ----------
    def _canvas_world_crs(self) -> Any:
        """CRS that this canvas's scene coordinates are currently expressed in
        - the active raster base's CRS, or the active vector layer's CRS when
        there is no raster base."""
        base = self.layer_manager.active_base_raster()
        if base is not None:
            return base.crs
        active = self.layer_manager.active_layer()
        return active.crs if active else None

    def visible_world_rect(self) -> Optional[Tuple[Tuple[float, float, float, float], Any]]:
        """Return ``(world_bbox, crs)`` for what is currently visible in the
        viewport, in real-world coordinates rather than this canvas's own
        scene/pixel space - so a different window (different raster
        resolution, or a vector-only synthetic scene) can be made to show the
        same real place."""
        if self.sceneRect().isNull():
            return None
        visible = self.mapToScene(self.viewport().rect()).boundingRect()
        corners = [
            (visible.left(), visible.top()), (visible.right(), visible.top()),
            (visible.right(), visible.bottom()), (visible.left(), visible.bottom()),
        ]
        base = self.layer_manager.active_base_raster()
        if base is not None:
            world_pts = [pixel_to_world(base.transform, cx, cy) for cx, cy in corners]
            crs = base.crs
        elif self._vector_bounds is not None:
            world_pts = [self._scene_to_vector_world(cx, cy, self._vector_bounds) for cx, cy in corners]
            crs = self._canvas_world_crs()
        else:
            return None
        xs = [p[0] for p in world_pts]
        ys = [p[1] for p in world_pts]
        return (min(xs), min(ys), max(xs), max(ys)), crs

    def set_visible_world_rect(self, world_rect: Tuple[float, float, float, float], src_crs: Any) -> bool:
        """Fit this canvas's view to the given real-world bbox, reprojecting
        from src_crs if it differs from this canvas's own CRS. Returns False
        when this canvas has no coordinate space to place the rect into."""
        minx, miny, maxx, maxy = world_rect
        corners = [(minx, miny), (maxx, miny), (maxx, maxy), (minx, maxy)]
        base = self.layer_manager.active_base_raster()
        if base is not None:
            transformer = self._get_transformer(src_crs, base.crs)
            scene_pts = [self._reproject_and_map(wx, wy, transformer, lambda tx, ty: world_to_pixel(base.transform, tx, ty)) for wx, wy in corners]
        elif self._vector_bounds is not None:
            transformer = self._get_transformer(src_crs, self._canvas_world_crs())
            bounds = self._vector_bounds
            scene_pts = [self._reproject_and_map(wx, wy, transformer, lambda tx, ty: self._vector_world_to_scene(tx, ty, bounds)) for wx, wy in corners]
        else:
            return False
        xs = [p[0] for p in scene_pts]
        ys = [p[1] for p in scene_pts]
        rect = QRectF(min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys))
        if rect.width() <= 0 or rect.height() <= 0:
            return False
        self._auto_fit = False
        self.fitInView(rect, Qt.AspectRatioMode.KeepAspectRatio)
        return True

    def _reproject_and_map(self, wx: float, wy: float, transformer: Any, to_scene) -> Tuple[float, float]:
        tx, ty = wx, wy
        if transformer is not None:
            try:
                tx, ty = transformer.transform(wx, wy)
            except Exception:
                tx, ty = wx, wy
        return to_scene(tx, ty)

    # ---------- Info and drawing ----------
    def _scene_info(self, scene_pos: QPointF) -> Dict[str, Any]:
        base = self.layer_manager.active_base_raster()
        if base:
            col = int(round(scene_pos.x()))
            row = int(round(scene_pos.y()))
            return build_click_info(base, col, row)

        if self._vector_bounds:
            x, y = self._scene_to_vector_world(scene_pos.x(), scene_pos.y(), self._vector_bounds)
            active = self.layer_manager.active_layer()
            top_layer = active if active and active.layer_type in {"vector"} else next((lyr for lyr in reversed(self.layer_manager.visible_layers()) if lyr.layer_type in {"vector"}), None)
            top_crs = getattr(top_layer, "crs", None)
            # Reproject through the vector's own CRS instead of guessing from
            # the raw coordinate range - a projected CRS (UTM, meters, ...)
            # has x/y values nowhere near -180..180/-90..90, so that guess
            # silently reported "no lat/lon" for anything but already-geographic
            # vector data even though the CRS was known and reprojectable.
            lat, lon = to_latlon(x, y, top_crs)
            return {
                "layer": top_layer.name if top_layer else "Vector",
                "pixel_x": int(scene_pos.x()),
                "pixel_y": int(scene_pos.y()),
                "x": x,
                "y": y,
                "latitude": lat,
                "longitude": lon,
                "crs": getattr(top_layer, "crs", None) or "Vector/point-cloud",
                "rgb": None,
                "hex": None,
                "elevation": None,
            }

        return {"layer": "None", "pixel_x": int(scene_pos.x()), "pixel_y": int(scene_pos.y())}

    def _draw_vector_layer(self, layer: Layer, base: Layer) -> QGraphicsItemGroup:
        group = QGraphicsItemGroup()
        for feat in layer.features:
            geom = feat.get("geometry")
            # Draw each ring/part separately so the virtual pen is lifted
            # between polygon rings. This prevents diagonal spider-web lines.
            for gtype, coords in iter_geometry_parts(geom):
                if not coords:
                    continue
                pix_coords = [self._world_to_base_pixel(x, y, layer.crs, base) for x, y in coords]
                self._add_geometry_to_group(group, gtype, pix_coords, self._vector_pen)
        return group

    def _draw_vector_layer_no_base(self, layer: Layer, bounds: Tuple[float, float, float, float]) -> QGraphicsItemGroup:
        group = QGraphicsItemGroup()
        for feat in layer.features:
            geom = feat.get("geometry")
            # Draw each ring/part separately so unrelated rings are not connected.
            for gtype, coords in iter_geometry_parts(geom):
                if not coords:
                    continue
                pix_coords = [self._vector_world_to_scene(x, y, bounds) for x, y in coords]
                self._add_geometry_to_group(group, gtype, pix_coords, self._vector_pen)
        return group

    # TCS overlay colour: bold red = detected tree - highly visible on top of
    # green palm-canopy imagery, distinct from any other vector overlay.
    _TCS_TREE_COLOR = "#DC2626"

    def _draw_tcs_style_layer(self, layer: Layer, base: Layer) -> QGraphicsItemGroup:
        """TCS-specific overlay rendering: a bold red circle per detected tree -
        completely separate from ``_draw_vector_layer``/``_add_geometry_to_group``,
        which stay unchanged for every other vector file."""
        group = QGraphicsItemGroup()
        pen = QPen(QColor(self._TCS_TREE_COLOR), 3)
        for feat in layer.features:
            geom = feat.get("geometry")
            if not geom or geom.get("type") != "Point":
                continue
            x, y = geom["coordinates"][:2]
            px, py = self._world_to_base_pixel(x, y, layer.crs, base)
            self._add_circle_marker(group, px, py, pen)
        return group

    def _add_circle_marker(self, group: QGraphicsItemGroup, x: float, y: float, pen: QPen, radius: float = 26.0) -> None:
        ellipse = QGraphicsEllipseItem(x - radius, y - radius, radius * 2, radius * 2)
        ellipse.setPen(pen)
        fill = QColor(pen.color())
        fill.setAlpha(255)
        ellipse.setBrush(fill)
        group.addToGroup(ellipse)

    def _draw_tcs_style_layer_no_base(self, layer: Layer, bounds: Tuple[float, float, float, float]) -> QGraphicsItemGroup:
        """Same red-circle rendering as ``_draw_tcs_style_layer``, but
        mapped into the synthetic vector-only coordinate space
        (``_vector_world_to_scene``) instead of a raster base's pixel space -
        TCS result windows never carry a raster base of their own, so this is
        the path they actually render through."""
        group = QGraphicsItemGroup()
        pen = QPen(QColor(self._TCS_TREE_COLOR), 3)
        for feat in layer.features:
            geom = feat.get("geometry")
            if not geom or geom.get("type") != "Point":
                continue
            x, y = geom["coordinates"][:2]
            px, py = self._vector_world_to_scene(x, y, bounds)
            self._add_circle_marker(group, px, py, pen)
        return group

    def _add_geometry_to_group(self, group: QGraphicsItemGroup, gtype: str, pix_coords: List[Tuple[float, float]], pen: QPen) -> None:
        if not pix_coords:
            return
        if gtype == "Point" or len(pix_coords) == 1:
            x, y = pix_coords[0]
            ellipse = QGraphicsEllipseItem(x - 3, y - 3, 6, 6)
            ellipse.setPen(pen)
            fill = QColor(pen.color())
            fill.setAlpha(120)
            ellipse.setBrush(fill)
            group.addToGroup(ellipse)
        else:
            path = QPainterPath(QPointF(pix_coords[0][0], pix_coords[0][1]))
            for x, y in pix_coords[1:]:
                path.lineTo(x, y)
            if gtype in {"Polygon", "PolygonRing"}:
                # Outline only: polygons show just their coloured boundary (no
                # fill) so the raster/imagery underneath stays fully visible.
                path.closeSubpath()
            item = QGraphicsPathItem(path)
            item.setPen(pen)
            # No setBrush(): a QGraphicsPathItem defaults to no fill, so polygons
            # render as outline only and the imagery underneath stays visible.
            group.addToGroup(item)

    def _get_transformer(self, source_crs: Any, dst_crs: Any) -> Any:
        """Return a cached pyproj Transformer for source_crs -> dst_crs, or None
        if no reprojection is needed/possible. Cached because building one is
        expensive and a single vector layer reuses the same pair for every point."""
        if not (source_crs and dst_crs and CRS is not None and Transformer is not None):
            return None
        key = (str(source_crs), str(dst_crs))
        if key in self._transformer_cache:
            return self._transformer_cache[key]
        transformer = None
        try:
            src = CRS.from_user_input(source_crs)
            dst = CRS.from_user_input(dst_crs)
            if src != dst:
                transformer = Transformer.from_crs(src, dst, always_xy=True)
        except Exception:
            transformer = None
        self._transformer_cache[key] = transformer
        return transformer

    def _world_to_base_pixel(self, x: float, y: float, source_crs: Any, base: Layer) -> Tuple[float, float]:
        tx, ty = float(x), float(y)
        transformer = self._get_transformer(source_crs, base.crs)
        if transformer is not None:
            try:
                tx, ty = transformer.transform(x, y)
            except Exception:
                tx, ty = float(x), float(y)
        return world_to_pixel(base.transform, tx, ty)

    def _draw_vector_preview_label(self, layer: Layer) -> None:
        # Intentionally hidden: Terra View follows ER Viewer style with a clean canvas.
        return

    def _visible_vector_bounds(self, layers: Optional[List[Layer]] = None) -> Optional[Tuple[float, float, float, float]]:
        bounds_list = []
        for layer in (layers if layers is not None else self.layer_manager.visible_layers()):
            if layer.layer_type in {"vector"} and layer.bounds:
                bounds_list.append(layer.bounds)
        if not bounds_list:
            return None
        minx = min(b[0] for b in bounds_list)
        miny = min(b[1] for b in bounds_list)
        maxx = max(b[2] for b in bounds_list)
        maxy = max(b[3] for b in bounds_list)
        if minx == maxx:
            minx -= 0.001
            maxx += 0.001
        if miny == maxy:
            miny -= 0.001
            maxy += 0.001
        return float(minx), float(miny), float(maxx), float(maxy)

    def _vector_world_to_scene(self, x: float, y: float, bounds: Tuple[float, float, float, float]) -> Tuple[float, float]:
        minx, miny, maxx, maxy = bounds
        w, h = self._vector_scene_size
        pad = 48.0
        scale = min((w - 2 * pad) / max(maxx - minx, 1e-9), (h - 2 * pad) / max(maxy - miny, 1e-9))
        sx = pad + (float(x) - minx) * scale
        sy = pad + (maxy - float(y)) * scale
        return sx, sy

    def _scene_to_vector_world(self, sx: float, sy: float, bounds: Tuple[float, float, float, float]) -> Tuple[float, float]:
        minx, miny, maxx, maxy = bounds
        w, h = self._vector_scene_size
        pad = 48.0
        scale = min((w - 2 * pad) / max(maxx - minx, 1e-9), (h - 2 * pad) / max(maxy - miny, 1e-9))
        x = minx + (float(sx) - pad) / scale
        y = maxy - (float(sy) - pad) / scale
        return float(x), float(y)

def numpy_rgb_to_qpixmap(image: np.ndarray) -> QPixmap:
    arr = np.ascontiguousarray(image)
    if arr.ndim == 2:
        arr = np.repeat(arr[:, :, None], 3, axis=2)
    if arr.shape[2] == 4:
        fmt = QImage.Format.Format_RGBA8888
    else:
        arr = arr[:, :, :3]
        fmt = QImage.Format.Format_RGB888
    h, w = arr.shape[:2]
    qimg = QImage(arr.data, w, h, arr.strides[0], fmt)
    return QPixmap.fromImage(qimg.copy())
