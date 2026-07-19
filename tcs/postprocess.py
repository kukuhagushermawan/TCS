"""Detections -> vector post-processing for TCS.

Runs cross-tile Non-Maximum Suppression on bounding boxes (so a tree sitting
in a tile-overlap region is not counted twice), converts surviving boxes to
georeferenced bounding-box (rectangle) features using the same affine-transform
convention as ``app.coordinate_tools`` - so the result layer draws a box around
each detected tree directly over the source raster - and optionally assigns
each box to a plantation block polygon for a per-block recap.

Saving reuses app.format_converter's existing GeoJSON/Shapefile writers so
TCS output loads through the exact same vector code path as any other file.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

from .inference import Detection

try:
    from pyproj import CRS, Transformer
except Exception:  # pragma: no cover
    CRS = None
    Transformer = None


def nms(detections: List[Detection], iou_threshold: float) -> List[Detection]:
    """Greedy IoU-based Non-Maximum Suppression, highest confidence first.

    Operates on detections already translated into full-raster pixel space,
    so it also suppresses duplicate detections of the same tree produced by
    neighboring overlapping tiles ("NMS lintas-tile").
    """
    if not detections:
        return []
    boxes = np.array([[d.x1, d.y1, d.x2, d.y2] for d in detections], dtype=np.float64)
    scores = np.array([d.confidence for d in detections], dtype=np.float64)
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    order = scores.argsort()[::-1]

    keep: List[int] = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        union = areas[i] + areas[rest] - inter
        iou = np.where(union > 0, inter / union, 0.0)
        order = rest[iou <= iou_threshold]
    return [detections[i] for i in keep]


def detections_to_point_features(detections: List[Detection], transform: Any) -> List[Dict[str, Any]]:
    """Bbox center -> point Feature in the raster's map CRS (pixel_to_world reused).

    TCS output is plain point/vector data (one point per detected tree),
    matching how any other vector layer opens/exports in this viewer - not a
    box burned into a copy of the raster's pixels.
    """
    from app.coordinate_tools import pixel_to_world

    features: List[Dict[str, Any]] = []
    for i, det in enumerate(detections, start=1):
        cx = (det.x1 + det.x2) / 2.0
        cy = (det.y1 + det.y2) / 2.0
        x, y = pixel_to_world(transform, cx, cy)
        features.append({
            "type": "Feature",
            "id": str(i),
            "geometry": {"type": "Point", "coordinates": [x, y]},
            "properties": {"tree_id": i, "confidence": round(float(det.confidence), 3)},
        })
    return features


def detections_to_box_features(detections: List[Detection], transform: Any) -> List[Dict[str, Any]]:
    """Bbox corners -> rectangle Polygon Feature in the raster's map CRS.

    Kept for callers that want the actual detected box instead of just its
    center point (not used by the default TCS output - see
    ``detections_to_point_features``).
    """
    from app.coordinate_tools import pixel_to_world

    features: List[Dict[str, Any]] = []
    for i, det in enumerate(detections, start=1):
        corners_px = [(det.x1, det.y1), (det.x2, det.y1), (det.x2, det.y2), (det.x1, det.y2)]
        ring = [list(pixel_to_world(transform, x, y)) for x, y in corners_px]
        ring.append(ring[0])
        features.append({
            "type": "Feature",
            "id": str(i),
            "geometry": {"type": "Polygon", "coordinates": [ring]},
            "properties": {"tree_id": i, "confidence": round(float(det.confidence), 3)},
        })
    return features


UNASSIGNED_BLOCK = "Tidak masuk blok"


def assign_blocks(features: List[Dict[str, Any]], source_crs: Any, boundary_layer: Any) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Tag each detection feature (Point or Polygon box) with the plantation
    block polygon that contains its centroid.

    ``boundary_layer`` is an already-loaded vector Layer with Polygon/MultiPolygon
    features (e.g. block boundaries). Reprojects boundary geometry into
    ``source_crs`` when the CRS differs, following the same pattern as
    ``app.map_canvas._world_to_base_pixel``.
    """
    from shapely.geometry import shape as shapely_shape

    transformer = None
    if source_crs and getattr(boundary_layer, "crs", None) and CRS is not None and Transformer is not None:
        try:
            src = CRS.from_user_input(boundary_layer.crs)
            dst = CRS.from_user_input(source_crs)
            if src != dst:
                transformer = Transformer.from_crs(src, dst, always_xy=True)
        except Exception:
            transformer = None

    def reproject_ring(ring):
        if transformer is None:
            return ring
        return [list(transformer.transform(x, y)) for x, y in ring]

    def reproject_geometry(geom: Dict[str, Any]) -> Dict[str, Any]:
        if transformer is None:
            return geom
        gtype = geom.get("type")
        coords = geom.get("coordinates")
        if gtype == "Polygon":
            return {"type": "Polygon", "coordinates": [reproject_ring(r) for r in coords]}
        if gtype == "MultiPolygon":
            return {"type": "MultiPolygon", "coordinates": [[reproject_ring(r) for r in poly] for poly in coords]}
        return geom

    blocks: List[Tuple[str, Any]] = []
    for idx, feat in enumerate(getattr(boundary_layer, "features", []) or []):
        geom = feat.get("geometry")
        if not geom or geom.get("type") not in {"Polygon", "MultiPolygon"}:
            continue
        try:
            poly = shapely_shape(reproject_geometry(geom))
        except Exception:
            continue
        props = feat.get("properties") or {}
        name = props.get("name") or props.get("Name") or props.get("NAMA") or props.get("BLOK") or props.get("blok")
        blocks.append((str(name) if name else f"Blok {idx + 1}", poly))

    counts: Dict[str, int] = {}
    for feat in features:
        point = shapely_shape(feat["geometry"]).centroid
        assigned = UNASSIGNED_BLOCK
        for name, poly in blocks:
            try:
                if poly.intersects(point):
                    assigned = name
                    break
            except Exception:
                continue
        feat["properties"]["block"] = assigned
        counts[assigned] = counts.get(assigned, 0) + 1
    return features, counts


_FONT_CACHE: Dict[int, Any] = {}


def _load_font(size: int) -> Any:
    """Best-effort TrueType font for confidence labels, falling back to PIL's
    tiny built-in bitmap font (still legible, just smaller) if none found."""
    if size in _FONT_CACHE:
        return _FONT_CACHE[size]
    from PIL import ImageFont

    font = None
    for candidate in ("arialbd.ttf", "Arial Bold.ttf", "DejaVuSans-Bold.ttf", "arial.ttf"):
        try:
            font = ImageFont.truetype(candidate, size)
            break
        except Exception:
            continue
    if font is None:
        font = ImageFont.load_default()
    _FONT_CACHE[size] = font
    return font


def draw_boxes_on_raster(
    image: np.ndarray,
    features: List[Dict[str, Any]],
    transform: Any,
    color: Tuple[int, int, int] = (255, 0, 0),
    line_width: int = 4,
    show_confidence: bool = True,
) -> np.ndarray:
    """Burn each detected box (and its confidence score) into a copy of the
    source raster's pixels.

    Produces a new standalone raster image (original photo + drawn boxes)
    so the TCS result can be opened as its own new raster layer/document -
    consistent with how every other file opens in this viewer - instead of
    depending on vector-over-raster overlay rendering.
    """
    from PIL import Image as PILImage, ImageDraw

    from app.coordinate_tools import world_to_pixel

    base = np.ascontiguousarray(image)
    if base.ndim == 2:
        base = np.repeat(base[:, :, None], 3, axis=2)
    base = base[:, :, :3].astype(np.uint8)

    pil_img = PILImage.fromarray(base).convert("RGB")
    draw = ImageDraw.Draw(pil_img)
    for feat in features:
        geom = feat.get("geometry")
        if not geom or geom.get("type") != "Polygon":
            continue
        ring = geom["coordinates"][0]
        pixel_pts = [world_to_pixel(transform, x, y) for x, y in ring]
        xs = [p[0] for p in pixel_pts]
        ys = [p[1] for p in pixel_pts]
        x0, x1 = min(xs), max(xs)
        y0, y1 = min(ys), max(ys)
        draw.rectangle([x0, y0, x1, y1], outline=color, width=line_width)

        if not show_confidence:
            continue
        confidence = feat.get("properties", {}).get("confidence")
        if confidence is None:
            continue
        label = f"{confidence:.2f}"
        # Scale the label to the box so it stays readable on both small and
        # huge canopies, capped at sane min/max sizes.
        font_size = int(max(14, min(48, (y1 - y0) * 0.18)))
        font = _load_font(font_size)
        text_bbox = draw.textbbox((0, 0), label, font=font)
        text_w, text_h = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
        pad = max(2, font_size // 6)
        label_x0, label_y0 = x0, max(0, y0 - text_h - 2 * pad)
        label_x1, label_y1 = label_x0 + text_w + 2 * pad, label_y0 + text_h + 2 * pad
        draw.rectangle([label_x0, label_y0, label_x1, label_y1], fill=color)
        draw.text((label_x0 + pad, label_y0 + pad - text_bbox[1]), label, fill=(255, 255, 255), font=font)
    return np.array(pil_img)


def feature_bounds(features: List[Dict[str, Any]]):
    from app.vector_loader import iter_geometry_coords

    xs: List[float] = []
    ys: List[float] = []
    for feat in features:
        for x, y in iter_geometry_coords(feat.get("geometry")):
            xs.append(x)
            ys.append(y)
    if not xs:
        return None
    return min(xs), min(ys), max(xs), max(ys)


def save_features(output_path: str, features: List[Dict[str, Any]], crs: Any) -> str:
    """Save TCS result features to GeoJSON or Shapefile, matching the extension."""
    from pathlib import Path

    from app.format_converter import write_geojson, write_shapefile

    ext = Path(output_path).suffix.lower()
    if ext in {".geojson", ".json"}:
        write_geojson(output_path, features, source_crs=crs)
    elif ext == ".shp":
        write_shapefile(output_path, features, source_crs=crs)
    else:
        raise RuntimeError(f"Format output TCS belum didukung: {ext}")
    return output_path
