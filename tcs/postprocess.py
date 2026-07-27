"""Detections -> vector post-processing for TCS.

Runs cross-tile Non-Maximum Suppression on bounding boxes (so a tree sitting
in a tile-overlap region is not counted twice), then converts each surviving
box to a georeferenced point feature (its centre, via the same
affine-transform convention as ``app.coordinate_tools``) - so TCS output is a
plain point vector layer, one point per detected tree.

Saving reuses app.format_converter's existing GeoJSON/Shapefile writers so
TCS output loads through the exact same vector code path as any other file.
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from .inference import Detection


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
