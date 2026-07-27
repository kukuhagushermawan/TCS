"""Geometry helpers for TCS's optional "Cari Lahan Kosong" (find empty
planting land) feature.

Idea: estimate the plantation's standard planting spacing from the
nearest-neighbor distance between already-detected trees, lay a staggered
triangular ("mata lima") planting grid at that spacing over the work area,
then keep only the grid slots that don't already have a real tree near
them - those are candidate coordinates for new planting.

The work area is either a user-supplied field boundary polygon layer, or -
when none is given - the convex hull of the detected tree points themselves,
so this feature never *requires* a boundary layer to be loaded first.

Pure numpy/shapely (both already core dependencies of this app) - no new
dependency such as scipy is introduced just for the nearest-neighbor lookup.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from shapely.geometry import MultiPoint, Point
from shapely.geometry import shape as shapely_shape
from shapely.ops import unary_union

try:
    from pyproj import CRS, Transformer
    from shapely.ops import transform as shapely_transform
except Exception:  # pragma: no cover
    CRS = None
    Transformer = None
    shapely_transform = None

# Row spacing for a staggered equilateral-triangle ("mata lima") grid: rows
# are offset by half the planting spacing and spaced sqrt(3)/2 of it apart,
# the classic layout Indonesian oil-palm plantations use to pack trees at
# maximum density rather than a plain square grid.
_TRIANGULAR_ROW_HEIGHT_FACTOR = math.sqrt(3) / 2.0


def is_boundary_layer(layer: Any) -> bool:
    """True for vector layers that look like a real polygon field boundary -
    not point layers such as TCS's own previous outputs (always Point
    features), which would be meaningless as a work-area boundary."""
    if getattr(layer, "layer_type", None) != "vector":
        return False
    if getattr(layer, "source_driver", None) == "TCS":
        return False
    features = getattr(layer, "features", None) or []
    if not features:
        return False
    geom = features[0].get("geometry") or {}
    return geom.get("type") in {"Polygon", "MultiPolygon"}


def points_from_features(features: List[Dict[str, Any]]) -> np.ndarray:
    """Extract plain (x, y) pairs from a list of Point-geometry features."""
    pts: List[Tuple[float, float]] = []
    for feat in features:
        geom = feat.get("geometry") or {}
        if geom.get("type") == "Point":
            x, y = geom["coordinates"][:2]
            pts.append((float(x), float(y)))
    return np.asarray(pts, dtype=np.float64) if pts else np.empty((0, 2), dtype=np.float64)


def _chunked_nearest_neighbor_distances(points: np.ndarray, chunk_size: int = 512) -> np.ndarray:
    """Distance from each point to its nearest *other* point.

    Computed in row chunks (not a full n x n matrix at once) so memory stays
    bounded even for several thousand trees.
    """
    n = points.shape[0]
    nearest = np.full(n, np.inf, dtype=np.float64)
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        chunk = points[start:end]
        diff = chunk[:, None, :] - points[None, :, :]
        dist = np.sqrt((diff ** 2).sum(axis=2))
        for i in range(end - start):
            dist[i, start + i] = np.inf  # exclude distance to itself
        nearest[start:end] = dist.min(axis=1)
    return nearest


def estimate_spacing(points: np.ndarray) -> Optional[float]:
    """Median nearest-neighbor distance between existing tree points - used
    as the estimated standard planting spacing. None when there are fewer
    than 2 points (no pair to measure a distance from)."""
    if points.shape[0] < 2:
        return None
    nearest = _chunked_nearest_neighbor_distances(points)
    return float(np.median(nearest))


def convex_hull_polygon(points: np.ndarray):
    """Fallback work-area boundary when no field boundary layer is given -
    the convex hull of the detected tree points themselves."""
    return MultiPoint([tuple(p) for p in points]).convex_hull


def boundary_polygon_from_layer(layer: Any, target_crs: Any) -> Optional[Any]:
    """Union every Polygon/MultiPolygon feature of `layer` into one shapely
    geometry, reprojected into `target_crs` if the layer's own CRS differs -
    the same reprojection pattern used elsewhere in this app (e.g.
    app.map_canvas._world_to_base_pixel). Returns None if the layer has no
    usable polygon geometry."""
    polys = []
    for feat in getattr(layer, "features", []) or []:
        geom = feat.get("geometry")
        if not geom or geom.get("type") not in {"Polygon", "MultiPolygon"}:
            continue
        try:
            polys.append(shapely_shape(geom))
        except Exception:
            continue
    if not polys:
        return None
    merged = unary_union(polys)

    layer_crs = getattr(layer, "crs", None)
    if target_crs and layer_crs and CRS is not None and Transformer is not None:
        try:
            src = CRS.from_user_input(layer_crs)
            dst = CRS.from_user_input(target_crs)
            if src != dst:
                transformer = Transformer.from_crs(src, dst, always_xy=True)
                merged = shapely_transform(transformer.transform, merged)
        except Exception:
            pass
    return merged


def generate_triangular_grid(boundary: Any, spacing: float) -> np.ndarray:
    """Candidate planting points on a staggered triangular grid at
    `spacing`, kept only where they fall inside `boundary`."""
    if boundary is None or boundary.is_empty or spacing <= 0:
        return np.empty((0, 2), dtype=np.float64)
    minx, miny, maxx, maxy = boundary.bounds
    row_height = spacing * _TRIANGULAR_ROW_HEIGHT_FACTOR
    points: List[Tuple[float, float]] = []
    row = 0
    y = miny
    while y <= maxy:
        x_offset = (spacing / 2.0) if row % 2 else 0.0
        x = minx + x_offset
        while x <= maxx:
            if boundary.contains(Point(x, y)):
                points.append((x, y))
            x += spacing
        y += row_height
        row += 1
    return np.asarray(points, dtype=np.float64) if points else np.empty((0, 2), dtype=np.float64)


def filter_empty_slots(grid_points: np.ndarray, tree_points: np.ndarray, min_dist: float, chunk_size: int = 512) -> np.ndarray:
    """Keep only grid points farther than `min_dist` from every existing
    tree - i.e. slots not already effectively occupied by a real tree."""
    if grid_points.shape[0] == 0 or tree_points.shape[0] == 0:
        return grid_points
    keep = np.ones(grid_points.shape[0], dtype=bool)
    for start in range(0, grid_points.shape[0], chunk_size):
        end = min(start + chunk_size, grid_points.shape[0])
        chunk = grid_points[start:end]
        diff = chunk[:, None, :] - tree_points[None, :, :]
        dist = np.sqrt((diff ** 2).sum(axis=2))
        keep[start:end] = dist.min(axis=1) > min_dist
    return grid_points[keep]


def points_to_features(points: np.ndarray) -> List[Dict[str, Any]]:
    """Plain Point features for the recommended-planting-slot result layer."""
    features: List[Dict[str, Any]] = []
    for i, (x, y) in enumerate(points, start=1):
        features.append({
            "type": "Feature",
            "id": str(i),
            "geometry": {"type": "Point", "coordinates": [float(x), float(y)]},
            "properties": {"slot_id": i},
        })
    return features
