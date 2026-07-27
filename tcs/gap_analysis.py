"""Geometry helpers for TCS's optional "Cari Lahan Kosong" (find empty
planting land) feature.

Idea: instead of laying a fixed abstract grid over the whole work area,
follow the actual planting pattern of the detected trees -
1. Estimate the dominant row direction from the histogram of
   nearest-neighbor direction vectors between detected trees.
2. Rotate the trees so rows become horizontal, then cluster them into rows
   by their rotated Y position.
3. Within each row, estimate the median along-row spacing and walk from
   that row's own first to last tree at that spacing - any expected
   position with no real tree nearby is a candidate empty slot.
4. Rotate candidates back, then drop any that sit too close to a real tree,
   outside the work area, or too near the raster's own edge (a row that
   simply continues past the captured image is not "missing").

The work area is either a user-supplied field boundary polygon layer, or -
when none is given - a small buffer around the convex hull of the detected
tree points themselves, so this feature never *requires* a boundary layer.

Pure numpy/shapely (both already core dependencies of this app) - no new
dependency such as scipy is introduced just for the nearest-neighbor lookup.
"""
from __future__ import annotations

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
    as a bootstrap distance estimate (e.g. the row-clustering threshold)
    before the more precise per-row along-row spacing is known. None when
    there are fewer than 2 points."""
    if points.shape[0] < 2:
        return None
    nearest = _chunked_nearest_neighbor_distances(points)
    return float(np.median(nearest))


def _nearest_neighbor_vectors(points: np.ndarray, k: int = 2, chunk_size: int = 256) -> np.ndarray:
    """Direction vectors (dx, dy) from each point to its k nearest other
    points - the raw material for estimating row direction."""
    n = points.shape[0]
    vectors: List[np.ndarray] = []
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        chunk = points[start:end]
        diff = points[None, :, :] - chunk[:, None, :]  # (c, n, 2): vector chunk[i] -> points[j]
        dist = np.sqrt((diff ** 2).sum(axis=2))
        for i in range(end - start):
            dist[i, start + i] = np.inf
        k_eff = min(k, dist.shape[1] - 1) if dist.shape[1] > 1 else 0
        if k_eff <= 0:
            continue
        nearest_idx = np.argpartition(dist, k_eff - 1, axis=1)[:, :k_eff]
        for row_i in range(end - start):
            for j in nearest_idx[row_i]:
                vectors.append(diff[row_i, j])
    return np.asarray(vectors, dtype=np.float64) if vectors else np.empty((0, 2), dtype=np.float64)


def estimate_row_direction(points: np.ndarray, k: int = 2) -> Optional[float]:
    """Dominant row-direction angle (radians, in [0, pi)) from a histogram of
    nearest-neighbor direction vectors across all trees.

    Oil-palm rows are locally the most common "shortest link" direction
    between neighboring trees, so the histogram's peak recovers row
    orientation even when the planted block isn't axis-aligned or
    perfectly rectangular. None if there isn't enough data to tell.
    """
    if points.shape[0] < 4:
        return None
    vectors = _nearest_neighbor_vectors(points, k=k)
    if vectors.shape[0] == 0:
        return None
    angles = np.arctan2(vectors[:, 1], vectors[:, 0]) % np.pi

    bins = 180
    hist, edges = np.histogram(angles, bins=bins, range=(0.0, np.pi))
    peak_bin = int(np.argmax(hist))
    peak_center = (edges[peak_bin] + edges[peak_bin + 1]) / 2.0

    # Refine using a circular mean (mod pi, via the standard doubled-angle
    # trick) of angles near the histogram peak, rather than the coarse bin
    # center alone - avoids the usual 0/pi wraparound issue since distance
    # is measured on the circle, not by plain subtraction.
    diff = np.abs(angles - peak_center)
    circ_dist = np.minimum(diff, np.pi - diff)
    tolerance = np.pi / 12  # 15 degrees
    selected = angles[circ_dist <= tolerance]
    if selected.size == 0:
        selected = angles
    mean_angle = np.arctan2(np.mean(np.sin(2 * selected)), np.mean(np.cos(2 * selected))) / 2.0
    return float(mean_angle % np.pi)


def rotate_points(points: np.ndarray, theta: float) -> np.ndarray:
    """Standard counter-clockwise rotation of 2D points by theta radians."""
    if points.shape[0] == 0:
        return points
    c, s = np.cos(theta), np.sin(theta)
    rot = np.array([[c, -s], [s, c]])
    return points @ rot.T


def cluster_rows(rotated_points: np.ndarray, row_gap_threshold: float) -> List[np.ndarray]:
    """Group already-rotated points (rows expected to be ~horizontal) into
    rows: sort by rotated Y, start a new row whenever the gap to the
    previous point's Y exceeds ``row_gap_threshold``."""
    if rotated_points.shape[0] == 0:
        return []
    order = np.argsort(rotated_points[:, 1])
    sorted_pts = rotated_points[order]
    rows: List[List[np.ndarray]] = [[sorted_pts[0]]]
    for i in range(1, len(sorted_pts)):
        if sorted_pts[i, 1] - sorted_pts[i - 1, 1] > row_gap_threshold:
            rows.append([])
        rows[-1].append(sorted_pts[i])
    return [np.asarray(r, dtype=np.float64) for r in rows]


def pooled_along_row_spacing(rows: List[np.ndarray]) -> Optional[float]:
    """Median distance between consecutive (along-row sorted) trees, pooled
    across every row - the estimated standard planting spacing."""
    gaps: List[float] = []
    for row in rows:
        if row.shape[0] < 2:
            continue
        xs = np.sort(row[:, 0])
        gaps.extend(np.diff(xs).tolist())
    if not gaps:
        return None
    return float(np.median(gaps))


def find_missing_in_rows(rows: List[np.ndarray], spacing: float, tolerance_ratio: float) -> np.ndarray:
    """Within each row's own span (its first to last tree - never
    extrapolated beyond what was actually detected), walk at ``spacing`` and
    flag positions with no real tree within ``tolerance_ratio * spacing``."""
    if spacing <= 0:
        return np.empty((0, 2), dtype=np.float64)
    tol = spacing * tolerance_ratio
    missing: List[Tuple[float, float]] = []
    for row in rows:
        if row.shape[0] < 2:
            continue
        xs = np.sort(row[:, 0])
        y = float(np.median(row[:, 1]))
        x = float(xs[0])
        x_max = float(xs[-1])
        while x <= x_max + 1e-6:
            if np.min(np.abs(xs - x)) > tol:
                missing.append((x, y))
            x += spacing
    return np.asarray(missing, dtype=np.float64) if missing else np.empty((0, 2), dtype=np.float64)


def filter_empty_slots(grid_points: np.ndarray, tree_points: np.ndarray, min_dist: float, chunk_size: int = 512) -> np.ndarray:
    """Keep only candidate points farther than `min_dist` from every existing
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


def filter_within_polygon(points: np.ndarray, polygon: Any) -> np.ndarray:
    """Keep only candidates that fall inside ``polygon`` (the work area)."""
    if points.shape[0] == 0 or polygon is None or polygon.is_empty:
        return points
    keep = np.array([polygon.contains(Point(x, y)) for x, y in points], dtype=bool)
    return points[keep]


def filter_within_raster_margin(points: np.ndarray, raster_bounds: Optional[Tuple[float, float, float, float]], margin: float) -> np.ndarray:
    """Drop candidates within ``margin`` of the source raster's own
    real-world edge - a row that simply continues past the captured image
    is not a genuinely "missing" tree."""
    if points.shape[0] == 0 or raster_bounds is None:
        return points
    minx, miny, maxx, maxy = raster_bounds
    keep = (
        (points[:, 0] >= minx + margin) & (points[:, 0] <= maxx - margin)
        & (points[:, 1] >= miny + margin) & (points[:, 1] <= maxy - margin)
    )
    return points[keep]


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
