"""Coordinate and pixel helper functions for Terra View."""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple
import math

try:
    from rasterio.transform import rowcol, xy
except Exception:  # pragma: no cover - optional dependency fallback
    rowcol = None
    xy = None

try:
    from pyproj import CRS, Transformer
except Exception:  # pragma: no cover
    CRS = None
    Transformer = None


def pixel_to_world(transform: Any, col: float, row: float) -> Tuple[float, float]:
    """Convert image pixel coordinate to map/world coordinate."""
    if transform is None:
        return float(col), float(row)
    try:
        if xy:
            x, y = xy(transform, row, col, offset="center")
            return float(x), float(y)
    except Exception:
        pass
    try:
        x = transform.c + col * transform.a + row * transform.b
        y = transform.f + col * transform.d + row * transform.e
        return float(x), float(y)
    except Exception:
        return float(col), float(row)


def world_to_pixel(transform: Any, x: float, y: float) -> Tuple[float, float]:
    """Convert map/world coordinate to image pixel coordinate."""
    if transform is None:
        return float(x), float(y)
    try:
        if rowcol:
            r, c = rowcol(transform, x, y)
            return float(c), float(r)
    except Exception:
        pass
    try:
        inv = ~transform
        col, row = inv * (x, y)
        return float(col), float(row)
    except Exception:
        return float(x), float(y)


def crs_to_text(crs: Any) -> str:
    if crs is None:
        return "Unknown / image pixel"
    try:
        return crs.to_string()
    except Exception:
        return str(crs)


def is_geographic_crs(crs: Any) -> bool:
    if crs is None or CRS is None:
        return False
    try:
        return CRS.from_user_input(crs).is_geographic
    except Exception:
        return False


def to_latlon(x: float, y: float, source_crs: Any) -> Tuple[Optional[float], Optional[float]]:
    """Return latitude, longitude from source CRS.

    If CRS is already geographic, x is treated as longitude and y as latitude.
    """
    if source_crs is None:
        return None, None
    if CRS is None or Transformer is None:
        return None, None
    try:
        crs = CRS.from_user_input(source_crs)
        if crs.is_geographic:
            return float(y), float(x)
        transformer = Transformer.from_crs(crs, CRS.from_epsg(4326), always_xy=True)
        lon, lat = transformer.transform(x, y)
        if math.isfinite(lat) and math.isfinite(lon):
            return float(lat), float(lon)
    except Exception:
        return None, None
    return None, None


def rgb_to_hex(rgb: Tuple[int, int, int]) -> str:
    r, g, b = [max(0, min(255, int(v))) for v in rgb]
    return f"#{r:02X}{g:02X}{b:02X}"


def safe_pixel_rgb(image, col: int, row: int) -> Optional[Tuple[int, int, int]]:
    if image is None:
        return None
    if row < 0 or col < 0 or row >= image.shape[0] or col >= image.shape[1]:
        return None
    px = image[row, col]
    if getattr(px, "ndim", 0) == 0:
        v = int(px)
        return v, v, v
    if len(px) == 1:
        v = int(px[0])
        return v, v, v
    return int(px[0]), int(px[1]), int(px[2])


def build_click_info(layer: Any, col: int, row: int) -> Dict[str, Any]:
    """Build standard information dictionary for map cursor/click."""
    x, y = pixel_to_world(getattr(layer, "transform", None), col, row)
    lat, lon = to_latlon(x, y, getattr(layer, "crs", None))
    rgb = safe_pixel_rgb(layer.display_image() if layer else None, col, row) if layer else None
    elevation = None
    if layer and getattr(layer, "dem_data", None) is not None:
        try:
            elevation = float(layer.dem_data[row, col])
        except Exception:
            elevation = None
    return {
        "pixel_x": col,
        "pixel_y": row,
        "x": x,
        "y": y,
        "latitude": lat,
        "longitude": lon,
        "crs": crs_to_text(getattr(layer, "crs", None) if layer else None),
        "rgb": rgb,
        "hex": rgb_to_hex(rgb) if rgb else None,
        "elevation": elevation,
        "layer": getattr(layer, "name", "None") if layer else "None",
    }
