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


PIXEL_UNIT_LABEL = "piksel (raster belum bergeoreferensi)"
DEGREE_UNIT_LABEL = "derajat (CRS geografis, bukan satuan jarak)"


def is_identity_transform(transform: Any) -> bool:
    """True when a geotransform carries no real georeferencing.

    GDAL/rasterio hand back this identity transform - 1 unit per pixel, origin
    at (0,0) - for an image with no georeferencing at all (a plain JPG/PNG
    without a world file) instead of failing. Coordinates derived from it are
    therefore raw pixel indices, not ground positions.
    """
    if transform is None:
        return True
    try:
        values = (transform.a, transform.b, transform.c, transform.d, transform.e, transform.f)
    except Exception:
        return False
    return values == (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)


def linear_unit_label(crs: Any, transform: Any = None) -> str:
    """Human-readable unit for distances measured in a layer's own coordinates.

    Any distance computed straight from layer coordinates (e.g. TCS's estimated
    planting spacing) silently inherits whatever unit that layer's CRS uses, so
    a bare number is ambiguous: metres for a projected CRS, degrees for a
    geographic one, and plain image pixels when the source had no georeferencing
    at all. Pass ``transform`` as well to also catch a file that declares a CRS
    but carries only an identity geotransform.
    """
    if transform is not None and is_identity_transform(transform):
        return PIXEL_UNIT_LABEL
    if crs is None:
        return PIXEL_UNIT_LABEL
    if CRS is None:
        return "unit CRS"
    try:
        crs_obj = CRS.from_user_input(crs)
    except Exception:
        return "unit CRS"
    if crs_obj.is_geographic:
        return DEGREE_UNIT_LABEL
    try:
        unit = (crs_obj.axis_info[0].unit_name or "").lower()
    except Exception:
        unit = ""
    if "metre" in unit or "meter" in unit:
        return "meter"
    if "foot" in unit or "feet" in unit:
        return "kaki"
    if "degree" in unit:
        return DEGREE_UNIT_LABEL
    return unit or "unit CRS"


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
