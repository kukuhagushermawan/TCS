"""Raster loader for GeoTIFF, JPG/JPEG, IMG, ECW when GDAL/rasterio support exists."""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple
import hashlib
import tempfile
import numpy as np
from PIL import Image

from .layer_manager import Layer
from .image_enhancement import to_uint8_rgb

try:
    import rasterio
    from rasterio.transform import Affine
    from rasterio.errors import RasterioIOError
except Exception:  # pragma: no cover
    rasterio = None
    Affine = None
    RasterioIOError = Exception


WORLD_FILE_EXTS = {
    ".jpg": [".jgw", ".jpgw"],
    ".jpeg": [".jgw", ".jpegw"],
    ".png": [".pgw", ".pngw"],
    ".tif": [".tfw"],
    ".tiff": [".tfw"],
}


def load_raster(path: str, as_dem: bool = False) -> Layer:
    p = Path(path)
    suffix = p.suffix.lower()
    if rasterio is not None:
        try:
            return _load_with_rasterio(path, as_dem=as_dem)
        except Exception as exc:
            # ECW is an optional GDAL driver. Many Conda/rasterio builds cannot
            # open ECW directly, while OSGeo4W/QGIS can. If the user has
            # installed OSGeo4W + gdal-ecw, Terra View automatically converts the
            # ECW to a temporary GeoTIFF and opens that cache, so the app feels
            # like it can open ECW directly.
            if suffix == ".ecw":
                try:
                    cache_tif = _convert_ecw_to_cache_geotiff(path)
                    layer = _load_with_rasterio(str(cache_tif), as_dem=as_dem)
                    layer.name = p.name
                    layer.path = path
                    layer.metadata.update({
                        "source_format": "ECW",
                        "ecw_open_mode": "Auto-converted via bundled/OSGeo4W/QGIS gdal_translate",
                        "cache_geotiff": str(cache_tif),
                    })
                    return layer
                except Exception as ecw_exc:
                    raise RuntimeError(
                        "File ECW membutuhkan GDAL ECW driver. Terra View sudah mencoba mencari "
                        "OSGeo4W/QGIS gdal_translate, tetapi konversi otomatis gagal.\n\n"
                        "Pastikan OSGeo4W sudah menginstall package: gdal dan gdal-ecw, lalu cek di OSGeo4W Shell:\n"
                        "gdalinfo --formats | findstr /I ECW\n\n"
                        f"Detail rasterio: {exc}\nDetail ECW fallback: {ecw_exc}"
                    ) from ecw_exc
            if suffix not in {".jpg", ".jpeg", ".png"}:
                raise RuntimeError(f"Gagal membuka raster dengan GDAL/rasterio: {exc}") from exc
    return _load_with_pillow(path, as_dem=as_dem)


def _convert_ecw_to_cache_geotiff(path: str) -> Path:
    from .format_converter import run_gdal_translate

    src = Path(path)
    digest = hashlib.sha1(str(src.resolve()).encode("utf-8", errors="ignore")).hexdigest()[:12]
    cache_dir = Path(tempfile.gettempdir()) / "Terra_View" / "ecw_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_tif = cache_dir / f"{src.stem}_{digest}_preview.tif"

    if out_tif.exists() and out_tif.stat().st_mtime >= src.stat().st_mtime:
        return out_tif

    # Viewer mode: create a downsampled GeoTIFF cache. This keeps ECW display
    # responsive, avoids massive temporary files, and prevents Windows from
    # showing "Not Responding" during normal preview use.
    run_gdal_translate(str(src), str(out_tif), fmt="GTiff", preview=True, max_size=3000)
    return out_tif


def _load_with_rasterio(path: str, as_dem: bool = False) -> Layer:
    assert rasterio is not None
    with rasterio.open(path) as src:
        profile = src.profile.copy()
        count = src.count
        if as_dem or count == 1:
            band = _masked_read_float(src, 1)
            display = _render_dem(band) if as_dem else to_uint8_rgb(band)
            # Single-band raster is kept as Raster in the viewer flow, but its
            # numeric band is exposed as elevation/DEM information so the user
            # can click and read DEM values when opening a DEM GeoTIFF via Raster.
            metadata = {"dem_info": "Single-band numeric raster; elevation/DEM value can be read on click."} if count == 1 else {}
            layer = Layer(
                name=Path(path).name,
                layer_type="dem" if as_dem else "raster",
                path=path,
                image=display,
                raw_data=band,
                dem_data=band.astype(float),
                transform=src.transform,
                crs=src.crs,
                profile=profile,
                nodata=src.nodata,
                width=src.width,
                height=src.height,
                extent=(src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top),
                metadata=metadata,
            )
            return layer
        indexes = [1, 2, 3] if count >= 3 else list(range(1, count + 1))
        data = _masked_read_float(src, indexes)
        data = np.moveaxis(data, 0, -1)
        display = to_uint8_rgb(data)
        return Layer(
            name=Path(path).name,
            layer_type="raster",
            path=path,
            image=display,
            raw_data=data,
            transform=src.transform,
            crs=src.crs,
            profile=profile,
            nodata=src.nodata,
            width=src.width,
            height=src.height,
            extent=(src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top),
        )


def _masked_read_float(src, indexes):
    """Read raster safely as float so nodata/masked pixels can become NaN.

    Rasterio may return uint16/uint8 masked arrays for GeoTIFF/DEM.
    Calling .filled(np.nan) directly on integer arrays raises:
    "Cannot convert fill_value nan to dtype uint16".
    Therefore the data is converted to float32 before filling masks with NaN.
    """
    arr = src.read(indexes, masked=True)
    if hasattr(arr, "astype"):
        arr = arr.astype("float32")
    if hasattr(arr, "filled"):
        return arr.filled(np.nan)
    return np.asarray(arr, dtype=np.float32)


def _load_with_pillow(path: str, as_dem: bool = False) -> Layer:
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img).astype(np.uint8)
    transform = _read_world_file(Path(path))
    layer = Layer(
        name=Path(path).name,
        layer_type="raster",
        path=path,
        image=arr,
        raw_data=arr,
        transform=transform,
        crs=None,
        width=img.width,
        height=img.height,
        extent=None,
        metadata={"note": "Opened as normal image. Coordinates use pixel space unless world file is present."},
    )
    if transform is not None:
        layer.metadata["world_file"] = "Detected"
    return layer


def _read_world_file(path: Path):
    if Affine is None:
        return None
    candidates = WORLD_FILE_EXTS.get(path.suffix.lower(), [])
    candidates.append(path.suffix + "w")
    for ext in candidates:
        wf = path.with_suffix(ext)
        if wf.exists():
            vals = [float(v.strip()) for v in wf.read_text().splitlines() if v.strip()]
            if len(vals) >= 6:
                # World file: A, D, B, E, C, F
                a, d, b, e, c, f = vals[:6]
                return Affine(a, b, c, d, e, f)
    return None


def _render_dem(dem: np.ndarray) -> np.ndarray:
    data = dem.astype(float)
    finite = np.isfinite(data)
    if not finite.any():
        return np.zeros((*data.shape, 3), dtype=np.uint8)
    lo, hi = np.nanpercentile(data[finite], (2, 98))
    if hi <= lo:
        lo, hi = np.nanmin(data[finite]), np.nanmax(data[finite])
    norm = np.zeros_like(data, dtype=np.float32)
    if hi > lo:
        norm = (data - lo) / (hi - lo)
    norm = np.clip(norm, 0, 1)
    gray = (norm * 255).astype(np.uint8)
    return np.repeat(gray[:, :, None], 3, axis=2)


def save_array_as_geotiff(output_path: str, image: np.ndarray, reference_layer: Layer) -> None:
    if rasterio is None:
        raise RuntimeError("rasterio/GDAL belum tersedia untuk export GeoTIFF")
    img = to_uint8_rgb(image)
    profile = reference_layer.profile.copy() if reference_layer.profile else {}
    profile.update({
        "driver": "GTiff",
        "height": img.shape[0],
        "width": img.shape[1],
        "count": 3,
        "dtype": "uint8",
        "transform": reference_layer.transform,
        "crs": reference_layer.crs,
    })
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(np.moveaxis(img, -1, 0))
