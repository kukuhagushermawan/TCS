"""Reliable raster/vector converters for Terra View.

This module avoids exposing raw GDAL/Fiona driver errors to the user.
Important design choices:
- KML output is written manually, so conversion does not depend on an optional GDAL KML write driver.
- GeoJSON I/O is written manually, so it works even without GDAL.
- Shapefile I/O uses pyshp (pure Python, no GDAL) instead of Fiona/GDAL, so the
  core app does not need to bundle a private GDAL copy just for Shapefiles -
  Fiona and rasterio each vendor their own GDAL/PROJ/spatialite build, which
  was the single largest source of installer bloat.
- DXF is read via ezdxf (pure Python, no GDAL/OGR DXF driver needed) - see
  app.vector_loader.read_dxf_features, shared by the vector loader and this
  module's conversion path.
- ECW/IMG raster conversion uses rasterio when available and falls back to OSGeo4W/QGIS gdal_translate.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import zipfile
import xml.etree.ElementTree as ET
from html import escape
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .resources import resource_path
from .vector_loader import geojson_crs_to_string, iter_geometry_parts, read_sidecar_prj

try:
    import rasterio
except Exception:  # pragma: no cover
    rasterio = None

try:
    import shapefile as pyshp
except Exception:  # pragma: no cover
    pyshp = None

try:
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None

try:
    from pyproj import CRS, Transformer
except Exception:  # pragma: no cover
    CRS = None
    Transformer = None


RASTER_INPUTS = {".tif", ".tiff", ".jpg", ".jpeg", ".png", ".img", ".ecw"}
VECTOR_INPUTS = {".shp", ".kml", ".kmz", ".dxf", ".geojson", ".json"}


# ---------------------------------------------------------------------------
# Raster conversion
# ---------------------------------------------------------------------------
def convert_raster(input_path: str, output_path: str) -> str:
    """Convert raster/image data to GeoTIFF, PNG, or JPEG.

    ECW/IMG support depends on the GDAL build. If rasterio cannot open the file,
    Terra View tries external OSGeo4W/QGIS `gdal_translate` automatically.
    """
    in_ext = Path(input_path).suffix.lower()
    out_ext = Path(output_path).suffix.lower()
    if in_ext not in RASTER_INPUTS:
        raise RuntimeError(f"Input raster belum didukung: {in_ext}")
    if out_ext not in {".tif", ".tiff", ".png", ".jpg", ".jpeg"}:
        raise RuntimeError(f"Output raster belum didukung: {out_ext}")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    if rasterio is not None and in_ext in {".tif", ".tiff", ".img", ".ecw"}:
        try:
            return _convert_raster_rasterio(input_path, output_path)
        except Exception as rasterio_error:
            # ECW often fails here if the internal GDAL lacks the ECW plugin.
            fallback = _convert_raster_gdal_translate(input_path, output_path)
            if fallback:
                return fallback
            raise RuntimeError(
                "Konversi raster gagal. Jika input ECW/IMG, pastikan OSGeo4W/QGIS GDAL driver tersedia.\n"
                f"Detail: {rasterio_error}"
            ) from rasterio_error

    # Plain image conversion path.
    if Image is None:
        fallback = _convert_raster_gdal_translate(input_path, output_path)
        if fallback:
            return fallback
        raise RuntimeError("Pillow/GDAL belum tersedia untuk konversi gambar biasa.")
    try:
        img = Image.open(input_path)
        if out_ext in {".jpg", ".jpeg"}:
            img = img.convert("RGB")
        img.save(output_path)
        return output_path
    except Exception as exc:
        fallback = _convert_raster_gdal_translate(input_path, output_path)
        if fallback:
            return fallback
        raise RuntimeError(f"Konversi raster gagal: {exc}") from exc


def _convert_raster_rasterio(input_path: str, output_path: str) -> str:
    assert rasterio is not None
    out_ext = Path(output_path).suffix.lower()
    driver = "GTiff" if out_ext in {".tif", ".tiff"} else "PNG" if out_ext == ".png" else "JPEG"
    with rasterio.open(input_path) as src:
        profile = src.profile.copy()
        profile.update(driver=driver)
        data = src.read()
        # PNG/JPEG cannot carry full geospatial profile; write display-friendly output.
        if driver in {"PNG", "JPEG"}:
            profile.pop("crs", None)
            profile.pop("transform", None)
            profile.pop("nodata", None)
            profile.pop("blockxsize", None)
            profile.pop("blockysize", None)
            profile.pop("tiled", None)
            profile.pop("compress", None)
            if driver == "JPEG" and data.shape[0] > 3:
                data = data[:3]
            profile.update(count=data.shape[0], dtype=data.dtype)
        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(data)
    return output_path


def _convert_raster_gdal_translate(input_path: str, output_path: str) -> Optional[str]:
    out_ext = Path(output_path).suffix.lower()
    fmt = "GTiff" if out_ext in {".tif", ".tiff"} else "PNG" if out_ext == ".png" else "JPEG"
    try:
        run_gdal_translate(input_path, output_path, fmt=fmt, preview=False)
        return output_path
    except Exception:
        return None


def gdal_runtime_dir() -> Path:
    base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
    return Path(base) / "TerraView" / "gdal_runtime"


def _registered_gdal_runtime_marker() -> Path:
    base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
    return Path(base) / "TerraView" / "gdal_runtime_path.txt"


def registered_gdal_runtime_folder() -> Optional[Path]:
    marker = _registered_gdal_runtime_marker()
    if not marker.exists():
        return None
    try:
        text = marker.read_text(encoding="utf-8").strip()
    except Exception:
        return None
    return Path(text) if text else None


def register_gdal_runtime_folder(folder: Path) -> None:
    marker = _registered_gdal_runtime_marker()
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(str(folder), encoding="utf-8")


def _candidate_gdal_translate_paths() -> List[Path]:
    """Return GDAL executables, preferring OSGeo4W/QGIS over Conda PATH.

    When Terra View is launched from Anaconda, shutil.which() often finds Conda's
    gdal_translate first. That build may not include ECW. OSGeo4W/QGIS is
    checked first because the user installs gdal + gdal-ecw there.
    """
    candidates: List[Path] = [
        # 1) Bundled GDAL/ECW runtime inside the installed app, if provided.
        # This enables ECW on client machines without installing OSGeo4W separately,
        # as long as the required GDAL + ECW runtime DLLs are legally bundled in
        # vendor/osgeo4w before building the EXE/installer.
        resource_path("vendor", "osgeo4w", "bin", "gdal_translate.exe"),
        resource_path("vendor", "gdal", "bin", "gdal_translate.exe"),
    ]
    registered = registered_gdal_runtime_folder()
    if registered is not None:
        candidates.append(registered / "bin" / "gdal_translate.exe")
        candidates.append(registered / "gdal_translate.exe")
    candidates += [
        gdal_runtime_dir() / "bin" / "gdal_translate.exe",
        Path(r"C:\OSGeo4W\bin\gdal_translate.exe"),
        Path(r"C:\OSGeo4W64\bin\gdal_translate.exe"),
    ]
    program_files = [
        Path(os.environ.get("PROGRAMFILES", r"C:\Program Files")),
        Path(os.environ.get("PROGRAMFILES(X86)", r"C:\Program Files (x86)")),
    ]
    for root in program_files:
        if root.exists():
            for qgis_dir in root.glob("QGIS*"):
                candidates.append(qgis_dir / "bin" / "gdal_translate.exe")
                candidates.append(qgis_dir / "apps" / "gdal" / "gdal_translate.exe")
    found = shutil.which("gdal_translate")
    if found:
        candidates.append(Path(found))

    unique: List[Path] = []
    seen = set()
    for cand in candidates:
        key = str(cand).lower()
        if key not in seen and cand.exists():
            unique.append(cand)
            seen.add(key)
    return unique


def find_gdal_translate() -> Optional[Path]:
    paths = _candidate_gdal_translate_paths()
    return paths[0] if paths else None


def _gdal_env_for(exe: Path) -> Dict[str, str]:
    env = os.environ.copy()
    bin_dir = exe.parent
    env["PATH"] = str(bin_dir) + os.pathsep + env.get("PATH", "")
    root = bin_dir.parent
    candidates = {
        "GDAL_DATA": [root / "apps" / "gdal" / "share" / "gdal", root / "share" / "gdal"],
        "GDAL_DRIVER_PATH": [root / "apps" / "gdal" / "lib" / "gdalplugins", root / "lib" / "gdalplugins"],
        "PROJ_LIB": [root / "share" / "proj", root / "apps" / "proj" / "share" / "proj"],
    }
    for key, paths in candidates.items():
        for path in paths:
            if path.exists():
                env[key] = str(path)
                break
    return env


def _run_gdal_direct(exe: Path, args: List[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [str(exe), *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=_gdal_env_for(exe),
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )


def _run_gdal_via_osgeo_shell(exe: Path, args: List[str]) -> subprocess.CompletedProcess:
    root = exe.parent.parent
    o4w_env = root / "bin" / "o4w_env.bat"
    if not o4w_env.exists():
        return _run_gdal_direct(exe, args)
    command = f'call "{o4w_env}" >nul && "{exe}" {subprocess.list2cmdline(args)}'
    return subprocess.run(
        ["cmd", "/d", "/s", "/c", command],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )


def run_gdal_translate(input_path: str, output_path: str, fmt: Optional[str] = None, preview: bool = False, max_size: int = 3000) -> str:
    """Run external gdal_translate using OSGeo4W/QGIS first.

    preview=True converts huge ECW images to a downsampled GeoTIFF cache for
    smooth viewing. Full-resolution conversion is still used from the Convert
    menu when preview=False.
    """
    paths = _candidate_gdal_translate_paths()
    if not paths:
        raise RuntimeError(
            "gdal_translate.exe tidak ditemukan. Untuk ECW tanpa install library lokal, "
            "bundel GDAL ECW runtime ke folder vendor/osgeo4w/bin sebelum build. "
            "Alternatif: install OSGeo4W/QGIS dengan package gdal + gdal-ecw."
        )

    out_ext = Path(output_path).suffix.lower()
    driver = fmt or ("GTiff" if out_ext in {".tif", ".tiff"} else "PNG" if out_ext == ".png" else "JPEG")
    args = ["-of", driver]
    if preview and driver == "GTiff":
        args += ["-outsize", str(max_size), "0", "-r", "bilinear", "-co", "TILED=YES", "-co", "COMPRESS=LZW", "-co", "BIGTIFF=IF_NEEDED"]
    args += [str(input_path), str(output_path)]

    errors = []
    for exe in paths:
        for runner in (_run_gdal_direct, _run_gdal_via_osgeo_shell):
            result = runner(exe, args)
            if result.returncode == 0 and Path(output_path).exists():
                return output_path
            msg = (result.stderr or result.stdout or "gdal_translate gagal tanpa pesan.").strip()
            errors.append(f"{exe}: {msg}")
    raise RuntimeError("Semua percobaan gdal_translate gagal:\n" + "\n\n".join(errors[-4:]))


# ---------------------------------------------------------------------------
# Vector conversion
# ---------------------------------------------------------------------------
def convert_vector(input_path: str, output_path: str) -> str:
    """Convert vector formats with stable fallbacks.

    Supported outputs:
    - .kml: manual KML writer, no Fiona KML driver required.
    - .geojson/.json: manual GeoJSON writer.
    - .shp: Fiona/GDAL writer.
    """
    in_ext = Path(input_path).suffix.lower()
    out_ext = Path(output_path).suffix.lower()
    if in_ext not in VECTOR_INPUTS:
        raise RuntimeError(f"Input vector belum didukung: {in_ext}")
    if out_ext not in {".shp", ".kml", ".geojson", ".json"}:
        raise RuntimeError(f"Output vector belum didukung: {out_ext}")

    features, source_crs = read_vector_features(input_path)
    if not features:
        raise RuntimeError("File vector tidak memiliki geometry/fitur yang bisa dikonversi.")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    if out_ext == ".kml":
        write_kml(output_path, features, source_crs=source_crs)
    elif out_ext in {".geojson", ".json"}:
        write_geojson(output_path, features, source_crs=source_crs)
    elif out_ext == ".shp":
        write_shapefile(output_path, features, source_crs=source_crs)
    return output_path


def read_vector_features(input_path: str) -> Tuple[List[Dict[str, Any]], Optional[Any]]:
    ext = Path(input_path).suffix.lower()
    if ext == ".kmz":
        return read_kmz_features(input_path)
    if ext == ".kml":
        return read_kml_features(input_path), "EPSG:4326"
    if ext in {".geojson", ".json"}:
        return _read_geojson_features(input_path)
    if ext == ".shp":
        return _read_shapefile_features(input_path)
    if ext == ".dxf":
        from .vector_loader import read_dxf_features

        return read_dxf_features(input_path), read_sidecar_prj(Path(input_path))
    raise RuntimeError(f"Format vector belum didukung: {ext}")


def _read_geojson_features(input_path: str) -> Tuple[List[Dict[str, Any]], Optional[Any]]:
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    raw_features = data.get("features", [data]) if data.get("type") == "FeatureCollection" else [data]
    features: List[Dict[str, Any]] = []
    for i, feat in enumerate(raw_features):
        geom = _plain_geometry(feat.get("geometry"))
        if not geom:
            continue
        props = _plain_properties(feat.get("properties") or {})
        features.append({"type": "Feature", "id": str(feat.get("id", i)), "properties": props, "geometry": geom})
    return features, geojson_crs_to_string(data)


def _read_shapefile_features(input_path: str) -> Tuple[List[Dict[str, Any]], Optional[Any]]:
    if pyshp is None:
        raise RuntimeError("pyshp belum tersedia untuk membaca Shapefile.")
    features: List[Dict[str, Any]] = []
    with pyshp.Reader(input_path) as reader:
        field_names = [f[0] for f in reader.fields if f[0] != "DeletionFlag"]
        for i, shape_rec in enumerate(reader.iterShapeRecords()):
            if not shape_rec.shape.points:
                continue
            geom = _plain_geometry(shape_rec.shape.__geo_interface__)
            if not geom:
                continue
            props = _plain_properties(dict(zip(field_names, shape_rec.record)))
            features.append({"type": "Feature", "id": str(i), "properties": props, "geometry": geom})
    source_crs = read_sidecar_prj(Path(input_path))
    return features, source_crs


def read_kmz_features(input_path: str) -> Tuple[List[Dict[str, Any]], Optional[Any]]:
    with tempfile.TemporaryDirectory() as tmp:
        with zipfile.ZipFile(input_path) as z:
            kml_files = [n for n in z.namelist() if n.lower().endswith(".kml")]
            if not kml_files:
                raise RuntimeError("KMZ tidak berisi file KML.")
            z.extract(kml_files[0], tmp)
            kml_path = Path(tmp) / kml_files[0]
            return read_kml_features(str(kml_path)), "EPSG:4326"


def _plain_geometry(geom: Any) -> Optional[Dict[str, Any]]:
    if geom is None:
        return None
    if isinstance(geom, dict):
        return json.loads(json.dumps(geom))
    if hasattr(geom, "__geo_interface__"):
        return json.loads(json.dumps(geom.__geo_interface__))
    try:
        return json.loads(json.dumps(dict(geom)))
    except Exception:
        return None


def _plain_properties(props: Any) -> Dict[str, Any]:
    try:
        props = dict(props)
    except Exception:
        props = {}
    out: Dict[str, Any] = {}
    for k, v in props.items():
        if isinstance(v, (str, int, float, bool)) or v is None:
            out[str(k)] = v
        else:
            out[str(k)] = str(v)
    return out


# ---------------------------------------------------------------------------
# KML/GeoJSON/Shapefile writers
# ---------------------------------------------------------------------------
def write_geojson(output_path: str, features: List[Dict[str, Any]], source_crs: Optional[Any] = None) -> None:
    collection: Dict[str, Any] = {"type": "FeatureCollection", "features": features}
    # Keep CRS summary as metadata without forcing coordinate reprojection.
    if source_crs:
        collection["name"] = Path(output_path).stem
        collection["Terra View_source_crs"] = crs_to_readable(source_crs)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(collection, f, ensure_ascii=False, indent=2)


def write_kml(output_path: str, features: List[Dict[str, Any]], source_crs: Optional[Any] = None) -> None:
    # KML expects lon/lat (EPSG:4326). Reproject when pyproj can understand the source CRS.
    transformer = _transformer_to_wgs84(source_crs)
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<kml xmlns="http://www.opengis.net/kml/2.2">',
        '<Document>',
        f'<name>{escape(Path(output_path).stem)}</name>',
    ]
    for idx, feat in enumerate(features, start=1):
        props = feat.get("properties") or {}
        name = props.get("name") or props.get("Name") or props.get("NAMA") or props.get("id") or f"Feature {idx}"
        lines.append("<Placemark>")
        lines.append(f"<name>{escape(str(name))}</name>")
        if props:
            lines.append("<ExtendedData>")
            for key, value in props.items():
                lines.append(f'<Data name="{escape(str(key))}"><value>{escape("" if value is None else str(value))}</value></Data>')
            lines.append("</ExtendedData>")
        geom_xml = geometry_to_kml(feat.get("geometry"), transformer=transformer)
        if geom_xml:
            lines.append(geom_xml)
        lines.append("</Placemark>")
    lines.extend(["</Document>", "</kml>"])
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


_PYSHP_TYPE_BY_GEOM = {}


def _pyshp_shape_type(geom_type: str) -> int:
    if not _PYSHP_TYPE_BY_GEOM:
        _PYSHP_TYPE_BY_GEOM.update({
            "Point": pyshp.POINT,
            "LineString": pyshp.POLYLINE,
            "Polygon": pyshp.POLYGON,
        })
    return _PYSHP_TYPE_BY_GEOM[geom_type]


def write_shapefile(output_path: str, features: List[Dict[str, Any]], source_crs: Optional[Any] = None) -> None:
    if pyshp is None:
        raise RuntimeError("pyshp belum tersedia untuk menulis Shapefile.")
    geom_type = _single_geometry_type(features)
    if geom_type in {"GeometryCollection", "Unknown"}:
        raise RuntimeError("Shapefile tidak mendukung geometry campuran pada data ini. Gunakan GeoJSON sebagai output.")
    schema_props = _infer_str_schema(features)

    # Remove existing shapefile sidecars to avoid stale schema conflicts.
    stem = Path(output_path).with_suffix("")
    stem.parent.mkdir(parents=True, exist_ok=True)
    for ext in [".shp", ".shx", ".dbf", ".prj", ".cpg"]:
        side = stem.with_suffix(ext)
        if side.exists():
            try:
                side.unlink()
            except Exception:
                pass

    with pyshp.Writer(str(stem), shapeType=_pyshp_shape_type(geom_type)) as writer:
        for key in schema_props:
            writer.field(key, "C", 254)
        for feat in features:
            geom = feat.get("geometry")
            if geometry_base_type(geom) != geom_type:
                continue
            _write_pyshp_geometry(writer, geom_type, geom)
            writer.record(**_schema_props(feat.get("properties") or {}, schema_props))

    if source_crs:
        wkt = _crs_to_wkt(source_crs)
        if wkt:
            stem.with_suffix(".prj").write_text(wkt, encoding="utf-8")


def _write_pyshp_geometry(writer, geom_type: str, geom: Dict[str, Any]) -> None:
    if geom_type == "Point":
        for _, coords in iter_geometry_parts(geom):
            x, y = coords[0]
            writer.point(x, y)
            return
    elif geom_type == "LineString":
        parts = [coords for kind, coords in iter_geometry_parts(geom) if kind == "LineString" and coords]
        writer.line(parts)
    elif geom_type == "Polygon":
        parts = [coords for kind, coords in iter_geometry_parts(geom) if kind == "PolygonRing" and coords]
        writer.poly(parts)


def _single_geometry_type(features: List[Dict[str, Any]]) -> str:
    types = [geometry_base_type(f.get("geometry")) for f in features if f.get("geometry")]
    types = [t for t in types if t != "Unknown"]
    if not types:
        return "Unknown"
    # Shapefile stores MultiPolygon as Polygon and MultiLineString as LineString.
    normalized = {"MultiPolygon": "Polygon", "MultiLineString": "LineString", "MultiPoint": "Point"}
    norm = [normalized.get(t, t) for t in types]
    return norm[0] if len(set(norm)) == 1 else "GeometryCollection"


def geometry_base_type(geom: Optional[Dict[str, Any]]) -> str:
    if not geom:
        return "Unknown"
    gtype = geom.get("type", "Unknown")
    return {"MultiPolygon": "Polygon", "MultiLineString": "LineString", "MultiPoint": "Point"}.get(gtype, gtype)


def _infer_str_schema(features: List[Dict[str, Any]]) -> Dict[str, str]:
    keys: List[str] = []
    for feat in features[:200]:
        for key in (feat.get("properties") or {}).keys():
            safe = _safe_field_name(str(key))
            if safe not in keys:
                keys.append(safe)
    if not keys:
        keys = ["name"]
    return {k: "str:254" for k in keys[:80]}


def _safe_field_name(name: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in name).strip("_") or "field"
    return cleaned[:10]


def _schema_props(props: Dict[str, Any], schema_props: Dict[str, str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    normalized = {_safe_field_name(str(k)): v for k, v in props.items()}
    for key in schema_props:
        value = normalized.get(key, "")
        out[key] = "" if value is None else str(value)[:254]
    return out


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
def _transformer_to_wgs84(source_crs: Optional[Any]) -> Optional[Any]:
    if not source_crs or CRS is None or Transformer is None:
        return None
    try:
        src = CRS.from_user_input(source_crs)
        dst = CRS.from_epsg(4326)
        if src == dst:
            return None
        return Transformer.from_crs(src, dst, always_xy=True)
    except Exception:
        return None


def transform_xy(x: Any, y: Any, transformer: Optional[Any]) -> Tuple[float, float]:
    x_f, y_f = float(x), float(y)
    if transformer is None:
        return x_f, y_f
    try:
        tx, ty = transformer.transform(x_f, y_f)
        return float(tx), float(ty)
    except Exception:
        return x_f, y_f


def coord_to_kml(coord: Sequence[Any], transformer: Optional[Any] = None) -> str:
    x, y = transform_xy(coord[0], coord[1], transformer)
    if len(coord) >= 3 and coord[2] is not None:
        return f"{x:.10f},{y:.10f},{float(coord[2]):.3f}"
    return f"{x:.10f},{y:.10f}"


def geometry_to_kml(geom: Optional[Dict[str, Any]], transformer: Optional[Any] = None) -> str:
    if not geom:
        return ""
    gtype = geom.get("type")
    coords = geom.get("coordinates")
    if gtype == "Point":
        return f"<Point><coordinates>{coord_to_kml(coords, transformer)}</coordinates></Point>"
    if gtype == "LineString":
        return f"<LineString><tessellate>1</tessellate><coordinates>{' '.join(coord_to_kml(c, transformer) for c in coords)}</coordinates></LineString>"
    if gtype == "Polygon":
        return polygon_to_kml(coords, transformer)
    if gtype == "MultiPoint":
        return "<MultiGeometry>" + "".join(geometry_to_kml({"type": "Point", "coordinates": c}, transformer) for c in coords) + "</MultiGeometry>"
    if gtype == "MultiLineString":
        return "<MultiGeometry>" + "".join(geometry_to_kml({"type": "LineString", "coordinates": c}, transformer) for c in coords) + "</MultiGeometry>"
    if gtype == "MultiPolygon":
        return "<MultiGeometry>" + "".join(polygon_to_kml(poly, transformer) for poly in coords) + "</MultiGeometry>"
    if gtype == "GeometryCollection":
        return "<MultiGeometry>" + "".join(geometry_to_kml(g, transformer) for g in geom.get("geometries", [])) + "</MultiGeometry>"
    return ""


def polygon_to_kml(rings: Sequence[Any], transformer: Optional[Any] = None) -> str:
    if not rings:
        return ""
    outer = " ".join(coord_to_kml(c, transformer) for c in rings[0])
    xml = ["<Polygon><tessellate>1</tessellate>", f"<outerBoundaryIs><LinearRing><coordinates>{outer}</coordinates></LinearRing></outerBoundaryIs>"]
    for inner_ring in rings[1:]:
        inner = " ".join(coord_to_kml(c, transformer) for c in inner_ring)
        xml.append(f"<innerBoundaryIs><LinearRing><coordinates>{inner}</coordinates></LinearRing></innerBoundaryIs>")
    xml.append("</Polygon>")
    return "".join(xml)


# ---------------------------------------------------------------------------
# Basic KML parser
# ---------------------------------------------------------------------------
def read_kml_features(path: str) -> List[Dict[str, Any]]:
    tree = ET.parse(path)
    root = tree.getroot()
    placemarks = [el for el in root.iter() if _local(el.tag) == "Placemark"]
    features: List[Dict[str, Any]] = []
    for idx, pm in enumerate(placemarks, start=1):
        name = _child_text(pm, "name") or f"Placemark {idx}"
        props = {"name": name}
        for data in [el for el in pm.iter() if _local(el.tag) == "Data"]:
            key = data.attrib.get("name")
            val = _child_text(data, "value")
            if key:
                props[key] = val
        geometries = _parse_geometries(pm)
        for geom in geometries:
            features.append({"type": "Feature", "id": str(len(features)), "properties": props.copy(), "geometry": geom})
    return features


def _parse_geometries(node: ET.Element) -> List[Dict[str, Any]]:
    geoms: List[Dict[str, Any]] = []
    # Avoid parsing coordinates nested inside already-parsed complex geometry twice by targeting geometry elements.
    for el in list(node.iter()):
        tag = _local(el.tag)
        if tag == "Point":
            text = _first_coordinates_text(el)
            coords = _parse_kml_coords(text)
            if coords:
                geoms.append({"type": "Point", "coordinates": coords[0]})
        elif tag == "LineString":
            text = _first_coordinates_text(el)
            coords = _parse_kml_coords(text)
            if coords:
                geoms.append({"type": "LineString", "coordinates": coords})
        elif tag == "Polygon":
            rings: List[List[Tuple[float, ...]]] = []
            for ring in [r for r in el.iter() if _local(r.tag) == "LinearRing"]:
                coords = _parse_kml_coords(_first_coordinates_text(ring))
                if coords:
                    rings.append(coords)
            if rings:
                geoms.append({"type": "Polygon", "coordinates": rings})
    # Remove geometries parsed from nested Point/LineString inside MultiGeometry duplicates is acceptable for simple conversions.
    return geoms


def _local(tag: str) -> str:
    return tag.split("}", 1)[-1] if "}" in tag else tag


def _child_text(node: ET.Element, local_name: str) -> Optional[str]:
    for child in node.iter():
        if _local(child.tag) == local_name and child.text:
            return child.text.strip()
    return None


def _first_coordinates_text(node: ET.Element) -> str:
    for child in node.iter():
        if _local(child.tag) == "coordinates" and child.text:
            return child.text.strip()
    return ""


def _parse_kml_coords(text: str) -> List[Tuple[float, ...]]:
    coords: List[Tuple[float, ...]] = []
    for token in text.replace("\n", " ").replace("\t", " ").split():
        parts = [p for p in token.split(",") if p != ""]
        if len(parts) >= 2:
            try:
                if len(parts) >= 3:
                    coords.append((float(parts[0]), float(parts[1]), float(parts[2])))
                else:
                    coords.append((float(parts[0]), float(parts[1])))
            except ValueError:
                continue
    return coords


# ---------------------------------------------------------------------------
# CRS helpers and UI text
# ---------------------------------------------------------------------------
def _crs_to_wkt(crs: Any) -> Optional[str]:
    if crs is None:
        return None
    if CRS is not None:
        try:
            return CRS.from_user_input(crs).to_wkt()
        except Exception:
            pass
    return str(crs)


def crs_to_readable(crs: Any) -> str:
    if not crs:
        return "Tidak tersedia"
    if CRS is not None:
        try:
            c = CRS.from_user_input(crs)
            epsg = c.to_epsg()
            name = c.name or "CRS"
            return f"EPSG:{epsg} - {name}" if epsg else name
        except Exception:
            pass
    text = str(crs)
    if 'AUTHORITY["EPSG"' in text:
        import re
        found = re.findall(r'AUTHORITY\["EPSG","(\d+)"\]', text)
        if found:
            return f"EPSG:{found[-1]}"
    return text[:120] + ("..." if len(text) > 120 else "")


def supported_conversion_text() -> str:
    return (
        "Raster: GeoTIFF/TIFF/JPG/PNG/IMG/ECW -> GeoTIFF/PNG/JPEG. "
        "ECW membutuhkan GDAL ECW driver; jika rasterio gagal, Terra View mencoba OSGeo4W/QGIS gdal_translate.\n"
        "Vector: SHP/DXF/GeoJSON/KML/KMZ -> SHP/KML/GeoJSON. "
        "KML ditulis manual sehingga tidak bergantung pada driver KML Fiona."
    )
