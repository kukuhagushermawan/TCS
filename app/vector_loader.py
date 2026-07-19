"""Vector loaders for SHP, KML, KMZ, and GeoJSON.

Uses pyshp (pure Python, no GDAL) for Shapefile and the stdlib json module
for GeoJSON, so the core app does not need to bundle a private GDAL copy
just to open vector data. DXF is not supported here - it needs GDAL's DXF
driver, which is only available through the optional GDAL runtime (see
app/gdal_runtime_dialog.py); a DXF file raises a clear message instead of
silently failing or crashing.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple
import zipfile
import tempfile
import xml.etree.ElementTree as ET

from .layer_manager import Layer

try:
    import shapefile as pyshp
except Exception:  # pragma: no cover
    pyshp = None


def load_vector(path: str) -> Layer:
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == ".kmz":
        return _load_kmz(path)
    if suffix == ".kml":
        return _load_kml_light(path)
    if suffix in {".geojson", ".json"}:
        return _load_geojson(path)
    if suffix == ".shp":
        return _load_shapefile(path)
    if suffix == ".dxf":
        raise RuntimeError(
            "Format DXF membutuhkan GDAL runtime tambahan.\n\n"
            "Gunakan menu Open Raster/Open Vector setelah memilih GDAL runtime "
            "yang mendukung DXF (misalnya OSGeo4W/QGIS), atau konversi DXF ke "
            "SHP/GeoJSON terlebih dahulu dengan QGIS."
        )
    raise RuntimeError(f"Format vector belum didukung: {suffix}")


def _load_shapefile(path: str) -> Layer:
    if pyshp is None:
        raise RuntimeError("pyshp belum tersedia untuk membuka Shapefile.")
    features: List[Dict[str, Any]] = []
    with pyshp.Reader(path) as reader:
        field_names = [f[0] for f in reader.fields if f[0] != "DeletionFlag"]
        bounds = tuple(reader.bbox) if reader.bbox else None
        for i, shape_rec in enumerate(reader.iterShapeRecords()):
            geom = shape_rec.shape.__geo_interface__ if shape_rec.shape.points else None
            if not geom:
                continue
            props = dict(zip(field_names, shape_rec.record))
            features.append({"id": str(i), "geometry": geom, "properties": props})
    crs = _read_shapefile_crs(Path(path))
    return Layer(
        name=Path(path).name,
        layer_type="vector",
        path=path,
        features=features,
        crs=crs,
        bounds=bounds,
        source_driver="ESRI Shapefile (pyshp)",
        metadata={"feature_count": len(features)},
    )


def _read_shapefile_crs(path: Path):
    prj_path = path.with_suffix(".prj")
    if prj_path.exists():
        try:
            return prj_path.read_text(encoding="utf-8", errors="ignore").strip()
        except Exception:
            return None
    return None


def _load_geojson(path: str) -> Layer:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    features: List[Dict[str, Any]] = []
    if data.get("type") == "FeatureCollection":
        raw_features = data.get("features", [])
    elif data.get("type") == "Feature":
        raw_features = [data]
    else:
        raw_features = [{"type": "Feature", "properties": {}, "geometry": data}]
    for i, feat in enumerate(raw_features):
        geom = feat.get("geometry")
        if not geom:
            continue
        features.append({
            "id": str(feat.get("id", i)),
            "geometry": geom,
            "properties": dict(feat.get("properties") or {}),
        })
    bounds = _feature_bounds(features)
    return Layer(
        name=Path(path).name,
        layer_type="vector",
        path=path,
        features=features,
        crs="EPSG:4326",
        bounds=bounds,
        source_driver="GeoJSON",
        metadata={"feature_count": len(features)},
    )


def _load_kmz(path: str) -> Layer:
    with tempfile.TemporaryDirectory() as tmp:
        with zipfile.ZipFile(path) as z:
            kml_files = [n for n in z.namelist() if n.lower().endswith(".kml")]
            if not kml_files:
                raise RuntimeError("KMZ tidak berisi file KML.")
            z.extract(kml_files[0], tmp)
            kml_path = str(Path(tmp) / kml_files[0])
            layer = _load_kml_light(kml_path)
            layer.name = Path(path).name
            layer.path = path
            return layer


def _load_kml_light(path: str) -> Layer:
    """Small KML parser for Point/LineString/Polygon coordinates.

    This fallback is intentionally basic. For full KML style/support, install GDAL/Fiona.
    """
    ns = {"kml": "http://www.opengis.net/kml/2.2"}
    tree = ET.parse(path)
    root = tree.getroot()
    features: List[Dict[str, Any]] = []

    for idx, placemark in enumerate(root.findall(".//kml:Placemark", ns)):
        name_node = placemark.find("kml:name", ns)
        name = name_node.text if name_node is not None else f"Placemark {idx+1}"
        for geom_tag, geom_type in [("Point", "Point"), ("LineString", "LineString"), ("Polygon", "Polygon")]:
            for geom in placemark.findall(f".//kml:{geom_tag}", ns):
                coords_node = geom.find(".//kml:coordinates", ns)
                if coords_node is None or not coords_node.text:
                    continue
                coords = _parse_kml_coords(coords_node.text)
                if geom_type == "Point" and coords:
                    geometry = {"type": "Point", "coordinates": coords[0]}
                elif geom_type == "Polygon" and coords:
                    geometry = {"type": "Polygon", "coordinates": [coords]}
                else:
                    geometry = {"type": "LineString", "coordinates": coords}
                features.append({"id": str(idx), "geometry": geometry, "properties": {"name": name}})
    bounds = _feature_bounds(features)
    return Layer(
        name=Path(path).name,
        layer_type="vector",
        path=path,
        features=features,
        crs="EPSG:4326",
        bounds=bounds,
        source_driver="KML-light",
        metadata={"feature_count": len(features)},
    )


def _parse_kml_coords(text: str) -> List[Tuple[float, float]]:
    coords = []
    for part in text.replace("\n", " ").split():
        vals = part.split(",")
        if len(vals) >= 2:
            coords.append((float(vals[0]), float(vals[1])))
    return coords


def _feature_bounds(features: List[Dict[str, Any]]):
    xs: List[float] = []
    ys: List[float] = []
    for feat in features:
        for x, y in iter_geometry_coords(feat.get("geometry")):
            xs.append(x)
            ys.append(y)
    if not xs:
        return None
    return min(xs), min(ys), max(xs), max(ys)



def iter_geometry_coords(geometry: Dict[str, Any]) -> Iterable[Tuple[float, float]]:
    if not geometry:
        return
    gtype = geometry.get("type")
    coords = geometry.get("coordinates")
    if gtype == "Point":
        yield float(coords[0]), float(coords[1])
    elif gtype in {"LineString", "MultiPoint"}:
        for xy in coords:
            yield float(xy[0]), float(xy[1])
    elif gtype == "Polygon":
        for ring in coords:
            for xy in ring:
                yield float(xy[0]), float(xy[1])
    elif gtype == "MultiLineString":
        for line in coords:
            for xy in line:
                yield float(xy[0]), float(xy[1])
    elif gtype == "MultiPolygon":
        for poly in coords:
            for ring in poly:
                for xy in ring:
                    yield float(xy[0]), float(xy[1])



def iter_geometry_parts(geometry: Dict[str, Any]) -> Iterable[Tuple[str, List[Tuple[float, float]]]]:
    """Yield independent drawable geometry parts.

    This is intentionally different from iter_geometry_coords(). For rendering,
    every polygon ring / multiline segment must be drawn as a separate path.
    Otherwise the UI connects the last coordinate of one ring to the first
    coordinate of the next ring, creating diagonal "spider web" artifacts.
    """
    if not geometry:
        return
    gtype = geometry.get("type")
    coords = geometry.get("coordinates")

    def as_xy_list(seq) -> List[Tuple[float, float]]:
        return [(float(xy[0]), float(xy[1])) for xy in seq if len(xy) >= 2]

    if gtype == "Point":
        yield "Point", [(float(coords[0]), float(coords[1]))]
    elif gtype == "MultiPoint":
        for xy in coords:
            yield "Point", [(float(xy[0]), float(xy[1]))]
    elif gtype == "LineString":
        yield "LineString", as_xy_list(coords)
    elif gtype == "MultiLineString":
        for line in coords:
            yield "LineString", as_xy_list(line)
    elif gtype == "Polygon":
        for ring in coords:
            yield "PolygonRing", as_xy_list(ring)
    elif gtype == "MultiPolygon":
        for poly in coords:
            for ring in poly:
                yield "PolygonRing", as_xy_list(ring)
    elif gtype == "GeometryCollection":
        for geom in geometry.get("geometries", []) or []:
            yield from iter_geometry_parts(geom)


def geometry_type(geometry: Dict[str, Any]) -> str:
    return geometry.get("type", "Unknown") if geometry else "Unknown"
