"""Vector loaders for SHP, KML, KMZ, GeoJSON, and DXF.

Uses pyshp (pure Python, no GDAL) for Shapefile, the stdlib json module for
GeoJSON, and ezdxf (pure Python, no GDAL) for DXF, so the core app does not
need to bundle a private GDAL copy just to open vector data.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
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
        return _load_dxf(path)
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
    return read_sidecar_prj(path)


def read_sidecar_prj(path: Path) -> Optional[str]:
    """Read a companion ``<name>.prj`` WKT file next to any vector/raster file.

    Shapefile always carries its CRS this way, but the same ``.prj`` sidecar
    convention is also used by some tools for DXF exports (e.g. Global
    Mapper/QGIS) and for georeferenced images (worldfile + .prj pairs), so
    every loader that gets a bare/no-CRS format checks here before giving up.
    """
    prj_path = Path(path).with_suffix(".prj")
    if prj_path.exists():
        try:
            text = prj_path.read_text(encoding="utf-8", errors="ignore").strip()
            return text or None
        except Exception:
            return None
    return None


def geojson_crs_to_string(data: Dict[str, Any]) -> str:
    """Resolve a GeoJSON object's legacy ``"crs"`` member to a CRS string.

    Defaults to EPSG:4326 (the RFC 7946 assumption) when no ``"crs"`` member
    is present, but honors an explicit legacy member (still produced by some
    GIS export tools) instead of silently overriding it.
    """
    crs_member = data.get("crs")
    if not isinstance(crs_member, dict):
        return "EPSG:4326"
    props = crs_member.get("properties")
    name = props.get("name") if isinstance(props, dict) else None
    if not name and isinstance(props, dict):
        name = props.get("code")
    if not name:
        return "EPSG:4326"
    name = str(name)
    if "CRS84" in name.upper():
        # OGC CRS84 is WGS84 in lon,lat order - the same order GeoJSON already uses.
        return "EPSG:4326"
    match = re.search(r"EPSG[:]{1,2}(\d+)", name, re.IGNORECASE)
    if match:
        return f"EPSG:{match.group(1)}"
    if name.upper().startswith("EPSG:"):
        return name.upper()
    return name


# ---------------------------------------------------------------------------
# DXF (ezdxf - pure Python, no GDAL/OGR DXF driver needed)
# ---------------------------------------------------------------------------

# Flattening tolerance (drawing units) for approximating curved entities
# (ARC/CIRCLE/ELLIPSE/SPLINE/bulged LWPOLYLINE) as straight-line vertices -
# our internal geometry model only knows Point/LineString/Polygon, the same
# limitation GeoJSON/Shapefile have. 0.1 gives a visually smooth boundary for
# typical plantation/survey-scale drawings (meters) without excessive points.
_DXF_FLATTEN_SAGITTA = 0.1


def _load_dxf(path: str) -> Layer:
    features = read_dxf_features(path)
    bounds = _feature_bounds(features)
    return Layer(
        name=Path(path).name,
        layer_type="vector",
        path=path,
        features=features,
        # DXF itself has no standard CRS tag (unlike Shapefile's .prj or
        # KML's fixed WGS84) - coordinates are just raw drawing units. Some
        # export tools (Global Mapper, QGIS DXF export) still drop a
        # companion .prj next to the .dxf, so check for that before giving up.
        crs=read_sidecar_prj(Path(path)),
        bounds=bounds,
        source_driver="DXF (ezdxf)",
        metadata={"feature_count": len(features)},
    )


def read_dxf_features(path: str) -> List[Dict[str, Any]]:
    """Flatten every drawable DXF entity - including entities inside INSERT
    block references, fully transformed - into plain Point/LineString/Polygon
    features, so DXF opens/converts through the same feature-dict shape as
    every other vector format here. Shared by format_converter.read_vector_features
    so DXF can also be exported once opened.
    """
    try:
        import ezdxf
        from ezdxf import path as ezpath
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("ezdxf belum tersedia untuk membuka DXF.") from exc

    try:
        doc = ezdxf.readfile(path)
    except Exception as exc:
        raise RuntimeError(f"Gagal membaca DXF: {exc}") from exc

    features: List[Dict[str, Any]] = []
    for entity in _iter_dxf_entities(doc.modelspace()):
        feat = _dxf_entity_to_feature(entity, ezpath, len(features) + 1)
        if feat is not None:
            features.append(feat)
    return features


def _iter_dxf_entities(layout):
    """Yield drawable entities, resolving INSERT block references into their
    already-transformed virtual entities (recursively - a block can itself
    contain nested INSERTs) instead of the untransformed block definition."""
    for entity in layout:
        if entity.dxftype() == "INSERT":
            yield from _iter_dxf_entities(entity.virtual_entities())
        else:
            yield entity


def _dxf_entity_to_feature(entity, ezpath, feature_id: int) -> Optional[Dict[str, Any]]:
    dxftype = entity.dxftype()
    props = {"entity_type": dxftype, "layer": entity.dxf.layer}

    if dxftype == "POINT":
        loc = entity.dxf.location
        return {
            "type": "Feature", "id": str(feature_id), "properties": props,
            "geometry": {"type": "Point", "coordinates": [loc.x, loc.y]},
        }

    try:
        flattened_path = ezpath.make_path(entity)
    except TypeError:
        # TEXT/MTEXT/HATCH/DIMENSION/3DFACE/... - not a line/curve geometry,
        # skip rather than fail the whole file over one unsupported entity.
        return None
    verts = [(v.x, v.y) for v in flattened_path.flattening(_DXF_FLATTEN_SAGITTA)]
    if len(verts) < 2:
        return None

    # CIRCLE/ELLIPSE are always closed loops; LWPOLYLINE/POLYLINE report their
    # own closed state. ezdxf's own flattening already repeats the first
    # vertex at the end for closed entities - the explicit check here is a
    # cheap safeguard in case a future entity type doesn't.
    is_closed = dxftype in {"CIRCLE", "ELLIPSE"} or bool(getattr(entity, "closed", False)) or bool(getattr(entity, "is_closed", False))
    if is_closed:
        if verts[0] != verts[-1]:
            verts.append(verts[0])
        if len(verts) < 4:
            return None
        geometry = {"type": "Polygon", "coordinates": [[list(v) for v in verts]]}
    else:
        geometry = {"type": "LineString", "coordinates": [list(v) for v in verts]}
    return {"type": "Feature", "id": str(feature_id), "properties": props, "geometry": geometry}


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
        crs=geojson_crs_to_string(data),
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
