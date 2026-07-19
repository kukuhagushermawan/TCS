"""Layer data model for Terra View.

This module intentionally keeps the layer model small and dependency-light. GUI code,
loaders, and exporters communicate through these classes so the project remains easy
to extend without turning into a heavy GIS suite.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import uuid

import numpy as np


BBox = Tuple[float, float, float, float]


@dataclass
class Layer:
    """Single visual layer in the map.

    layer_type values used by the app:
    - raster: RGB/single band raster rendered as image
    - dem: elevation raster rendered as grayscale/color ramp
    - vector: SHP/KML/KMZ/DXF geometries
    - marker: click marker/overlay
    """

    name: str
    layer_type: str
    path: Optional[str] = None
    visible: bool = True
    opacity: float = 1.0
    id: str = field(default_factory=lambda: str(uuid.uuid4()))

    # Raster/DEM fields
    image: Optional[np.ndarray] = None          # uint8 RGB for display
    raw_data: Optional[np.ndarray] = None       # original data/bands when available
    enhanced_image: Optional[np.ndarray] = None # uint8 RGB after processing
    dem_data: Optional[np.ndarray] = None       # float elevation values
    transform: Any = None                       # affine transform from rasterio
    crs: Any = None                             # rasterio/pyproj CRS object or string
    profile: Dict[str, Any] = field(default_factory=dict)
    nodata: Optional[float] = None
    width: Optional[int] = None
    height: Optional[int] = None
    extent: Optional[BBox] = None               # world bounds: minx, miny, maxx, maxy

    # Vector/LAS fields
    features: List[Dict[str, Any]] = field(default_factory=list)
    bounds: Optional[BBox] = None
    source_driver: Optional[str] = None

    # UI/metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def display_image(self) -> Optional[np.ndarray]:
        """Return enhanced image when available, otherwise original display image."""
        return self.enhanced_image if self.enhanced_image is not None else self.image

    def short_info(self) -> str:
        parts = [self.layer_type.upper()]
        if self.width and self.height:
            parts.append(f"{self.width}x{self.height}px")
        if self.crs:
            parts.append(f"CRS={self.crs}")
        if self.path:
            parts.append(Path(self.path).name)
        return " | ".join(parts)


class LayerManager:
    """Small in-memory layer stack."""

    def __init__(self) -> None:
        self.layers: List[Layer] = []
        self.active_layer_id: Optional[str] = None

    def add_layer(self, layer: Layer) -> Layer:
        self.layers.append(layer)
        self.active_layer_id = layer.id
        return layer

    def remove_layer(self, layer_id: str) -> None:
        self.layers = [lyr for lyr in self.layers if lyr.id != layer_id]
        if self.active_layer_id == layer_id:
            self.active_layer_id = self.layers[-1].id if self.layers else None

    def get(self, layer_id: str) -> Optional[Layer]:
        for layer in self.layers:
            if layer.id == layer_id:
                return layer
        return None

    def clear(self) -> None:
        self.layers.clear()
        self.active_layer_id = None

    def move_up(self, layer_id: str) -> None:
        idx = self._index(layer_id)
        if idx is not None and idx < len(self.layers) - 1:
            self.layers[idx], self.layers[idx + 1] = self.layers[idx + 1], self.layers[idx]

    def move_down(self, layer_id: str) -> None:
        idx = self._index(layer_id)
        if idx is not None and idx > 0:
            self.layers[idx], self.layers[idx - 1] = self.layers[idx - 1], self.layers[idx]

    def set_visible(self, layer_id: str, visible: bool) -> None:
        layer = self.get(layer_id)
        if layer:
            layer.visible = visible

    def set_opacity(self, layer_id: str, opacity: float) -> None:
        layer = self.get(layer_id)
        if layer:
            layer.opacity = max(0.0, min(1.0, opacity))

    def visible_layers(self) -> List[Layer]:
        return [lyr for lyr in self.layers if lyr.visible]

    def raster_layers(self) -> List[Layer]:
        return [lyr for lyr in self.layers if lyr.layer_type in {"raster", "dem"}]

    def set_active(self, layer_id: Optional[str]) -> None:
        self.active_layer_id = layer_id

    def active_layer(self) -> Optional[Layer]:
        if self.active_layer_id:
            layer = self.get(self.active_layer_id)
            if layer and layer.visible:
                return layer
        return self.layers[-1] if self.layers else None

    def active_base_raster(self) -> Optional[Layer]:
        """Return selected visible raster/DEM first, then top-most visible raster/DEM."""
        active = self.active_layer()
        if active and active.visible and active.layer_type in {"raster", "dem"} and active.display_image() is not None:
            return active
        for layer in reversed(self.layers):
            if layer.visible and layer.layer_type in {"raster", "dem"} and layer.display_image() is not None:
                return layer
        return None

    def _index(self, layer_id: str) -> Optional[int]:
        for i, layer in enumerate(self.layers):
            if layer.id == layer_id:
                return i
        return None
