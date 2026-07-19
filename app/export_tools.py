"""Export helpers for raster/vector images."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .image_enhancement import to_uint8_rgb
from .raster_loader import save_array_as_geotiff


def export_image_array(path: str, image: np.ndarray) -> str:
    img = Image.fromarray(to_uint8_rgb(image))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if Path(path).suffix.lower() in {".jpg", ".jpeg"}:
        img = img.convert("RGB")
    img.save(path)
    return path


def export_enhancement(path: str, layer: Any) -> str:
    image = layer.display_image()
    if image is None:
        raise RuntimeError("Layer tidak memiliki image untuk diekspor.")
    if Path(path).suffix.lower() in {".tif", ".tiff"}:
        save_array_as_geotiff(path, image, layer)
    else:
        export_image_array(path, image)
    return path
