"""Runtime resource path helpers for Terra View."""
from __future__ import annotations

import sys
from pathlib import Path


def app_root() -> Path:
    """Return project root in source mode or PyInstaller runtime root in frozen mode."""
    if hasattr(sys, "_MEIPASS"):
        return Path(getattr(sys, "_MEIPASS"))
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[1]


def resource_path(*parts: str) -> Path:
    return app_root().joinpath(*parts)
