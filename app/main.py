"""TCS (Tree Counting Sawit) entry point.

Built on the Terra View raster/vector viewer and adds the TCS Analysis module
for automatic oil-palm tree detection/counting from raster imagery.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Allow running from source root: python app/main.py
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PyQt6.QtWidgets import QApplication
from app.ui_main import create_main_window


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("TCS")
    app.setOrganizationName("TCS")
    window = create_main_window()
    window.showMaximized()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
