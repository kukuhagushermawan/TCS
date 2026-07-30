# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for TCS (Tree Counting Sawit).

Clean build rule:
- Raster + Vector viewer plus the TCS Analysis module (oil-palm tree counting).
- TCS's bundled detection model is ONNX (models/*.onnx), run through
  onnxruntime (CPU-only, ~15-20 MB) - no PyTorch/ultralytics/other large ML
  framework is needed at build or runtime, so the installed app runs fully
  self-contained without any extra install on the client machine.
- Do NOT use collect_all() for PyQt6/rasterio/pyproj/shapely/PIL - it drags
  in unused Qt Quick/QML/3D/WebEngine/Designer/Pdf plugins and package
  docs/tests, which is most of the previous installer's size.
- Do NOT use Fiona (private GDAL/PROJ/spatialite duplicate of rasterio's own
  copy) - Shapefile I/O uses pyshp instead.
"""
from pathlib import Path

project_root = Path.cwd()
block_cipher = None

# NOTE: vendor/ (the bundled GDAL-ECW runtime) is deliberately NOT listed here.
# Putting it in datas makes PyInstaller copy its ~150 MB of DLLs into
# _internal/ AND keep a second copy, doubling size. It is copied into
# dist/TCS/_internal/vendor/ AFTER PyInstaller by REBUILD_EXE.bat /
# BUILD_INSTALLER.bat instead (see those scripts).
datas = [
    (str(project_root / "assets" / "icons"), "assets/icons"),
    (str(project_root / "assets" / "logos"), "assets/logos"),
    (str(project_root / "assets" / "opening"), "assets/opening"),
    (str(project_root / "models"), "models"),
    (str(project_root / "legal"), "legal"),
]
binaries = []
hiddenimports = [
    "PyQt6.QtCore",
    "PyQt6.QtGui",
    "PyQt6.QtWidgets",
    "rasterio.sample",
    "rasterio.vrt",
    "rasterio.features",
    "rasterio._features",
    "rasterio._warp",
    "shapefile",
    "pyproj.datadir",
    "shapely.geometry",
    "PIL.Image",
    "PIL.ImageQt",
    "PIL.ImageDraw",
    "PIL.ImageFont",
    # TCS detection backend (tcs/inference.py OnnxTreeDetector). The
    # pyinstaller-hooks-contrib "onnxruntime" hook collects its native
    # onnxruntime.dll / onnxruntime_providers_shared.dll / pybind11 .pyd
    # automatically once the package itself is included here.
    "onnxruntime",
    # DXF reader (app/vector_loader.py read_dxf_features) - pure Python,
    # pulls in fontTools as its own transitive dependency.
    "ezdxf",
    # ezdxf.sections.blocks imports pyparsing unconditionally as part of
    # normal document loading (not just its optional query() DSL) - but
    # PyInstaller's static analysis misses it (ezdxf lazy-loads many of its
    # own submodules), so it must be forced here or opening any DXF crashes
    # in the frozen app with "No module named 'pyparsing'".
    "pyparsing",
]

_blocked_prefixes = (
    "realesrgan", "basicsr", "facexlib", "gfpgan", "filterpy", "torchvision",
    "tensorboard", "skimage", "cv2", "sklearn", "numba", "IPython", "jupyter",
    "notebook", "pytest", "tkinter", "torch", "torchgen", "matplotlib",
    "PyQt6.QtQml", "PyQt6.QtQuick", "PyQt6.QtQuick3D", "PyQt6.QtMultimedia",
    "PyQt6.QtWebEngineWidgets", "PyQt6.QtWebEngineCore", "PyQt6.QtPdf",
    "PyQt6.QtDesigner", "PyQt6.QtBluetooth", "PyQt6.QtNfc", "PyQt6.QtPositioning",
    "PyQt6.QtSensors", "PyQt6.QtSerialPort", "PyQt6.QtSql", "PyQt6.QtHelp", "PyQt6.QtTest",
)
hiddenimports = [item for item in hiddenimports if not any(item == p or item.startswith(p + ".") for p in _blocked_prefixes)]

analysis = Analysis(
    [str(project_root / "app" / "main.py")],
    pathex=[str(project_root)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "IPython", "jupyter", "notebook", "pytest", "tkinter", "matplotlib", "matplotlib.*",
        "skimage", "skimage.*", "cv2", "cv2.*", "sklearn", "sklearn.*", "numba", "numba.*",
        "torch", "torch.*", "torchvision", "torchvision.*", "torch.utils.tensorboard",
        "tensorboard", "tensorboard.*",
        "realesrgan", "realesrgan.*", "basicsr", "basicsr.*",
        "facexlib", "facexlib.*", "gfpgan", "gfpgan.*",
        "filterpy", "filterpy.*", "filterpy.kalman",
        "torch.testing", "torch.testing.*", "torchgen", "torchgen.*",
        "PyQt6.QtQml", "PyQt6.QtQuick", "PyQt6.QtQuick3D", "PyQt6.QtMultimedia",
        "PyQt6.QtWebEngineWidgets", "PyQt6.QtWebEngineCore", "PyQt6.QtPdf",
        "PyQt6.QtDesigner", "PyQt6.QtBluetooth", "PyQt6.QtNfc", "PyQt6.QtPositioning",
        "PyQt6.QtSensors", "PyQt6.QtSerialPort", "PyQt6.QtSql", "PyQt6.QtHelp", "PyQt6.QtTest",
        "PIL._avif", "PIL._webp", "PIL.AvifImagePlugin", "PIL.WebPImagePlugin",
        "fiona", "fiona.*",
        "reportlab", "reportlab.*",
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

# Some Qt6 DLLs are still copied as transitive plugin dependencies even
# though the corresponding PyQt6.Qt* submodule is excluded above (they are
# binary data, not a hidden import). This app never uses hardware/software
# OpenGL rendering or PDF viewing, so drop them explicitly.
_DROP_BINARY_NAME_SUBSTRINGS = ("opengl32sw", "qt6pdf", "d3dcompiler_47")
analysis.binaries = [
    b for b in analysis.binaries
    if not any(s in b[0].lower() for s in _DROP_BINARY_NAME_SUBSTRINGS)
]

pyz = PYZ(analysis.pure, analysis.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    analysis.scripts,
    [],
    exclude_binaries=True,
    name="TCS",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(project_root / "assets" / "logos" / "tcs_icon.ico"),
    version=str(project_root / "build" / "version_info.txt"),
    manifest=str(project_root / "build" / "app.manifest"),
    uac_admin=False,
)

coll = COLLECT(
    exe,
    analysis.binaries,
    analysis.zipfiles,
    analysis.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="TCS",
)
