# TCS - Tree Counting Sawit

Aplikasi desktop (Windows) untuk membuka data raster/vector sekaligus menghitung otomatis jumlah pohon sawit dari citra, oleh PT Terramitra Citra Persada. Dibangun di atas viewer Terra View dengan tambahan modul **Analysis TCS**.

Fitur viewer: Open Raster, Open Vector, Open DEM (raster elevation), Open ECW (via GDAL runtime jika tersedia), Export Raster, Export Vector, Reset / Data Extents, Zoom Box, Window Link, Metadata / Properties (menu View), MDI window (minimize/maximize/close), status bar Lat/Lon, dan popup promosi PT Terramitra Citra Persada.

Fitur analysis (menu **Analysis** / grup toolbar **Analysis**):
- **TCS Panel** - deteksi/hitung pohon sawit otomatis dari raster (pilih raster, pilih model, Run Counting). AOI full extent / bounding box / boundary blok, threshold confidence & IoU, tiling, dan penyimpanan hasil sebagai layer vector (GeoJSON/Shapefile).
- **Clear All** - bersihkan semua layer dan jendela.

Model deteksi memakai YOLO (`models/*.pt`) bila `ultralytics` + `torch` terpasang; tanpa keduanya, TCS tetap berjalan penuh memakai placeholder detector bawaan.

## Setup environment (sekali saja)

```bat
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt pyinstaller
```

Untuk memakai model YOLO terlatih: `.venv\Scripts\python -m pip install ultralytics torch`.

## Menjalankan dan build

- Jalankan dari source: `run_app.bat`
- Build EXE: `REBUILD_EXE.bat` -> `dist\TCS\TCS.exe`
- Build EXE + Installer: `BUILD_INSTALLER.bat` -> `release\TCS_Setup.exe`
- Cek ukuran build: `CHECK_BUILD_SIZE.bat`
- Buat portable ZIP: `MAKE_PORTABLE_ZIP_SMALL.bat`

## ECW / GDAL

Fitur raster non-ECW (GeoTIFF/JPG/PNG/IMG, termasuk DEM) selalu berjalan tanpa GDAL runtime tambahan. ECW butuh GDAL runtime dengan driver ECW - aplikasi mencari `vendor/osgeo4w` -> folder yang dipilih user -> `%LOCALAPPDATA%\TerraView\gdal_runtime` -> `C:\OSGeo4W` -> PATH sistem. Jika tidak ditemukan, muncul dialog **Select GDAL Runtime Folder** / **Open Installation Guide**, bukan crash. Cek runtime: `CHECK_GDAL_ECW_RUNTIME.bat`.

## Vector

SHP/GeoJSON/KML/KMZ dibaca dan ditulis tanpa GDAL (pyshp + parser bawaan). DXF butuh GDAL runtime tambahan (driver DXF) - belum didukung di core app.

## Script

| Script | Kegunaan |
| --- | --- |
| `CHECK_GDAL_ECW_RUNTIME.bat` | Cek gdal_translate ditemukan + driver ECW tersedia atau tidak |
| `CHECK_BUILD_SIZE.bat` | Ukuran folder dist + installer, 30 file terbesar, PASS/FAIL <100MB |
| `MAKE_PORTABLE_ZIP_SMALL.bat` | Zip `dist\TCS` ke `release\TCS_Portable.zip` |
| `installer\build_installer.bat` | Build installer (Inno Setup 6) |

## Distribusi

Distribusi portable **harus** mengirim folder lengkap `dist\TCS\` (bukan hanya file `.exe`-nya saja), karena runtime Python/Qt/GDAL ada di folder `_internal` di sebelahnya.
