@echo off
setlocal
cd /d "%~dp0"
echo ==================================================
echo TCS - Check GDAL / ECW Runtime
echo ==================================================
set PY=.venv\Scripts\python.exe
if not exist "%PY%" (
  echo [FAILED] Virtual environment .venv belum ada. Jalankan: python -m venv .venv
  pause
  exit /b 1
)
"%PY%" -c "import subprocess; from app.format_converter import find_gdal_translate; exe = find_gdal_translate(); print('[OK] gdal_translate ditemukan: ' + str(exe)) if exe else print('[GAGAL] gdal_translate tidak ditemukan.'); info = (exe.parent / 'gdalinfo.exe') if exe else None; formats = subprocess.run([str(info), '--formats'], capture_output=True, text=True).stdout if info and info.exists() else ''; has_ecw = 'ECW' in formats.upper(); print('[OK] Driver ECW tersedia pada GDAL ini.' if has_ecw else '[INFO] Driver ECW TIDAK tersedia pada GDAL yang ditemukan. Raster biasa tetap bisa dibuka; ECW butuh OSGeo4W/QGIS dengan package gdal-ecw, atau bundel runtime ke vendor/osgeo4w.'); raise SystemExit(0 if exe else 1)"
if errorlevel 1 (
  echo.
  echo [FAILED] Tidak ada gdal_translate.exe ditemukan sama sekali.
  echo Install OSGeo4W/QGIS, atau bundel GDAL runtime ke vendor\osgeo4w\bin sebelum build.
  pause
  exit /b 1
)
echo.
echo [OK] Cek GDAL/ECW selesai. Raster GeoTIFF/JPG/PNG tetap bisa dibuka walau ECW driver tidak ada.
pause
