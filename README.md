# TCS - Tree Counting Sawit

Aplikasi desktop (Windows) untuk membuka data raster/vector sekaligus menghitung otomatis jumlah pohon sawit dari citra, oleh PT Terramitra Citra Persada. Dibangun di atas viewer Terra View dengan tambahan modul **Analysis TCS**.

Fitur viewer: Open Raster, Open Vector, Open DEM (raster elevation), Open ECW (via GDAL runtime jika tersedia), Export Raster, Export Vector, Reset / Data Extents, Zoom Box, Window Link, Metadata / Properties (menu View), MDI window (minimize/maximize/close), status bar Lat/Lon, dan popup promosi PT Terramitra Citra Persada.

Fitur analysis (menu **Analysis** / grup toolbar **Analysis**):
- **TCS Panel** - deteksi/hitung pohon sawit otomatis dari raster (pilih raster, atur tingkat keyakinan, Run Counting). Selalu memproses seluruh raster (full extent); hasil berupa layer vector titik (satu titik per pohon) yang bisa disimpan/diexport (GeoJSON/Shapefile).
- **Cari Lahan Kosong...** - langkah opsional setelah counting selesai (tidak wajib, tidak mengubah alur counting biasa). Menghitung estimasi jarak tanam dari titik pohon yang terdeteksi, lalu menandai koordinat lahan kosong yang bisa ditanami pada grid segitiga ("mata lima") - bisa memakai polygon batas lahan yang dipilih, atau otomatis pakai convex hull dari titik pohon kalau tidak ada. Hasilnya juga layer vector titik yang bisa diekspor.
- **Clear All** - bersihkan semua layer dan jendela.

Model deteksi bawaan (`models/*.onnx`) berjalan lewat `onnxruntime` (CPU-only, ringan) - selalu tersedia begitu dependency ter-install, tanpa perlu setup tambahan. Jika model tidak ditemukan, TCS tetap berjalan memakai placeholder detector bawaan.

## Menjalankan dari source (repo ini)

```bat
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
run_app.bat
```

Ini semua yang dibutuhkan untuk menjalankan TCS dari source - tidak perlu install Python/GDAL/OSGeo4W tambahan, tidak perlu `pyinstaller`, dan tidak perlu `ultralytics`/`torch` (modul deteksi TCS memakai `onnxruntime`, sudah termasuk di `requirements.txt`).

## ECW / GDAL

Fitur raster non-ECW (GeoTIFF/JPG/PNG/IMG, termasuk DEM) selalu berjalan tanpa GDAL runtime tambahan. ECW butuh GDAL runtime dengan driver ECW - aplikasi mencari `vendor/osgeo4w` -> folder yang dipilih user -> `%LOCALAPPDATA%\TerraView\gdal_runtime` -> `C:\OSGeo4W` -> PATH sistem. Repo ini tidak menyertakan `vendor/osgeo4w` (ukurannya besar); tanpa itu, ECW butuh OSGeo4W/QGIS terinstall terpisah di komputer. Jika tidak ditemukan, muncul dialog **Select GDAL Runtime Folder** / **Open Installation Guide**, bukan crash.

## Vector

SHP/GeoJSON/KML/KMZ dibaca dan ditulis tanpa GDAL (pyshp + parser bawaan). DXF butuh GDAL runtime tambahan (driver DXF) - belum didukung di core app.

## Build EXE/installer (developer, tidak termasuk repo ini)

Skrip build (`REBUILD_EXE.bat`, `BUILD_INSTALLER.bat`, `CHECK_BUILD_SIZE.bat`, `CHECK_GDAL_ECW_RUNTIME.bat`, `MAKE_PORTABLE_ZIP_SMALL.bat`) beserta folder `build/`, `installer/`, `legal/`, `vendor/`, dan `environment.yml` **sengaja tidak ada di repo ini** - repo ini fokus untuk jalan dari source saja. File-file itu ada di copy kerja developer untuk membuat installer/EXE mandiri (PyInstaller + Inno Setup, dengan GDAL/ECW runtime dibundel dari OSGeo4W). Kalau butuh installer siap pakai, minta ke developer secara terpisah - bukan lewat clone repo ini.
