@echo off
setlocal
cd /d "%~dp0"
echo ==================================================
echo TCS - Check Build Size
echo ==================================================
set PY=.venv\Scripts\python.exe
if not exist "%PY%" (
  echo [FAILED] Virtual environment .venv belum ada. Jalankan: python -m venv .venv
  pause
  exit /b 1
)
"%PY%" check_build_size.py
if errorlevel 1 (
  echo.
  echo [FAILED] Salah satu atau kedua installer >= 100 MB. Lihat rekomendasi di atas.
  pause
  exit /b 1
)
echo.
echo [OK] Semua installer yang ditemukan di bawah 100 MB.
pause
