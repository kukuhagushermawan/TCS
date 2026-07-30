@echo off
setlocal
cd /d "%~dp0"
echo ==================================================
echo TCS - Build EXE
echo ==================================================
echo.

set PY=.venv\Scripts\python.exe
if not exist "%PY%" (
  echo [FAILED] Virtual environment .venv belum ada.
  echo Buat dulu: python -m venv .venv
  echo Lalu install: .venv\Scripts\python -m pip install -r requirements.txt pyinstaller
  pause
  exit /b 1
)

if not exist "build\TCS.spec" (
  echo [FAILED] File build\TCS.spec tidak ditemukan.
  echo Jangan hapus folder build seluruhnya karena file .spec ada di dalamnya.
  echo Extract ulang ZIP final jika file ini hilang.
  pause
  exit /b 1
)

echo [CLEAN] Menghapus hasil build lama tanpa menghapus build\TCS.spec...
if exist "dist\TCS" rmdir /s /q "dist\TCS"
if exist "build\TCS" rmdir /s /q "build\TCS"

"%PY%" -m PyInstaller --noconfirm --clean build\TCS.spec
if errorlevel 1 (
  echo.
  echo [FAILED] Build EXE gagal.
  pause
  exit /b 1
)

echo.
echo [VENDOR] Copy GDAL/ECW runtime ke dist\TCS\_internal\vendor ...
if exist "dist\TCS\_internal" (
  xcopy /E /I /Y /Q "vendor" "dist\TCS\_internal\vendor" >nul
  echo [OK] Runtime GDAL/ECW tersalin ke _internal\vendor.
) else (
  echo [WARN] dist\TCS\_internal tidak ditemukan; vendor tidak dicopy.
)

echo.
echo [OK] EXE: dist\TCS\TCS.exe
pause
