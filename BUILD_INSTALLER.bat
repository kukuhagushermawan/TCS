@echo off
setlocal
cd /d "%~dp0"
echo ==================================================
echo TCS - Build EXE + Installer
echo ==================================================

set PY=.venv\Scripts\python.exe
if not exist "%PY%" (
  echo [FAILED] Virtual environment .venv belum ada. Jalankan REBUILD_EXE.bat dulu.
  pause
  exit /b 1
)

"%PY%" -m PyInstaller --noconfirm --clean build\TCS.spec
if errorlevel 1 (
  echo [FAILED] Build EXE gagal.
  pause
  exit /b 1
)
if not exist "dist\TCS\TCS.exe" (
  echo [ERROR] EXE tidak ditemukan.
  pause
  exit /b 1
)

echo [VENDOR] Copy GDAL/ECW runtime ke dist\TCS\_internal\vendor ...
xcopy /E /I /Y /Q "vendor" "dist\TCS\_internal\vendor" >nul
echo [OK] Runtime GDAL/ECW tersalin ke _internal\vendor.

if exist "C:\Program Files (x86)\Inno Setup 6\ISCC.exe" (
  "C:\Program Files (x86)\Inno Setup 6\ISCC.exe" installer\setup_script.iss
) else if exist "C:\Program Files\Inno Setup 6\ISCC.exe" (
  "C:\Program Files\Inno Setup 6\ISCC.exe" installer\setup_script.iss
) else (
  echo [WARNING] Inno Setup 6 tidak ditemukan. EXE sudah dibuat, installer belum dibuat.
  echo Install Inno Setup 6 lalu jalankan installer\build_installer.bat
)
echo [OK] Build selesai.
pause
