@echo off
setlocal
cd /d "%~dp0.."
echo ==================================================
echo TCS - Build Installer
echo ==================================================
if not exist "dist\TCS\TCS.exe" (
  echo [ERROR] File EXE belum ada.
  echo Jalankan REBUILD_EXE.bat dulu dari folder utama.
  pause
  exit /b 1
)
if exist "C:\Program Files (x86)\Inno Setup 6\ISCC.exe" (
  "C:\Program Files (x86)\Inno Setup 6\ISCC.exe" installer\setup_script.iss
) else if exist "C:\Program Files\Inno Setup 6\ISCC.exe" (
  "C:\Program Files\Inno Setup 6\ISCC.exe" installer\setup_script.iss
) else (
  echo [ERROR] ISCC tidak ditemukan. Install Inno Setup 6.
  pause
  exit /b 1
)
pause
