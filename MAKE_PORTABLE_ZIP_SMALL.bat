@echo off
setlocal
cd /d "%~dp0"
echo ==================================================
echo TCS - Make Portable ZIP
echo ==================================================
if not exist "release" mkdir "release"

rem Uses the built-in Windows tar.exe (bsdtar) to zip. It is much faster than
rem PowerShell Compress-Archive on folders with thousands of small files.

if not exist "dist\TCS\TCS.exe" (
  echo [FAILED] dist\TCS\TCS.exe tidak ditemukan. Jalankan REBUILD_EXE.bat dulu.
  pause
  exit /b 1
)

echo [ZIP] dist\TCS -^> release\TCS_Portable.zip
if exist "release\TCS_Portable.zip" del /f /q "release\TCS_Portable.zip"
tar.exe -a -c -f "release\TCS_Portable.zip" -C dist TCS
if errorlevel 1 (
  echo [FAILED] Gagal membuat TCS_Portable.zip.
  pause
  exit /b 1
)

echo.
echo [OK] release\TCS_Portable.zip
pause
