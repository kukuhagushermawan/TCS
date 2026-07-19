@echo off
cd /d "%~dp0"
set PY=.venv\Scripts\python.exe
if not exist "%PY%" (
  echo [FAILED] Virtual environment .venv belum ada.
  echo Buat dulu: python -m venv .venv
  echo Lalu install: .venv\Scripts\python -m pip install -r requirements.txt
  pause
  exit /b 1
)
"%PY%" app\main.py
pause
