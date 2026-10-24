@echo off
chcp 65001 >nul
setlocal
rem Set this path to the Python environment with torch and timm installed.
set "GRAIN_UI_PYTHON=D:\ANACONDA\envs\grain-defect-py311\python.exe"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0start_ui.ps1"
if errorlevel 1 pause
endlocal
