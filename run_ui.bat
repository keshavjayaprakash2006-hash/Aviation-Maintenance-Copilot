@echo off
title Aviation Maintenance Copilot - Streamlit UI
echo ========================================================
echo Starting Aviation Maintenance Copilot - Streamlit UI
echo App URL: http://localhost:8501
echo ========================================================
cd /d "%~dp0"

set "PY_CMD=python"
if exist "%~dp0venv\Scripts\python.exe" (
    set "PY_CMD=%~dp0venv\Scripts\python.exe"
) else if exist "%~dp0.venv\Scripts\python.exe" (
    set "PY_CMD=%~dp0.venv\Scripts\python.exe"
)

%PY_CMD% -m streamlit run ui.py
pause
