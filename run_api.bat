@echo off
title Aviation Maintenance Copilot - Backend API
echo ========================================================
echo Starting Aviation Maintenance Copilot - FastAPI Backend
echo Local Web UI : http://127.0.0.1:8000
echo Local API    : http://127.0.0.1:8000/chat
echo Swagger Docs : http://127.0.0.1:8000/docs
echo ========================================================
cd /d "%~dp0"

set "PY_CMD=python"
if exist "%~dp0venv\Scripts\python.exe" (
    set "PY_CMD=%~dp0venv\Scripts\python.exe"
) else if exist "%~dp0.venv\Scripts\python.exe" (
    set "PY_CMD=%~dp0.venv\Scripts\python.exe"
)

%PY_CMD% -m uvicorn app:app --reload --host 127.0.0.1 --port 8000
pause
