@echo off
title Thai Character Classification Web App (Solution 3)
echo =====================================================================
echo  Thai Character Classification Web App - Solution 3 (Pai + Pooh)
echo =====================================================================
echo.
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment not found. Please create it first.
    pause
    exit /b 1
)

echo Starting Flask server on http://127.0.0.1:5000 ...
start http://127.0.0.1:5000
.\.venv\Scripts\python.exe web\app.py
pause
