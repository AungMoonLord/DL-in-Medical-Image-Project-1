@echo off
title Thai Character Classifier WebUI - Solution 3
echo ============================================================
echo   Thai Character Classifier Web Application
echo   Solution 3 (Pai + Pooh) - Universal Inference System
echo ============================================================
echo.

set PYTHON_CMD=.\ver1\.venv\Scripts\python.exe

if not exist "%PYTHON_CMD%" (
    echo [ERROR] Virtual environment Python not found at %PYTHON_CMD%
    pause
    exit /b 1
)

echo Starting Flask Inference Web Server on http://127.0.0.1:5000 ...
"%PYTHON_CMD%" web\app.py
pause
