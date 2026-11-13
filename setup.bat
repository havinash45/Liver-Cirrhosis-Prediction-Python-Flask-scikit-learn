@echo off
REM =====================================================================
REM  Liver Cirrhosis Prediction System - Automated Environment Setup
REM =====================================================================

title Liver Cirrhosis Prediction - Setup
echo =====================================================================
echo  Setting up Liver Cirrhosis Prediction Full-Stack System
echo =====================================================================

REM 1. Check Python installation
python --version >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Python is not installed or not in PATH.
    echo Please install Python 3.10+ from https://www.python.org/
    pause
    exit /b 1
)

REM 2. Create Virtual Environment
if not exist ".venv" (
    echo [1/4] Creating virtual environment (.venv)...
    python -m venv .venv
    if %ERRORLEVEL% neq 0 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
) else (
    echo [1/4] Virtual environment (.venv) already exists.
)

REM 3. Activate Virtual Environment
echo [2/4] Activating virtual environment...
call .venv\Scripts\activate.bat

REM 4. Install Dependencies
echo [3/4] Installing project requirements...
python -m pip install --upgrade pip
pip install -r requirements.txt
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Failed to install requirements.
    pause
    exit /b 1
)

REM 5. Initialize Database (MySQL with SQLite Fallback)
echo [4/4] Initializing Database (MySQL / SQLite)...
if not exist ".env" (
    if exist ".env.example" (
        copy .env.example .env >nul
        echo Created .env file from .env.example
    )
)
python setup_database.py

echo.
echo =====================================================================
echo  SETUP COMPLETED SUCCESSFULLY!
echo =====================================================================
echo  To start the application server, run:
echo      run.bat
echo =====================================================================
pause
