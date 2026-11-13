@echo off
REM =====================================================================
REM  Liver Cirrhosis Prediction System - Application Launcher
REM =====================================================================

title Liver Cirrhosis Prediction App
echo =====================================================================
echo  Starting Liver Cirrhosis Prediction Full-Stack System...
echo =====================================================================

REM If .venv doesn't exist, run setup first
if not exist ".venv" (
    echo Virtual environment (.venv) not found. Running setup.bat first...
    call setup.bat
)

REM Activate virtual environment and start app
echo Activating .venv...
call .venv\Scripts\activate.bat

echo Starting server...
python run.py

pause
