# =====================================================================
#  Liver Cirrhosis Prediction System - PowerShell Launcher
# =====================================================================

Write-Host "=====================================================================" -ForegroundColor Cyan
Write-Host " Starting Liver Cirrhosis Prediction Full-Stack System..." -ForegroundColor Cyan
Write-Host "=====================================================================" -ForegroundColor Cyan

if (-Not (Test-Path ".venv")) {
    Write-Host "Virtual environment (.venv) not found. Running setup.ps1 first..." -ForegroundColor Yellow
    & ".\setup.ps1"
}

& ".\.venv\Scripts\Activate.ps1"
python run.py
