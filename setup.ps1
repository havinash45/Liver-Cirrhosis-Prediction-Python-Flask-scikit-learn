# =====================================================================
#  Liver Cirrhosis Prediction System - PowerShell Setup Script
# =====================================================================

Write-Host "=====================================================================" -ForegroundColor Cyan
Write-Host " Setting up Liver Cirrhosis Prediction Full-Stack System" -ForegroundColor Cyan
Write-Host "=====================================================================" -ForegroundColor Cyan

# 1. Check Python
try {
    $pyVersion = python --version 2>&1
    Write-Host "Detected Python: $pyVersion" -ForegroundColor Green
} catch {
    Write-Host "[ERROR] Python is not installed or not in PATH." -ForegroundColor Red
    Exit 1
}

# 2. Virtual environment
if (-Not (Test-Path ".venv")) {
    Write-Host "[1/4] Creating virtual environment (.venv)..." -ForegroundColor Yellow
    python -m venv .venv
} else {
    Write-Host "[1/4] Virtual environment (.venv) already exists." -ForegroundColor Green
}

# 3. Activate venv
Write-Host "[2/4] Activating virtual environment..." -ForegroundColor Yellow
& ".\.venv\Scripts\Activate.ps1"

# 4. Install dependencies
Write-Host "[3/4] Installing requirements..." -ForegroundColor Yellow
python -m pip install --upgrade pip
pip install -r requirements.txt

# 5. Initialize database
Write-Host "[4/4] Initializing Database..." -ForegroundColor Yellow
if (-Not (Test-Path ".env")) {
    if (Test-Path ".env.example") {
        Copy-Item ".env.example" ".env"
        Write-Host "Created .env configuration file." -ForegroundColor Green
    }
}
python setup_database.py

Write-Host "=====================================================================" -ForegroundColor Cyan
Write-Host " SETUP COMPLETED SUCCESSFULLY! Run: .\run.ps1" -ForegroundColor Green
Write-Host "=====================================================================" -ForegroundColor Cyan
