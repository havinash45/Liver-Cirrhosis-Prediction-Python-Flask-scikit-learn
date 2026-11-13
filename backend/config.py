"""Configuration — one place for every setting the user may change."""

from __future__ import annotations

import os
from pathlib import Path

# Load .env file if available
BASE_DIR = Path(__file__).resolve().parents[1]      # project root
BACKEND_DIR = Path(__file__).resolve().parent

env_file = BASE_DIR / ".env"
if env_file.exists():
    with open(env_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

# Server settings
PORT = int(os.environ.get("PORT", "8080"))
HOST = os.environ.get("HOST", "127.0.0.1")
DEBUG = os.environ.get("FLASK_DEBUG", "0") == "1"

# MySQL Database configuration (with defaults for quick setup)
MYSQL_HOST = os.environ.get("MYSQL_HOST", "localhost")
MYSQL_PORT = int(os.environ.get("MYSQL_PORT", "3306"))
MYSQL_USER = os.environ.get("MYSQL_USER", "root")
MYSQL_PASSWORD = os.environ.get("MYSQL_PASSWORD", "rootpassword")
MYSQL_DATABASE = os.environ.get("MYSQL_DATABASE", "liver_cirrhosis_db")
DB_ENGINE = os.environ.get("DB_ENGINE", "mysql").lower()

# Trained artifacts (built by ml/train.py and ml/train_risk.py)
MODELS_DIR = BASE_DIR / "ml" / "models"
STAGE_MODEL_PATH = MODELS_DIR / "model.pkl"
STAGE_SCALER_PATH = MODELS_DIR / "scaler.bin"
STAGE_METRICS_PATH = MODELS_DIR / "metrics.json"
RISK_MODEL_PATH = MODELS_DIR / "risk_model.pkl"
RISK_SCALER_PATH = MODELS_DIR / "risk_scaler.bin"
RISK_METRICS_PATH = MODELS_DIR / "risk_metrics.json"

# SQLite history log (used as local fallback)
INSTANCE_DIR = BACKEND_DIR / "instance"
DB_PATH = INSTANCE_DIR / "predictions.db"
