"""Master execution script for the Liver Cirrhosis Prediction Full-Stack System.

Checks environment dependencies, initializes database connection, and launches
the Flask application server.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Set root directory
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from backend import config
from backend.app import create_app
from database.db import check_db_status, init_db


def check_models():
    """Verify that required model artifacts exist."""
    missing = []
    if not config.STAGE_MODEL_PATH.exists():
        missing.append(str(config.STAGE_MODEL_PATH))
    if not config.RISK_MODEL_PATH.exists():
        missing.append(str(config.RISK_MODEL_PATH))
    if missing:
        print("[WARNING] The following trained model files were not found:")
        for m in missing:
            print(f"  - {m}")
        print("Run training scripts first:")
        print("  python ml/train.py")
        print("  python ml/train_risk.py\n")


def print_banner(db_info):
    banner = f"""
======================================================================
  LIVER CIRRHOSIS PREDICTION SYSTEM - FULL-STACK PLATFORM
======================================================================
  * Framework    : Python Flask + Scikit-Learn
  * Web Server   : Running on http://{config.HOST}:{config.PORT}
  * Database     : {db_info['active_engine'].upper()}
  * DB Status    : {'MySQL Server LIVE' if db_info['mysql_server_reachable'] else 'SQLite Fallback Active'}
  * Database Name: {db_info['mysql_database'] if db_info['active_engine'] == 'mysql' else 'predictions.db'}
  * REST API     :
      - POST http://{config.HOST}:{config.PORT}/api/predict
      - POST http://{config.HOST}:{config.PORT}/api/risk-check
      - GET  http://{config.HOST}:{config.PORT}/api/models
      - GET  http://{config.HOST}:{config.PORT}/api/history
      - GET  http://{config.HOST}:{config.PORT}/api/db-status
======================================================================
 Press CTRL+C to gracefully stop the server.
"""
    print(banner)


def main():
    check_models()
    try:
        init_db()
    except Exception as exc:
        print(f"[Database Initialization Warning] {exc}")

    db_info = check_db_status()
    print_banner(db_info)

    app = create_app()
    app.run(host=config.HOST, port=config.PORT, debug=config.DEBUG)


if __name__ == "__main__":
    main()
