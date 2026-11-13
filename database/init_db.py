"""Standalone database initialization and test utility."""

import sys
from pathlib import Path

# Ensure project root is in sys.path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from database.db import check_db_status, init_db, log_model_metrics


def main():
    print("=" * 65)
    print(" Liver Cirrhosis Prediction System - Database Initializer")
    print("=" * 65)

    print("\n[1/3] Testing database connection and creating tables...")
    active_engine = init_db()
    print(f"      Active Database Engine: {active_engine.upper()}")

    print("\n[2/3] Checking system database diagnostics...")
    status = check_db_status()
    print(f"      Active Engine       : {status['active_engine']}")
    print(f"      MySQL Reachable     : {status['mysql_server_reachable']}")
    print(f"      MySQL Host/Port     : {status['mysql_host']}:{status['mysql_port']}")
    print(f"      MySQL User/DB       : {status['mysql_user']} / {status['mysql_database']}")
    print(f"      Diagnostic Message  : {status['mysql_message']}")
    print(f"      Total Past Records  : {status['total_predictions_stored']}")

    if status["active_engine"] == "mysql":
        print("\n>>> SUCCESS: Connected to MySQL database successfully!")
        print("    All tables ('patients', 'predictions', 'model_metrics_logs') are verified.")
    else:
        print("\n>>> NOTE: Currently using SQLite fallback engine.")
        print(f"    SQLite File: {status['sqlite_path']}")
        print("    To switch to MySQL:")
        print("    1. Ensure your MySQL server service is started.")
        print("    2. Verify credentials in .env file (MYSQL_USER, MYSQL_PASSWORD).")
        print("    3. Re-run: python setup_database.py")

    print("=" * 65)


if __name__ == "__main__":
    main()
