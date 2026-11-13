"""Database connection and management module.

Supports MySQL (via mysql.connector or pymysql) with automatic table creation,
connection pooling, and a resilient SQLite fallback when MySQL is offline.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logger = logging.getLogger("liver_cirrhosis.database")

# Paths and configuration
BASE_DIR = Path(__file__).resolve().parents[1]
INSTANCE_DIR = BASE_DIR / "backend" / "instance"
SQLITE_DB_PATH = INSTANCE_DIR / "predictions.db"

MYSQL_HOST = os.environ.get("MYSQL_HOST", "localhost")
MYSQL_PORT = int(os.environ.get("MYSQL_PORT", "3306"))
MYSQL_USER = os.environ.get("MYSQL_USER", "root")
MYSQL_PASSWORD = os.environ.get("MYSQL_PASSWORD", "rootpassword")
MYSQL_DATABASE = os.environ.get("MYSQL_DATABASE", "liver_cirrhosis_db")
PREFERRED_ENGINE = os.environ.get("DB_ENGINE", "mysql").lower()

# Internal connection cache & state tracking
_ACTIVE_ENGINE: Optional[str] = None
_MYSQL_LAST_CHECK: float = 0.0
_MYSQL_IS_REACHABLE: Optional[bool] = None
_MYSQL_LAST_ERROR: str = ""
_CHECK_COOLDOWN_SECONDS: float = 10.0  # Avoid reconnect retry storm on every single query


def _get_mysql_driver():
    """Attempt to import mysql.connector or pymysql."""
    try:
        import mysql.connector
        return "mysql.connector", mysql.connector
    except ImportError:
        pass
    try:
        import pymysql
        return "pymysql", pymysql
    except ImportError:
        pass
    return None, None


def _probe_mysql() -> Tuple[bool, str]:
    """Probe MySQL availability with a short timeout and cooldown."""
    global _MYSQL_LAST_CHECK, _MYSQL_IS_REACHABLE, _MYSQL_LAST_ERROR
    now = time.time()
    if _MYSQL_IS_REACHABLE is not None and (now - _MYSQL_LAST_CHECK) < _CHECK_COOLDOWN_SECONDS:
        return _MYSQL_IS_REACHABLE, _MYSQL_LAST_ERROR

    _MYSQL_LAST_CHECK = now
    driver_name, driver = _get_mysql_driver()
    if not driver:
        _MYSQL_IS_REACHABLE = False
        _MYSQL_LAST_ERROR = "Neither 'mysql-connector-python' nor 'pymysql' is installed."
        return _MYSQL_IS_REACHABLE, _MYSQL_LAST_ERROR

    try:
        conn = driver.connect(
            host=MYSQL_HOST,
            port=MYSQL_PORT,
            user=MYSQL_USER,
            password=MYSQL_PASSWORD,
            connection_timeout=2
        )
        conn.close()
        _MYSQL_IS_REACHABLE = True
        _MYSQL_LAST_ERROR = ""
        return True, "Connected successfully"
    except Exception as exc:
        _MYSQL_IS_REACHABLE = False
        _MYSQL_LAST_ERROR = str(exc)
        return False, _MYSQL_LAST_ERROR


def _ensure_mysql_database_exists() -> bool:
    """Create target MySQL database if missing."""
    driver_name, driver = _get_mysql_driver()
    if not driver:
        return False

    try:
        conn = driver.connect(
            host=MYSQL_HOST,
            port=MYSQL_PORT,
            user=MYSQL_USER,
            password=MYSQL_PASSWORD,
            connection_timeout=2
        )
        cursor = conn.cursor()
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS `{MYSQL_DATABASE}` CHARACTER SET utf8mb4;")
        conn.commit()
        cursor.close()
        conn.close()
        return True
    except Exception as exc:
        return False


def get_connection():
    """Return an active connection to MySQL, or SQLite if MySQL is unavailable."""
    global _ACTIVE_ENGINE

    if PREFERRED_ENGINE == "mysql":
        reachable, _ = _probe_mysql()
        if reachable:
            driver_name, driver = _get_mysql_driver()
            try:
                _ensure_mysql_database_exists()
                if driver_name == "mysql.connector":
                    conn = driver.connect(
                        host=MYSQL_HOST,
                        port=MYSQL_PORT,
                        user=MYSQL_USER,
                        password=MYSQL_PASSWORD,
                        database=MYSQL_DATABASE
                    )
                    _ACTIVE_ENGINE = "mysql"
                    return conn
                else:
                    conn = driver.connect(
                        host=MYSQL_HOST,
                        port=MYSQL_PORT,
                        user=MYSQL_USER,
                        password=MYSQL_PASSWORD,
                        database=MYSQL_DATABASE,
                        cursorclass=driver.cursors.DictCursor
                    )
                    _ACTIVE_ENGINE = "mysql"
                    return conn
            except Exception:
                pass

    # Fallback to SQLite
    INSTANCE_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(SQLITE_DB_PATH)
    conn.row_factory = sqlite3.Row
    _ACTIVE_ENGINE = "sqlite"
    return conn


def init_db() -> str:
    """Initialize schema tables in the active database engine."""
    conn = get_connection()
    engine = _ACTIVE_ENGINE

    if engine == "mysql":
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS patients (
                id INT AUTO_INCREMENT PRIMARY KEY,
                patient_code VARCHAR(50) UNIQUE NOT NULL,
                full_name VARCHAR(100) NOT NULL,
                age FLOAT NOT NULL,
                gender VARCHAR(20) NOT NULL,
                contact_phone VARCHAR(50),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            ) ENGINE=InnoDB;
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INT AUTO_INCREMENT PRIMARY KEY,
                patient_id INT NULL,
                patient_name VARCHAR(100) DEFAULT 'Anonymous Patient',
                model_type VARCHAR(50) NOT NULL,
                prediction_result VARCHAR(100) NOT NULL,
                stage_number INT NULL,
                risk_label VARCHAR(10) NULL,
                confidence FLOAT NULL,
                probability_scores JSON NULL,
                inputs JSON NOT NULL,
                doctor_notes TEXT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            ) ENGINE=InnoDB;
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS model_metrics_logs (
                id INT AUTO_INCREMENT PRIMARY KEY,
                model_type VARCHAR(50) NOT NULL,
                algorithm_name VARCHAR(100) NOT NULL,
                accuracy FLOAT NULL,
                precision_macro FLOAT NULL,
                recall_macro FLOAT NULL,
                f1_macro FLOAT NULL,
                cv_mean_f1 FLOAT NULL,
                cv_std_f1 FLOAT NULL,
                recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            ) ENGINE=InnoDB;
        """)
        conn.commit()
        cursor.close()
        conn.close()
        return "mysql"
    else:
        # SQLite
        with conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS patients (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    patient_code TEXT UNIQUE NOT NULL,
                    full_name TEXT NOT NULL,
                    age REAL NOT NULL,
                    gender TEXT NOT NULL,
                    contact_phone TEXT,
                    created_at TEXT NOT NULL
                );
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    patient_id INTEGER,
                    patient_name TEXT DEFAULT 'Anonymous Patient',
                    model_type TEXT NOT NULL,
                    prediction_result TEXT NOT NULL,
                    stage_number INTEGER,
                    risk_label TEXT,
                    confidence REAL,
                    probability_scores TEXT,
                    inputs TEXT NOT NULL,
                    doctor_notes TEXT,
                    created_at TEXT NOT NULL
                );
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS model_metrics_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_type TEXT NOT NULL,
                    algorithm_name TEXT NOT NULL,
                    accuracy REAL,
                    precision_macro REAL,
                    recall_macro REAL,
                    f1_macro REAL,
                    cv_mean_f1 REAL,
                    cv_std_f1 REAL,
                    recorded_at TEXT NOT NULL
                );
            """)
        conn.close()
        return "sqlite"


def save_prediction(
    model_type: str,
    inputs: Dict[str, Any],
    prediction_result: str,
    stage_number: Optional[int] = None,
    risk_label: Optional[str] = None,
    confidence: Optional[float] = None,
    probability_scores: Optional[Dict[Any, float]] = None,
    patient_name: str = "Anonymous Patient",
    doctor_notes: Optional[str] = None
) -> int:
    """Save an inference record into the active database."""
    init_db()
    conn = get_connection()
    engine = _ACTIVE_ENGINE
    created_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

    inputs_json = json.dumps(inputs, default=str)
    probs_json = json.dumps(probability_scores, default=str) if probability_scores else None

    inserted_id = 0

    if engine == "mysql":
        cursor = conn.cursor()
        query = """
            INSERT INTO predictions (
                patient_name, model_type, prediction_result, stage_number,
                risk_label, confidence, probability_scores, inputs, doctor_notes, created_at
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        cursor.execute(query, (
            patient_name, model_type, prediction_result, stage_number,
            risk_label, confidence, probs_json, inputs_json, doctor_notes, created_at
        ))
        conn.commit()
        inserted_id = cursor.lastrowid
        cursor.close()
        conn.close()
    else:
        with conn:
            cursor = conn.cursor()
            query = """
                INSERT INTO predictions (
                    patient_name, model_type, prediction_result, stage_number,
                    risk_label, confidence, probability_scores, inputs, doctor_notes, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
            cursor.execute(query, (
                patient_name, model_type, prediction_result, stage_number,
                risk_label, confidence, probs_json, inputs_json, doctor_notes, created_at
            ))
            inserted_id = cursor.lastrowid
        conn.close()

    return inserted_id


def get_prediction_history(limit: int = 100) -> List[Dict[str, Any]]:
    """Retrieve logged prediction records."""
    init_db()
    conn = get_connection()
    engine = _ACTIVE_ENGINE
    records = []

    try:
        if engine == "mysql":
            cursor = conn.cursor(dictionary=True) if hasattr(conn, "cursor") else conn.cursor()
            query = "SELECT * FROM predictions ORDER BY id DESC LIMIT %s"
            cursor.execute(query, (limit,))
            rows = cursor.fetchall()
            for r in rows:
                row_dict = dict(r)
                if isinstance(row_dict.get("inputs"), str):
                    try:
                        row_dict["inputs"] = json.loads(row_dict["inputs"])
                    except Exception:
                        pass
                if isinstance(row_dict.get("probability_scores"), str):
                    try:
                        row_dict["probability_scores"] = json.loads(row_dict["probability_scores"])
                    except Exception:
                        pass
                records.append(row_dict)
            cursor.close()
        else:
            with conn:
                cursor = conn.cursor()
                rows = cursor.execute("SELECT * FROM predictions ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
                for r in rows:
                    row_dict = dict(r)
                    if isinstance(row_dict.get("inputs"), str):
                        try:
                            row_dict["inputs"] = json.loads(row_dict["inputs"])
                        except Exception:
                            pass
                    records.append(row_dict)
    finally:
        conn.close()

    return records


def log_model_metrics(
    model_type: str,
    algorithm_name: str,
    accuracy: float,
    precision_macro: float,
    recall_macro: float,
    f1_macro: float,
    cv_mean_f1: float,
    cv_std_f1: float
) -> None:
    """Log model performance metrics into the database."""
    init_db()
    conn = get_connection()
    engine = _ACTIVE_ENGINE
    recorded_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

    if engine == "mysql":
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO model_metrics_logs (
                model_type, algorithm_name, accuracy, precision_macro, recall_macro,
                f1_macro, cv_mean_f1, cv_std_f1, recorded_at
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (model_type, algorithm_name, accuracy, precision_macro, recall_macro, f1_macro, cv_mean_f1, cv_std_f1, recorded_at))
        conn.commit()
        cursor.close()
    else:
        with conn:
            conn.execute("""
                INSERT INTO model_metrics_logs (
                    model_type, algorithm_name, accuracy, precision_macro, recall_macro,
                    f1_macro, cv_mean_f1, cv_std_f1, recorded_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (model_type, algorithm_name, accuracy, precision_macro, recall_macro, f1_macro, cv_mean_f1, cv_std_f1, recorded_at))
    conn.close()


def check_db_status() -> Dict[str, Any]:
    """Return diagnostic information about database connectivity."""
    is_mysql_live, mysql_msg = _probe_mysql()
    init_db()

    conn = get_connection()
    engine = _ACTIVE_ENGINE
    total_records = 0
    try:
        if engine == "mysql":
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM predictions;")
            total_records = cursor.fetchone()[0]
            cursor.close()
        else:
            with conn:
                cursor = conn.cursor()
                res = cursor.execute("SELECT COUNT(*) FROM predictions;").fetchone()
                total_records = res[0] if res else 0
    except Exception:
        total_records = 0
    finally:
        conn.close()

    return {
        "active_engine": engine,
        "mysql_server_reachable": is_mysql_live,
        "mysql_host": MYSQL_HOST,
        "mysql_port": MYSQL_PORT,
        "mysql_user": MYSQL_USER,
        "mysql_database": MYSQL_DATABASE,
        "mysql_message": "Connected" if is_mysql_live else mysql_msg,
        "sqlite_path": str(SQLITE_DB_PATH),
        "total_predictions_stored": total_records,
        "status": "connected_mysql" if engine == "mysql" else "active_sqlite_fallback"
    }
