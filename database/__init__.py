"""Database package initialization."""

from database.db import (
    check_db_status,
    get_connection,
    get_prediction_history,
    init_db,
    log_model_metrics,
    save_prediction,
)

__all__ = [
    "get_connection",
    "init_db",
    "save_prediction",
    "get_prediction_history",
    "log_model_metrics",
    "check_db_status",
]
