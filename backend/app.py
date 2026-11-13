"""Flask application factory and server entry point.

Run: python backend/app.py (or python run.py)
Configured to serve frontend templates and static assets, with REST APIs
and MySQL/SQLite database integration.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from flask import Flask, jsonify  # noqa: E402

from backend import config  # noqa: E402
from backend.routes import api, web  # noqa: E402
from backend.validation import ValidationError  # noqa: E402
from database.db import check_db_status, init_db  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("liver_cirrhosis.backend")


def create_app() -> Flask:
    app = Flask(
        __name__,
        template_folder=str(ROOT / "frontend" / "templates"),
        static_folder=str(ROOT / "frontend" / "static"),
    )
    app.config["SECRET_KEY"] = "liver-cirrhosis-secret-key-batch-14"

    # Initialize database tables
    try:
        active_db = init_db()
        logger.info(f"Database initialized successfully with engine: {active_db.upper()}")
    except Exception as exc:
        logger.warning(f"Database initialization warning: {exc}")

    # Register blueprints
    app.register_blueprint(web)
    app.register_blueprint(api)

    # Enable CORS headers for API blueprint
    @app.after_request
    def add_cors_headers(response):
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
        return response

    @app.errorhandler(ValidationError)
    def handle_validation(exc: ValidationError):
        return jsonify({"success": False, "error": str(exc)}), 400

    @app.errorhandler(404)
    def handle_404(_exc):
        return jsonify({"success": False, "error": "Not found"}), 404

    @app.errorhandler(500)
    def handle_500(exc):
        logger.error(f"Internal server error: {exc}", exc_info=True)
        return jsonify({"success": False, "error": "Internal server error"}), 500

    return app


app = create_app()

if __name__ == "__main__":
    db_info = check_db_status()
    print("=" * 65)
    print("  Liver Cirrhosis Prediction Full-Stack System")
    print(f"  Web Interface : http://{config.HOST}:{config.PORT}")
    print(f"  REST API      : http://{config.HOST}:{config.PORT}/api")
    print(f"  Database      : {db_info['active_engine'].upper()} "
          f"({db_info['mysql_database'] if db_info['active_engine'] == 'mysql' else db_info['sqlite_path']})")
    print("=" * 65)
    app.run(host=config.HOST, port=config.PORT, debug=config.DEBUG)
