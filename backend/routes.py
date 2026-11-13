"""Web + API routes.

Routes never touch sklearn directly — they delegate to backend/services/prediction_service.py.
Handles web UI rendering, RESTful JSON APIs, database status checks, and model retraining.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from flask import Blueprint, jsonify, render_template, request

from backend.services import prediction_service as svc
from backend.validation import ValidationError

ROOT = Path(__file__).resolve().parents[1]

web = Blueprint("web", __name__)
api = Blueprint("api", __name__, url_prefix="/api")


def _form_or_json() -> dict:
    if request.is_json:
        return dict(request.get_json(silent=True) or {})
    return dict(request.form)


@web.route("/")
def home():
    db_status = svc.get_db_diagnostics()
    return render_template("home.html", db_status=db_status)


@web.route("/predict", methods=["GET", "POST"])
def predict():
    """GET returns the form; POST runs the stage prediction and shows result page."""
    db_status = svc.get_db_diagnostics()
    if request.method == "GET":
        return render_template("predict.html", active_tab="stage", db_status=db_status)
    try:
        result = svc.predict_stage(_form_or_json())
    except ValidationError as exc:
        return render_template(
            "predict.html",
            active_tab="stage",
            error=str(exc),
            db_status=db_status
        ), 400
    return render_template("result.html", result=result, kind="stage", db_status=db_status)


@web.route("/risk-check", methods=["GET", "POST"])
def risk_check():
    db_status = svc.get_db_diagnostics()
    if request.method == "GET":
        return render_template("predict.html", active_tab="risk", db_status=db_status)
    try:
        result = svc.predict_risk(_form_or_json())
    except ValidationError as exc:
        return render_template(
            "predict.html",
            active_tab="risk",
            error=str(exc),
            db_status=db_status
        ), 400
    return render_template("result.html", result=result, kind="risk", db_status=db_status)


@web.route("/models")
def models_page():
    stage_metrics = svc.get_stage_metrics()
    risk_metrics = svc.get_risk_metrics()
    db_status = svc.get_db_diagnostics()
    return render_template(
        "models.html",
        stage_metrics=stage_metrics,
        risk_metrics=risk_metrics,
        db_status=db_status
    )


@web.route("/models/retrain", methods=["POST"])
def retrain():
    """Re-run training scripts and reload model artifacts."""
    out = []
    for script in ("ml/train.py", "ml/train_risk.py"):
        proc = subprocess.run(
            [sys.executable, str(ROOT / script)],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=600,
        )
        out.append(f"$ python {script}\n{proc.stdout}\n{proc.stderr}")
        if proc.returncode != 0:
            return (
                render_template(
                    "models.html",
                    stage_metrics=svc._stage["metrics"],
                    risk_metrics=svc._risk["metrics"],
                    error=f"Retraining failed in {script}:\n{proc.stderr}",
                    db_status=svc.get_db_diagnostics(),
                ),
                500,
            )

    # Invalidate cache so next request loads fresh artifacts
    svc._stage["model"] = None
    svc._risk["model"] = None
    _, _, stage_metrics = svc.load_stage()
    _, _, risk_metrics = svc.load_risk()

    return render_template(
        "models.html",
        stage_metrics=stage_metrics,
        risk_metrics=risk_metrics,
        message="Retraining complete. Models successfully reloaded.",
        retrain_log="\n".join(out),
        db_status=svc.get_db_diagnostics(),
    )


@web.route("/history")
def history():
    entries = svc.get_history(limit=100)
    db_status = svc.get_db_diagnostics()
    return render_template("history.html", entries=entries, db_status=db_status)


# ---------------------------------------------------------------------------
# JSON REST API
# ---------------------------------------------------------------------------

@api.route("/predict", methods=["POST"])
def api_predict():
    """Run stage prediction via JSON API."""
    try:
        result = svc.predict_stage(_form_or_json())
    except ValidationError as exc:
        return jsonify({"success": False, "error": str(exc)}), 400
    except Exception as exc:
        return jsonify({"success": False, "error": f"Internal error: {exc}"}), 500
    return jsonify({
        "success": True,
        **{k: v for k, v in result.items() if k != "inputs"},
        "inputs": result.get("inputs")
    })


@api.route("/risk-check", methods=["POST"])
def api_risk_check():
    """Run cirrhosis risk prediction via JSON API."""
    try:
        result = svc.predict_risk(_form_or_json())
    except ValidationError as exc:
        return jsonify({"success": False, "error": str(exc)}), 400
    except Exception as exc:
        return jsonify({"success": False, "error": f"Internal error: {exc}"}), 500
    return jsonify({
        "success": True,
        **{k: v for k, v in result.items() if k != "inputs"},
        "inputs": result.get("inputs")
    })


@api.route("/models", methods=["GET"])
def api_models():
    """Retrieve current model performance metrics."""
    stage_metrics = svc.get_stage_metrics()
    risk_metrics = svc.get_risk_metrics()
    return jsonify({
        "success": True,
        "stage": stage_metrics,
        "risk": risk_metrics
    })


@api.route("/history", methods=["GET"])
def api_history():
    """Retrieve prediction history from database as JSON."""
    limit = request.args.get("limit", default=50, type=int)
    history_records = svc.get_history(limit=limit)
    return jsonify({
        "success": True,
        "count": len(history_records),
        "history": history_records
    })


@api.route("/db-status", methods=["GET"])
def api_db_status():
    """Retrieve current database connectivity and statistics."""
    status = svc.get_db_diagnostics()
    return jsonify({
        "success": True,
        "database": status
    })


@api.errorhandler(404)
def api_404(_exc):
    return jsonify({"success": False, "error": "Endpoint not found"}), 404
