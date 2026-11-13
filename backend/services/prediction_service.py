"""Prediction service — the dedicated Machine Learning serving layer.

Routes never import sklearn/joblib directly; they call the functions here.
Loads the trained artifacts lazily, validates through backend/validation.py,
encodes through ml/predict.py (single source of truth), performs model inference,
and logs every prediction to the database (MySQL with SQLite fallback).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Add ml and database paths
ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR / "ml"))
sys.path.insert(0, str(ROOT_DIR))

from predict import (RISK_FEATURE_KEYS, STAGE_FEATURES, STAGE_LABELS,  # noqa: E402
                     STAGE_SHORT, encode_risk_row, encode_stage_row)

from backend import config  # noqa: E402
from backend.validation import ValidationError, validate_risk, validate_stage  # noqa: E402
from database.db import check_db_status, get_prediction_history, save_prediction  # noqa: E402

_stage = {"model": None, "scaler": None, "metrics": None}
_risk = {"model": None, "scaler": None, "metrics": None}


def _load_metrics(path: Path) -> dict | None:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return None


def load_stage():
    """Load the multiclass staging model (Mayo Clinic PBC dataset)."""
    if _stage["model"] is None:
        try:
            import joblib
        except ImportError:
            raise ImportError(
                "Package 'joblib' is required for model inference. "
                "Please run: pip install -r requirements.txt"
            )

        if not config.STAGE_MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Stage model not found at {config.STAGE_MODEL_PATH}. "
                "Run: python ml/train.py"
            )
        _stage["model"] = joblib.load(config.STAGE_MODEL_PATH)
        _stage["scaler"] = joblib.load(config.STAGE_SCALER_PATH)
        _stage["metrics"] = _load_metrics(config.STAGE_METRICS_PATH)
    return _stage["model"], _stage["scaler"], _stage["metrics"]


def load_risk():
    """Load the binary cirrhosis risk check model (HealthCareData dataset)."""
    if _risk["model"] is None:
        try:
            import joblib
        except ImportError:
            raise ImportError(
                "Package 'joblib' is required for model inference. "
                "Please run: pip install -r requirements.txt"
            )

        if not config.RISK_MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Risk model not found at {config.RISK_MODEL_PATH}. "
                "Run: python ml/train_risk.py"
            )
        _risk["model"] = joblib.load(config.RISK_MODEL_PATH)
        _risk["scaler"] = joblib.load(config.RISK_SCALER_PATH)
        _risk["metrics"] = _load_metrics(config.RISK_METRICS_PATH)
    return _risk["model"], _risk["scaler"], _risk["metrics"]


def stage_model_name() -> str:
    metrics = _load_metrics(config.STAGE_METRICS_PATH)
    return (metrics or {}).get("winner", "SVM (Support Vector Machine)")


def risk_model_name() -> str:
    metrics = _load_metrics(config.RISK_METRICS_PATH)
    return (metrics or {}).get("winner", "Random Forest Classifier")


def get_stage_metrics() -> dict | None:
    return _load_metrics(config.STAGE_METRICS_PATH)


def get_risk_metrics() -> dict | None:
    return _load_metrics(config.RISK_METRICS_PATH)


def _probabilities(model, X_scaled) -> dict[int, float] | None:
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_scaled)[0]
        return {int(c): round(float(p), 4) for c, p in zip(model.classes_, probs)}
    return None


def get_history(limit: int = 100) -> list[dict]:
    """Retrieve history from the database (MySQL with SQLite fallback)."""
    records = get_prediction_history(limit=limit)
    formatted = []
    for r in records:
        formatted.append({
            "id": r.get("id"),
            "created_at": r.get("created_at"),
            "patient_name": r.get("patient_name", "Anonymous Patient"),
            "model": r.get("model_type"),
            "result": r.get("prediction_result"),
            "probability": r.get("confidence"),
            "inputs": r.get("inputs"),
            "notes": r.get("doctor_notes", "")
        })
    return formatted


def get_db_diagnostics() -> dict:
    """Return database engine status for UI display."""
    return check_db_status()


def predict_stage(raw: dict) -> dict:
    """Validate -> encode -> scale -> predict Stage 1-4, then save to DB."""
    patient_name = raw.get("patient_name", "").strip() or "Anonymous Patient"
    clean = validate_stage(raw)
    try:
        import pandas as pd
    except ImportError:
        raise ImportError(
            "Package 'pandas' is required for model inference. "
            "Please run: pip install -r requirements.txt"
        )
    model, scaler, _ = load_stage()
    vector = encode_stage_row(clean)
    
    # Named DataFrame: scaler was fitted on named columns
    X = scaler.transform(pd.DataFrame([vector], columns=STAGE_FEATURES))
    stage = int(model.predict(X)[0])
    probs = _probabilities(model, X)
    confidence = probs.get(stage) if probs else None

    result = {
        "stage": stage,
        "label": STAGE_LABELS[stage],
        "short_label": STAGE_SHORT[stage],
        "probabilities": probs,
        "confidence": confidence,
        "model": stage_model_name(),
        "inputs": clean,
        "patient_name": patient_name
    }

    # Persist to database (MySQL with SQLite fallback)
    save_prediction(
        model_type="stage",
        inputs=clean,
        prediction_result=f"Stage {stage}: {STAGE_SHORT[stage]}",
        stage_number=stage,
        risk_label=None,
        confidence=confidence,
        probability_scores=probs,
        patient_name=patient_name
    )

    return result


def predict_risk(raw: dict) -> dict:
    """Validate -> encode -> scale -> predict cirrhosis YES/no, then save to DB."""
    patient_name = raw.get("patient_name", "").strip() or "Anonymous Patient"
    clean = validate_risk(raw)
    try:
        import pandas as pd
    except ImportError:
        raise ImportError(
            "Package 'pandas' is required for model inference. "
            "Please run: pip install -r requirements.txt"
        )
    model, scaler, _ = load_risk()
    vector = encode_risk_row(clean)
    X = scaler.transform(pd.DataFrame([vector], columns=RISK_FEATURE_KEYS))
    pred = int(model.predict(X)[0])
    probs = _probabilities(model, X)
    prob_yes = probs.get(1) if probs else None

    risk_str = "YES" if pred == 1 else "no"
    result = {
        "risk": risk_str,
        "label": (
            "The model predicts the patient is at risk of liver cirrhosis."
            if pred == 1
            else "The model predicts the patient is NOT at risk of liver cirrhosis."
        ),
        "probability_yes": prob_yes,
        "probabilities": probs,
        "model": risk_model_name(),
        "inputs": clean,
        "patient_name": patient_name
    }

    # Persist to database (MySQL with SQLite fallback)
    save_prediction(
        model_type="risk",
        inputs=clean,
        prediction_result=f"Cirrhosis Risk: {risk_str.upper()}",
        stage_number=None,
        risk_label=risk_str,
        confidence=prob_yes,
        probability_scores=probs,
        patient_name=patient_name
    )

    return result
