"""Tests — run:  python -m pytest tests/ -v   (from the project root)."""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "ml"))

from predict import encode_edema, encode_stage_row  # noqa: E402
from backend.app import create_app  # noqa: E402

DEMO_STAGE = {
    "drug": "D-penicillamine", "age": 56.4, "sex": "Female",
    "ascites": "No", "hepatomegaly": "Yes", "spiders": "Yes",
    "edema": "No edema and no diuretic therapy for edema",
    "bilirubin": 1.1, "cholesterol": 302, "albumin": 4.14, "copper": 54,
    "alk_phos": 7394.8, "sgot": 113.52, "tryglicerides": 88,
    "platelets": 221, "prothrombin": 10.6,
}

DEMO_RISK = {
    "age": 55, "gender": "Male", "alcohol_years": 12, "alcohol_qty": 2,
    "hepatitis_b": "negative", "hepatitis_c": "negative",
    "diabetes": "Yes", "obesity": "Yes",
    "hemoglobin": 12, "platelet_count": 1.5, "total_bilirubin": 7,
    "direct_bilirubin": 4, "albumin": 3, "alk_phos": 150, "sgot": 56, "sgpt": 34,
}


@pytest.fixture()
def client():
    app = create_app()
    app.config["TESTING"] = True
    return app.test_client()


# --- encodings (must match app.py, incl. the fixed edema) -------------------

def test_edema_encoding_fixed():
    # app.py's bug: middle option was -1.0, same as the default.
    assert encode_edema("No edema and no diuretic therapy for edema") == 0.0
    assert encode_edema(
        "Edema present without diuretics, or edema resolved by diuretics") == 0.5
    assert encode_edema("Edema despite diuretic therapy") == 1.0
    assert encode_edema("N") == 0.0
    assert encode_edema("S") == 0.5
    assert encode_edema("Y") == 1.0


def test_stage_feature_order_matches_app_py():
    vec = encode_stage_row(DEMO_STAGE)
    assert len(vec) == 16
    # [drug, age, sex, ascites, hepatomegaly, spiders, edema, ...]
    assert vec[:7] == [0.0, 56.4, 0.0, 0.0, 1.0, 1.0, 0.0]
    assert vec[7:] == [1.1, 302.0, 4.14, 54.0, 7394.8, 113.52, 88.0,
                       221.0, 10.6]


# --- models + metrics files -------------------------------------------------

def test_models_load_and_metrics_exist():
    from backend.services import prediction_service as svc
    model, scaler, metrics = svc.load_stage()
    assert model is not None and scaler is not None
    assert metrics["winner"] in metrics["comparison"]
    assert metrics["n_rows_used"] == 412
    rmodel, rscaler, rmetrics = svc.load_risk()
    assert rmodel is not None and rscaler is not None
    assert rmetrics["n_rows_used"] == 896
    assert (ROOT / "ml" / "models" / "metrics.json").exists()
    assert (ROOT / "ml" / "models" / "risk_metrics.json").exists()


# --- web routes --------------------------------------------------------------

def test_home_200(client):
    assert client.get("/").status_code == 200


def test_get_predict_returns_form(client):
    # Original app.py returned None for GET /predict.
    resp = client.get("/predict")
    assert resp.status_code == 200
    assert b"Liver Stage" in resp.data


def test_stage_happy_path_demo_row(client):
    resp = client.post("/predict", data=DEMO_STAGE)
    assert resp.status_code == 200
    assert b"Prediction Result" in resp.data
    # True stage for ID 2 is 3; we only assert a stage label is shown,
    # never hard-code the model's answer.
    assert any(s in resp.data for s in
               [b"Normal liver", b"Fatty liver", b"Fibrosis", b"Cirrhosis"])


def test_stage_bad_value_400(client):
    bad = dict(DEMO_STAGE, bilirubin=99)
    resp = client.post("/predict", data=bad)
    assert resp.status_code == 400
    assert b"bilirubin" in resp.data.lower() or b"Bilirubin" in resp.data


def test_stage_missing_field_400(client):
    bad = dict(DEMO_STAGE)
    del bad["copper"]
    assert client.post("/predict", data=bad).status_code == 400


def test_api_predict_json(client):
    resp = client.post("/api/predict", json=DEMO_STAGE)
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["stage"] in (1, 2, 3, 4)
    assert "label" in data


def test_api_predict_bad_json_400(client):
    resp = client.post("/api/predict", json=dict(DEMO_STAGE, bilirubin=99))
    assert resp.status_code == 400
    assert "error" in resp.get_json()


def test_api_risk_check(client):
    resp = client.post("/api/risk-check", json=DEMO_RISK)
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["risk"] in ("YES", "no")
    assert data["probability_yes"] is not None


def test_api_models_real_metrics(client):
    resp = client.get("/api/models")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["stage"]["winner"]
    assert data["risk"]["winner"]
    assert data["stage"]["winner_metrics"]["f1_macro"] > 0
    assert data["risk"]["target_counts"]["YES"] == 876


def test_models_page_200(client):
    assert client.get("/models").status_code == 200


def test_history_page_200(client):
    assert client.get("/history").status_code == 200
