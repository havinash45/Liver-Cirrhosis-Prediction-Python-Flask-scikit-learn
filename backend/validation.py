"""Server-side validation for both prediction forms.

Original app.py had NO validation: a missing or bad field crashed with a
500.  Here every field is checked and a bad value returns HTTP 400 with a
clear message — never a 500.

Range note (important): the ranges enforced here are the OBSERVED ranges
of the training data.  app.py's code comments quoted narrower ranges
(e.g. alk_phos 289–2745, bilirubin 0.30–7.30, cholesterol 160.875–459.875,
sgot 26.35–202.88, tryglicerides 45.875–176.875, platelets 62–503.75,
prothrombin 9–12.75), but the real cirrhosis.csv goes well beyond them —
alk_phos up to 13,862.4, bilirubin to 28.0 — and the worked demo row
(ID 2) has alk_phos 7394.8.  Enforcing app.py's comment ranges would
reject real patient rows, so the data ranges are used instead and the
comment ranges are listed below for reference.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ml"))

from predict import (DRUG_MAP, EDEMA_MAP, EDEMA_SHORT_MAP, RISK_RANGES,  # noqa: E402
                     SEX_MAP, STAGE_RANGES, YES_NO_MAP)

# app.py's comment ranges (kept for reference — NOT enforced, see docstring).
APP_PY_COMMENT_RANGES = {
    "age": (26.0, 78.0),
    "bilirubin": (0.30, 7.30),
    "cholesterol": (160.875, 459.875),
    "albumin": (2.45, 4.56),
    "copper": (4.0, 175.0),
    "alk_phos": (289.0, 2745.0),
    "sgot": (26.35, 202.88),
    "tryglicerides": (45.875, 176.875),
    "platelets": (62.0, 503.75),
    "prothrombin": (9.0, 12.75),
}

STAGE_REQUIRED = [
    "drug", "age", "sex", "ascites", "hepatomegaly", "spiders", "edema",
    "bilirubin", "cholesterol", "albumin", "copper", "alk_phos", "sgot",
    "tryglicerides", "platelets", "prothrombin",
]

STAGE_NUMERIC = {
    "age": ("age_years", "Age (years)"),
    "bilirubin": ("bilirubin", "Bilirubin (mg/dl)"),
    "cholesterol": ("cholesterol", "Cholesterol (mg/dl)"),
    "albumin": ("albumin", "Albumin (g/dl)"),
    "copper": ("copper", "Copper (µg/dl)"),
    "alk_phos": ("alk_phos", "Alkaline Phosphatase (U/L)"),
    "sgot": ("sgot", "SGOT / AST (U/L)"),
    "tryglicerides": ("tryglicerides", "Triglycerides (mg/dl)"),
    "platelets": ("platelets", "Platelets (×10³/µL)"),
    "prothrombin": ("prothrombin", "Prothrombin time (sec)"),
}

RISK_REQUIRED = [
    "age", "gender", "alcohol_years", "alcohol_qty", "hepatitis_b",
    "hepatitis_c", "diabetes", "obesity", "hemoglobin",
    "platelet_count", "total_bilirubin", "direct_bilirubin", "albumin",
    "alk_phos", "sgot", "sgpt",
]

RISK_NUMERIC = {
    "age": ("age", "Age (years)"),
    "alcohol_years": ("alcohol_years", "Duration of alcohol use (years)"),
    "alcohol_qty": ("alcohol_qty", "Alcohol quantity (quarters/day)"),
    "hemoglobin": ("hemoglobin", "Hemoglobin (g/dl)"),
    "platelet_count": ("platelet_count", "Platelet count (lakhs/mm³)"),
    "total_bilirubin": ("total_bilirubin", "Total bilirubin (mg/dl)"),
    "direct_bilirubin": ("direct_bilirubin", "Direct bilirubin (mg/dl)"),
    "albumin": ("albumin", "Albumin (g/dl)"),
    "alk_phos": ("alk_phos", "ALP (U/L)"),
    "sgot": ("sgot", "SGOT / AST (U/L)"),
    "sgpt": ("sgpt", "SGPT / ALT (U/L)"),
}


class ValidationError(ValueError):
    """Raised for any bad input; the routes turn it into HTTP 400."""


def _missing(data: dict, required: list[str]) -> list[str]:
    return [f for f in required
            if f not in data or data[f] in (None, "", " ")]


def _number(data: dict, field: str, label: str, lo: float, hi: float) -> float:
    raw = data[field]
    try:
        value = float(raw)
    except (TypeError, ValueError):
        raise ValidationError(
            f"{label} must be a number, got {raw!r}.") from None
    if not (lo <= value <= hi):
        raise ValidationError(
            f"{label} must be between {lo:g} and {hi:g}, got {value:g}.")
    return value


def _choice(data: dict, field: str, label: str,
            allowed: dict) -> str:
    value = str(data[field]).strip()
    if value not in allowed:
        raise ValidationError(
            f"{label} must be one of {sorted(allowed)}, got {value!r}.")
    return value


def validate_stage(data: dict) -> dict:
    """Validate a stage-form payload; returns a clean dict for encoding."""
    missing = _missing(data, STAGE_REQUIRED)
    if missing:
        raise ValidationError(
            f"Missing required field(s): {', '.join(missing)}.")
    clean = {
        "drug": _choice(data, "drug", "Drug", DRUG_MAP),
        "sex": _choice(data, "sex", "Sex", SEX_MAP),
        "ascites": _choice(data, "ascites", "Ascites", YES_NO_MAP),
        "hepatomegaly": _choice(data, "hepatomegaly", "Hepatomegaly", YES_NO_MAP),
        "spiders": _choice(data, "spiders", "Spiders", YES_NO_MAP),
        "edema": _choice(data, "edema", "Edema",
                         {**EDEMA_MAP, **EDEMA_SHORT_MAP}),
    }
    for field, (range_key, label) in STAGE_NUMERIC.items():
        lo, hi = STAGE_RANGES[range_key]
        clean[field] = _number(data, field, label, lo, hi)
    return clean


def validate_risk(data: dict) -> dict:
    """Validate a risk-check payload; returns a clean dict for encoding."""
    missing = _missing(data, RISK_REQUIRED)
    if missing:
        raise ValidationError(
            f"Missing required field(s): {', '.join(missing)}.")
    clean = {
        "gender": _choice(data, "gender", "Gender",
                          {"Male": "Male", "Female": "Female",
                           "male": "Male", "female": "Female"}),
    }
    for field in ["hepatitis_b", "hepatitis_c", "diabetes", "obesity"]:
        clean[field] = _choice(
            data, field, field.replace("_", " ").title(),
            {"Yes": "Yes", "No": "No", "yes": "Yes", "no": "No",
             "positive": "Yes", "negative": "No"})
    for field, (range_key, label) in RISK_NUMERIC.items():
        lo, hi = RISK_RANGES[range_key]
        clean[field] = _number(data, field, label, lo, hi)
    return clean
