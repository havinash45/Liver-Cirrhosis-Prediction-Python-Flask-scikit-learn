"""Shared feature-encoding functions — the SINGLE source of truth.

Used by BOTH the training scripts (ml/train.py, ml/train_risk.py) and the
live backend (backend/services/prediction_service.py).  Never duplicate an
encoder anywhere else: if a mapping changes, change it here only.

Stage model (cirrhosis.csv, Mayo Clinic PBC data)
--------------------------------------------------
Feature order MUST match the original app.py exactly::

    [drug, age_years, sex, ascites, hepatomegaly, spiders, edema,
     bilirubin, cholesterol, albumin, copper, alk_phos, sgot,
     tryglicerides, platelets, prothrombin]

Encodings (kept from app.py, with ONE fix — see ``encode_edema``):

* Drug:            D-penicillamine = 0, Placebo = 1
* Sex:             Female = 0, Male = 1
* Ascites / Hepatomegaly / Spiders:  No = 0, Yes = 1
* Edema:           N = 0.0, S = 0.5, Y = 1.0  (Mayo Clinic standard)
* Age:             the CSV stores Age in DAYS (9,598–28,650).  The original
  app.py form expects age in YEARS (comment range 26–78).  Training MUST
  convert ``age_years = Age_days / 365.25`` so the form matches the model.

Original bug fixed here: app.py encoded the middle edema option
("Edema present without diuretics, or edema resolved by diuretics") as
-1.0 — the same value as its uninitialised default.  That made "mild
edema" indistinguishable from "no value given".  Fixed to 0.5.

Risk model (HealthCareData.xlsx, Kaggle Indian alcohol/liver data)
-------------------------------------------------------------------
Binary target: YES / no cirrhosis.  Curated subset of 16 features
(see RISK_FEATURES); the full sheet has 41 feature columns, most of them
redundant blood-count variants.  See ml/train_risk.py for the
feature-importance justification.  Two columns the importance probe
ranks highly were deliberately LEFT OUT: RBC (top-ranked, but missing
in 552/950 rows — 58% — so it cannot be a reliable form field;
Haemoglobin, present in every row, carries the same red-cell signal)
and Family history (values include free-text "husband", and it adds
almost nothing once alcohol history is in).
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Stage model (cirrhosis.csv)
# ---------------------------------------------------------------------------

STAGE_FEATURES = [
    "drug", "age_years", "sex", "ascites", "hepatomegaly", "spiders",
    "edema", "bilirubin", "cholesterol", "albumin", "copper", "alk_phos",
    "sgot", "tryglicerides", "platelets", "prothrombin",
]

STAGE_LABELS = {
    1: "The person with the given details has a normal liver.",
    2: "The person with the given details has a fatty liver.",
    3: "The person with the given details is suffering from Liver Fibrosis.",
    4: "The person with the given details is suffering from Liver Cirrhosis.",
}

STAGE_SHORT = {1: "Normal liver", 2: "Fatty liver",
               3: "Liver fibrosis", 4: "Liver cirrhosis"}

# app.py's comment ranges are TOO NARROW for the real data (e.g. it says
# alk_phos 289–2745 but the CSV goes to 13,862.4 and the worked demo row,
# ID 2, has alk_phos 7394.8; bilirubin goes to 28.0, not 7.30).
# Enforcement ranges below are the OBSERVED data ranges, so a real patient
# row is never rejected.  app.py's comment ranges are kept in
# backend/validation.py for reference.  Bilirubin 99 is still rejected.
STAGE_RANGES = {
    "age_years": (26.0, 79.0),
    "bilirubin": (0.30, 28.0),
    "cholesterol": (120.0, 1775.0),
    "albumin": (1.96, 4.64),
    "copper": (4.0, 588.0),
    "alk_phos": (289.0, 13862.4),
    "sgot": (26.35, 457.25),
    "tryglicerides": (33.0, 598.0),
    "platelets": (62.0, 721.0),
    "prothrombin": (9.0, 18.0),
}

DRUG_MAP = {"D-penicillamine": 0.0, "Placebo": 1.0}
SEX_MAP = {"Female": 0.0, "Male": 1.0}
YES_NO_MAP = {"No": 0.0, "Yes": 1.0}
# Fixed Mayo encoding (app.py had the middle option as -1.0 — a bug).
EDEMA_MAP = {
    "No edema and no diuretic therapy for edema": 0.0,
    "Edema present without diuretics, or edema resolved by diuretics": 0.5,
    "Edema despite diuretic therapy": 1.0,
}
# Short keys accepted by the JSON API as well as the long form labels.
EDEMA_SHORT_MAP = {"N": 0.0, "S": 0.5, "Y": 1.0}


def encode_drug(value: str) -> float:
    return DRUG_MAP[value]


def encode_sex(value: str) -> float:
    return SEX_MAP[value]


def encode_yes_no(value: str) -> float:
    return YES_NO_MAP[value]


def encode_edema(value: str) -> float:
    """Mayo encoding N=0.0, S=0.5, Y=1.0.

    Accepts the full form label, the CSV code (N/S/Y) or the words
    "No edema" / "Mild edema" / "Edema despite diuretics".
    """
    if value in EDEMA_MAP:
        return EDEMA_MAP[value]
    if value in EDEMA_SHORT_MAP:
        return EDEMA_SHORT_MAP[value]
    lowered = str(value).strip().lower()
    if lowered in {"no edema", "none", "no"}:
        return 0.0
    if lowered in {"mild edema", "edema present", "s"}:
        return 0.5
    if lowered in {"severe edema", "edema despite diuretics", "y"}:
        return 1.0
    raise ValueError(f"Unknown edema value: {value!r}")


def age_days_to_years(age_days: float) -> float:
    """CSV Age is in DAYS; the form and the model use YEARS."""
    return float(age_days) / 365.25


def encode_stage_row(row: dict) -> list[float]:
    """Encode one stage-model input dict (form/API keys) -> feature vector.

    Accepted keys: drug, age (years), sex, ascites, hepatomegaly, spiders,
    edema, bilirubin, cholesterol, albumin, copper, alk_phos, sgot,
    tryglicerides, platelets, prothrombin.
    """
    return [
        encode_drug(row["drug"]),
        float(row["age"]),
        encode_sex(row["sex"]),
        encode_yes_no(row["ascites"]),
        encode_yes_no(row["hepatomegaly"]),
        encode_yes_no(row["spiders"]),
        encode_edema(row["edema"]),
        float(row["bilirubin"]),
        float(row["cholesterol"]),
        float(row["albumin"]),
        float(row["copper"]),
        float(row["alk_phos"]),
        float(row["sgot"]),
        float(row["tryglicerides"]),
        float(row["platelets"]),
        float(row["prothrombin"]),
    ]


# ---------------------------------------------------------------------------
# Risk model (HealthCareData.xlsx) — curated 16-feature subset
# ---------------------------------------------------------------------------

# Exact column names in HealthCareData.xlsx.
RISK_TARGET_COL = "Predicted Value(Out Come-Patient suffering from liver  cirrosis or not)"

RISK_FEATURES = [
    ("Age", "age"),
    ("Gender", "gender"),
    ("Duration of alcohol consumption(years)", "alcohol_years"),
    ("Quantity of alcohol consumption (quarters/day)", "alcohol_qty"),
    ("Hepatitis B infection", "hepatitis_b"),
    ("Hepatitis C infection", "hepatitis_c"),
    ("Diabetes Result", "diabetes"),
    ("Obesity", "obesity"),
    ("Hemoglobin  (g/dl)", "hemoglobin"),
    ("Platelet Count  (lakhs/mm)", "platelet_count"),
    ("Total Bilirubin    (mg/dl)", "total_bilirubin"),
    ("Direct    (mg/dl)", "direct_bilirubin"),
    ("Albumin   (g/dl)", "albumin"),
    ("AL.Phosphatase      (U/L)", "alk_phos"),
    ("SGOT/AST      (U/L)", "sgot"),
    ("SGPT/ALT (U/L)", "sgpt"),
]

RISK_FEATURE_KEYS = [key for _, key in RISK_FEATURES]

# Plausible clinical ranges for the risk form (observed data ranges).
RISK_RANGES = {
    "age": (18.0, 90.0),
    "alcohol_years": (0.0, 50.0),
    "alcohol_qty": (0.0, 10.0),
    "hemoglobin": (4.0, 20.0),
    "platelet_count": (0.2, 6.0),
    "total_bilirubin": (0.2, 40.0),
    "direct_bilirubin": (0.8, 25.0),
    "albumin": (1.0, 6.0),
    "alk_phos": (40.0, 600.0),
    "sgot": (5.0, 500.0),
    "sgpt": (5.0, 500.0),
}

RISK_YES_NO = {"yes": 1.0, "no": 0.0, "YES": 1.0, "NO": 0.0,
               "positive": 1.0, "negative": 0.0}


def _risk_binary(value) -> float:
    key = str(value).strip().lower()
    if key in {"yes", "positive"}:
        return 1.0
    if key in {"no", "negative"}:
        return 0.0
    raise ValueError(f"Expected yes/no, got {value!r}")


def encode_risk_row(row: dict) -> list[float]:
    """Encode one risk-model input dict (API/form keys) -> feature vector."""
    gender = str(row["gender"]).strip().lower()
    sex = 1.0 if gender in {"male", "m"} else 0.0
    return [
        float(row["age"]),
        sex,
        float(row["alcohol_years"]),
        float(row["alcohol_qty"]),
        _risk_binary(row["hepatitis_b"]),
        _risk_binary(row["hepatitis_c"]),
        _risk_binary(row["diabetes"]),
        _risk_binary(row["obesity"]),
        float(row["hemoglobin"]),
        float(row["platelet_count"]),
        float(row["total_bilirubin"]),
        float(row["direct_bilirubin"]),
        float(row["albumin"]),
        float(row["alk_phos"]),
        float(row["sgot"]),
        float(row["sgpt"]),
    ]
