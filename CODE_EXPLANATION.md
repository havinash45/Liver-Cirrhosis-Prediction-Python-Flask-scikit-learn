# Code Explanation — Liver Cirrhosis Prediction (Python rebuild)

Maps every part of this Python rebuild to the original project part it
replaces. Original parts: `app.py` (Jupyter-exported Flask), `index.css`,
`cirrhosis.csv`, `HealthCareData.xlsx`, `Final_Report.pdf`.

## `ml/predict.py` — shared encodings (the single source of truth)

Replaces the inline encoding code scattered through original `app.py`.
Both the training scripts and the live backend import from here, so an
encoding can never drift between training and serving.

- `STAGE_FEATURES` — the 16-feature order, exactly as in `app.py`:
  drug, age_years, sex, ascites, hepatomegaly, spiders, edema,
  bilirubin, cholesterol, albumin, copper, alk_phos, sgot,
  tryglicerides, platelets, prothrombin.
- `encode_edema()` — Mayo encoding N=0.0, S=0.5, Y=1.0. This is the
  **fix** for original bug (c): `app.py` set the middle edema option to
  `-1.0`, the same as its uninitialised default.
- `age_days_to_years()` — the CSV stores Age in days; the form and the
  model use years (`Age / 365.25`).
- `RISK_FEATURES` — the curated 16 risk features and their exact Excel
  column names, plus `encode_risk_row()`.

## `ml/train.py` — stage model training

Replaces the missing training step (the original shipped only
`model.pkl`/`scaler.bin`, which were never provided, and no training
code at all).

- Loads `ml/data/cirrhosis.csv`, drops the 6 rows with no Stage
  (412 left), converts Age days→years, encodes categories, imputes
  missing values with median/mode.
- Stratified 80/20 split + 5-fold CV; trains Logistic Regression,
  Random Forest, SVM, GaussianNB, KNN; picks the winner by **CV
  F1-macro**; saves `ml/models/model.pkl`, `scaler.bin`,
  `metrics.json`, `confusion_matrix.png`.
- Winner on this data: **SVM** (CV F1-macro 0.3630).

## `ml/train_risk.py` — risk model training

New in this rebuild (the original had no second dataset/model). Same
pattern as `train.py` on `HealthCareData.xlsx`: drops 54 blank-target
rows (896 left), encodes the curated 16-feature subset, stratified
split, `class_weight='balanced'` where supported, winner by CV
F1-macro. Saves `risk_model.pkl`, `risk_scaler.bin`,
`risk_metrics.json`, `risk_confusion_matrix.png`.
Winner: **Random Forest** — see README for why its 1.0000 must not be
quoted as real-world accuracy.

## `ml/eda.py` — exploratory analysis

Prints and saves `ml/EDA_SUMMARY.md`: row counts, Stage counts, the
Age-in-days finding, missing-value counts, numeric ranges, and the
risk-target imbalance (YES 876 / no 20 / blank 54).

## `backend/app.py` — entry point (thin)

Replaces original `app.py`'s top level. Creates the Flask app with
`template_folder=../frontend/templates` and
`static_folder=../frontend/static`, registers the blueprints, and runs
on **port 8080** like the original. No model code lives here.

## `backend/config.py` — settings in one place

Port/host/debug, artifact paths (`ml/models/...`), SQLite history path
(`backend/instance/predictions.db`). Change values here or via
environment variables (`PORT`, `HOST`, `FLASK_DEBUG`).

## `backend/routes.py` — web + API routes

Replaces original `app.py`'s two routes, split into a `web` blueprint
(HTML pages) and an `api` blueprint (JSON):

- `GET /` — home page (original `home.html` was never provided; rebuilt
  around the original CSS classes).
- `GET /predict` — **fix for original bug (a)**: the original returned
  `None` for GET. Now returns the form with both tabs.
- `POST /predict`, `POST /risk-check` — validate, predict, render
  `result.html` with the stage label (exactly `app.py`'s wording),
  confidence/probabilities, risk badge and printable report.
- `GET /models` + `POST /models/retrain` — real metrics page; Retrain
  re-runs both training scripts and reloads the artifacts.
- `GET /history` — predictions logged in SQLite.
- `POST /api/predict`, `POST /api/risk-check`, `GET /api/models` —
  JSON in/out; bad input → `{"error": "..."}` with HTTP 400.

## `backend/validation.py` — server-side validation

**Fix for original bug (b)**: `app.py` had no validation, so a missing
or bad field crashed with a 500. Every field is checked here (required,
choice, numeric range) and a `ValidationError` becomes HTTP 400 with a
clear message. Ranges are the *observed data ranges* — see the note in
the file: `app.py`'s comment ranges are too narrow for the real data
(e.g. alk_phos 289–2745 vs actual max 13,862.4; the demo row has
7394.8), so enforcing them would reject real patients.

## `backend/services/prediction_service.py` — the only place that touches sklearn

Routes never import sklearn/joblib. This module lazily loads the four
artifacts, calls `ml/predict.py` to encode, scales, predicts, returns
stage label + probabilities (or YES/no + probability), and logs every
prediction to SQLite (`predictions` table: timestamp, model, inputs,
result, probability).

## `frontend/` — templates + static

- `static/css/index.css` — the original 168-line stylesheet, copied
  **verbatim**. Templates are built around its classes: `.header`,
  `.container`, `.result`, `.report`, `.image-container` /
  `.overlay-text` (home hero).
- `templates/home.html` — hero + "how it works" (the journey from the
  Customer Journey Map) + disclaimer.
- `templates/predict.html` — both tabs: "Liver Stage (1–4)" (16 fields)
  and "Risk Check (Yes/No)" (16 fields), with min/max hints and
  "Fill demo patient" buttons.
- `templates/result.html` — stage label, confidence, probabilities,
  risk badge, and the `.report` block (inputs + prediction + date)
  with a Print/Download button.
- `templates/models.html` — comparison tables, per-class reports,
  confusion matrices, feature importances, Retrain button.
- `templates/history.html` — SQLite prediction log.
- `static/js/main.js` — tab switching, demo-patient fillers, nothing else.

## `tests/test_app.py` — pytest (14 tests)

Encoding matches `app.py` (including the fixed edema value and the
exact 16-feature order); both models load and metrics files exist;
home 200; GET /predict returns the form; stage happy path with the
ID-2 demo row; bad bilirubin → 400; missing field → 400; API JSON
happy path + 400; risk-check API; `/api/models` returns real metrics;
models and history pages 200.

Run: `python -m pytest tests/ -v`
