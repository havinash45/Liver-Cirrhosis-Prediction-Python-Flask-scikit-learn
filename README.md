# Liver Cirrhosis Prediction — Python (Flask + scikit-learn)

A complete, runnable Python rebuild of **"Revolutionizing Liver Care:
Predicting Liver Cirrhosis using Advanced Machine Learning Techniques"**
(Team ID LTVIP2025TMID38618, May–June 2025).

Two models, one app:

1. **Liver Stage (1–4)** — trained on `cirrhosis.csv` (Mayo Clinic PBC
   data, 412 patients): 1 normal liver, 2 fatty liver, 3 liver fibrosis,
   4 liver cirrhosis.
2. **Risk Check (YES/no)** — trained on `HealthCareData.xlsx` (Kaggle
   Indian alcohol/liver data, 896 patients): is the patient at risk of
   liver cirrhosis?

> **Decision-support tool only — not a diagnosis. Always consult a doctor.**

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows
# source .venv/bin/activate       # macOS / Linux
python -m pip install -r requirements.txt
python ml/train.py                # builds ml/models/model.pkl + scaler.bin
python ml/train_risk.py           # builds ml/models/risk_model.pkl + risk_scaler.bin
python backend/app.py             # runs on http://127.0.0.1:8080
```

The trained models are already included in `ml/models/`, so the app runs
immediately after `pip install`. Re-run the two training scripts any time
to retrain (or use the **Retrain** button on the Models page).

Full step-by-step instructions (including VS Code + Anaconda) are in
[VSCODE_GUIDE.md](VSCODE_GUIDE.md). What each file does is in
[CODE_EXPLANATION.md](CODE_EXPLANATION.md).

## How to use

1. **Home** — project overview, "Check Liver Stage" and "Risk Check".
2. **Check Liver Stage** — enter the 16 clinical values
   (drug, age in years, sex, ascites, hepatomegaly, spiders, edema,
   bilirubin, cholesterol, albumin, copper, alkaline phosphatase, SGOT,
   triglycerides, platelets, prothrombin). The "Fill demo patient" button
   loads the worked example below.
3. **Risk Check** — the second tab: age, gender, alcohol history,
   hepatitis B/C, diabetes, obesity, haemoglobin, platelet count,
   bilirubin (total + direct), albumin, ALP, SGOT, SGPT.
4. **Result** — stage label, model confidence / probabilities, a risk
   badge, and a printable report (inputs + prediction + date).
5. **Models** — real measured metrics, confusion matrices, feature
   importances, Retrain button. **History** — every prediction logged
   to SQLite (`backend/instance/predictions.db`).

API: `POST /api/predict` (stage, JSON), `POST /api/risk-check`,
`GET /api/models`. Bad input returns HTTP 400 with a clear message.

## Worked demo (real row from cirrhosis.csv — ID 2, true Stage 3)

| Field | Value |
|---|---|
| Drug | D-penicillamine |
| Age | 20,617 days = **56.4 years** |
| Sex / Ascites / Hepatomegaly / Spiders | Female / No / Yes / Yes |
| Edema | No edema |
| Bilirubin / Cholesterol / Albumin / Copper | 1.1 / 302 / 4.14 / 54 |
| Alk_Phos / SGOT / Triglycerides / Platelets / Prothrombin | 7394.8 / 113.52 / 88 / 221 / 10.6 |
| **True stage** | **3 — Liver Fibrosis** |

**What the model actually predicts for this row: Stage 4 (liver
cirrhosis), confidence 49.9%** — probabilities Stage 1: 5.0%,
Stage 2: 20.0%, Stage 3: 25.1%, Stage 4: 49.9%. The model gets this
one wrong, and that is reported honestly: the stage model is modest
(see metrics below). Never hard-code the expected answer — run it and
see.

## Real measured metrics (not the report's blank table)

The original report's §6.1 performance-metrics table is **blank** — no
numbers were published. Every number below was measured by
`ml/train.py` / `ml/train_risk.py` on a stratified 80/20 held-out test
set with 5-fold cross-validation. The winner in each case was picked by
**best 5-fold CV F1-macro** (a single small test split is too noisy to
pick a winner — the stage test set is only 83 rows, the risk test set
holds only 4 "no" rows).

### Stage model — cirrhosis.csv (412 rows: Stage 1=21, 2=92, 3=155, 4=144)

| Model | Test accuracy | Precision (macro) | Recall (macro) | F1-macro (test) | CV F1-macro |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.4699 | 0.3613 | 0.3449 | 0.3423 | 0.3573 ± 0.0423 |
| Random Forest | **0.5422** | 0.3770 | 0.3893 | 0.3749 | 0.3484 ± 0.0405 |
| **SVM (winner)** | 0.4819 | 0.3425 | 0.3422 | 0.3255 | **0.3630 ± 0.0225** |
| Naive Bayes | 0.3253 | 0.4096 | 0.3021 | 0.2961 | 0.3097 ± 0.0678 |
| KNN | 0.4337 | 0.4235 | 0.3933 | 0.4004 | 0.3590 ± 0.0344 |

**Winner: SVM** (best CV F1-macro 0.3630, the most stable across folds;
KNN has the best single-test F1, 0.4004, but is less stable). The stage
model is honest but modest: Stage 1 has only 21 patients, and the
winner never predicts Stage 1 on the test set (recall 0.00 for Stage 1).
Per-class test scores (SVM): Stage 1 P 0.00 / R 0.00, Stage 2
0.25 / 0.11, Stage 3 0.41 / 0.68, Stage 4 0.71 / 0.59.
Confusion matrix (rows true, cols predicted, stages 1–4):
`[[0,1,3,0],[0,2,15,2],[0,5,21,5],[0,0,12,17]]`.

### Risk model — HealthCareData.xlsx (896 rows: YES 876, no 20)

| Model | Test accuracy | Precision (macro) | Recall (macro) | F1-macro (test) | CV F1-macro |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9278 | 0.6176 | 0.9631 | 0.6713 | 0.6946 ± 0.0381 |
| **Random Forest (winner)** | 1.0000 | 1.0000 | 1.0000 | 1.0000 | **1.0000 ± 0.0000** |
| SVM | 0.9833 | 0.7857 | 0.9915 | 0.8593 | 0.8207 ± 0.0977 |
| Naive Bayes | 0.9167 | 0.6053 | 0.9574 | 0.6517 | 0.6667 ± 0.0503 |
| KNN | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.8642 ± 0.1007 |

**Winner: Random Forest.** Per-class test scores: "no" P 1.00 / R 1.00
(support 4), YES P 1.00 / R 1.00 (support 176).

⚠️ **Do NOT quote the risk model's 1.0000 as real-world accuracy.**
The target is 97.8% YES (a model that always answers YES already scores
97.8% accuracy), the test set contains only **4** "no" patients, and the
sheet contains many near-identical rows, so test/CV rows can be
near-copies of training rows and a tree model can memorise them. The
honest headline is the per-class metrics above *plus* this caveat, and
the more conservative models (SVM CV F1 0.82, Logistic Regression
0.69) show the plausible real range. Top risk features (Random Forest
importance): total bilirubin 0.29, direct bilirubin 0.17, albumin 0.13,
alcohol duration 0.12, SGOT 0.08.

## What was in the source files

| Source file | What it was |
|---|---|
| `Final_Report.pdf` / `.docx` (19 pp.) | Project report; Flask + scikit-learn; models NB/SVM/RF/LR/KNN; §6.1 metrics table blank |
| `app.py` (131 lines, Jupyter export) | Original Flask app — loads `model.pkl` + `scaler.bin` (never provided) and renders `home.html` (never provided); port 8080 |
| `cirrhosis.csv` (418 rows) | Mayo Clinic PBC data; target Stage 1–4 |
| `HealthCareData.xlsx` (950 rows × 42 cols) | Kaggle Indian alcohol/liver data; binary YES/no target |
| `index.css` (168 lines) | Original stylesheet — copied **verbatim** to `frontend/static/css/index.css` |
| Empathy Map + Customer Journey Map (27 May 2025) | Users are doctors/healthcare professionals; journey: scheduling → data entry → prediction → receiving results |

## User journey (from the Customer Journey Map)

Scheduling → data entry → model prediction → receiving results. The app
mirrors it: Home → enter details → prediction → downloadable report
(FR-3), with every prediction logged for later review on /history.

## Bugs in the original app.py — fixed here

1. **GET /predict returned `None`** (no return for GET) — now returns the form.
2. **No validation** — a missing/bad field crashed with a 500. Now every
   field is validated server-side and bad input returns HTTP 400 with a
   clear message.
3. **Edema encoding broken** — the middle option ("Edema present without
   diuretics, or edema resolved by diuretics") was set to `-1.0`, the
   same as the uninitialised default. Fixed to the standard Mayo
   encoding N=0.0, S=0.5, Y=1.0 (in `ml/predict.py`, the single source
   of truth).
4. **`model.pkl`, `scaler.bin`, `home.html` were missing**, so the
   original could not run at all. Both models are now trained from the
   provided datasets and shipped in `ml/models/`.

## Two data gotchas — read before the viva

- **Age is in DAYS in cirrhosis.csv** (9,598–28,650 days = 26.3–78.4
  years), but the original form expects age in **years** (26–78).
  Training converts `Age / 365.25`; if you skip this, the model sees
  ages ~500× too large and every prediction is wrong.
- **app.py's comment ranges are too narrow for the real data**
  (e.g. alk_phos "289–2745" but the CSV goes to 13,862.4; the demo row
  has 7394.8; bilirubin goes to 28.0, not 7.30). Validation here uses
  the *observed data ranges* so real patient rows are never rejected;
  the comment ranges are listed in `backend/validation.py`.
- **Missing values**: Drug/Ascites/Hepatomegaly/Spiders/Alk_Phos/SGOT/
  Copper missing in 106 rows, Cholesterol 134, Triglycerides 136,
  Platelets 11, Prothrombin 2 — imputed with median (numeric) / mode
  (categorical). 6 rows with no Stage were dropped (412 used).
- **Risk-model imbalance**: YES 876 / no 20 / 54 blank (dropped).
  Stratified split + `class_weight='balanced'` where supported; judged
  on per-class precision/recall, never raw accuracy alone.
- **Risk feature subset**: 16 of 41 columns, chosen from a
  Random-Forest importance ranking over all columns (alcohol duration,
  bilirubin, albumin, SGOT, haemoglobin, platelets top the list).
  RBC ranked first in that probe but is missing in 552/950 rows (58%),
  so it was left out — Haemoglobin carries the same red-cell signal.
  Family history was dropped (free-text values like "husband").

## Project layout

```
liver-cirrhosis-python/
  README.md  CODE_EXPLANATION.md  VSCODE_GUIDE.md  requirements.txt
  backend/    app.py  config.py  routes.py  validation.py  services/prediction_service.py
  frontend/   templates/ (home, predict, result, models, history)
              static/css/index.css  (verbatim original)  static/js/main.js
  ml/         predict.py (shared encodings)  eda.py  train.py  train_risk.py
              data/ (cirrhosis.csv, HealthCareData.xlsx)
              models/ (model.pkl, scaler.bin, risk_model.pkl, risk_scaler.bin,
                       metrics.json, risk_metrics.json, confusion matrices)
              EDA_SUMMARY.md
  tests/      test_app.py  (pytest, 14 tests)
  screenshots/
```

## Viva notes (honest)

- The stage model is modest (CV F1-macro 0.36) — 4-class staging from
  412 patients with only 21 Stage-1 cases is hard; the demo row itself
  is misclassified (true 3, predicted 4). Say so; do not claim the
  paper's accuracy, because the report published none.
- The risk model's perfect score is an artefact of a 97.8%-YES dataset
  with near-duplicate rows and only 4 negative test cases — quote the
  per-class metrics and the caveat, not "100% accuracy".
- No passwords or auth in this project; none is needed.
