"""Train + compare the stage model (cirrhosis.csv, Stage 1–4).

Run:  python ml/train.py        (from the project root)

* Drops the 6 rows with no Stage (412 rows left).
* Converts Age from DAYS to years (Age/365.25) — see ml/predict.py.
* Imputes missing values: median (numeric) / mode (categorical).
* Stratified 80/20 train/test split (random_state=42) + 5-fold CV.
* Compares Logistic Regression, Random Forest, SVM, GaussianNB, KNN;
  picks the winner by 5-fold CV F1-macro (the test set is only 83 rows,
  so a single split is noisy).
* Saves ml/models/model.pkl, ml/models/scaler.bin,
  ml/models/metrics.json and ml/models/confusion_matrix.png.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ml"))

from predict import (DRUG_MAP, EDEMA_SHORT_MAP, STAGE_FEATURES,  # noqa: E402
                     age_days_to_years)

MODELS_DIR = ROOT / "ml" / "models"
DATA = ROOT / "ml" / "data" / "cirrhosis.csv"


def load_stage_data() -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(DATA)
    df = df.dropna(subset=["Stage"]).copy()          # 412 rows
    df["age_years"] = df["Age"].apply(age_days_to_years)
    df["drug"] = df["Drug"].map(DRUG_MAP)
    df["sex"] = df["Sex"].map({"F": 0.0, "M": 1.0})
    for lower, orig in [("ascites", "Ascites"),
                       ("hepatomegaly", "Hepatomegaly"),
                       ("spiders", "Spiders")]:
        df[lower] = df[orig].map({"N": 0.0, "Y": 1.0})
    df["edema"] = df["Edema"].map(EDEMA_SHORT_MAP)
    df = df.rename(columns={
        "Bilirubin": "bilirubin", "Cholesterol": "cholesterol",
        "Albumin": "albumin", "Copper": "copper", "Alk_Phos": "alk_phos",
        "SGOT": "sgot", "Tryglicerides": "tryglicerides",
        "Platelets": "platelets", "Prothrombin": "prothrombin",
    })
    # Median / mode imputation (missing: Drug/Ascites/Hepatomegaly/Spiders/
    # Alk_Phos/SGOT/Copper in 106 rows, Cholesterol 134, Tryglicerides 136,
    # Platelets 11, Prothrombin 2).
    for col in STAGE_FEATURES:
        if df[col].isna().any():
            fill = df[col].median() if df[col].dtype != object else df[col].mode()[0]
            df[col] = df[col].fillna(fill)
    X = df[STAGE_FEATURES].astype(float)
    y = df["Stage"].astype(int)
    return X, y


def make_models() -> dict[str, object]:
    return {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000))]),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, random_state=42, n_jobs=-1),
        "SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(probability=True, random_state=42))]),
        "Naive Bayes": GaussianNB(),
        "KNN": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier(n_neighbors=5))]),
    }


def main() -> None:
    X, y = load_stage_data()
    print(f"Stage data: {len(X)} rows, {len(STAGE_FEATURES)} features; "
          f"classes: {y.value_counts().sort_index().to_dict()}")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y)
    print(f"Train {len(X_train)} / test {len(X_test)} (stratified 80/20)")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results: dict[str, dict] = {}
    fitted: dict[str, object] = {}
    for name, model in make_models().items():
        model.fit(X_train, y_train)
        fitted[name] = model
        pred = model.predict(X_test)
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv,
                                    scoring="f1_macro")
        results[name] = {
            "accuracy": round(accuracy_score(y_test, pred), 4),
            "precision_macro": round(precision_score(y_test, pred, average="macro", zero_division=0), 4),
            "recall_macro": round(recall_score(y_test, pred, average="macro", zero_division=0), 4),
            "f1_macro": round(f1_score(y_test, pred, average="macro", zero_division=0), 4),
            "cv_f1_macro_mean": round(float(cv_scores.mean()), 4),
            "cv_f1_macro_std": round(float(cv_scores.std()), 4),
        }
        print(f"{name:22s} acc={results[name]['accuracy']:.4f} "
              f"F1-macro={results[name]['f1_macro']:.4f} "
              f"CV F1={results[name]['cv_f1_macro_mean']:.4f} "
              f"+/- {results[name]['cv_f1_macro_std']:.4f}")

    winner = max(results, key=lambda n: results[n]["cv_f1_macro_mean"])
    print(f"\nWinner (best 5-fold CV F1-macro): {winner}")
    best = fitted[winner]
    pred = best.predict(X_test)
    report = classification_report(y_test, pred, digits=4, zero_division=0)
    print(report)
    cm = confusion_matrix(y_test, pred, labels=[1, 2, 3, 4])

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    # Save the winner as a plain (scaler, model) pair, matching the
    # original app.py's model.pkl + scaler.bin layout.
    if isinstance(best, Pipeline):
        scaler = best.named_steps["scaler"]
        clf = best.named_steps["clf"]
    else:
        scaler = StandardScaler().fit(X_train)
        clf = best
    joblib.dump(clf, MODELS_DIR / "model.pkl")
    joblib.dump(scaler, MODELS_DIR / "scaler.bin")

    # Feature importances (only meaningful for tree / linear winners).
    importances = None
    if hasattr(clf, "feature_importances_"):
        importances = {f: round(float(v), 4) for f, v in
                       sorted(zip(STAGE_FEATURES, clf.feature_importances_),
                              key=lambda kv: -kv[1])}
    elif hasattr(clf, "coef_"):
        importances = {f: round(float(v), 4) for f, v in
                       sorted(zip(STAGE_FEATURES,
                                  np.abs(clf.coef_).mean(axis=0)),
                              key=lambda kv: -kv[1])}

    metrics = {
        "task": "Liver stage prediction (Stage 1-4)",
        "dataset": "cirrhosis.csv (Mayo Clinic PBC data)",
        "n_rows_total": 418,
        "n_rows_used": int(len(X)),
        "n_rows_dropped_no_stage": 6,
        "features": STAGE_FEATURES,
        "test_size": int(len(X_test)),
        "train_size": int(len(X_train)),
        "split": "stratified 80/20, random_state=42, 5-fold CV",
        "winner": winner,
        "winner_selection": "best 5-fold CV F1-macro (more robust than the "
                            "83-row test set)",
        "comparison": results,
        "winner_metrics": results[winner],
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_labels": [1, 2, 3, 4],
        "feature_importances": importances,
        "stage_counts": {str(k): int(v) for k, v in
                         y.value_counts().sort_index().items()},
        "notes": [
            "Age converted from days to years (Age/365.25) before training.",
            "Missing values imputed with median (numeric) / mode (categorical).",
            "Edema encoding fixed to Mayo N=0.0, S=0.5, Y=1.0 "
            "(original app.py used -1.0 for the middle option).",
        ],
    }
    (MODELS_DIR / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(4)); ax.set_yticks(range(4))
    ax.set_xticklabels([1, 2, 3, 4]); ax.set_yticklabels([1, 2, 3, 4])
    ax.set_xlabel("Predicted stage"); ax.set_ylabel("True stage")
    ax.set_title(f"Confusion matrix — {winner} (stage model)")
    for i in range(4):
        for j in range(4):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(MODELS_DIR / "confusion_matrix.png", dpi=150)
    print(f"Saved model.pkl, scaler.bin, metrics.json, confusion_matrix.png "
          f"to {MODELS_DIR}")


if __name__ == "__main__":
    main()
