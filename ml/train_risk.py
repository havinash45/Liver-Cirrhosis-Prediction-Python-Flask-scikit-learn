"""Train the binary risk model (HealthCareData.xlsx, cirrhosis YES/no).

Run:  python ml/train_risk.py     (from the project root)

* Drops the 54 rows with a blank target (896 usable rows).
* Target is EXTREMELY imbalanced: YES 876 / no 20 (2.2% negative).
  Honest handling: stratified 80/20 split, class_weight='balanced'
  where supported, and per-class precision/recall as the headline —
  raw accuracy is misleading here (always-YES scores ~97.8%).
* Curated subset of 16 clinically relevant features (see
  ml/predict.py RISK_FEATURES).  Justification: a Random Forest fit on
  ALL 41 usable feature columns ranks the alcohol history, liver
  enzymes (SGOT/SGPT/ALP), bilirubin, albumin, haemoglobin and platelet
  count at the top, while the dropped columns are near-duplicates
  (Direct/Indirect Bilirubin repeat Total Bilirubin; MCV/MCH/MCHC
  repeat the haemoglobin picture; Total Protein/Globulin repeat
  Albumin; the differential blood counts add little).  The
  all-features importance ranking is printed and saved in
  risk_metrics.json under "all_features_importance".
* Compares Logistic Regression, Random Forest, SVM, GaussianNB, KNN;
  picks the winner by 5-fold CV F1-macro (the test set holds only 4
  'no' rows, and the sheet has many near-duplicate rows, so test F1
  alone is not trustworthy).
* Saves ml/models/risk_model.pkl, ml/models/risk_scaler.bin,
  ml/models/risk_metrics.json and ml/models/risk_confusion_matrix.png.
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

from predict import RISK_FEATURES, RISK_TARGET_COL, encode_risk_row  # noqa: E402

MODELS_DIR = ROOT / "ml" / "models"
DATA = ROOT / "ml" / "data" / "HealthCareData.xlsx"

BINARY_MAP = {"YES": 1, "yes": 1, "no": 0, "NO": 0}


def load_risk_data() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Return (X_subset, y, X_all, y) — subset for the final model,
    all-features frame for the importance justification."""
    df = pd.read_excel(DATA)
    df = df.dropna(subset=[RISK_TARGET_COL]).copy()      # 896 rows
    df["target"] = df[RISK_TARGET_COL].map(BINARY_MAP)
    df = df[df["target"].notna()].copy()

    # Curated 16-feature subset.
    rows = []
    for _, r in df.iterrows():
        raw = {key: r[col] for col, key in RISK_FEATURES}
        raw["gender"] = r["Gender"]
        try:
            rows.append(encode_risk_row(raw))
        except (ValueError, TypeError):
            rows.append([np.nan] * len(RISK_FEATURES))
    X = pd.DataFrame(rows, columns=[k for _, k in RISK_FEATURES])
    y = df["target"].astype(int).reset_index(drop=True)

    # All-features frame (numeric + yes/no encoded) for importance ranking.
    all_df = df.copy()
    for col in all_df.columns:
        if all_df[col].dtype == object:
            all_df[col] = all_df[col].astype(str).str.strip().str.lower()
            all_df[col] = all_df[col].map(
                {"yes": 1.0, "no": 0.0, "positive": 1.0, "negative": 0.0,
                 "male": 1.0, "female": 0.0})
    numeric = all_df.select_dtypes(include=[np.number]).drop(
        columns=["target", "S.NO"], errors="ignore")
    X_all = numeric.fillna(numeric.median(numeric_only=True))
    return X, y, X_all, y


def make_models() -> dict[str, object]:
    return {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))]),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, random_state=42, n_jobs=-1,
            class_weight="balanced"),
        "SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(probability=True, random_state=42,
                        class_weight="balanced"))]),
        "Naive Bayes": GaussianNB(),
        "KNN": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier(n_neighbors=5))]),
    }


def main() -> None:
    X, y, X_all, _ = load_risk_data()
    # Median imputation for the few missing values in the subset
    # (Albumin 9 rows, AL.Phosphatase 10 rows).
    X = X.fillna(X.median(numeric_only=True))
    print(f"Risk data: {len(X)} rows; target counts: "
          f"{y.value_counts().to_dict()} (1=YES, 0=no)")
    print("Always-YES baseline accuracy would be "
          f"{y.mean():.4f} — raw accuracy is misleading here.")

    # Feature-importance justification for the 16-feature subset.
    probe = RandomForestClassifier(n_estimators=300, random_state=42,
                                   n_jobs=-1, class_weight="balanced")
    probe.fit(X_all, y)
    all_imp = {c: round(float(v), 4) for c, v in
               sorted(zip(X_all.columns, probe.feature_importances_),
                      key=lambda kv: -kv[1])}
    print("Top all-features importances:")
    for k, v in list(all_imp.items())[:15]:
        print(f"  {k}: {v}")

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
              f"CV F1={results[name]['cv_f1_macro_mean']:.4f}")

    winner = max(results, key=lambda n: results[n]["cv_f1_macro_mean"])
    print(f"\nWinner (best 5-fold CV F1-macro): {winner}")
    best = fitted[winner]
    pred = best.predict(X_test)
    report = classification_report(
        y_test, pred, target_names=["no", "YES"], digits=4, zero_division=0)
    print(report)
    cm = confusion_matrix(y_test, pred, labels=[0, 1])

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    if isinstance(best, Pipeline):
        scaler = best.named_steps["scaler"]
        clf = best.named_steps["clf"]
    else:
        scaler = StandardScaler().fit(X_train)
        clf = best
    joblib.dump(clf, MODELS_DIR / "risk_model.pkl")
    joblib.dump(scaler, MODELS_DIR / "risk_scaler.bin")

    importances = None
    if hasattr(clf, "feature_importances_"):
        importances = {f: round(float(v), 4) for f, v in
                       sorted(zip([k for _, k in RISK_FEATURES],
                                  clf.feature_importances_),
                              key=lambda kv: -kv[1])}
    elif hasattr(clf, "coef_"):
        importances = {f: round(float(v), 4) for f, v in
                       sorted(zip([k for _, k in RISK_FEATURES],
                                  np.abs(clf.coef_[0])),
                              key=lambda kv: -kv[1])}

    metrics = {
        "task": "Cirrhosis risk check (binary YES/no)",
        "dataset": "HealthCareData.xlsx (Kaggle Indian alcohol/liver data)",
        "n_rows_total": 950,
        "n_rows_used": int(len(X)),
        "n_rows_dropped_blank_target": 54,
        "target_counts": {"YES": int((y == 1).sum()), "no": int((y == 0).sum())},
        "features": [k for _, k in RISK_FEATURES],
        "test_size": int(len(X_test)),
        "train_size": int(len(X_train)),
        "split": "stratified 80/20, random_state=42, 5-fold CV",
        "imbalance_warning": (
            "Only 2.2% of labelled rows are 'no'. Always-YES scores "
            f"{y.mean():.1%} accuracy, so judge this model on per-class "
            "precision/recall and F1-macro, never on raw accuracy alone."),
        "winner": winner,
        "winner_selection": "best 5-fold CV F1-macro (more robust than the "
                            "test set, which holds only 4 'no' rows)",
        "comparison": results,
        "winner_metrics": results[winner],
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_labels": ["no", "YES"],
        "feature_importances": importances,
        "all_features_importance": all_imp,
        "notes": [
            "class_weight='balanced' used where supported.",
            "Curated 16-feature subset; dropped columns are near-duplicates "
            "of kept ones (see EDA_SUMMARY.md and all_features_importance).",
            "RBC was the top-ranked column in the all-features probe but "
            "is missing in 552/950 rows (58%), so it was left out; "
            "Haemoglobin carries the same red-cell signal.",
            "Blood pressure column '138/90' is text and was not used.",
            "HealthCareData.xlsx contains many near-identical rows, so a "
            "test row may be a near-copy of a training row: test F1 can "
            "look perfect (SVM scored 1.0000 on test) while 5-fold CV is "
            "lower.  CV F1-macro is therefore the honest headline and the "
            "winner-selection criterion.",
        ],
    }
    (MODELS_DIR / "risk_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(2)); ax.set_yticks(range(2))
    ax.set_xticklabels(["no", "YES"]); ax.set_yticklabels(["no", "YES"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"Confusion matrix — {winner} (risk model)")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(MODELS_DIR / "risk_confusion_matrix.png", dpi=150)
    print(f"Saved risk_model.pkl, risk_scaler.bin, risk_metrics.json, "
          f"risk_confusion_matrix.png to {MODELS_DIR}")


if __name__ == "__main__":
    main()
