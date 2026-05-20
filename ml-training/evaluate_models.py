"""
Model evaluation — compare Decision Tree v1 vs Ensemble v2.

Loads both the original DecisionTreeClassifier from the legacy chatbot
and the new ensemble model, evaluates them on Testing.csv, and generates
a comprehensive comparison report with metrics, confusion matrices,
and per-class analysis.

Outputs:
  model_artifacts/evaluation_report.json  — full metrics comparison
  model_artifacts/confusion_matrix.png   — side-by-side confusion matrices

Usage:
  python evaluate_models.py --data-dir ../chatbot/Data
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    top_k_accuracy_score,
)
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

ARTIFACTS_DIR = Path(__file__).parent / "model_artifacts"


# ── Data Loading (reuse logic from train_ensemble) ──────────────────────

def _load_data(data_dir: str) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Load Testing.csv with duplicate-column handling."""
    csv_path = os.path.join(data_dir, "Testing.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Testing.csv not found at {csv_path}")

    df = pd.read_csv(csv_path)
    df.columns = [c.strip().replace(" ", "_") for c in df.columns]

    target_col = "prognosis"
    for c in df.columns:
        if c.lower() == "prognosis":
            target_col = c
            break

    y = df[target_col].str.strip()
    feature_cols = [c for c in df.columns if c != target_col]

    # Deduplicate
    dedup: list[str] = []
    seen: set[str] = set()
    for c in feature_cols:
        base = c.split(".")[0] if "." in c and c.split(".")[-1].isdigit() else c
        if base not in seen:
            seen.add(base)
            dedup.append(c)

    X = df[dedup].astype(int)
    X.columns = [
        c.split(".")[0] if "." in c and c.split(".")[-1].isdigit() else c
        for c in X.columns
    ]

    return X, y, list(X.columns)


# ── Decision Tree v1 (train from scratch for baseline) ──────────────────

def _train_baseline_dt(data_dir: str, feature_names: list[str]) -> tuple[Any, LabelEncoder]:
    """Train a basic Decision Tree on Training.csv as the v1 baseline."""
    csv_path = os.path.join(data_dir, "Training.csv")
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().replace(" ", "_") for c in df.columns]

    target_col = "prognosis"
    for c in df.columns:
        if c.lower() == "prognosis":
            target_col = c
            break

    y = df[target_col].str.strip()

    # Deduplicate feature columns
    feature_cols = [c for c in df.columns if c != target_col]
    dedup: list[str] = []
    seen: set[str] = set()
    for c in feature_cols:
        base = c.split(".")[0] if "." in c and c.split(".")[-1].isdigit() else c
        if base not in seen:
            seen.add(base)
            dedup.append(c)

    X = df[dedup].astype(int)
    X.columns = [
        c.split(".")[0] if "." in c and c.split(".")[-1].isdigit() else c
        for c in X.columns
    ]

    # Align features
    for c in feature_names:
        if c not in X.columns:
            X[c] = 0
    X = X[feature_names]

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    dt = DecisionTreeClassifier(random_state=42)
    dt.fit(X.values, y_enc)

    return dt, le


# ── Evaluation Metrics ──────────────────────────────────────────────────

def _compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None,
    class_names: list[str],
) -> dict[str, Any]:
    """Compute comprehensive metrics for a model's predictions."""
    acc = accuracy_score(y_true, y_pred)
    f1_w = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_true, y_pred, average="weighted", zero_division=0)

    metrics: dict[str, Any] = {
        "accuracy": round(acc, 4),
        "f1_weighted": round(f1_w, 4),
        "f1_macro": round(f1_macro, 4),
        "precision_weighted": round(prec, 4),
        "recall_weighted": round(rec, 4),
    }

    # Top-k accuracy (if probabilities available)
    if y_proba is not None and y_proba.ndim == 2:
        for k in [3, 5]:
            if y_proba.shape[1] >= k:
                topk = top_k_accuracy_score(y_true, y_proba, k=k)
                metrics[f"top_{k}_accuracy"] = round(topk, 4)

    # Per-class report
    report = classification_report(
        y_true, y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    metrics["per_class"] = {
        name: {
            "precision": round(v["precision"], 4),
            "recall": round(v["recall"], 4),
            "f1": round(v["f1-score"], 4),
            "support": int(v["support"]),
        }
        for name, v in report.items()
        if name not in ("accuracy", "macro avg", "weighted avg")
    }

    return metrics


# ── Confusion Matrix Visualization ──────────────────────────────────────

def _save_confusion_matrices(
    y_true: np.ndarray,
    y_pred_dt: np.ndarray,
    y_pred_ens: np.ndarray,
    class_names: list[str],
) -> str | None:
    """Generate side-by-side confusion matrix plot."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import ConfusionMatrixDisplay
    except ImportError:
        print("  ⚠️  matplotlib not installed — skipping confusion matrix plot")
        return None

    fig, axes = plt.subplots(1, 2, figsize=(24, 10))

    # Limit labels if too many classes
    labels = class_names if len(class_names) <= 20 else None

    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred_dt, ax=axes[0],
        display_labels=labels,
        cmap="Blues",
        xticks_rotation=90,
        colorbar=False,
    )
    axes[0].set_title("Decision Tree (v1 Baseline)", fontsize=14)

    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred_ens, ax=axes[1],
        display_labels=labels,
        cmap="Greens",
        xticks_rotation=90,
        colorbar=False,
    )
    axes[1].set_title("Ensemble (v2 — XGB+RF+MLP)", fontsize=14)

    plt.tight_layout()
    path = str(ARTIFACTS_DIR / "confusion_matrix.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ── Main Evaluation ─────────────────────────────────────────────────────

def evaluate(data_dir: str) -> dict[str, Any]:
    """Run full model comparison.

    1. Load Testing.csv
    2. Train/load Decision Tree v1 baseline
    3. Load Ensemble v2
    4. Compute metrics for both
    5. Generate comparison report + confusion matrix

    Returns:
        Full evaluation report dictionary.
    """
    print("=" * 60)
    print("  Healthcare AI — Model Evaluation & Comparison")
    print("=" * 60)

    # ── Load Test Data ──
    print("\n📂 Loading test data…")
    X_test, y_test, feature_names = _load_data(data_dir)
    print(f"  Test set: {len(X_test)} samples, {len(feature_names)} features")

    # ── Decision Tree v1 Baseline ──
    print("\n🌲 Training Decision Tree v1 baseline…")
    dt_model, dt_le = _train_baseline_dt(data_dir, feature_names)
    y_test_enc_dt = dt_le.transform(y_test)

    y_pred_dt = dt_model.predict(X_test[feature_names].values)
    y_proba_dt = (
        dt_model.predict_proba(X_test[feature_names].values)
        if hasattr(dt_model, "predict_proba")
        else None
    )

    dt_metrics = _compute_metrics(y_test_enc_dt, y_pred_dt, y_proba_dt, list(dt_le.classes_))
    print(f"  DT Accuracy: {dt_metrics['accuracy']:.4f}")

    # ── Ensemble v2 ──
    ensemble_path = ARTIFACTS_DIR / "ensemble_model.joblib"
    if not ensemble_path.exists():
        print(f"\n❌ Ensemble model not found at {ensemble_path}")
        print("   Run train_ensemble.py first.")
        return {"error": "Ensemble model not found"}

    print(f"\n🤖 Loading Ensemble v2 from {ensemble_path}…")
    artifacts = joblib.load(ensemble_path)
    ens_model = artifacts["model"]
    ens_le = artifacts["label_encoder"]
    ens_features = artifacts["feature_names"]

    # Align test features with ensemble's expected features
    X_ens = X_test.copy()
    for c in ens_features:
        if c not in X_ens.columns:
            X_ens[c] = 0
    X_ens = X_ens[ens_features]

    y_test_enc_ens = ens_le.transform(y_test)
    y_pred_ens = ens_model.predict(X_ens.values)
    y_proba_ens = (
        ens_model.predict_proba(X_ens.values)
        if hasattr(ens_model, "predict_proba")
        else None
    )

    ens_metrics = _compute_metrics(y_test_enc_ens, y_pred_ens, y_proba_ens, list(ens_le.classes_))
    print(f"  Ensemble Accuracy: {ens_metrics['accuracy']:.4f}")

    # ── Per-estimator Scores ──
    estimator_scores: dict[str, float] = {}
    if hasattr(ens_model, "named_estimators_"):
        print("\n  Per-estimator breakdown:")
        for name, est in ens_model.named_estimators_.items():
            score = accuracy_score(y_test_enc_ens, est.predict(X_ens.values))
            estimator_scores[name] = round(score, 4)
            print(f"    {name}: {score:.4f}")

    # ── Improvement Analysis ──
    improvement = ens_metrics["accuracy"] - dt_metrics["accuracy"]
    f1_improvement = ens_metrics["f1_weighted"] - dt_metrics["f1_weighted"]

    print(f"\n📈 Improvement:")
    print(f"  Accuracy: {improvement:+.4f} ({improvement * 100:+.2f}%)")
    print(f"  F1 Score: {f1_improvement:+.4f} ({f1_improvement * 100:+.2f}%)")

    # ── Per-class Improvement Analysis ──
    improved_classes: list[dict[str, Any]] = []
    degraded_classes: list[dict[str, Any]] = []
    for cls in list(ens_le.classes_):
        dt_f1 = dt_metrics["per_class"].get(cls, {}).get("f1", 0)
        ens_f1 = ens_metrics["per_class"].get(cls, {}).get("f1", 0)
        delta = ens_f1 - dt_f1
        entry = {"class": cls, "dt_f1": dt_f1, "ens_f1": ens_f1, "delta": round(delta, 4)}
        if delta > 0.01:
            improved_classes.append(entry)
        elif delta < -0.01:
            degraded_classes.append(entry)

    improved_classes.sort(key=lambda x: x["delta"], reverse=True)
    degraded_classes.sort(key=lambda x: x["delta"])

    if improved_classes:
        print(f"\n  ✅ Improved on {len(improved_classes)} classes (top 5):")
        for c in improved_classes[:5]:
            print(f"    {c['class']}: {c['dt_f1']:.3f} → {c['ens_f1']:.3f} ({c['delta']:+.3f})")

    if degraded_classes:
        print(f"\n  ⚠️  Degraded on {len(degraded_classes)} classes:")
        for c in degraded_classes[:5]:
            print(f"    {c['class']}: {c['dt_f1']:.3f} → {c['ens_f1']:.3f} ({c['delta']:+.3f})")

    # ── Confusion Matrix Plot ──
    print("\n📊 Generating confusion matrix…")
    cm_path = _save_confusion_matrices(
        y_test_enc_ens, y_pred_dt, y_pred_ens, list(ens_le.classes_)
    )

    # ── Save Report ──
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    report = {
        "test_samples": len(X_test),
        "n_classes": len(ens_le.classes_),
        "decision_tree_v1": dt_metrics,
        "ensemble_v2": ens_metrics,
        "estimator_scores": estimator_scores,
        "improvement": {
            "accuracy_delta": round(improvement, 4),
            "f1_delta": round(f1_improvement, 4),
            "improved_classes": len(improved_classes),
            "degraded_classes": len(degraded_classes),
        },
        "top_improved_classes": improved_classes[:10],
        "degraded_classes": degraded_classes[:10],
        "confusion_matrix_path": cm_path,
    }

    report_path = ARTIFACTS_DIR / "evaluation_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n📋 Evaluation report saved to {report_path}")

    print("\n" + "=" * 60)
    print("  COMPARISON SUMMARY")
    print("-" * 60)
    print(f"  {'Metric':<25} {'DT v1':>10} {'Ensemble v2':>12} {'Δ':>8}")
    print("-" * 60)
    for metric in ["accuracy", "f1_weighted", "precision_weighted", "recall_weighted"]:
        dt_v = dt_metrics[metric]
        ens_v = ens_metrics[metric]
        delta = ens_v - dt_v
        print(f"  {metric:<25} {dt_v:>10.4f} {ens_v:>12.4f} {delta:>+8.4f}")

    if "top_3_accuracy" in ens_metrics:
        print(f"  {'top_3_accuracy':<25} {'N/A':>10} {ens_metrics['top_3_accuracy']:>12.4f}")
    if "top_5_accuracy" in ens_metrics:
        print(f"  {'top_5_accuracy':<25} {'N/A':>10} {ens_metrics['top_5_accuracy']:>12.4f}")

    print("=" * 60)

    return report


# ── CLI ─────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate and compare healthcare ML models")
    parser.add_argument(
        "--data-dir",
        default=str(Path(__file__).parent.parent / "chatbot" / "Data"),
        help="Path to directory containing Testing.csv",
    )
    args = parser.parse_args()
    evaluate(args.data_dir)


if __name__ == "__main__":
    main()
