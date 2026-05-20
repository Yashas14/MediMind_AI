"""
Ensemble model training — XGBoost + RandomForest + Neural Network.

Trains a VotingClassifier with soft voting over three base estimators
on the 132-feature binary symptom dataset (Training.csv).

Outputs:
  model_artifacts/ensemble_model.joblib  — serialised model + metadata
  model_artifacts/training_report.json   — cross-validation metrics

Usage:
  python train_ensemble.py --data-dir ../chatbot/Data

Requirements:
  pip install scikit-learn xgboost pandas numpy joblib
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None  # type: ignore[misc]
    print("⚠️  xgboost not installed — falling back to GradientBoosting")
    from sklearn.ensemble import GradientBoostingClassifier

# ── Constants ───────────────────────────────────────────────────────────
CANONICAL_SYMPTOMS: list[str] = [
    "itching", "skin_rash", "nodal_skin_eruptions", "continuous_sneezing",
    "shivering", "chills", "joint_pain", "stomach_pain", "acidity",
    "ulcers_on_tongue", "muscle_wasting", "vomiting", "burning_micturition",
    "spotting_urination", "fatigue", "weight_gain", "anxiety",
    "cold_hands_and_feets", "mood_swings", "weight_loss", "restlessness",
    "lethargy", "patches_in_throat", "irregular_sugar_level", "cough",
    "high_fever", "sunken_eyes", "breathlessness", "sweating", "dehydration",
    "indigestion", "headache", "yellowish_skin", "dark_urine", "nausea",
    "loss_of_appetite", "pain_behind_the_eyes", "back_pain", "constipation",
    "abdominal_pain", "diarrhoea", "mild_fever", "yellow_urine",
    "yellowing_of_eyes", "acute_liver_failure", "fluid_overload",
    "swelling_of_stomach", "swelled_lymph_nodes", "malaise",
    "blurred_and_distorted_vision", "phlegm", "throat_irritation",
    "redness_of_eyes", "sinus_pressure", "runny_nose", "congestion",
    "chest_pain", "weakness_in_limbs", "fast_heart_rate",
    "pain_during_bowel_movements", "pain_in_anal_region", "bloody_stool",
    "irritation_in_anus", "neck_pain", "dizziness", "cramps", "bruising",
    "obesity", "swollen_legs", "swollen_blood_vessels", "puffy_face_and_eyes",
    "enlarged_thyroid", "brittle_nails", "swollen_extremeties",
    "excessive_hunger", "extra_marital_contacts", "drying_and_tingling_lips",
    "slurred_speech", "knee_pain", "hip_joint_pain", "muscle_weakness",
    "stiff_neck", "swelling_joints", "movement_stiffness",
    "spinning_movements", "loss_of_balance", "unsteadiness",
    "weakness_of_one_body_side", "loss_of_smell", "bladder_discomfort",
    "foul_smell_of_urine", "continuous_feel_of_urine", "passage_of_gases",
    "internal_itching", "toxic_look_(typhos)", "depression", "irritability",
    "muscle_pain", "altered_sensorium", "red_spots_over_body", "belly_pain",
    "abnormal_menstruation", "dischromic_patches", "watering_from_eyes",
    "increased_appetite", "polyuria", "family_history", "mucoid_sputum",
    "rusty_sputum", "lack_of_concentration", "visual_disturbances",
    "receiving_blood_transfusion", "receiving_unsterile_injections", "coma",
    "stomach_bleeding", "distention_of_abdomen",
    "history_of_alcohol_consumption", "blood_in_sputum",
    "prominent_veins_on_calf", "palpitations", "painful_walking",
    "pus_filled_pimples", "blackheads", "scurring", "skin_peeling",
    "silver_like_dusting", "small_dents_in_nails", "inflammatory_nails",
    "blister", "red_sore_around_nose", "yellow_crust_ooze",
]

ARTIFACTS_DIR = Path(__file__).parent / "model_artifacts"


# ── Data Loading ────────────────────────────────────────────────────────

def load_training_data(data_dir: str) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Load Training.csv, handling the duplicate 'fluid_overload' column.

    The raw CSV has 133 columns (132 symptoms + prognosis), but two
    columns share the name 'fluid_overload'. We keep the first one and
    drop the duplicate to align with the 132-length CANONICAL_SYMPTOMS list.

    Returns:
        (X, y, feature_names) — feature matrix, labels, and ordered feature names.
    """
    csv_path = os.path.join(data_dir, "Training.csv")
    print(f"📂 Loading training data from {csv_path}")

    # Read with pandas — duplicates get auto-suffixed (.1, .2, …)
    df = pd.read_csv(csv_path)

    # Clean up column names: strip whitespace and normalise spaces
    df.columns = [c.strip().replace(" ", "_") for c in df.columns]

    # The last column is 'prognosis' (the target)
    target_col = "prognosis"
    if target_col not in df.columns:
        # Try case-insensitive lookup
        for c in df.columns:
            if c.lower() == "prognosis":
                target_col = c
                break

    y = df[target_col].str.strip()

    # Drop the target and any duplicate columns (e.g. 'fluid_overload.1')
    feature_cols = [c for c in df.columns if c != target_col]

    # Remove auto-generated duplicate suffixes (.1, .2, etc.)
    dedup_cols: list[str] = []
    seen: set[str] = set()
    for c in feature_cols:
        base = c.split(".")[0] if "." in c and c.split(".")[-1].isdigit() else c
        if base not in seen:
            seen.add(base)
            dedup_cols.append(c)

    X = df[dedup_cols].astype(int)
    # Rename back to base names (strip .0 suffixes if any)
    X.columns = [c.split(".")[0] if "." in c and c.split(".")[-1].isdigit() else c for c in X.columns]

    feature_names = list(X.columns)

    print(f"  ✅ Loaded {len(X)} samples, {len(feature_names)} features, "
          f"{y.nunique()} classes")

    return X, y, feature_names


def load_testing_data(data_dir: str, feature_names: list[str]) -> tuple[pd.DataFrame, pd.Series]:
    """Load Testing.csv with the same column handling."""
    csv_path = os.path.join(data_dir, "Testing.csv")
    if not os.path.exists(csv_path):
        print("  ⚠️  Testing.csv not found — will use cross-validation only")
        return pd.DataFrame(), pd.Series(dtype=str)

    df = pd.read_csv(csv_path)
    df.columns = [c.strip().replace(" ", "_") for c in df.columns]

    target_col = "prognosis"
    for c in df.columns:
        if c.lower() == "prognosis":
            target_col = c
            break

    y = df[target_col].str.strip()

    # Use only the feature columns we trained on
    avail = [c for c in feature_names if c in df.columns]
    X = df[avail].astype(int)

    # Fill any missing columns with 0
    for c in feature_names:
        if c not in X.columns:
            X[c] = 0
    X = X[feature_names]

    print(f"  ✅ Testing set: {len(X)} samples")
    return X, y


# ── Model Building ──────────────────────────────────────────────────────

def build_ensemble(n_classes: int) -> VotingClassifier:
    """Build a VotingClassifier with XGBoost, RandomForest, and MLP.

    Uses soft voting (probability-based) so the ensemble can produce
    calibrated confidence scores for the diagnosis agent.
    """
    # ── Estimator 1: XGBoost / GradientBoosting ──
    if XGBClassifier is not None:
        xgb = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            use_label_encoder=False,
            eval_metric="mlogloss",
            n_jobs=-1,
        )
    else:
        xgb = GradientBoostingClassifier(
            n_estimators=150,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42,
        )

    # ── Estimator 2: Random Forest ──
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1,
    )

    # ── Estimator 3: Neural Network (MLP) ──
    mlp = MLPClassifier(
        hidden_layer_sizes=(256, 128, 64),
        activation="relu",
        solver="adam",
        alpha=0.001,
        batch_size=64,
        learning_rate="adaptive",
        learning_rate_init=0.001,
        max_iter=300,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=15,
    )

    # ── Voting Ensemble (soft) ──
    ensemble = VotingClassifier(
        estimators=[("xgb", xgb), ("rf", rf), ("mlp", mlp)],
        voting="soft",
        weights=[2, 2, 1],  # Slight emphasis on tree models
        n_jobs=-1,
    )

    return ensemble


# ── Training ────────────────────────────────────────────────────────────

def train_and_evaluate(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame | None,
    y_test: pd.Series | None,
    feature_names: list[str],
) -> dict[str, Any]:
    """Train the ensemble and compute comprehensive metrics.

    Steps:
    1. Encode labels
    2. Run stratified 5-fold cross-validation
    3. Fit on full training set
    4. Evaluate on held-out test set (if available)
    5. Serialize model + artifacts

    Returns:
        Report dictionary with all metrics and file paths.
    """
    t0 = time.time()

    # ── Label Encoding ──
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    n_classes = len(le.classes_)
    print(f"\n🏷️  {n_classes} disease classes: {list(le.classes_[:10])}…")

    # ── Build Ensemble ──
    ensemble = build_ensemble(n_classes)

    # ── Cross-Validation ──
    print("\n🔄 Running 5-fold stratified cross-validation…")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(ensemble, X_train, y_train_enc, cv=cv,
                                scoring="accuracy", n_jobs=-1)
    print(f"  CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"  Per-fold:    {[f'{s:.4f}' for s in cv_scores]}")

    # ── Full Training ──
    print("\n🚀 Training ensemble on full training set…")
    ensemble.fit(X_train.values, y_train_enc)
    train_time = time.time() - t0
    print(f"  Training completed in {train_time:.1f}s")

    # ── Training Set Accuracy ──
    y_train_pred = ensemble.predict(X_train.values)
    train_acc = accuracy_score(y_train_enc, y_train_pred)
    print(f"  Training accuracy: {train_acc:.4f}")

    # ── Test Set Evaluation ──
    test_metrics: dict[str, Any] = {}
    if X_test is not None and len(X_test) > 0 and y_test is not None:
        y_test_enc = le.transform(y_test)
        y_test_pred = ensemble.predict(X_test.values)

        test_acc = accuracy_score(y_test_enc, y_test_pred)
        test_f1 = f1_score(y_test_enc, y_test_pred, average="weighted")
        test_prec = precision_score(y_test_enc, y_test_pred, average="weighted")
        test_rec = recall_score(y_test_enc, y_test_pred, average="weighted")

        print(f"\n📊 Test Set Metrics:")
        print(f"  Accuracy:  {test_acc:.4f}")
        print(f"  F1 (wtd):  {test_f1:.4f}")
        print(f"  Precision: {test_prec:.4f}")
        print(f"  Recall:    {test_rec:.4f}")

        test_metrics = {
            "accuracy": round(test_acc, 4),
            "f1_weighted": round(test_f1, 4),
            "precision_weighted": round(test_prec, 4),
            "recall_weighted": round(test_rec, 4),
            "classification_report": classification_report(
                y_test_enc, y_test_pred,
                target_names=le.classes_,
                output_dict=True,
            ),
        }

    # ── Per-estimator accuracy (on test set if available, else training) ──
    estimator_scores: dict[str, float] = {}
    eval_X = X_test.values if X_test is not None and len(X_test) > 0 else X_train.values
    eval_y = le.transform(y_test) if y_test is not None and len(y_test) > 0 else y_train_enc
    for name, estimator in ensemble.named_estimators_.items():
        score = accuracy_score(eval_y, estimator.predict(eval_X))
        estimator_scores[name] = round(score, 4)
        print(f"  {name}: {score:.4f}")

    # ── Save Artifacts ──
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    model_path = ARTIFACTS_DIR / "ensemble_model.joblib"
    artifacts = {
        "model": ensemble,
        "label_encoder": le,
        "feature_names": feature_names,
        "n_classes": n_classes,
        "classes": list(le.classes_),
        "canonical_symptoms": CANONICAL_SYMPTOMS,
    }
    joblib.dump(artifacts, model_path, compress=3)
    model_size_mb = model_path.stat().st_size / (1024 * 1024)
    print(f"\n💾 Model saved to {model_path} ({model_size_mb:.1f} MB)")

    # ── Build Report ──
    report = {
        "model_type": "VotingClassifier(XGBoost+RandomForest+MLP)",
        "n_features": len(feature_names),
        "n_classes": n_classes,
        "n_training_samples": len(X_train),
        "n_test_samples": len(X_test) if X_test is not None else 0,
        "cv_accuracy_mean": round(cv_scores.mean(), 4),
        "cv_accuracy_std": round(cv_scores.std(), 4),
        "cv_scores": [round(s, 4) for s in cv_scores],
        "train_accuracy": round(train_acc, 4),
        "test_metrics": test_metrics,
        "estimator_scores": estimator_scores,
        "training_time_seconds": round(train_time, 1),
        "model_path": str(model_path),
        "model_size_mb": round(model_size_mb, 1),
    }

    report_path = ARTIFACTS_DIR / "training_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"📋 Report saved to {report_path}")

    return report


# ── CLI ─────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train healthcare ensemble model (XGBoost + RF + MLP)"
    )
    parser.add_argument(
        "--data-dir",
        default=str(Path(__file__).parent.parent / "chatbot" / "Data"),
        help="Path to directory containing Training.csv and Testing.csv",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  Healthcare AI — Ensemble Model Training")
    print("=" * 60)

    # Load data
    X_train, y_train, feature_names = load_training_data(args.data_dir)
    X_test, y_test = load_testing_data(args.data_dir, feature_names)

    # Validate feature alignment with CANONICAL_SYMPTOMS
    if len(feature_names) != len(CANONICAL_SYMPTOMS):
        print(f"\n⚠️  Feature count mismatch: CSV has {len(feature_names)}, "
              f"expected {len(CANONICAL_SYMPTOMS)}. Aligning…")
        # Use intersection
        common = [s for s in CANONICAL_SYMPTOMS if s in feature_names]
        X_train = X_train[[c for c in common if c in X_train.columns]]
        if len(X_test) > 0:
            X_test = X_test[[c for c in common if c in X_test.columns]]
        feature_names = common
        print(f"  Using {len(feature_names)} aligned features")
    else:
        print(f"  ✅ Feature alignment verified: {len(feature_names)} features")

    # Train
    report = train_and_evaluate(
        X_train, y_train,
        X_test if len(X_test) > 0 else None,
        y_test if len(y_test) > 0 else None,
        feature_names,
    )

    print("\n" + "=" * 60)
    print(f"  ✅ Training complete — {report['model_type']}")
    print(f"  CV Accuracy: {report['cv_accuracy_mean']:.2%} ± {report['cv_accuracy_std']:.2%}")
    if report["test_metrics"]:
        print(f"  Test Accuracy: {report['test_metrics']['accuracy']:.2%}")
    print("=" * 60)


if __name__ == "__main__":
    main()
