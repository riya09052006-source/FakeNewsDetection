from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import load_npz

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

FEATURE_DIR = BASE_DIR / "data" / "processed" / "features"
MODEL_DIR = BASE_DIR / "models" / "optimized"
EVALUATION_DIR = BASE_DIR / "data" / "processed" / "evaluation"

EVALUATION_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 70)
print("PHASE 6 - OPTIMIZED MODEL VALIDATION")
print("=" * 70)

model_path = MODEL_DIR / "linear_svm_optimized.joblib"

model = joblib.load(model_path)


# ============================================================
# LOAD VALIDATION DATA
# ============================================================

X_validation = load_npz(
    FEATURE_DIR / "X_validation.npz"
)

y_validation = joblib.load(
    FEATURE_DIR / "y_validation.joblib"
)


print("\nValidation samples:", f"{X_validation.shape[0]:,}")
print("Features:", f"{X_validation.shape[1]:,}")


# ============================================================
# PREDICTION
# ============================================================

print("\nRunning predictions...")

start_time = time.time()

predictions = model.predict(
    X_validation
)

decision_scores = model.decision_function(
    X_validation
)

inference_time = time.time() - start_time


# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(
    y_validation,
    predictions,
)

precision = precision_score(
    y_validation,
    predictions,
    zero_division=0,
)

recall = recall_score(
    y_validation,
    predictions,
    zero_division=0,
)

f1 = f1_score(
    y_validation,
    predictions,
    zero_division=0,
)

roc_auc = roc_auc_score(
    y_validation,
    decision_scores,
)


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n" + "=" * 70)
print("OPTIMIZED MODEL RESULTS")
print("=" * 70)

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
print(f"ROC-AUC  : {roc_auc:.4f}")

print(f"\nInference time: {inference_time:.4f} seconds")

print("\n" + "=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

print(
    classification_report(
        y_validation,
        predictions,
        target_names=["Fake", "Real"],
        digits=4,
        zero_division=0,
    )
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_validation,
    predictions,
)

print("\n" + "=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

print(cm)


# ============================================================
# SAVE RESULTS
# ============================================================

results = pd.DataFrame(
    [
        {
            "model": "Optimized Linear SVM",
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "roc_auc": roc_auc,
            "inference_time_seconds": inference_time,
        }
    ]
)

output_path = (
    EVALUATION_DIR /
    "phase6_optimized_validation_results.csv"
)

results.to_csv(
    output_path,
    index=False,
)

print("\nValidation results saved to:")
print(output_path)


# ============================================================
# SAVE CONFUSION MATRIX
# ============================================================

cm_path = (
    EVALUATION_DIR /
    "phase6_optimized_confusion_matrix.csv"
)

pd.DataFrame(
    cm,
    index=["Actual Fake", "Actual Real"],
    columns=["Predicted Fake", "Predicted Real"],
).to_csv(
    cm_path
)

print("\nConfusion matrix saved to:")
print(cm_path)


print("\n" + "=" * 70)
print("VALIDATION EVALUATION COMPLETED")
print("=" * 70)
