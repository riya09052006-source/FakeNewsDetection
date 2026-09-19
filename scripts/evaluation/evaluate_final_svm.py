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
    roc_curve,
    precision_recall_curve,
    average_precision_score,
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

FEATURE_DIR = (
    BASE_DIR /
    "data" /
    "processed" /
    "features"
)

MODEL_PATH = (
    BASE_DIR /
    "models" /
    "optimized" /
    "linear_svm_optimized.joblib"
)

TEST_FEATURE_PATH = (
    FEATURE_DIR /
    "X_test.npz"
)

TEST_LABEL_PATH = (
    FEATURE_DIR /
    "y_test.joblib"
)

OUTPUT_DIR = (
    BASE_DIR /
    "data" /
    "processed" /
    "evaluation" /
    "final"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# HEADER
# ============================================================

print("=" * 75)
print("PHASE 8 - FINAL TEST EVALUATION")
print("OPTIMIZED LINEAR SVM")
print("=" * 75)


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading optimized model...")

model = joblib.load(
    MODEL_PATH
)


# ============================================================
# LOAD TEST DATA
# ============================================================

print("Loading untouched test data...")

X_test = load_npz(
    TEST_FEATURE_PATH
)

y_test = joblib.load(
    TEST_LABEL_PATH
)

print("\nTest samples :", f"{X_test.shape[0]:,}")
print("Features     :", f"{X_test.shape[1]:,}")


# ============================================================
# PREDICTION
# ============================================================

print("\nRunning final predictions...")

start_time = time.perf_counter()

y_pred = model.predict(
    X_test
)

decision_scores = model.decision_function(
    X_test
)

inference_time = (
    time.perf_counter() -
    start_time
)


# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_test,
    decision_scores
)

average_precision = average_precision_score(
    y_test,
    decision_scores
)


# ============================================================
# PRINT METRICS
# ============================================================

print("\n" + "=" * 75)
print("FINAL TEST RESULTS")
print("=" * 75)

print(f"Accuracy            : {accuracy:.4f}")
print(f"Precision           : {precision:.4f}")
print(f"Recall              : {recall:.4f}")
print(f"F1 Score            : {f1:.4f}")
print(f"ROC-AUC             : {roc_auc:.4f}")
print(f"Average Precision   : {average_precision:.4f}")
print(f"Inference Time      : {inference_time:.4f} sec")


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 75)
print("CLASSIFICATION REPORT")
print("=" * 75)

report = classification_report(
    y_test,
    y_pred,
    target_names=[
        "Fake",
        "Real"
    ],
    digits=4,
    zero_division=0
)

print(report)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred
)

print("\n" + "=" * 75)
print("CONFUSION MATRIX")
print("=" * 75)

print(cm)


# ============================================================
# SAVE CONFUSION MATRIX
# ============================================================

cm_df = pd.DataFrame(
    cm,
    index=[
        "Actual Fake",
        "Actual Real"
    ],
    columns=[
        "Predicted Fake",
        "Predicted Real"
    ]
)

cm_df.to_csv(
    OUTPUT_DIR /
    "svm_confusion_matrix.csv"
)


# ============================================================
# ROC CURVE DATA
# ============================================================

fpr, tpr, roc_thresholds = roc_curve(
    y_test,
    decision_scores
)

roc_df = pd.DataFrame(
    {
        "false_positive_rate": fpr,
        "true_positive_rate": tpr,
        "threshold": roc_thresholds,
    }
)

roc_df.to_csv(
    OUTPUT_DIR /
    "svm_roc_curve.csv",
    index=False
)


# ============================================================
# PRECISION-RECALL CURVE
# ============================================================

precision_values, recall_values, pr_thresholds = (
    precision_recall_curve(
        y_test,
        decision_scores
    )
)

pr_df = pd.DataFrame(
    {
        "precision": precision_values,
        "recall": recall_values,
    }
)

pr_df.to_csv(
    OUTPUT_DIR /
    "svm_precision_recall_curve.csv",
    index=False
)


# ============================================================
# FINAL SUMMARY
# ============================================================

summary = pd.DataFrame(
    [
        {
            "model": "Optimized Linear SVM",
            "dataset": "Untouched Test Set",
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "roc_auc": roc_auc,
            "average_precision": average_precision,
            "inference_time_seconds": inference_time,
            "test_samples": len(y_test),
        }
    ]
)

summary.to_csv(
    OUTPUT_DIR /
    "svm_final_test_results.csv",
    index=False
)


# ============================================================
# SAVE TEXT REPORT
# ============================================================

with open(
    OUTPUT_DIR /
    "svm_classification_report.txt",
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "Optimized Linear SVM - Final Test Evaluation\n"
    )

    file.write(
        "=" * 60 +
        "\n\n"
    )

    file.write(
        f"Accuracy: {accuracy:.4f}\n"
    )

    file.write(
        f"Precision: {precision:.4f}\n"
    )

    file.write(
        f"Recall: {recall:.4f}\n"
    )

    file.write(
        f"F1 Score: {f1:.4f}\n"
    )

    file.write(
        f"ROC-AUC: {roc_auc:.4f}\n"
    )

    file.write(
        f"Average Precision: "
        f"{average_precision:.4f}\n"
    )

    file.write(
        f"Inference Time: "
        f"{inference_time:.4f} seconds\n\n"
    )

    file.write(
        "Classification Report\n"
    )

    file.write(
        report
    )

    file.write(
        "\nConfusion Matrix\n"
    )

    file.write(
        str(cm)
    )


# ============================================================
# COMPLETION
# ============================================================

print("\nFiles saved to:")

print(OUTPUT_DIR)

print("\nFinal SVM test evaluation completed.")
