from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

EVALUATION_DIR = (
    BASE_DIR /
    "data" /
    "processed" /
    "evaluation" /
    "final"
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm_file = (
    EVALUATION_DIR /
    "svm_confusion_matrix.csv"
)

cm = pd.read_csv(
    cm_file,
    index_col=0
)


plt.figure(
    figsize=(7, 6)
)

plt.imshow(
    cm.values,
    interpolation="nearest"
)

plt.title(
    "Optimized Linear SVM - Confusion Matrix"
)

plt.xlabel(
    "Predicted Label"
)

plt.ylabel(
    "Actual Label"
)

plt.xticks(
    range(2),
    [
        "Fake",
        "Real"
    ]
)

plt.yticks(
    range(2),
    [
        "Fake",
        "Real"
    ]
)

for i in range(2):

    for j in range(2):

        plt.text(
            j,
            i,
            cm.iloc[i, j],
            ha="center",
            va="center"
        )

plt.tight_layout()

plt.savefig(
    EVALUATION_DIR /
    "svm_confusion_matrix.png",
    dpi=200
)

plt.close()


# ============================================================
# ROC CURVE
# ============================================================

roc_file = (
    EVALUATION_DIR /
    "svm_roc_curve.csv"
)

roc_df = pd.read_csv(
    roc_file
)


# Read final ROC-AUC
results = pd.read_csv(
    EVALUATION_DIR /
    "svm_final_test_results.csv"
)

roc_auc = results.loc[
    0,
    "roc_auc"
]


plt.figure(
    figsize=(7, 6)
)

plt.plot(
    roc_df["false_positive_rate"],
    roc_df["true_positive_rate"],
    label=f"ROC-AUC = {roc_auc:.4f}"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.xlabel(
    "False Positive Rate"
)

plt.ylabel(
    "True Positive Rate"
)

plt.title(
    "Optimized Linear SVM - ROC Curve"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    EVALUATION_DIR /
    "svm_roc_curve.png",
    dpi=200
)

plt.close()


# ============================================================
# PRECISION-RECALL CURVE
# ============================================================

pr_file = (
    EVALUATION_DIR /
    "svm_precision_recall_curve.csv"
)

pr_df = pd.read_csv(
    pr_file
)


plt.figure(
    figsize=(7, 6)
)

plt.plot(
    pr_df["recall"],
    pr_df["precision"]
)

plt.xlabel(
    "Recall"
)

plt.ylabel(
    "Precision"
)

plt.title(
    "Optimized Linear SVM - Precision-Recall Curve"
)

plt.tight_layout()

plt.savefig(
    EVALUATION_DIR /
    "svm_precision_recall_curve.png",
    dpi=200
)

plt.close()


# ============================================================
# METRIC BAR CHART
# ============================================================

metrics = [
    "accuracy",
    "precision",
    "recall",
    "f1_score",
    "roc_auc"
]

values = [
    results.loc[0, metric]
    for metric in metrics
]


plt.figure(
    figsize=(8, 6)
)

plt.bar(
    metrics,
    values
)

plt.ylim(
    0,
    1
)

plt.ylabel(
    "Score"
)

plt.title(
    "Optimized Linear SVM - Final Test Metrics"
)

plt.xticks(
    rotation=20
)

for index, value in enumerate(values):

    plt.text(
        index,
        value + 0.02,
        f"{value:.3f}",
        ha="center"
    )

plt.tight_layout()

plt.savefig(
    EVALUATION_DIR /
    "svm_final_metrics.png",
    dpi=200
)

plt.close()


# ============================================================
# DONE
# ============================================================

print("=" * 70)
print("EVALUATION PLOTS CREATED")
print("=" * 70)

print("\nSaved to:")

print(EVALUATION_DIR)
