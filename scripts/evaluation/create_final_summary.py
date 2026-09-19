from pathlib import Path

import joblib
import pandas as pd


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

MODEL_DIR = (
    BASE_DIR /
    "models" /
    "optimized"
)


# ============================================================
# LOAD FINAL RESULTS
# ============================================================

results_file = (
    EVALUATION_DIR /
    "svm_final_test_results.csv"
)

results = pd.read_csv(
    results_file
)

result = results.iloc[0]


# ============================================================
# LOAD MODEL METADATA
# ============================================================

metadata_file = (
    MODEL_DIR /
    "linear_svm_optimized_metadata.joblib"
)

metadata = joblib.load(
    metadata_file
)


# ============================================================
# BUILD REPORT
# ============================================================

report = f"""
============================================================
FAKE NEWS DETECTION
FINAL MODEL EVALUATION SUMMARY
============================================================

FINAL MODEL
-----------
Model: Optimized Linear SVM
Features: Word TF-IDF + Character TF-IDF

OPTIMIZATION
------------
Best C: {metadata["C"]}
Cross-validation folds: {metadata["cv_folds"]}

CROSS-VALIDATION
----------------
Accuracy: {metadata["cv_accuracy"]:.4f}
Precision: {metadata["cv_precision"]:.4f}
Recall: {metadata["cv_recall"]:.4f}
F1 Score: {metadata["cv_f1"]:.4f}

FINAL TEST SET
--------------
Dataset: Untouched Test Set
Samples: {int(result["test_samples"])}

Accuracy: {result["accuracy"]:.4f}
Precision: {result["precision"]:.4f}
Recall: {result["recall"]:.4f}
F1 Score: {result["f1_score"]:.4f}
ROC-AUC: {result["roc_auc"]:.4f}
Average Precision: {result["average_precision"]:.4f}

Inference Time
--------------
{result["inference_time_seconds"]:.4f} seconds

IMPORTANT LIMITATION
--------------------
The model predicts patterns learned from the training dataset.
A prediction of "Fake" or "Real" should not be treated as
independent verification of factual truth.

============================================================
END OF REPORT
============================================================
"""


# ============================================================
# SAVE
# ============================================================

output_file = (
    EVALUATION_DIR /
    "FINAL_MODEL_REPORT.txt"
)

with open(
    output_file,
    "w",
    encoding="utf-8"
) as file:

    file.write(report)


print(report)

print(
    f"\nFinal report saved to:\n{output_file}"
)
