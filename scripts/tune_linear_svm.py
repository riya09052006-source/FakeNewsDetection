from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd

from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

FEATURE_DIR = BASE_DIR / "data" / "processed" / "features"
MODEL_DIR = BASE_DIR / "models" / "optimized"
EVALUATION_DIR = BASE_DIR / "data" / "processed" / "evaluation"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
EVALUATION_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("PHASE 6 - LINEAR SVM HYPERPARAMETER TUNING")
print("=" * 70)

print("\nLoading training features...")

X_train = __import__("scipy.sparse").sparse.load_npz(
    FEATURE_DIR / "X_train.npz"
)

y_train = joblib.load(FEATURE_DIR / "y_train.joblib")

print(f"Training samples : {X_train.shape[0]:,}")
print(f"Features         : {X_train.shape[1]:,}")


# ============================================================
# HYPERPARAMETERS
# ============================================================

C_VALUES = [
    0.25,
    0.5,
    1.0,
    2.0,
    4.0,
]

CV_FOLDS = 3


# ============================================================
# CROSS VALIDATION
# ============================================================

cv = StratifiedKFold(
    n_splits=CV_FOLDS,
    shuffle=True,
    random_state=42,
)


results = []


print("\n" + "=" * 70)
print("STARTING CROSS-VALIDATION")
print("=" * 70)

print("\nC values:", C_VALUES)
print(f"Cross-validation folds: {CV_FOLDS}")


for c_value in C_VALUES:

    print("\n" + "-" * 70)
    print(f"Testing LinearSVC with C = {c_value}")
    print("-" * 70)

    model = LinearSVC(
        C=c_value,
        class_weight="balanced",
        max_iter=5000,
        random_state=42,
    )

    start_time = time.time()

    scores = cross_validate(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring={
            "accuracy": "accuracy",
            "precision": "precision",
            "recall": "recall",
            "f1": "f1",
        },
        n_jobs=-1,
        return_train_score=False,
    )

    elapsed = time.time() - start_time

    accuracy = scores["test_accuracy"].mean()
    precision = scores["test_precision"].mean()
    recall = scores["test_recall"].mean()
    f1 = scores["test_f1"].mean()

    accuracy_std = scores["test_accuracy"].std()
    f1_std = scores["test_f1"].std()

    print(f"Accuracy : {accuracy:.4f} (+/- {accuracy_std:.4f})")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f} (+/- {f1_std:.4f})")
    print(f"Time     : {elapsed:.2f} seconds")

    results.append(
        {
            "model": "LinearSVC",
            "C": c_value,
            "cv_accuracy_mean": accuracy,
            "cv_accuracy_std": accuracy_std,
            "cv_precision_mean": precision,
            "cv_recall_mean": recall,
            "cv_f1_mean": f1,
            "cv_f1_std": f1_std,
            "training_time_seconds": elapsed,
        }
    )


# ============================================================
# RESULTS
# ============================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by=["cv_f1_mean", "cv_accuracy_mean"],
    ascending=False,
).reset_index(drop=True)


print("\n" + "=" * 70)
print("CROSS-VALIDATION RESULTS")
print("=" * 70)

print(
    results_df[
        [
            "C",
            "cv_accuracy_mean",
            "cv_precision_mean",
            "cv_recall_mean",
            "cv_f1_mean",
        ]
    ].to_string(index=False)
)


# ============================================================
# BEST PARAMETER
# ============================================================

best_row = results_df.iloc[0]

best_c = float(best_row["C"])

print("\n" + "=" * 70)
print("BEST CONFIGURATION")
print("=" * 70)

print(f"Best C       : {best_c}")
print(f"CV Accuracy  : {best_row['cv_accuracy_mean']:.4f}")
print(f"CV Precision : {best_row['cv_precision_mean']:.4f}")
print(f"CV Recall    : {best_row['cv_recall_mean']:.4f}")
print(f"CV F1        : {best_row['cv_f1_mean']:.4f}")


# ============================================================
# SAVE RESULTS
# ============================================================

results_path = EVALUATION_DIR / "phase6_svm_tuning_results.csv"

results_df.to_csv(
    results_path,
    index=False,
)

print(f"\nResults saved to:")
print(results_path)


# ============================================================
# TRAIN FINAL OPTIMIZED MODEL
# ============================================================

print("\n" + "=" * 70)
print("TRAINING OPTIMIZED LINEAR SVM")
print("=" * 70)

final_model = LinearSVC(
    C=best_c,
    class_weight="balanced",
    max_iter=5000,
    random_state=42,
)

start_time = time.time()

final_model.fit(
    X_train,
    y_train,
)

training_time = time.time() - start_time


# ============================================================
# SAVE MODEL
# ============================================================

model_path = MODEL_DIR / "linear_svm_optimized.joblib"

joblib.dump(
    final_model,
    model_path,
)


# ============================================================
# SAVE METADATA
# ============================================================

metadata = {
    "model": "LinearSVC",
    "C": best_c,
    "class_weight": "balanced",
    "max_iter": 5000,
    "random_state": 42,
    "cv_folds": CV_FOLDS,
    "cv_accuracy": float(best_row["cv_accuracy_mean"]),
    "cv_precision": float(best_row["cv_precision_mean"]),
    "cv_recall": float(best_row["cv_recall_mean"]),
    "cv_f1": float(best_row["cv_f1_mean"]),
    "training_time_seconds": training_time,
    "feature_type": "Word TF-IDF + Character TF-IDF",
}


metadata_path = MODEL_DIR / "linear_svm_optimized_metadata.joblib"

joblib.dump(
    metadata,
    metadata_path,
)


# ============================================================
# FINAL MESSAGE
# ============================================================

print("\n" + "=" * 70)
print("PHASE 6.1 COMPLETED")
print("=" * 70)

print(f"\nOptimized model saved:")
print(model_path)

print(f"\nMetadata saved:")
print(metadata_path)

print(f"\nTraining time: {training_time:.2f} seconds")

print("\nNext step:")
print("Evaluate the optimized model on the validation dataset.")
