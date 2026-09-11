import time
from pathlib import Path

import joblib
import pandas as pd
import matplotlib.pyplot as plt

from scipy.sparse import load_npz

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)


# ============================================================
# PATH CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

FEATURE_DIR = (
    BASE_DIR
    / "data"
    / "processed"
    / "features"
)

MODEL_DIR = (
    BASE_DIR
    / "models"
    / "baseline"
)

RESULT_DIR = (
    BASE_DIR
    / "data"
    / "processed"
    / "evaluation"
)

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD VALIDATION DATA
# ============================================================

print("=" * 70)
print("BASELINE MODEL EVALUATION")
print("=" * 70)


X_validation = load_npz(
    FEATURE_DIR / "X_validation.npz"
)

y_validation = joblib.load(
    FEATURE_DIR / "y_validation.joblib"
)


# ============================================================
# MODELS
# ============================================================

model_names = [
    "naive_bayes",
    "logistic_regression",
    "linear_svm"
]


results = []


# ============================================================
# EVALUATION
# ============================================================

for model_name in model_names:

    print("\n" + "=" * 70)
    print(f"EVALUATING: {model_name.upper()}")
    print("=" * 70)

    model_path = (
        MODEL_DIR
        / f"{model_name}.joblib"
    )

    model = joblib.load(
        model_path
    )

    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    start_time = time.perf_counter()

    predictions = model.predict(
        X_validation
    )

    end_time = time.perf_counter()

    inference_time = (
        end_time - start_time
    )

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_validation,
        predictions
    )

    precision = precision_score(
        y_validation,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_validation,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_validation,
        predictions,
        zero_division=0
    )

    # --------------------------------------------------------
    # ROC-AUC
    # --------------------------------------------------------

    roc_auc = None

    if hasattr(model, "predict_proba"):

        probabilities = model.predict_proba(
            X_validation
        )[:, 1]

        roc_auc = roc_auc_score(
            y_validation,
            probabilities
        )

    elif hasattr(model, "decision_function"):

        scores = model.decision_function(
            X_validation
        )

        roc_auc = roc_auc_score(
            y_validation,
            scores
        )

    # --------------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------------

    cm = confusion_matrix(
        y_validation,
        predictions
    )

    print("\nAccuracy:")
    print(f"{accuracy:.4f}")

    print("\nPrecision:")
    print(f"{precision:.4f}")

    print("\nRecall:")
    print(f"{recall:.4f}")

    print("\nF1 Score:")
    print(f"{f1:.4f}")

    print("\nROC-AUC:")
    print(
        f"{roc_auc:.4f}"
        if roc_auc is not None
        else "N/A"
    )

    print("\nInference time:")
    print(
        f"{inference_time:.4f} seconds"
    )

    print("\nClassification Report:")

    print(
        classification_report(
            y_validation,
            predictions,
            target_names=[
                "Fake",
                "Real"
            ],
            zero_division=0
        )
    )

    print("\nConfusion Matrix:")

    print(cm)

    # --------------------------------------------------------
    # SAVE CONFUSION MATRIX
    # --------------------------------------------------------

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=[
            "Fake",
            "Real"
        ]
    )

    display.plot()

    plt.title(
        f"{model_name} - Confusion Matrix"
    )

    plt.tight_layout()

    figure_path = (
        RESULT_DIR
        / f"{model_name}_confusion_matrix.png"
    )

    plt.savefig(
        figure_path,
        dpi=150
    )

    plt.close()

    # --------------------------------------------------------
    # STORE RESULTS
    # --------------------------------------------------------

    results.append({
        "model": model_name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc,
        "inference_time_seconds": inference_time
    })


# ============================================================
# RESULTS DATAFRAME
# ============================================================

results_df = pd.DataFrame(
    results
)


# ============================================================
# SORT BY F1
# ============================================================

results_df = results_df.sort_values(
    by="f1_score",
    ascending=False
)


# ============================================================
# DISPLAY
# ============================================================

print("\n" + "=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

print(
    results_df.to_string(
        index=False
    )
)


# ============================================================
# SAVE RESULTS
# ============================================================

results_file = (
    RESULT_DIR
    / "baseline_results.csv"
)

results_df.to_csv(
    results_file,
    index=False
)


# ============================================================
# BEST MODEL
# ============================================================

best_model = results_df.iloc[0]

print("\n" + "=" * 70)
print("BEST VALIDATION MODEL")
print("=" * 70)

print(
    "Model:",
    best_model["model"]
)

print(
    "Accuracy:",
    f"{best_model['accuracy']:.4f}"
)

print(
    "F1:",
    f"{best_model['f1_score']:.4f}"
)

print(
    "ROC-AUC:",
    f"{best_model['roc_auc']:.4f}"
)


print("\nResults saved to:")

print(results_file)


print("\n" + "=" * 70)
print("EVALUATION COMPLETED")
print("=" * 70)
