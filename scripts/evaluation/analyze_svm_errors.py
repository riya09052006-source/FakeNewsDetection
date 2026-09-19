from pathlib import Path

import joblib
import pandas as pd


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

TEST_FILE = (
    BASE_DIR /
    "data" /
    "processed" /
    "splits" /
    "test.csv"
)

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

OUTPUT_DIR = (
    BASE_DIR /
    "data" /
    "processed" /
    "evaluation" /
    "final"
)


# ============================================================
# LOAD
# ============================================================

print("=" * 70)
print("SVM ERROR ANALYSIS")
print("=" * 70)

test_df = pd.read_csv(
    TEST_FILE
)

X_test = __import__(
    "scipy.sparse"
).sparse.load_npz(
    FEATURE_DIR /
    "X_test.npz"
)

model = joblib.load(
    MODEL_PATH
)

y_test = joblib.load(
    FEATURE_DIR /
    "y_test.joblib"
)


# ============================================================
# PREDICTION
# ============================================================

predictions = model.predict(
    X_test
)

scores = model.decision_function(
    X_test
)


# ============================================================
# ALIGN DATA
# ============================================================

test_df = test_df.reset_index(
    drop=True
)

test_df["actual_label"] = y_test

test_df["predicted_label"] = predictions

test_df["decision_score"] = scores


# ============================================================
# LABEL NAMES
# ============================================================

test_df["actual"] = (
    test_df["actual_label"]
    .map(
        {
            0: "Fake",
            1: "Real"
        }
    )
)

test_df["predicted"] = (
    test_df["predicted_label"]
    .map(
        {
            0: "Fake",
            1: "Real"
        }
    )
)


# ============================================================
# FIND ERRORS
# ============================================================

errors = test_df[
    test_df["actual_label"]
    !=
    test_df["predicted_label"]
].copy()


# ============================================================
# ERROR TYPE
# ============================================================

errors["error_type"] = errors.apply(
    lambda row:
        "False Positive"
        if (
            row["actual_label"] == 0
            and
            row["predicted_label"] == 1
        )
        else "False Negative",
    axis=1
)


# ============================================================
# SORT BY UNCERTAINTY
# ============================================================

errors["absolute_score"] = (
    errors["decision_score"]
    .abs()
)

errors = errors.sort_values(
    by="absolute_score",
    ascending=True
)


# ============================================================
# SAVE ALL ERRORS
# ============================================================

errors.to_csv(
    OUTPUT_DIR /
    "svm_misclassified_articles.csv",
    index=False
)


# ============================================================
# FALSE POSITIVES
# ============================================================

false_positives = errors[
    errors["error_type"]
    ==
    "False Positive"
]

false_positives.to_csv(
    OUTPUT_DIR /
    "svm_false_positives.csv",
    index=False
)


# ============================================================
# FALSE NEGATIVES
# ============================================================

false_negatives = errors[
    errors["error_type"]
    ==
    "False Negative"
]

false_negatives.to_csv(
    OUTPUT_DIR /
    "svm_false_negatives.csv",
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print(
    f"\nTotal test articles: "
    f"{len(test_df):,}"
)

print(
    f"Misclassified articles: "
    f"{len(errors):,}"
)

print(
    f"False positives: "
    f"{len(false_positives):,}"
)

print(
    f"False negatives: "
    f"{len(false_negatives):,}"
)


print("\nSaved error analysis files to:")

print(OUTPUT_DIR)
