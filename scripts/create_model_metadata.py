from pathlib import Path
import json

import joblib


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

MODEL_DIR = (
    BASE_DIR /
    "models" /
    "optimized"
)

JOBLIB_METADATA = (
    MODEL_DIR /
    "linear_svm_optimized_metadata.joblib"
)

JSON_METADATA = (
    MODEL_DIR /
    "model_metadata.json"
)


# ============================================================
# LOAD
# ============================================================

metadata = joblib.load(
    JOBLIB_METADATA
)


# ============================================================
# ADD APPLICATION INFORMATION
# ============================================================

metadata["application"] = (
    "Fake News Detection Using Machine Learning"
)

metadata["prediction_labels"] = {
    "0": "FAKE",
    "1": "REAL",
}

metadata["warning"] = (
    "Predictions indicate model classification patterns "
    "and should not be treated as independent verification "
    "of factual truth."
)


# ============================================================
# SAVE JSON
# ============================================================

with open(
    JSON_METADATA,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        metadata,
        file,
        indent=4
    )


# ============================================================
# DISPLAY
# ============================================================

print("=" * 70)
print("MODEL METADATA CREATED")
print("=" * 70)

print(
    JSON_METADATA
)

print("\nMetadata:")

print(
    json.dumps(
        metadata,
        indent=4
    )
)
