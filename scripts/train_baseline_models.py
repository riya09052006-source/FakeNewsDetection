import time
from pathlib import Path

import joblib

from scipy.sparse import load_npz

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC


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

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD FEATURES
# ============================================================

print("=" * 70)
print("FAKE NEWS DETECTION")
print("BASELINE MODEL TRAINING")
print("=" * 70)


print("\nLoading training features...")

X_train = load_npz(
    FEATURE_DIR / "X_train.npz"
)

X_validation = load_npz(
    FEATURE_DIR / "X_validation.npz"
)


print("Loading labels...")

y_train = joblib.load(
    FEATURE_DIR / "y_train.joblib"
)

y_validation = joblib.load(
    FEATURE_DIR / "y_validation.joblib"
)


# ============================================================
# DISPLAY DATASET INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("FEATURE INFORMATION")
print("=" * 70)

print("\nTraining features:")
print(X_train.shape)

print("\nValidation features:")
print(X_validation.shape)

print("\nTraining labels:")
print(y_train.shape)

print("\nValidation labels:")
print(y_validation.shape)


# ============================================================
# MODEL DEFINITIONS
# ============================================================

models = {

    "naive_bayes": MultinomialNB(
        alpha=0.1
    ),

    "logistic_regression": LogisticRegression(
        C=2.0,
        max_iter=1000,
        class_weight="balanced",
        solver="liblinear"
    ),

    "linear_svm": LinearSVC(
        C=1.0,
        class_weight="balanced",
        max_iter=5000
    )
}


# ============================================================
# TRAIN MODELS
# ============================================================

training_times = {}


for model_name, model in models.items():

    print("\n" + "=" * 70)
    print(f"TRAINING: {model_name.upper()}")
    print("=" * 70)

    start_time = time.perf_counter()

    model.fit(
        X_train,
        y_train
    )

    end_time = time.perf_counter()

    training_time = (
        end_time - start_time
    )

    training_times[model_name] = training_time

    print(
        f"\nTraining time: "
        f"{training_time:.2f} seconds"
    )

    # --------------------------------------------------------
    # SAVE MODEL
    # --------------------------------------------------------

    model_path = (
        MODEL_DIR
        / f"{model_name}.joblib"
    )

    joblib.dump(
        model,
        model_path
    )

    print("\nModel saved:")
    print(model_path)


# ============================================================
# TRAINING SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("TRAINING SUMMARY")
print("=" * 70)

for model_name, training_time in training_times.items():

    print(
        f"{model_name}: "
        f"{training_time:.2f} seconds"
    )


print("\n" + "=" * 70)
print("BASELINE TRAINING COMPLETED")
print("=" * 70)
