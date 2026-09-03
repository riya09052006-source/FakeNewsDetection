import sys

from pathlib import Path

import joblib

from scipy.sparse import load_npz

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))


# ============================================================
# PATH
# ============================================================

FEATURE_DIR = (
    BASE_DIR
    / "data"
    / "processed"
    / "features"
)


# ============================================================
# LOAD
# ============================================================

print("=" * 70)
print("FEATURE VERIFICATION")
print("=" * 70)

if not (FEATURE_DIR / "X_train.npz").exists():
    print(f"\nERROR: X_train.npz not found at {FEATURE_DIR / 'X_train.npz'}")
    print("Please run scripts/build_features.py first.")
    sys.exit(1)


X_train = load_npz(
    FEATURE_DIR / "X_train.npz"
)

X_validation = load_npz(
    FEATURE_DIR / "X_validation.npz"
)

X_test = load_npz(
    FEATURE_DIR / "X_test.npz"
)


y_train = joblib.load(
    FEATURE_DIR / "y_train.joblib"
)

y_validation = joblib.load(
    FEATURE_DIR / "y_validation.joblib"
)

y_test = joblib.load(
    FEATURE_DIR / "y_test.joblib"
)


# ============================================================
# DISPLAY
# ============================================================

print("\nTraining features:")
print(X_train.shape)

print("\nValidation features:")
print(X_validation.shape)

print("\nTest features:")
print(X_test.shape)


print("\nTraining labels:")
print(y_train.shape)

print("\nValidation labels:")
print(y_validation.shape)

print("\nTest labels:")
print(y_test.shape)


# ============================================================
# CHECK
# ============================================================

print("\n" + "=" * 70)
print("VERIFICATION")
print("=" * 70)


if X_train.shape[0] == len(y_train):

    print("Training features/labels: OK")

else:

    print("Training features/labels: ERROR")


if X_validation.shape[0] == len(y_validation):

    print("Validation features/labels: OK")

else:

    print("Validation features/labels: ERROR")


if X_test.shape[0] == len(y_test):

    print("Test features/labels: OK")

else:

    print("Test features/labels: ERROR")


print("\n" + "=" * 70)
print("FEATURE VERIFICATION COMPLETED")
print("=" * 70)
