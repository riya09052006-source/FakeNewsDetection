import sys

from pathlib import Path

import joblib
import pandas as pd

from scipy.sparse import hstack, save_npz

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from ml.preprocessing.text_cleaner import clean_text
from ml.features.tfidf_features import TfidfFeatureExtractor


# ============================================================
# PROJECT PATHS
# ============================================================

SPLIT_DIR = (
    BASE_DIR
    / "data"
    / "processed"
    / "splits"
)

FEATURE_DIR = (
    BASE_DIR
    / "data"
    / "processed"
    / "features"
)

VECTOR_DIR = (
    BASE_DIR
    / "models"
    / "vectorizers"
)


FEATURE_DIR.mkdir(
    parents=True,
    exist_ok=True
)

VECTOR_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("FAKE NEWS DETECTION - FEATURE ENGINEERING")
print("=" * 70)

if not (SPLIT_DIR / "train.csv").exists():
    print(f"\nERROR: train.csv not found at {SPLIT_DIR / 'train.csv'}")
    print("Please make sure dataset splits exist before running feature engineering.")
    sys.exit(1)

train_df = pd.read_csv(
    SPLIT_DIR / "train.csv"
)

validation_df = pd.read_csv(
    SPLIT_DIR / "validation.csv"
)

test_df = pd.read_csv(
    SPLIT_DIR / "test.csv"
)


print("\nDataset sizes:")

print("Training:", len(train_df))
print("Validation:", len(validation_df))
print("Testing:", len(test_df))


# ============================================================
# CLEAN TEXT
# ============================================================

print("\n" + "=" * 70)
print("TEXT PREPROCESSING")
print("=" * 70)


print("\nCleaning training text...")

train_text = (
    train_df["content"]
    .fillna("")
    .apply(clean_text)
)


print("Cleaning validation text...")

validation_text = (
    validation_df["content"]
    .fillna("")
    .apply(clean_text)
)


print("Cleaning test text...")

test_text = (
    test_df["content"]
    .fillna("")
    .apply(clean_text)
)


# ============================================================
# CREATE FEATURE EXTRACTOR
# ============================================================

extractor = TfidfFeatureExtractor()


# ============================================================
# FIT ONLY ON TRAINING DATA
# ============================================================

print("\n" + "=" * 70)
print("FITTING TF-IDF")
print("=" * 70)


word_train, char_train = (
    extractor.fit_transform(train_text)
)


# ============================================================
# TRANSFORM VALIDATION AND TEST
# ============================================================

print("\n" + "=" * 70)
print("TRANSFORMING VALIDATION DATA")
print("=" * 70)


word_validation, char_validation = (
    extractor.transform(validation_text)
)


print("\n" + "=" * 70)
print("TRANSFORMING TEST DATA")
print("=" * 70)


word_test, char_test = (
    extractor.transform(test_text)
)


# ============================================================
# COMBINE WORD + CHARACTER FEATURES
# ============================================================

print("\n" + "=" * 70)
print("COMBINING FEATURES")
print("=" * 70)


train_features = hstack(
    [word_train, char_train]
).tocsr()


validation_features = hstack(
    [word_validation, char_validation]
).tocsr()


test_features = hstack(
    [word_test, char_test]
).tocsr()


# ============================================================
# PRINT SHAPES
# ============================================================

print("\nTraining feature matrix:")
print(train_features.shape)


print("\nValidation feature matrix:")
print(validation_features.shape)


print("\nTest feature matrix:")
print(test_features.shape)


# ============================================================
# SAVE FEATURES
# ============================================================

print("\n" + "=" * 70)
print("SAVING FEATURE MATRICES")
print("=" * 70)


save_npz(
    FEATURE_DIR / "X_train.npz",
    train_features
)


save_npz(
    FEATURE_DIR / "X_validation.npz",
    validation_features
)


save_npz(
    FEATURE_DIR / "X_test.npz",
    test_features
)


# ============================================================
# SAVE LABELS
# ============================================================

joblib.dump(
    train_df["label"].values,
    FEATURE_DIR / "y_train.joblib"
)


joblib.dump(
    validation_df["label"].values,
    FEATURE_DIR / "y_validation.joblib"
)


joblib.dump(
    test_df["label"].values,
    FEATURE_DIR / "y_test.joblib"
)


# ============================================================
# SAVE VECTORIZERS
# ============================================================

extractor.save(
    VECTOR_DIR
)


# ============================================================
# FINAL MESSAGE
# ============================================================

print("\n" + "=" * 70)
print("FEATURE ENGINEERING COMPLETED")
print("=" * 70)

print("\nFeature files saved to:")

print(FEATURE_DIR)

print("\nVectorizers saved to:")

print(VECTOR_DIR)
