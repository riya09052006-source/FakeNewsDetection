import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

import joblib

from scipy.sparse import hstack

from ml.preprocessing.text_cleaner import clean_text
from ml.features.tfidf_features import TfidfFeatureExtractor


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

VECTOR_DIR = (
    BASE_DIR
    / "models"
    / "vectorizers"
)

MODEL_DIR = (
    BASE_DIR
    / "models"
    / "baseline"
)


# ============================================================
# LOAD VECTORIZERS
# ============================================================

extractor = TfidfFeatureExtractor()

extractor.load(
    VECTOR_DIR
)


# ============================================================
# LOAD MODEL
# ============================================================

model = joblib.load(
    MODEL_DIR / "linear_svm.joblib"
)


# ============================================================
# TEST NEWS
# ============================================================

news = """
The government announced a new national education
program that will introduce modern digital learning
facilities in schools.
"""


# ============================================================
# CLEAN TEXT
# ============================================================

cleaned_text = clean_text(
    news
)


# ============================================================
# CREATE FEATURES
# ============================================================

word_features, char_features = (
    extractor.transform(
        [cleaned_text]
    )
)


features = hstack(
    [
        word_features,
        char_features
    ]
).tocsr()


# ============================================================
# PREDICTION
# ============================================================

prediction = model.predict(
    features
)[0]


# ============================================================
# RESULT
# ============================================================

if prediction == 0:

    result = "FAKE"

else:

    result = "REAL"


print("=" * 70)
print("FAKE NEWS DETECTION TEST")
print("=" * 70)

print("\nArticle:")
print(news)

print("\nPrediction:")
print(result)

print("\n" + "=" * 70)
print("PREDICTION TEST COMPLETED")
print("=" * 70)
