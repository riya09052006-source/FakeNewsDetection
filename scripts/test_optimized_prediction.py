from pathlib import Path

import joblib
from scipy.sparse import hstack

from ml.preprocessing.text_cleaner import clean_text
from ml.features.tfidf_features import TfidfFeatureExtractor


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

MODEL_DIR = BASE_DIR / "models" / "optimized"
VECTOR_DIR = BASE_DIR / "models" / "vectorizers"


# ============================================================
# LOAD MODEL
# ============================================================

model = joblib.load(
    MODEL_DIR / "linear_svm_optimized.joblib"
)


# ============================================================
# LOAD VECTORIZERS
# ============================================================

word_vectorizer = joblib.load(
    VECTOR_DIR / "word_tfidf_vectorizer.joblib"
)

char_vectorizer = joblib.load(
    VECTOR_DIR / "char_tfidf_vectorizer.joblib"
)


# ============================================================
# SAMPLE NEWS
# ============================================================

news_text = """
Scientists have announced a new breakthrough in renewable
energy technology after testing a new solar energy system.
Researchers say the system may improve energy efficiency
and reduce electricity costs.
"""


# ============================================================
# CLEAN TEXT
# ============================================================

cleaned_text = clean_text(
    news_text
)


# ============================================================
# CREATE FEATURES
# ============================================================

word_features = word_vectorizer.transform(
    [cleaned_text]
)

char_features = char_vectorizer.transform(
    [cleaned_text]
)

features = hstack(
    [
        word_features,
        char_features,
    ]
).tocsr()


# ============================================================
# PREDICT
# ============================================================

prediction = model.predict(
    features
)[0]

decision_score = model.decision_function(
    features
)[0]


# ============================================================
# RESULT
# ============================================================

print("=" * 60)
print("FAKE NEWS DETECTION")
print("=" * 60)

if prediction == 0:
    print("Prediction: FAKE")
else:
    print("Prediction: REAL")

print(f"Decision score: {decision_score:.4f}")

print("\nNote:")
print("This is a machine-learning prediction, not a guaranteed")
print("determination of factual truth.")
