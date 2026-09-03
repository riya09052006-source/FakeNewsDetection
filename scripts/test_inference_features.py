import sys
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from ml.features.tfidf_features import TfidfFeatureExtractor
from ml.preprocessing.text_cleaner import clean_text


# ============================================================
# PATH
# ============================================================

VECTOR_DIR = (
    BASE_DIR
    / "models"
    / "vectorizers"
)


# ============================================================
# LOAD VECTORIZERS
# ============================================================

print("=" * 70)
print("INFERENCE FEATURE TEST")
print("=" * 70)

if not (VECTOR_DIR / "word_tfidf_vectorizer.joblib").exists():
    print(f"\nERROR: Vectorizers not found at {VECTOR_DIR}")
    print("Please run scripts/build_features.py first.")
    sys.exit(1)

extractor = TfidfFeatureExtractor()

extractor.load(
    VECTOR_DIR
)


# ============================================================
# NEW NEWS ARTICLE
# ============================================================

news = """
The government announced a new technology program
to improve digital education across the country.
"""


# ============================================================
# CLEAN
# ============================================================

cleaned_news = clean_text(news)


print("Cleaned news:")
print(cleaned_news)


# ============================================================
# TRANSFORM
# ============================================================

word_features, char_features = (
    extractor.transform(
        [cleaned_news]
    )
)


# ============================================================
# DISPLAY
# ============================================================

print("\nWord feature shape:")
print(word_features.shape)


print("\nCharacter feature shape:")
print(char_features.shape)


print("\nTotal feature count:")
print(
    word_features.shape[1]
    + char_features.shape[1]
)


print("\n" + "=" * 70)
print("INFERENCE FEATURE TEST SUCCESSFUL")
print("=" * 70)
