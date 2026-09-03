from pathlib import Path

import joblib

from sklearn.feature_extraction.text import TfidfVectorizer


class TfidfFeatureExtractor:

    def __init__(self):

        # ----------------------------------------------------
        # WORD LEVEL TF-IDF
        # ----------------------------------------------------

        self.word_vectorizer = TfidfVectorizer(
            lowercase=True,

            ngram_range=(1, 2),

            min_df=2,

            max_df=0.98,

            sublinear_tf=True,

            max_features=200000
        )

        # ----------------------------------------------------
        # CHARACTER LEVEL TF-IDF
        # ----------------------------------------------------

        self.char_vectorizer = TfidfVectorizer(
            lowercase=True,

            analyzer="char",

            ngram_range=(3, 5),

            min_df=3,

            max_features=100000,

            sublinear_tf=True
        )

    # ========================================================
    # FIT
    # ========================================================

    def fit(self, texts):

        print("Fitting word-level TF-IDF...")

        self.word_vectorizer.fit(texts)

        print(
            "Word vocabulary size:",
            len(self.word_vectorizer.vocabulary_)
        )

        print("\nFitting character-level TF-IDF...")

        self.char_vectorizer.fit(texts)

        print(
            "Character vocabulary size:",
            len(self.char_vectorizer.vocabulary_)
        )

        return self

    # ========================================================
    # TRANSFORM
    # ========================================================

    def transform(self, texts):

        print("Transforming word features...")

        word_features = self.word_vectorizer.transform(
            texts
        )

        print("Transforming character features...")

        char_features = self.char_vectorizer.transform(
            texts
        )

        return word_features, char_features

    # ========================================================
    # FIT + TRANSFORM
    # ========================================================

    def fit_transform(self, texts):

        print("Fitting and transforming word TF-IDF...")

        word_features = (
            self.word_vectorizer.fit_transform(texts)
        )

        print(
            "Word vocabulary size:",
            len(self.word_vectorizer.vocabulary_)
        )

        print("\nFitting and transforming character TF-IDF...")

        char_features = (
            self.char_vectorizer.fit_transform(texts)
        )

        print(
            "Character vocabulary size:",
            len(self.char_vectorizer.vocabulary_)
        )

        return word_features, char_features

    # ========================================================
    # SAVE
    # ========================================================

    def save(self, output_directory):

        output_directory = Path(output_directory)

        output_directory.mkdir(
            parents=True,
            exist_ok=True
        )

        word_path = (
            output_directory
            / "word_tfidf_vectorizer.joblib"
        )

        char_path = (
            output_directory
            / "char_tfidf_vectorizer.joblib"
        )

        joblib.dump(
            self.word_vectorizer,
            word_path
        )

        joblib.dump(
            self.char_vectorizer,
            char_path
        )

        print("\nVectorizers saved:")

        print(word_path)
        print(char_path)

    # ========================================================
    # LOAD
    # ========================================================

    def load(self, input_directory):

        input_directory = Path(input_directory)

        word_path = (
            input_directory
            / "word_tfidf_vectorizer.joblib"
        )

        char_path = (
            input_directory
            / "char_tfidf_vectorizer.joblib"
        )

        self.word_vectorizer = joblib.load(
            word_path
        )

        self.char_vectorizer = joblib.load(
            char_path
        )

        print("Vectorizers loaded successfully.")

        return self
