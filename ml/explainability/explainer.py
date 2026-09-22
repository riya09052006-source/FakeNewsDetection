from pathlib import Path
from typing import Dict, List

import joblib
from scipy.sparse import hstack

from ml.preprocessing.text_cleaner import clean_text


class FakeNewsExplainer:
    """
    Explain predictions made by the TF-IDF + Linear SVM model.

    The explanation is based on feature contribution:
        contribution = TF-IDF value * SVM coefficient

    Positive contributions push toward REAL.
    Negative contributions push toward FAKE.
    """

    def __init__(self):
        self.project_root = Path(__file__).resolve().parents[2]

        self.model_path = (
            self.project_root
            / "models"
            / "optimized"
            / "linear_svm_optimized.joblib"
        )

        self.word_vectorizer_path = (
            self.project_root
            / "models"
            / "vectorizers"
            / "word_tfidf_vectorizer.joblib"
        )

        self.char_vectorizer_path = (
            self.project_root
            / "models"
            / "vectorizers"
            / "char_tfidf_vectorizer.joblib"
        )

        self.model = None
        self.word_vectorizer = None
        self.char_vectorizer = None

        self._load_models()

    def _load_models(self):
        """Load model and TF-IDF vectorizers."""

        self.model = joblib.load(self.model_path)

        self.word_vectorizer = joblib.load(
            self.word_vectorizer_path
        )

        self.char_vectorizer = joblib.load(
            self.char_vectorizer_path
        )

    def is_ready(self) -> bool:
        """Check whether all required components are loaded."""

        return all(
            [
                self.model is not None,
                self.word_vectorizer is not None,
                self.char_vectorizer is not None,
            ]
        )

    def _build_feature_names(self) -> List[str]:
        """
        Build combined feature names for word and character TF-IDF.
        """

        word_features = [
            f"word:{feature}"
            for feature in self.word_vectorizer.get_feature_names_out()
        ]

        char_features = [
            f"char:{feature}"
            for feature in self.char_vectorizer.get_feature_names_out()
        ]

        return word_features + char_features

    def explain(self, text: str, top_k: int = 8) -> Dict:
        """
        Generate a local explanation for one article.
        """

        if not isinstance(text, str):
            raise TypeError("Text must be a string.")

        text = text.strip()

        if len(text) < 20:
            raise ValueError(
                "Text must contain at least 20 characters."
            )

        cleaned_text = clean_text(text)

        word_features = self.word_vectorizer.transform(
            [cleaned_text]
        )

        char_features = self.char_vectorizer.transform(
            [cleaned_text]
        )

        combined_features = hstack(
            [word_features, char_features]
        ).tocsr()

        prediction_id = int(
            self.model.predict(combined_features)[0]
        )

        decision_score = float(
            self.model.decision_function(combined_features)[0]
        )

        feature_names = self._build_feature_names()

        coefficients = self.model.coef_[0]

        feature_values = combined_features.toarray()[0]

        contributions = feature_values * coefficients

        active_indices = contributions.nonzero()[0]

        feature_contributions = []

        for index in active_indices:
            contribution = float(contributions[index])

            feature_contributions.append(
                {
                    "feature": feature_names[index],
                    "contribution": contribution,
                }
            )

        fake_features = sorted(
            [
                item
                for item in feature_contributions
                if item["contribution"] < 0
            ],
            key=lambda x: x["contribution"],
        )[:top_k]

        real_features = sorted(
            [
                item
                for item in feature_contributions
                if item["contribution"] > 0
            ],
            key=lambda x: x["contribution"],
            reverse=True,
        )[:top_k]

        prediction = (
            "REAL"
            if prediction_id == 1
            else "FAKE"
        )

        return {
            "prediction": prediction,
            "label_id": prediction_id,
            "decision_score": decision_score,
            "top_real_features": real_features,
            "top_fake_features": fake_features,
            "explanation_note": (
                "These features represent model signals learned "
                "from the training data. They are not evidence "
                "that the article is factually true or false."
            ),
        }
