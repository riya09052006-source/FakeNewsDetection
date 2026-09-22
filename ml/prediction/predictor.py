from pathlib import Path
from typing import Any, Dict, List

import joblib
from scipy.sparse import hstack

from ml.preprocessing.text_cleaner import clean_text


class FakeNewsPredictor:
    """
    Reusable prediction engine for the Fake News Detection system.

    Pipeline:

        Raw Text
            ↓
        Text Cleaning
            ↓
        Word TF-IDF
            +
        Character TF-IDF
            ↓
        Optimized Linear SVM
            ↓
        Prediction
    """

    def __init__(self, base_dir: Path | None = None):
        """
        Initialize the prediction engine.

        Parameters
        ----------
        base_dir:
            Project root directory.
            If not provided, it is detected automatically.
        """

        if base_dir is None:
            base_dir = Path(__file__).resolve().parents[2]

        self.base_dir = Path(base_dir)

        self.model_dir = (
            self.base_dir /
            "models" /
            "optimized"
        )

        self.vectorizer_dir = (
            self.base_dir /
            "models" /
            "vectorizers"
        )

        self.model_path = (
            self.model_dir /
            "linear_svm_optimized.joblib"
        )

        self.word_vectorizer_path = (
            self.vectorizer_dir /
            "word_tfidf_vectorizer.joblib"
        )

        self.char_vectorizer_path = (
            self.vectorizer_dir /
            "char_tfidf_vectorizer.joblib"
        )

        self.model = None
        self.word_vectorizer = None
        self.char_vectorizer = None

        self._load_components()

    # ========================================================
    # LOAD MODEL COMPONENTS
    # ========================================================

    def _load_components(self) -> None:
        """
        Load the trained model and TF-IDF vectorizers.
        """

        missing_files = []

        required_files = {
            "Optimized model": self.model_path,
            "Word vectorizer": self.word_vectorizer_path,
            "Character vectorizer": self.char_vectorizer_path,
        }

        for name, path in required_files.items():

            if not path.exists():

                missing_files.append(
                    f"{name}: {path}"
                )

        if missing_files:

            message = (
                "Required ML files are missing:\n"
                +
                "\n".join(missing_files)
            )

            raise FileNotFoundError(message)

        self.model = joblib.load(
            self.model_path
        )

        self.word_vectorizer = joblib.load(
            self.word_vectorizer_path
        )

        self.char_vectorizer = joblib.load(
            self.char_vectorizer_path
        )

    # ========================================================
    # VALIDATE TEXT
    # ========================================================

    @staticmethod
    def _validate_text(text: Any) -> str:
        """
        Validate incoming news text.
        """

        if not isinstance(text, str):

            raise TypeError(
                "News text must be a string."
            )

        text = text.strip()

        if not text:

            raise ValueError(
                "News text cannot be empty."
            )

        if len(text) < 20:

            raise ValueError(
                "Please provide a longer news article."
            )

        return text

    # ========================================================
    # CREATE FEATURES
    # ========================================================

    def _create_features(self, text: str):
        """
        Convert cleaned text into the same feature representation
        used during model training.
        """

        cleaned_text = clean_text(text)

        if not cleaned_text:

            raise ValueError(
                "Text became empty after preprocessing."
            )

        word_features = (
            self.word_vectorizer.transform(
                [cleaned_text]
            )
        )

        char_features = (
            self.char_vectorizer.transform(
                [cleaned_text]
            )
        )

        combined_features = hstack(
            [
                word_features,
                char_features,
            ]
        ).tocsr()

        return combined_features

    # ========================================================
    # PREDICT SINGLE ARTICLE
    # ========================================================

    def predict(self, text: str) -> Dict[str, Any]:
        """
        Predict whether a news article is likely fake or real.

        Returns
        -------
        dict
            Structured prediction result.
        """

        text = self._validate_text(text)

        features = self._create_features(
            text
        )

        prediction = self.model.predict(
            features
        )[0]

        decision_score = float(
            self.model.decision_function(
                features
            )[0]
        )

        if prediction == 0:

            label = "FAKE"

        else:

            label = "REAL"

        return {
            "prediction": label,
            "label_id": int(prediction),
            "decision_score": decision_score,
            "model": "Optimized Linear SVM",
            "feature_type": (
                "Word TF-IDF + Character TF-IDF"
            ),
        }

    # ========================================================
    # PREDICT MULTIPLE ARTICLES
    # ========================================================

    def predict_batch(
        self,
        texts: List[str],
    ) -> List[Dict[str, Any]]:
        """
        Predict multiple news articles.
        """

        if not isinstance(texts, list):

            raise TypeError(
                "texts must be a list of strings."
            )

        results = []

        for index, text in enumerate(texts):

            try:

                result = self.predict(
                    text
                )

                result["index"] = index

                results.append(result)

            except (
                TypeError,
                ValueError,
            ) as error:

                results.append(
                    {
                        "index": index,
                        "error": str(error),
                    }
                )

        return results

    # ========================================================
    # HEALTH CHECK
    # ========================================================

    def is_ready(self) -> bool:
        """
        Check whether the prediction engine is loaded.
        """

        return (
            self.model is not None
            and
            self.word_vectorizer is not None
            and
            self.char_vectorizer is not None
        )
