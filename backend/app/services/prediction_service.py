from pathlib import Path
from typing import Any

from ml.prediction.predictor import FakeNewsPredictor


class PredictionService:
    """
    Application service responsible for ML predictions.

    The predictor is loaded once and reused for all requests.
    """

    def __init__(self):

        self.predictor = FakeNewsPredictor(
            base_dir=Path(__file__).resolve().parents[3]
        )

    def predict(
        self,
        text: str,
    ) -> dict[str, Any]:

        return self.predictor.predict(
            text
        )

    def predict_batch(
        self,
        texts: list[str],
    ) -> list[dict[str, Any]]:

        return self.predictor.predict_batch(
            texts
        )

    def is_ready(self) -> bool:

        return self.predictor.is_ready()
