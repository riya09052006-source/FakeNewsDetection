from ml.explainability.explainer import FakeNewsExplainer
from ml.prediction.predictor import FakeNewsPredictor


class PredictionService:

    def __init__(self):
        self.predictor = FakeNewsPredictor()
        self.explainer = FakeNewsExplainer()

    def predict(self, text: str):
        return self.predictor.predict(text)

    def predict_batch(self, texts):
        return self.predictor.predict_batch(texts)

    def explain(self, text: str):
        return self.explainer.explain(text)

    def is_ready(self) -> bool:
        return (
            self.predictor.is_ready()
            and self.explainer.is_ready()
        )


prediction_service = PredictionService()
