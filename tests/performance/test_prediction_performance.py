import time

from ml.prediction.predictor import (
    FakeNewsPredictor,
)


ARTICLE = """
Researchers published a detailed report after conducting
multiple experiments. Independent researchers reviewed
the findings and recommended additional studies before
drawing broader conclusions.
"""


def test_prediction_performance():

    predictor = FakeNewsPredictor()

    start = time.perf_counter()

    for _ in range(10):

        result = predictor.predict(
            ARTICLE
        )

        assert result["prediction"] in [
            "FAKE",
            "REAL",
        ]

    elapsed = (
        time.perf_counter()
        - start
    )

    average_ms = (
        elapsed / 10
    ) * 1000

    print(
        f"\nAverage inference time: "
        f"{average_ms:.2f} ms"
    )

    # Generous CPU-friendly threshold.
    # This is a regression guard, not a hardware guarantee.
    assert average_ms < 5000
