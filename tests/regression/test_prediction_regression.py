from ml.prediction.predictor import (
    FakeNewsPredictor,
)


ARTICLES = [

    """
    Researchers published a detailed scientific report after
    conducting controlled experiments. The researchers stated
    that additional independent studies are needed to confirm
    the findings.
    """,

    """
    A viral post claims that a secret drink can instantly
    cure every disease. The post provides no scientific
    evidence or credible medical research.
    """,

    """
    The national statistics agency released its latest
    economic report containing updated employment and inflation
    figures based on officially collected data.
    """,

]


def test_prediction_contract():

    predictor = FakeNewsPredictor()

    for article in ARTICLES:

        result = predictor.predict(
            article
        )

        assert result["prediction"] in [
            "FAKE",
            "REAL",
        ]

        assert result["label_id"] in [
            0,
            1,
        ]

        assert isinstance(
            result["decision_score"],
            float,
        )

        assert result["model"] == (
            "Optimized Linear SVM"
        )

        assert result["feature_type"] == (
            "Word TF-IDF + Character TF-IDF"
        )
