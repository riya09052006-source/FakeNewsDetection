from ml.prediction.predictor import (
    FakeNewsPredictor,
)


VALID_ARTICLE = """
Researchers published a detailed report after conducting
multiple experiments. Independent researchers reviewed the
results and recommended additional studies to confirm the
findings.
"""


def test_predictor_initialization():

    predictor = FakeNewsPredictor()

    assert predictor.is_ready()


def test_prediction_result():

    predictor = FakeNewsPredictor()

    result = predictor.predict(
        VALID_ARTICLE
    )

    assert isinstance(result, dict)

    assert "prediction" in result
    assert "label_id" in result
    assert "decision_score" in result
    assert "model" in result
    assert "feature_type" in result


def test_prediction_label():

    predictor = FakeNewsPredictor()

    result = predictor.predict(
        VALID_ARTICLE
    )

    assert result["prediction"] in [
        "FAKE",
        "REAL",
    ]


def test_prediction_label_id():

    predictor = FakeNewsPredictor()

    result = predictor.predict(
        VALID_ARTICLE
    )

    assert result["label_id"] in [
        0,
        1,
    ]


def test_short_text_rejected():

    predictor = FakeNewsPredictor()

    try:

        predictor.predict("Too short")

        assert False

    except ValueError:

        assert True


def test_empty_text_rejected():

    predictor = FakeNewsPredictor()

    try:

        predictor.predict("")

        assert False

    except ValueError:

        assert True


def test_batch_prediction():

    predictor = FakeNewsPredictor()

    articles = [
        VALID_ARTICLE,
        VALID_ARTICLE,
    ]

    results = predictor.predict_batch(
        articles
    )

    assert len(results) == 2

    for result in results:

        assert result["prediction"] in [
            "FAKE",
            "REAL",
        ]
