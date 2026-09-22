from ml.prediction.predictor import FakeNewsPredictor


VALID_ARTICLE = """
Researchers have published a detailed scientific report after
conducting multiple experiments. The researchers explained that
additional independent studies are necessary to confirm the
findings and understand their broader implications.
"""


def test_predictor_loads():

    predictor = FakeNewsPredictor()

    assert predictor.is_ready() is True


def test_prediction_result():

    predictor = FakeNewsPredictor()

    result = predictor.predict(
        VALID_ARTICLE
    )

    assert "prediction" in result

    assert "label_id" in result

    assert "decision_score" in result

    assert result["prediction"] in {
        "FAKE",
        "REAL",
    }

    assert result["label_id"] in {
        0,
        1,
    }


def test_empty_text():

    predictor = FakeNewsPredictor()

    try:

        predictor.predict("")

        assert False

    except ValueError:

        assert True


def test_short_text():

    predictor = FakeNewsPredictor()

    try:

        predictor.predict("hello")

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

        assert (
            result["prediction"]
            in {"FAKE", "REAL"}
        )
