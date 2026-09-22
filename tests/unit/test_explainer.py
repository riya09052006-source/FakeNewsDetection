from ml.explainability.explainer import (
    FakeNewsExplainer,
)


ARTICLE = """
A viral social media post claims that a secret drink
can instantly cure every known disease. The post provides
no scientific evidence or credible source to support
the claim.
"""


def test_explainer_ready():

    explainer = FakeNewsExplainer()

    assert explainer.is_ready()


def test_explanation_result():

    explainer = FakeNewsExplainer()

    result = explainer.explain(
        ARTICLE
    )

    assert isinstance(result, dict)

    assert "prediction" in result
    assert "label_id" in result
    assert "decision_score" in result
    assert "top_real_features" in result
    assert "top_fake_features" in result
    assert "explanation_note" in result


def test_explanation_prediction():

    explainer = FakeNewsExplainer()

    result = explainer.explain(
        ARTICLE
    )

    assert result["prediction"] in [
        "FAKE",
        "REAL",
    ]


def test_explanation_features_are_lists():

    explainer = FakeNewsExplainer()

    result = explainer.explain(
        ARTICLE
    )

    assert isinstance(
        result["top_real_features"],
        list,
    )

    assert isinstance(
        result["top_fake_features"],
        list,
    )


def test_short_text_rejected():

    explainer = FakeNewsExplainer()

    try:

        explainer.explain(
            "Too short"
        )

        assert False

    except ValueError:

        assert True
