from ml.explainability.explainer import FakeNewsExplainer


def main():
    print("=" * 60)
    print("FAKE NEWS DETECTION - XAI TEST")
    print("=" * 60)

    explainer = FakeNewsExplainer()

    print("\nExplainer ready:", explainer.is_ready())

    article = """
    A viral social media post claims that a secret drink
    can instantly cure every known disease. The post provides
    no scientific evidence, medical study, or credible source
    to support the claim.
    """

    result = explainer.explain(article)

    print("\nPrediction:")
    print(result["prediction"])

    print("\nLabel ID:")
    print(result["label_id"])

    print("\nDecision Score:")
    print(result["decision_score"])

    print("\nTop REAL signals:")

    for item in result["top_real_features"]:
        print(
            f"  {item['feature']}: "
            f"{item['contribution']:.6f}"
        )

    print("\nTop FAKE signals:")

    for item in result["top_fake_features"]:
        print(
            f"  {item['feature']}: "
            f"{item['contribution']:.6f}"
        )

    print("\nExplanation note:")
    print(result["explanation_note"])


if __name__ == "__main__":
    main()
