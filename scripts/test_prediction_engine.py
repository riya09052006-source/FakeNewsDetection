from ml.prediction.predictor import FakeNewsPredictor


def main():

    print("=" * 70)
    print("FAKE NEWS PREDICTION ENGINE TEST")
    print("=" * 70)

    # ========================================================
    # LOAD ENGINE
    # ========================================================

    predictor = FakeNewsPredictor()

    print("\nPrediction engine ready:")
    print(predictor.is_ready())

    # ========================================================
    # SAMPLE NEWS
    # ========================================================

    news_text = """
    Researchers have announced the results of a new scientific
    study after conducting several controlled experiments.
    The researchers stated that further independent studies
    are required before the findings can be applied broadly.
    """

    # ========================================================
    # PREDICT
    # ========================================================

    result = predictor.predict(
        news_text
    )

    print("\n" + "=" * 70)
    print("PREDICTION RESULT")
    print("=" * 70)

    for key, value in result.items():

        print(
            f"{key}: {value}"
        )


if __name__ == "__main__":
    main()
