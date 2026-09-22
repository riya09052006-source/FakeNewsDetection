from ml.prediction.predictor import FakeNewsPredictor


def main():

    print("=" * 70)
    print("FAKE NEWS DETECTION")
    print("=" * 70)

    print("\nEnter a news article.")
    print("Type 'exit' to close the program.\n")

    predictor = FakeNewsPredictor()

    while True:

        news_text = input(
            "\nNews: "
        ).strip()

        if news_text.lower() == "exit":

            print("\nExiting...")

            break

        if not news_text:

            print(
                "Please enter some news text."
            )

            continue

        try:

            result = predictor.predict(
                news_text
            )

            print("\n" + "-" * 50)

            print(
                "Prediction:",
                result["prediction"]
            )

            print(
                "Decision Score:",
                f"{result['decision_score']:.4f}"
            )

            print(
                "Model:",
                result["model"]
            )

            print("-" * 50)

        except (
            TypeError,
            ValueError,
        ) as error:

            print(
                f"\nInput error: {error}"
            )


if __name__ == "__main__":

    main()
