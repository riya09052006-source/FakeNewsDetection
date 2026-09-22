from ml.prediction.predictor import FakeNewsPredictor


def main():

    predictor = FakeNewsPredictor()

    articles = [

        """
        Government researchers published a report describing
        the results of a multi-year scientific study. The report
        explains that additional research is needed before the
        findings can be generalized.
        """,

        """
        A viral social media post claims that drinking a special
        liquid can instantly cure every known disease. The post
        provides no scientific evidence or credible sources.
        """,

        """
        The national statistics agency released its latest
        economic report containing updated employment and
        inflation figures based on officially collected data.
        """,
    ]

    results = predictor.predict_batch(
        articles
    )

    print("=" * 70)
    print("BATCH PREDICTION TEST")
    print("=" * 70)

    for result in results:

        print("\n" + "-" * 50)

        for key, value in result.items():

            print(
                f"{key}: {value}"
            )


if __name__ == "__main__":

    main()
