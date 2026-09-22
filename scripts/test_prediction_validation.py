from ml.prediction.predictor import FakeNewsPredictor


def test_empty_text():

    predictor = FakeNewsPredictor()

    try:

        predictor.predict("")

        print("ERROR: Empty text was accepted.")

    except ValueError as error:

        print("PASS: Empty text rejected.")
        print("Message:", error)


def test_short_text():

    predictor = FakeNewsPredictor()

    try:

        predictor.predict("Fake news")

        print("ERROR: Short text was accepted.")

    except ValueError as error:

        print("PASS: Short text rejected.")
        print("Message:", error)


def test_non_string():

    predictor = FakeNewsPredictor()

    try:

        predictor.predict(12345)

        print("ERROR: Non-string input was accepted.")

    except TypeError as error:

        print("PASS: Non-string input rejected.")
        print("Message:", error)


if __name__ == "__main__":

    print("=" * 70)
    print("PREDICTION INPUT VALIDATION TEST")
    print("=" * 70)

    test_empty_text()

    print()

    test_short_text()

    print()

    test_non_string()
