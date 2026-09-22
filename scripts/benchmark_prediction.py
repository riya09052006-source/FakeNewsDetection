import statistics
import time

from ml.prediction.predictor import (
    FakeNewsPredictor,
)


ARTICLES = [

    """
    Researchers published a detailed scientific report
    after conducting multiple experiments and reviewing
    independent evidence.
    """,

    """
    A viral post claims that a secret method can instantly
    solve every disease without any scientific evidence.
    """,

    """
    The government statistics agency released updated
    employment and inflation figures based on officially
    collected data.
    """,

]


def main():

    predictor = FakeNewsPredictor()

    times = []

    for article in ARTICLES:

        start = time.perf_counter()

        result = predictor.predict(
            article
        )

        elapsed = (
            time.perf_counter()
            - start
        )

        times.append(
            elapsed * 1000
        )

        print(
            f"{result['prediction']:5s} "
            f"{elapsed * 1000:.2f} ms"
        )

    print("\nPerformance Summary")

    print(
        f"Average: "
        f"{statistics.mean(times):.2f} ms"
    )

    print(
        f"Minimum: "
        f"{min(times):.2f} ms"
    )

    print(
        f"Maximum: "
        f"{max(times):.2f} ms"
    )


if __name__ == "__main__":
    main()
