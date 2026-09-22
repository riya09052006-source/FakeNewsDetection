import os
import platform
import time

from ml.prediction.predictor import (
    FakeNewsPredictor,
)


def main():

    print("=" * 60)
    print("FAKEGUARD AI PERFORMANCE CHECK")
    print("=" * 60)

    print("\nPlatform:")
    print(platform.platform())

    print("\nCPU threads:")
    print(os.cpu_count())

    predictor = FakeNewsPredictor()

    article = """
    Researchers published a detailed report after conducting
    multiple experiments. The researchers explained that
    additional independent studies are necessary to confirm
    the findings and understand their broader implications.
    """

    start = time.perf_counter()

    result = predictor.predict(
        article
    )

    elapsed = (
        time.perf_counter() - start
    )

    print("\nPrediction:")
    print(result["prediction"])

    print("\nInference time:")
    print(
        f"{elapsed * 1000:.2f} ms"
    )

    print("\nModel ready:")
    print(
        predictor.is_ready()
    )


if __name__ == "__main__":
    main()
