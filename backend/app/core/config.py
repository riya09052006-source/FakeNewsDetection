from pathlib import Path


class Settings:
    """
    Application configuration.
    """

    APP_NAME = "Fake News Detection API"

    VERSION = "1.0.0"

    DESCRIPTION = (
        "Local machine-learning API for fake news classification."
    )

    BASE_DIR = Path(__file__).resolve().parents[3]

    MODEL_DIR = (
        BASE_DIR /
        "models" /
        "optimized"
    )

    ALLOWED_ORIGINS = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ]

    MAX_TEXT_LENGTH = 100_000

    MAX_BATCH_SIZE = 20


settings = Settings()
