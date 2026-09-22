import re


def sanitize_text(text: str) -> str:
    """
    Basic input sanitization.

    This does not alter the ML meaning unnecessarily.
    It mainly removes dangerous control characters.
    """

    if not isinstance(text, str):
        return ""

    text = text.replace("\x00", " ")

    text = re.sub(
        r"[\x01-\x08\x0B\x0C\x0E-\x1F\x7F]",
        " ",
        text,
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


def validate_text_length(
    text: str,
    minimum: int = 20,
    maximum: int = 100_000,
):
    if len(text) < minimum:
        raise ValueError(
            f"Text must contain at least {minimum} characters."
        )

    if len(text) > maximum:
        raise ValueError(
            f"Text cannot exceed {maximum:,} characters."
        )

    return True
