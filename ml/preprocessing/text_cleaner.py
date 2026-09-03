import re
import unicodedata


def normalize_unicode(text: str) -> str:
    """
    Normalize Unicode characters.
    """

    if not isinstance(text, str):
        return ""

    return unicodedata.normalize("NFKC", text)


def normalize_whitespace(text: str) -> str:
    """
    Replace multiple spaces, tabs and newlines
    with a single space.
    """

    text = re.sub(r"\s+", " ", text)

    return text.strip()


def normalize_urls(text: str) -> str:
    """
    Replace URLs with a common token.
    """

    url_pattern = r"https?://\S+|www\.\S+"

    text = re.sub(
        url_pattern,
        " URL ",
        text,
        flags=re.IGNORECASE
    )

    return text


def normalize_emails(text: str) -> str:
    """
    Replace email addresses with a common token.
    """

    email_pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"

    text = re.sub(
        email_pattern,
        " EMAIL ",
        text
    )

    return text


def normalize_repeated_characters(text: str) -> str:
    """
    Reduce excessive repeated characters.

    Example:
    amaaaaazing -> amaazing
    """

    text = re.sub(
        r"(.)\1{3,}",
        r"\1\1",
        text
    )

    return text


def clean_text(text: str) -> str:
    """
    Main text preprocessing function.

    This intentionally performs LIGHT preprocessing.
    We do not remove stopwords or punctuation aggressively
    because those features may contain useful information
    for fake-news classification.
    """

    if not isinstance(text, str):
        return ""

    # Unicode normalization
    text = normalize_unicode(text)

    # URL normalization
    text = normalize_urls(text)

    # Email normalization
    text = normalize_emails(text)

    # Repeated character normalization
    text = normalize_repeated_characters(text)

    # Convert whitespace
    text = normalize_whitespace(text)

    return text
