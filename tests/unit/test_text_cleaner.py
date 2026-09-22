from ml.preprocessing.text_cleaner import (
    clean_text,
    normalize_unicode,
    normalize_whitespace,
    normalize_urls,
    normalize_emails,
    normalize_repeated_characters,
)


def test_unicode_normalization():

    text = "café"

    result = normalize_unicode(text)

    assert isinstance(result, str)
    assert len(result) > 0


def test_whitespace_normalization():

    text = "Hello     world\n\nthis   is   news"

    result = normalize_whitespace(text)

    assert result == "Hello world this is news"


def test_url_normalization():

    text = "Visit https://example.com for information."

    result = normalize_urls(text)

    assert "https://example.com" not in result
    assert "URL" in result


def test_email_normalization():

    text = "Contact test@example.com for details."

    result = normalize_emails(text)

    assert "test@example.com" not in result
    assert "EMAIL" in result


def test_repeated_characters():

    text = "This is amaaaaazing"

    result = normalize_repeated_characters(text)

    assert result != text


def test_clean_text():

    text = """
    Visit https://example.com

    Contact test@example.com
    """

    result = clean_text(text)

    assert isinstance(result, str)
    assert len(result) > 0
