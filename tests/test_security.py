from backend.app.core.security import (
    sanitize_text,
    validate_text_length,
)


def test_null_character_removed():

    text = "Hello\x00World"

    result = sanitize_text(
        text
    )

    assert "\x00" not in result


def test_whitespace_normalized():

    text = "Hello     World"

    result = sanitize_text(
        text
    )

    assert result == "Hello World"


def test_valid_text_length():

    text = "A" * 50

    assert validate_text_length(
        text
    ) is True


def test_short_text_rejected():

    text = "Too short"

    try:

        validate_text_length(
            text,
            minimum=20,
        )

        assert False

    except ValueError:

        assert True


def test_long_text_rejected():

    text = "A" * 101

    try:

        validate_text_length(
            text,
            maximum=100,
        )

        assert False

    except ValueError:

        assert True
