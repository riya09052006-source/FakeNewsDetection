from backend.app.core.security import (
    sanitize_text,
    validate_text_length,
)


def test_null_character_removed():

    result = sanitize_text(
        "Hello\x00World"
    )

    assert "\x00" not in result


def test_control_characters_removed():

    result = sanitize_text(
        "Hello\x01World"
    )

    assert "\x01" not in result


def test_whitespace_normalized():

    result = sanitize_text(
        "Hello     World"
    )

    assert result == "Hello World"


def test_valid_length():

    result = validate_text_length(
        "A" * 50
    )

    assert result is True


def test_short_text():

    try:

        validate_text_length(
            "Short",
            minimum=20,
        )

        assert False

    except ValueError:

        assert True


def test_long_text():

    try:

        validate_text_length(
            "A" * 101,
            maximum=100,
        )

        assert False

    except ValueError:

        assert True
