import pytest

from hms_api.engine.llm_wrapper import sanitize_llm_output


@pytest.mark.parametrize(
    "input_text, expected",
    [
        # Null bytes stripped
        ("hello\x00world", "helloworld"),
        ("FIRST\u0000PAGE", "FIRSTPAGE"),
        # Multiple null bytes
        ("\x00\x00text\x00", "text"),
        # Other control characters stripped (non-whitespace)
        ("text\x01\x02\x03end", "textend"),
        ("text\x08end", "textend"),  # backspace
        ("text\x0cend", "textend"),  # form feed
        ("text\x0bend", "textend"),  # vertical tab
        ("text\x1fend", "textend"),  # unit separator
        ("text\x7fend", "textend"),  # DEL
        # Whitespace preserved
        ("hello\tworld", "hello\tworld"),
        ("hello\nworld", "hello\nworld"),
        ("hello\r\nworld", "hello\r\nworld"),
        # Unicode surrogates stripped
        ("text\ud800end", "textend"),
        ("text\udfffend", "textend"),
        # Clean text unchanged
        ("normal text", "normal text"),
        ("unicode: café naïve", "unicode: café naïve"),
        # Edge cases
        ("", ""),
        (None, None),
    ],
)
def test_sanitize_llm_output(input_text, expected):
    assert sanitize_llm_output(input_text) == expected
