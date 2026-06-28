"""Unit tests for _normalize_indentation_tabs helper."""

from obsidian_rag.parsing.frontmatter import _normalize_indentation_tabs


def test_normalize_pure_tab_indentation() -> None:
    """Leading tabs converted to 2 spaces each."""
    result = _normalize_indentation_tabs("\tkey: value")
    assert result == "  key: value"


def test_normalize_tab_after_spaces() -> None:
    """Mixed space+tab indentation: tabs replaced, spaces preserved."""
    result = _normalize_indentation_tabs("  \tkey: value")
    assert result == "    key: value"


def test_normalize_multi_level_nesting_preserved() -> None:
    """Multiple tab levels produce correct 2-space nesting."""
    result = _normalize_indentation_tabs("\tkey:\n\t\tsub: val")
    assert result == "  key:\n    sub: val"


def test_normalize_quoted_value_tabs_untouched() -> None:
    """Tabs inside quoted string values are NOT converted."""
    result = _normalize_indentation_tabs('key: "a\tb"')
    assert result == 'key: "a\tb"'


def test_normalize_mid_line_tab_untouched() -> None:
    """Tabs after non-whitespace characters in mid-line are NOT converted."""
    result = _normalize_indentation_tabs("key: a\tb")
    assert result == "key: a\tb"


def test_normalize_blank_tab_only_lines() -> None:
    """Lines containing only tab characters are normalized harmlessly."""
    result = _normalize_indentation_tabs("\t")
    assert result == "  "


def test_normalize_no_tabs_passthrough() -> None:
    """Content without tabs returns identical string."""
    input_str = "key: value\n  sub: val"
    result = _normalize_indentation_tabs(input_str)
    assert result == input_str


def test_normalize_empty_string() -> None:
    """Empty string input returns empty string."""
    result = _normalize_indentation_tabs("")
    assert result == ""


def test_normalize_consecutive_tabs() -> None:
    """Multiple consecutive tabs in leading run each get 2 spaces."""
    result = _normalize_indentation_tabs("\t\tkey: value")
    assert result == "    key: value"


def test_normalize_preserves_non_leading_content() -> None:
    """Content after leading whitespace is byte-for-byte identical."""
    result = _normalize_indentation_tabs("\tkey: a\tb")
    assert result == "  key: a\tb"
