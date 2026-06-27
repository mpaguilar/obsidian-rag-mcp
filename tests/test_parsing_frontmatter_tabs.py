"""Tests for frontmatter tab indentation detection."""

import pytest

from obsidian_rag.parsing.frontmatter import (
    FrontMatterParsingError,
    _has_indentation_tabs,
    extract_frontmatter,
    parse_frontmatter,
)


def test_has_indentation_tabs_detects_tabs_at_line_start() -> None:
    """Tabs at indentation position are detected on all lines."""
    yaml_content = "\tkey: value\n\tnested:\n\t\tchild: true"
    result = _has_indentation_tabs(yaml_content)
    assert result == [1, 2, 3]


def test_has_indentation_tabs_accepts_tabs_in_quoted_values() -> None:
    """Tabs inside quoted string values are not flagged."""
    yaml_content = 'key: "value\twith\ttabs"'
    result = _has_indentation_tabs(yaml_content)
    assert result == []


def test_has_indentation_tabs_mixed_indentation_and_value_tabs() -> None:
    """Only indentation tabs are flagged; value tabs are ignored."""
    yaml_content = 'key: "value\twith\ttabs"\n\tnested: true\n  \tmixed: value'
    result = _has_indentation_tabs(yaml_content)
    assert result == [2, 3]


def test_has_indentation_tabs_no_tabs() -> None:
    """Clean YAML without tabs returns empty list."""
    yaml_content = "key: value\nnnested:\n  child: true"
    result = _has_indentation_tabs(yaml_content)
    assert result == []


def test_has_indentation_tabs_empty_content() -> None:
    """Empty string returns empty list."""
    result = _has_indentation_tabs("")
    assert result == []


def test_has_indentation_tabs_tabs_at_line_start_after_spaces() -> None:
    """Mixed space and tab indentation is flagged."""
    yaml_content = "  \tkey: value"
    result = _has_indentation_tabs(yaml_content)
    assert result == [1]


def test_has_indentation_tabs_only_tab_indentation() -> None:
    """A line starting with a tab is flagged."""
    yaml_content = "\tkey: value"
    result = _has_indentation_tabs(yaml_content)
    assert result == [1]


def test_has_indentation_tabs_multiple_lines_with_tabs() -> None:
    """Multiple lines with tab indentation are all reported."""
    yaml_content = "\tone\n\ttwo\n  three\nfour"
    result = _has_indentation_tabs(yaml_content)
    assert result == [1, 2]


def test_extract_frontmatter_raises_error_for_indentation_tabs() -> None:
    """extract_frontmatter raises FrontMatterParsingError for tab-indented frontmatter."""
    content = "---\n\tkey: value\n---\nBody text"
    with pytest.raises(FrontMatterParsingError):
        extract_frontmatter(content)


def test_extract_frontmatter_error_message_contains_line_numbers() -> None:
    """Error message includes line numbers where tabs were found."""
    content = "---\n\tkey: value\n---\nBody text"
    with pytest.raises(FrontMatterParsingError) as exc_info:
        extract_frontmatter(content)
    assert "1" in str(exc_info.value)


def test_extract_frontmatter_error_message_recommends_spaces() -> None:
    """Error message recommends replacing tabs with spaces."""
    content = "---\n\tkey: value\n---\nBody text"
    with pytest.raises(FrontMatterParsingError) as exc_info:
        extract_frontmatter(content)
    assert "replace tabs with spaces" in str(exc_info.value).lower()


def test_extract_frontmatter_accepts_tabs_in_quoted_string_values() -> None:
    """Tabs inside quoted string values are accepted."""
    content = '---\nkey: "value\twith\ttabs"\n---\nBody text'
    frontmatter, remaining = extract_frontmatter(content)
    assert frontmatter == {"key": "value\twith\ttabs"}
    assert remaining == "Body text"


def test_extract_frontmatter_still_handles_yaml_error_for_non_tab_issues() -> None:
    """Non-tab YAML errors are caught by the yaml.YAMLError handler."""
    content = '---\nkey: "unclosed string\n---\nBody text'
    frontmatter, remaining = extract_frontmatter(content)
    assert frontmatter == {}
    assert remaining == "Body text"


def test_parse_frontmatter_propagates_tab_error() -> None:
    """parse_frontmatter propagates FrontMatterParsingError."""
    content = "---\n\tkey: value\n---\nBody text"
    with pytest.raises(FrontMatterParsingError):
        parse_frontmatter(content)


def test_extract_frontmatter_no_frontmatter_not_checked_for_tabs() -> None:
    """Content without frontmatter delimiters is not checked for tabs."""
    content = "\tkey: value\nBody text"
    frontmatter, remaining = extract_frontmatter(content)
    assert frontmatter == {}
    assert remaining == content
