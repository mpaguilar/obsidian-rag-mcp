"""Comprehensive edge case tests for body tag extraction.

Tests ``extract_body_tags`` and ``_merge_tags`` with real implementations
to verify edge cases specified in the body-tag feature requirements.
"""

from obsidian_rag.parsing.body_tags import extract_body_tags
from obsidian_rag.services.tag_merging import _merge_tags


def test_document_no_frontmatter_no_body_tags() -> None:
    """Empty body with no frontmatter tags yields None from extract_body_tags."""
    result = extract_body_tags("")
    assert result is None


def test_document_no_frontmatter_body_tags_only() -> None:
    """Body containing only inline tags returns those tags."""
    result = extract_body_tags("#tag1 and #tag2 here")
    assert result == ["tag1", "tag2"]


def test_document_frontmatter_and_body_same_tag() -> None:
    """Frontmatter 'Work' merged with body '#work' yields single 'work' tag."""
    frontmatter_tags = ["Work"]
    body_tags = extract_body_tags("Some text #work here")
    assert body_tags == ["work"]

    merged = _merge_tags(frontmatter_tags, body_tags)
    assert merged == ["work"]
    assert merged.count("work") == 1


def test_document_case_insensitive_dedup() -> None:
    """Frontmatter 'Personal/Expenses' and body '#personal/expenses' collapse."""
    frontmatter_tags = ["Personal/Expenses"]
    body_tags = extract_body_tags("Spending #personal/expenses today")
    assert body_tags == ["personal/expenses"]

    merged = _merge_tags(frontmatter_tags, body_tags)
    assert merged == ["personal/expenses"]


def test_heading_h1_not_tag() -> None:
    """A markdown H1 heading is not extracted as a tag."""
    result = extract_body_tags("# Heading\n")
    assert result is None


def test_heading_h2_not_tag() -> None:
    """A markdown H2 heading is not extracted as a tag."""
    result = extract_body_tags("## Sub-heading\n")
    assert result is None


def test_heading_h3_not_tag() -> None:
    """A markdown H3 heading is not extracted as a tag."""
    result = extract_body_tags("### Deep heading\n")
    assert result is None


def test_heading_with_tag_same_line() -> None:
    """Heading text with a tag on the same line extracts only the tag."""
    result = extract_body_tags("# Heading text #tag")
    assert result == ["tag"]
    assert "heading" not in (result or [])


def test_fenced_code_block_tags_excluded() -> None:
    """Tags inside fenced code blocks are NOT extracted."""
    result = extract_body_tags("```\n#code-tag\n```\n#real-tag")
    assert result == ["real-tag"]
    assert "code-tag" not in (result or [])


def test_inline_code_tag_excluded() -> None:
    """Tags inside inline code spans are NOT extracted."""
    result = extract_body_tags("Use `#inline` and #real-tag")
    assert result == ["real-tag"]
    assert "inline" not in (result or [])


def test_all_numeric_1984_not_tag() -> None:
    """All-numeric token like #1984 is NOT extracted as a tag."""
    result = extract_body_tags("Read #1984 today.")
    assert result is None or "1984" not in result


def test_y1984_is_tag() -> None:
    """Mixed alphanumeric token #y1984 IS extracted as a tag."""
    result = extract_body_tags("#y1984")
    assert result == ["y1984"]


def test_tag_in_blockquote() -> None:
    """Tags inside markdown blockquotes ARE extracted."""
    result = extract_body_tags("> #blockquote-tag")
    assert result == ["blockquote-tag"]


def test_tag_in_callout() -> None:
    """Tags inside Obsidian callouts ARE extracted."""
    result = extract_body_tags("> [!note] #callout-tag")
    assert result == ["callout-tag"]


def test_tag_at_line_start_no_space() -> None:
    """Tag at line start without space after # IS extracted."""
    result = extract_body_tags("#personal/expenses\nmore text")
    assert result == ["personal/expenses"]


def test_multiple_hierarchical_tags() -> None:
    """Multiple hierarchical tags are returned in order."""
    result = extract_body_tags("#a/b #c/d/e")
    assert result == ["a/b", "c/d/e"]


def test_unclosed_code_block_defensive() -> None:
    """Unclosed fenced code block strips to EOF; tags after are excluded."""
    result = extract_body_tags("#before-tag\n```\n#after-tag")
    assert result == ["before-tag"]
    assert "after-tag" not in result
