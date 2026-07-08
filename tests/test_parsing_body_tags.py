"""Tests for body tag extraction module."""

from obsidian_rag.parsing.body_tags import extract_body_tags


def test_extract_body_tags_no_tags() -> None:
    """Content with no tags returns None."""
    result = extract_body_tags("Just plain text.")
    assert result is None


def test_extract_body_tags_none_content() -> None:
    """None content returns None."""
    assert extract_body_tags(None) is None


def test_extract_body_tags_single_tag() -> None:
    """Single #tag returns [tag]."""
    result = extract_body_tags("This has a #tag.")
    assert result == ["tag"]


def test_extract_body_tags_multiple_tags() -> None:
    """Multiple different tags are returned."""
    result = extract_body_tags("#first and #second here.")
    assert result == ["first", "second"]


def test_extract_body_tags_hierarchical_tag() -> None:
    """Hierarchical tag personal/expenses is extracted."""
    result = extract_body_tags("#personal/expenses")
    assert result == ["personal/expenses"]


def test_extract_body_tags_deduplication() -> None:
    """Same tag appearing twice becomes a single entry."""
    result = extract_body_tags("#tag #tag")
    assert result == ["tag"]


def test_extract_body_tags_case_insensitive_dedup() -> None:
    """#Work and #work are deduplicated case-insensitively."""
    result = extract_body_tags("#Work #work")
    assert result == ["work"]


def test_extract_body_tags_empty_content() -> None:
    """Empty string returns None."""
    assert extract_body_tags("") is None


def test_extract_body_tags_task_line() -> None:
    """Tag inside a task line is extracted as a body tag."""
    result = extract_body_tags("- [ ] do something #important")
    assert result == ["important"]


def test_extract_body_tags_excludes_heading() -> None:
    """# Heading is not extracted as a tag."""
    result = extract_body_tags("# Heading\n")
    assert result is None


def test_extract_body_tags_excludes_h2_heading() -> None:
    """## Heading is not extracted as a tag."""
    result = extract_body_tags("## Heading\n")
    assert result is None


def test_extract_body_tags_excludes_all_numeric() -> None:
    """#1984 is not extracted as a tag."""
    result = extract_body_tags("Read #1984 today.")
    assert "1984" not in (result or [])


def test_extract_body_tags_includes_mixed_alphanumeric() -> None:
    """#y1984 is extracted as a tag."""
    result = extract_body_tags("#y1984")
    assert result == ["y1984"]


def test_extract_body_tags_excludes_tags_in_code_block() -> None:
    """Tags inside fenced code blocks are not extracted."""
    result = extract_body_tags("```\n#code-tag\n```\n#real-tag")
    assert result == ["real-tag"]


def test_extract_body_tags_excludes_tags_in_inline_code() -> None:
    """Tags inside inline code are not extracted."""
    result = extract_body_tags("Use `#inline` and #real-tag")
    assert result == ["real-tag"]


def test_extract_body_tags_includes_tags_in_blockquote() -> None:
    """Tags in blockquotes are extracted."""
    result = extract_body_tags("> #blockquote-tag")
    assert result == ["blockquote-tag"]


def test_extract_body_tags_heading_excludes_tag_same_line() -> None:
    """Headings are excluded entirely, so #tag inside a heading line is skipped."""
    result = extract_body_tags("# Heading with #tag")
    assert result is None


def test_extract_body_tags_tag_at_line_start() -> None:
    """Tag at start of line without space after # is extracted."""
    result = extract_body_tags("#personal/expenses at line start")
    assert result == ["personal/expenses"]


def test_extract_body_tags_dot_tags() -> None:
    """Version-style tag with dot is extracted."""
    result = extract_body_tags("#v1.0/release")
    assert result == ["v1.0/release"]
