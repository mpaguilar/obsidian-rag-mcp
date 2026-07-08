"""Comprehensive edge case tests for body tag extraction.

Tests ``extract_body_tags`` and ``_merge_tags`` with real implementations
to verify edge cases specified in the body-tag feature requirements.
"""

from unittest.mock import patch

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
    """Heading text with a tag on the same line excludes the tag (heading skipped)."""
    result = extract_body_tags("# Heading text #tag")
    assert result is None
    assert "tag" not in (result or [])


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


def test_four_backtick_fence_tags_excluded() -> None:
    """4-backtick fenced code block (length-matching fence) excludes inner tags.

    mistune emits block_code with marker='````' for 4-backtick fences — a
    capability the old regex could not express (no backreference). Only the
    real tag after the fence is extracted.
    """
    result = extract_body_tags("before\n````\n#code\n````\nafter #real")
    assert result == ["real"]
    assert "code" not in (result or [])


def test_indented_code_block_tags_excluded() -> None:
    """4-space indented code block excludes inner tags (NEW capability).

    The old regex did NOT handle indented code blocks at all (leaked tags).
    mistune emits block_code with style='indent'. Only the real tag is extracted.
    """
    result = extract_body_tags("    #indented\n\n#real")
    assert result == ["real"]
    assert "indented" not in (result or [])


def test_setext_heading_excludes_tag() -> None:
    """Setext heading (underline ===) excludes the tag on the heading line (REQ-005).

    mistune emits a heading token with style='setext' for 'Heading #tag\n==='.
    """
    result = extract_body_tags("Heading #tag\n===")
    assert result is None
    assert "tag" not in (result or [])


def test_escaped_tag_excluded() -> None:
    """Escaped \\#tag is cleanly excluded via mistune token split (REQ-002).

    mistune splits \\#tag into text{'#'} + text{'tag ...'}; the bare '#'
    token has no tag chars after it -> no match. A real #tag in the same
    line IS extracted.
    """
    result = extract_body_tags("\\#tag and #real")
    assert result == ["real"]
    assert "tag" not in (result or [])


def test_decimal_entity_tag_excluded() -> None:
    """Decimal HTML entity &#35;tag is excluded (REQ-002).

    mistune preserves &#35;tag literally; regex matches #35 (all-numeric)
    -> excluded by isdigit(). The ';' boundary also breaks the tag match.
    """
    result = extract_body_tags("&#35;tag here")
    assert result is None
    assert "35" not in (result or [])


def test_hex_entity_tag_limitation() -> None:
    """Hex HTML entity &#x23;tag leaks x23 as a false-positive (DOCUMENTED LIMITATION, REQ-002).

    mistune preserves &#x23;tag literally; regex matches #x23 (NOT
    all-numeric) -> x23 is extracted. This is an accepted, documented
    limitation (extremely rare in real Obsidian notes).
    """
    result = extract_body_tags("&#x23;tag here")
    assert result == ["x23"], f"Expected documented limitation ['x23'], got {result}"


def test_block_html_no_children_skipped() -> None:
    """Raw HTML block (block_html token with no children) is skipped gracefully.

    mistune emits ``block_html`` tokens for raw HTML blocks; these have
    no ``children`` key. The walker must not crash and should simply skip
    them (hits the ``not children`` guard in ``_walk_block``).
    """
    result = extract_body_tags("<div></div>\n\n#real")
    assert result == ["real"]
    assert "div" not in (result or [])


def test_ast_returns_string_defensive() -> None:
    """Defensive path when mistune returns a string instead of AST list.

    The ``_MD_AST`` renderer normally returns ``list[dict]``, but the
    code defensively handles a string return to avoid crashing during
    bulk ingestion.
    """
    from obsidian_rag.parsing import body_tags

    with patch.object(body_tags, "_MD_AST", return_value="not a list"):
        result = extract_body_tags("some #tag text")
        assert result is None
