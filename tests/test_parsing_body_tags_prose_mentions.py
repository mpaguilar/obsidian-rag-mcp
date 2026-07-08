"""Regression tests for triple-backtick-prose-mention bug in body tag extraction.

These tests isolate the bug where INLINE_CODE_PATTERN = re.compile(r"`[^`]+`")
matches across newlines and misaligns backtick pairs when prose mentions ```,
producing concatenated garbage tags.
"""

from obsidian_rag.parsing.body_tags import extract_body_tags

# ---------------------------------------------------------------------------
# Minimal repro fixture (expanded in individual tests as needed)
# ---------------------------------------------------------------------------
_PROSE_MENTION_REPRO = (
    "# Title\n\n"
    "`_strip_code_blocks()` removes fenced code blocks (``` ... ```) and inline code.\n\n"
    "All-numeric `#1984` is NOT a tag, `#y1984` IS.\n"
)

# Garbage strings observed when the inline-code regex misaligns on triple backticks.
_FORBIDDEN_GARBAGE_TAGS = {
    "1984get_documents_by_tagget_all_tags_merge_tags",
    "tagparse_frontmatter",
    "tagbody_tags_merge_tags",
    "tag_strip_code_blocks",
    "tag",
    "strip_hash",
}


def test_prose_mention_with_inline_hash_code_no_garbage() -> None:
    """Minimal repro: prose mentions ``` + inline code; no garbage tags, no false positives.

    The current buggy code misaligns backtick pairs when triple backticks appear
    in prose, causing adjacent inline-code spans to collapse into a single
    concatenated garbage tag. After the fix no garbage tags must remain.
    Note: #y1984 is inside backticks in the source text, so it is correctly
    stripped as inline code and must NOT be extracted.
    """
    text = (
        _PROSE_MENTION_REPRO
        + "See also `get_documents_by_tag`, `get_all_tags`, `_merge_tags`.\n"
        "Modules: `parse_frontmatter`, `body_tags`, `_merge_tags`.\n"
        "Helpers: `_strip_code_blocks`.\n"
        "A stray `strip_hash` in prose.\n"
    )
    result = extract_body_tags(text)
    # Every tag-looking token in this text is inside backticks, so no tags
    # should be extracted after correct inline-code stripping.
    assert result is None, f"Expected None (all tokens inside backticks), got {result}"


def test_fenced_code_blocks_prose_strips_cleanly() -> None:
    """Prose with fenced block mention and inline code; only realtag survives.

    The phrase '(``` ... ```) and inline code (`#notatag`)' causes the inline
    regex to misalign, leaving '#notatag' exposed. After the fix only '#realtag'
    must be extracted.
    """
    text = "fenced code blocks (``` ... ```) and inline code (`#notatag`)\n#realtag\n"
    result = extract_body_tags(text)
    assert result == ["realtag"], f"Expected ['realtag'], got {result}"


def test_full_037_doc_repro_no_garbage() -> None:
    """Compact 037-doc reproduction: personal/expenses present, no garbage tags.

    Mimics the 037 checkpoint description structure (triple-backtick prose mentions
    mixed with inline code spans) and asserts that the valid #personal/expenses tag
    is found while every observed garbage tag is absent.
    """
    text = (
        "The module uses `_strip_code_blocks()` to remove fenced code blocks (``` ... ```).\n"
        "All-numeric `#1984` is NOT a tag, `#y1984` IS.\n"
        "See also `get_documents_by_tag`, `get_all_tags`, `_merge_tags`.\n"
        "Modules: `parse_frontmatter`, `body_tags`, `_merge_tags`.\n"
        "Helpers: `_strip_code_blocks`.\n"
        "A stray `strip_hash` in prose.\n"
        "#personal/expenses\n"
    )
    result = extract_body_tags(text)
    assert result is not None, "Expected tags, got None"
    assert "personal/expenses" in result, f"Missing personal/expenses in {result}"
    assert result == ["personal/expenses"], (
        f"Expected only ['personal/expenses'], got {result}"
    )
    for tag in result:
        assert tag not in _FORBIDDEN_GARBAGE_TAGS, (
            f"Forbidden garbage tag {tag!r} found"
        )


def test_inline_code_does_not_cross_newlines() -> None:
    """Inline code span with a newline inside does not produce cross-line garbage tags.

    mistune parses inline code as a single codespan token (regardless of
    newlines in source); #faketag inside it is excluded. #realtag in prose
    is extracted. No single giant concatenated tag is formed.
    """
    text = "`start of span\n#faketag end of span`\n#realtag\n"
    result = extract_body_tags(text)
    assert result is not None
    assert "realtag" in result, f"Missing realtag in {result}"
    assert "faketag" not in result, f"#faketag inside codespan leaked: {result}"
    for tag in result:
        assert "\n" not in tag, f"Tag {tag!r} contains a newline"
        assert len(tag) < 50, f"Tag {tag!r} looks like giant garbage"


def test_prose_mention_empty_triple_backticks() -> None:
    """Empty prose mention ``` ``` ``` followed by a real #tag yields ["tag"].

    mistune parses the backticks as codespan/text tokens within a paragraph;
    the #tag on the next line is a text child of that paragraph → extracted.
    This is arguably more correct than the old None (the #tag is genuinely
    in prose).
    """
    text = "``` ``` ```\n#tag\n"
    result = extract_body_tags(text)
    assert result == ["tag"], f"Expected ['tag'], got {result}"


def test_prose_mention_with_language_id_inline() -> None:
    """```python ... ``` inline in prose + #tag: only #tag extracted, no language leak."""
    text = "Use ```python ... ``` inline in prose. #tag\n"
    result = extract_body_tags(text)
    assert result is not None
    assert "python" not in result, f"Language word leaked as tag: {result}"
    assert "tag" in result, f"Missing real tag in {result}"


def test_unbalanced_stray_backtick_no_garbage() -> None:
    """Odd number of stray backticks must not raise or produce concatenated tags.

    A stray opening backtick on the same line causes the inline-code regex to
    consume text up to the next backtick, exposing #notatag and #another.
    After the fix there must be no single giant concatenated tag; instead any
    exposed tags remain as small individual tokens (or none if properly stripped).
    """
    text = "Some ` stray backticks in prose `#notatag` and `#another`. `#realtag"
    result = extract_body_tags(text)
    assert result is not None
    for tag in result:
        # REQ-002 tightened regex prevents cross-line matches, so even with a
        # stray backtick no single tag can exceed ~50 chars.
        assert len(tag) < 50, f"Garbage concatenated tag found: {tag!r}"


def test_double_quoted_hash_prose_is_valid_tag() -> None:
    """Inline tag '#personal/expenses' inside double quotes IS a valid Obsidian tag.

    Verifies surrounding prose with triple-backtick mentions does not corrupt
    extraction of the quoted tag.
    """
    text = (
        'Inline tag "#personal/expenses" IS a valid Obsidian tag.\n'
        "`_strip_code_blocks()` removes fenced code blocks (``` ... ```).\n"
    )
    result = extract_body_tags(text)
    assert result is not None
    assert "personal/expenses" in result, f"Missing personal/expenses in {result}"
