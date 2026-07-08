"""Regression tests for triple-backtick-prose-mention bug in body tag extraction.

These tests isolate the bug where INLINE_CODE_PATTERN = re.compile(r"`[^`]+`")
matches across newlines and misaligns backtick pairs when prose mentions ```,
producing concatenated garbage tags.
"""

from obsidian_rag.parsing.body_tags import _strip_code_blocks, extract_body_tags

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
    """REQ-002: inline code with a newline inside must not cross lines.

    The current buggy regex `` `[^`]+` `` matches across newlines, incorrectly
    removing '#faketag'. After the fix the pattern cannot cross \\n, so the span
    is broken and '#faketag' remains as bare prose text. The critical invariant
    is that no single giant concatenated tag is formed across the newline.
    """
    text = "`start of span\n#faketag end of span`\n#realtag\n"
    stripped = _strip_code_blocks(text)
    assert "#faketag" in stripped, (
        "Inline-code regex crossed newline and incorrectly removed #faketag. "
        f"Stripped: {stripped!r}"
    )
    result = extract_body_tags(text)
    assert result is not None, "Expected at least realtag"
    for tag in result:
        assert "\n" not in tag, f"Tag {tag!r} contains a newline"
        assert len(tag) < 50, f"Tag {tag!r} looks like giant garbage"


def test_prose_mention_empty_triple_backticks() -> None:
    """Empty prose mention ``` ``` ``` followed by a real #tag.

    The third ``` is followed by a newline, which the unclosed-fenced-block
    layer correctly treats as an opening fence (defensive behavior). That
    consumes the entire remainder of the text, including #tag, so no tags
    survive stripping. This is the correct behavior for ambiguous input.
    """
    text = "``` ``` ```\n#tag\n"
    result = extract_body_tags(text)
    assert result is None, (
        f"Expected None (unclosed fence consumes tag line), got {result}"
    )


def test_prose_mention_with_language_id_inline() -> None:
    """```python ... ``` inline in prose + #tag.

    The current code misaligns on the triple backticks, leaving stray backticks
    and potentially leaking the language word. After the fix no stray backticks
    must remain and only #tag must be extracted.
    """
    text = "Use ```python ... ``` inline in prose. #tag\n"
    stripped = _strip_code_blocks(text)
    assert "````" not in stripped, (
        f"Misaligned backtick stripping left stray backticks. Stripped: {stripped!r}"
    )
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

    Also verifies that surrounding prose with triple-backtick mentions does not
    corrupt extraction of the quoted tag.
    """
    text = (
        'Inline tag "#personal/expenses" IS a valid Obsidian tag.\n'
        "`_strip_code_blocks()` removes fenced code blocks (``` ... ```).\n"
    )
    stripped = _strip_code_blocks(text)
    assert "````" not in stripped, (
        f"Triple-backtick mention left stray backticks. Stripped: {stripped!r}"
    )
    result = extract_body_tags(text)
    assert result is not None
    assert "personal/expenses" in result, f"Missing personal/expenses in {result}"
