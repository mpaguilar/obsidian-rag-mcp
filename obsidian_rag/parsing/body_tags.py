"""Body tag extraction module."""

import logging
import re

log = logging.getLogger(__name__)

# Regex pattern for Obsidian tags (NOT headings, NOT all-numeric)
# Key rules: # must not be followed by whitespace (heading),
# tag must contain at least one non-numerical character,
# valid chars: letters, digits, underscore, hyphen, forward slash, dot;
# dots are only kept when followed by more tag characters (so #tag. yields "tag")
INLINE_TAG_PATTERN: re.Pattern[str] = re.compile(
    r"#([a-zA-Z0-9_/-]+(?:\.[a-zA-Z0-9_/-]+)*)",
    re.MULTILINE,
)

# Regex pattern for fenced code blocks (``` ... ```)
FENCED_CODE_PATTERN: re.Pattern[str] = re.compile(r"```[\w]*\n.*?```", re.DOTALL)

# Regex pattern for inline code (` ... `) — single-line only (excludes newlines)
INLINE_CODE_PATTERN: re.Pattern[str] = re.compile(r"`[^`\n]+`")

# Regex pattern for triple-backtick prose mentions (```...``` appearing as literal text)
PROSE_MENTION_PATTERN: re.Pattern[str] = re.compile(r"```[^`\n]*```")


def _strip_code_blocks(content: str) -> str:
    """Remove fenced code blocks, prose triple-backtick mentions, and inline code.

    Args:
        content: Raw markdown content.

    Returns:
        Content with code blocks, triple-backtick prose mentions, and inline
        code removed. Inline code spans are single-line (newline-excluded)
        per Obsidian's inline-code behavior.

    Notes:
        Layered stripping order:
        1. Properly closed fenced code blocks (multi-line, DOTALL).
        2. Unclosed fenced blocks (opening ``` to EOF, defensive).
        3. Triple-backtick prose mentions (```...``` appearing as literal text on a single line of prose that merely DESCRIBES fenced syntax rather than being one).
        4. Single-backtick inline code (single-line only — excludes newlines).

    """
    _msg = "_strip_code_blocks starting"
    log.debug(_msg)

    result = FENCED_CODE_PATTERN.sub("", content)
    unclosed_pattern = re.compile(r"```[\w]*\n.*$", re.DOTALL)
    result = unclosed_pattern.sub("", result)
    result = PROSE_MENTION_PATTERN.sub("", result)  # NEW prose-mention layer
    result = INLINE_CODE_PATTERN.sub("", result)  # tightened: no newlines

    _msg = "_strip_code_blocks returning"
    log.debug(_msg)
    return result


def _collect_unique_tags(stripped: str) -> list[str]:
    """Collect unique, lowercased tags from stripped markdown content.

    Args:
        stripped: Markdown content with code blocks removed.

    Returns:
        List of unique tags in the order they first appear.

    """
    tags: list[str] = []
    seen: set[str] = set()

    for match in INLINE_TAG_PATTERN.finditer(stripped):
        tag_text = match.group(1)

        # REQ-008: Exclude all-numeric tags (#1984 is NOT a tag)
        if tag_text.isdigit():
            continue

        # Normalize: lowercase (the # prefix is already stripped in group(1))
        normalized = tag_text.lower()
        if normalized not in seen:
            seen.add(normalized)
            tags.append(normalized)

    return tags


def extract_body_tags(content: str | None) -> list[str] | None:
    """Extract inline Obsidian tags from markdown body content.

    Args:
        content: Markdown body content (after frontmatter removal).
            None or empty content returns None.

    Returns:
        Deduplicated list of tags (with # prefix stripped), or None
        if no tags found. Tags are lowercased.

    Notes:
        Follows Obsidian tag recognition rules:
        - # followed by space is a heading (NOT a tag)
        - All-numeric #1984 is NOT a tag
        - Tags inside code blocks/inline code are NOT extracted
        - Tags in blockquotes/callouts ARE extracted
        - Hierarchical tags (personal/expenses) ARE extracted

    """
    _msg = "extract_body_tags starting"
    log.debug(_msg)

    if not content or not content.strip():
        _msg = "extract_body_tags returning None (empty content)"
        log.debug(_msg)
        return None

    stripped = _strip_code_blocks(content)
    tags = _collect_unique_tags(stripped)

    if tags:
        _msg = f"extract_body_tags returning {len(tags)} tags"
        log.debug(_msg)
        return tags

    _msg = "extract_body_tags returning None (no tags found)"
    log.debug(_msg)
    return None
