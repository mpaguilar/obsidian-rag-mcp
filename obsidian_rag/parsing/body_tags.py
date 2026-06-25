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

# Regex pattern for inline code (` ... `)
INLINE_CODE_PATTERN: re.Pattern[str] = re.compile(r"`[^`]+`")


def _strip_code_blocks(content: str) -> str:
    """Remove fenced code blocks and inline code from content.

    Args:
        content: Raw markdown content.

    Returns:
        Content with code blocks and inline code removed.

    Notes:
        Handles unclosed fenced code blocks defensively:
        strips from opening ``` to end of content.
        Preserves content outside code blocks.

    """
    _msg = "_strip_code_blocks starting"
    log.debug(_msg)

    # Step 1: Remove properly closed fenced code blocks (including empty ones)
    result = FENCED_CODE_PATTERN.sub("", content)

    # Step 2: Remove unclosed fenced blocks (opening ``` to EOF)
    unclosed_pattern = re.compile(r"```[\w]*\n.*$", re.DOTALL)
    result = unclosed_pattern.sub("", result)

    # Step 3: Remove inline code
    result = INLINE_CODE_PATTERN.sub("", result)

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
