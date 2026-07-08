"""Body tag extraction module."""

import logging
import re

import mistune

log = logging.getLogger(__name__)

# Retained: Obsidian tag regex (scans parser-filtered prose text)
INLINE_TAG_PATTERN: re.Pattern[str] = re.compile(
    r"#([a-zA-Z0-9_/-]+(?:\.[a-zA-Z0-9_/-]+)*)",
    re.MULTILINE,
)

# Module-level singleton (Detail Q5): amortizes parser construction across
# bulk ingestion. renderer='ast' returns list[dict] of plain block tokens.
_MD_AST: mistune.Markdown = mistune.create_markdown(renderer="ast")

# Block-token types whose text children are eligible for tag extraction
# (prose containers). Headings, code blocks, blank lines, thematic breaks,
# and inline_html are NOT in this set.
_PROSE_BLOCK_TYPES: frozenset[str] = frozenset(
    {"paragraph", "block_quote", "list", "list_item", "block_text"}
)

_SKIP_BLOCK_TYPES: frozenset[str] = frozenset(
    {"block_code", "heading", "blank_line", "thematic_break", "inline_html"}
)


def _walk_block(token: dict, texts: list[str]) -> None:
    """Recursively walk a block token, delegating to inline walk for prose containers.

    Args:
        token: A mistune AST block-token dict.
        texts: Mutable list to collect `raw` strings from eligible `text` tokens.

    """
    t_type = token.get("type", "")
    if t_type in _SKIP_BLOCK_TYPES:
        return
    children = token.get("children")
    if not children:
        return
    walk_fn = _walk_inline if t_type in _PROSE_BLOCK_TYPES else _walk_block
    for child in children:
        walk_fn(child, texts)


def _walk_inline(token: dict, texts: list[str]) -> None:
    """Recursively walk an inline token, collecting `raw` text and skipping codespan.

    Args:
        token: A mistune AST inline-token dict.
        texts: Mutable list to collect `raw` strings from eligible `text` tokens.

    """
    t_type = token.get("type", "")
    if t_type == "codespan":
        return
    raw = token.get("raw", "")
    if t_type == "text" and raw:
        texts.append(raw)
        return
    children = token.get("children")
    if children:
        for child in children:
            _walk_inline(child, texts)


def _extract_prose_text(tokens: list[dict]) -> list[str]:
    """Recursively collect raw text from prose inline children, skipping code/headings.

    Args:
        tokens: List of mistune AST block-token dicts (or nested children).

    Returns:
        List of `raw` strings from `text` inline tokens found inside
        eligible prose block types. Code blocks, codespans, and headings
        contribute no text.

    """
    _msg = "_extract_prose_text starting"
    log.debug(_msg)

    texts: list[str] = []
    for token in tokens:
        _walk_block(token, texts)

    _msg = f"_extract_prose_text returning {len(texts)} text tokens"
    log.debug(_msg)
    return texts


def _collect_unique_tags(prose_text: str) -> list[str]:
    """Collect unique, lowercased tags from parser-filtered prose text.

    Args:
        prose_text: Concatenated raw text from eligible mistune `text` tokens.

    Returns:
        List of unique tags in the order they first appear.

    """
    tags: list[str] = []
    seen: set[str] = set()

    for match in INLINE_TAG_PATTERN.finditer(prose_text):
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
        Uses a mistune AST walk (renderer='ast') to extract tags only
        from prose (paragraphs, blockquotes, list items). Headings,
        fenced code blocks (any backtick length), indented code blocks,
        and inline code spans are excluded. Escaped \\#tag and decimal
        &#35;tag are cleanly excluded; hex &#x23;tag leaks x23 (known
        limitation).

    """
    _msg = "extract_body_tags starting"
    log.debug(_msg)

    if not content or not content.strip():
        _msg = "extract_body_tags returning None (empty content)"
        log.debug(_msg)
        return None

    ast = _MD_AST(content)
    if isinstance(ast, str):
        _msg = "extract_body_tags returning None (unexpected string output from parser)"
        log.debug(_msg)
        return None
    prose_texts = _extract_prose_text(ast)
    joined = "\n".join(prose_texts)
    tags = _collect_unique_tags(joined)

    if tags:
        _msg = f"extract_body_tags returning {len(tags)} tags"
        log.debug(_msg)
        return tags

    _msg = "extract_body_tags returning None (no tags found)"
    log.debug(_msg)
    return None
