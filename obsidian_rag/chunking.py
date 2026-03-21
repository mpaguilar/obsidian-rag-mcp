"""Document chunking module for splitting large documents.

This module provides functionality to split large documents into overlapping
chunks that fit within embedding token limits.
"""

import logging
from dataclasses import dataclass

log = logging.getLogger(__name__)

# Default chunking parameters
# 24,000 characters ≈ 6,000 tokens (assuming ~4 chars per token)
DEFAULT_MAX_CHUNK_CHARS = 24000
DEFAULT_CHUNK_OVERLAP_CHARS = 800  # ~200 tokens overlap

# Paragraph and sentence delimiters
PARAGRAPH_DELIMITER = "\n\n"
SENTENCE_DELIMITERS = [". ", "! ", "? ", "\n"]


@dataclass
class Chunk:
    """Represents a chunk of text from a document.

    Attributes:
        text: The text content of the chunk.
        start_char: Starting character position in the original document.
        end_char: Ending character position in the original document.
        index: Index of this chunk within the document (0-based).

    """

    text: str
    start_char: int
    end_char: int
    index: int


def _estimate_tokens(text: str) -> int:
    """Estimate the number of tokens in text.

    Uses a simple heuristic of ~4 characters per token.
    This is a rough estimate for English text.

    Args:
        text: The text to estimate tokens for.

    Returns:
        Estimated token count.

    Notes:
        This is a rough estimate. Actual token counts depend on
        the specific tokenizer used by the embedding model.

    """
    _msg = "_estimate_tokens starting"
    log.debug(_msg)
    # Rough estimate: ~4 characters per token for English text
    result = len(text) // 4
    _msg = "_estimate_tokens returning"
    log.debug(_msg)
    return result


def _find_split_point(
    text: str,
    target_pos: int,
    max_pos: int,
) -> int:
    """Find the best split point near the target position.

    Tries to split on paragraph boundaries first, then sentence boundaries,
    then falls back to the target position.

    Args:
        text: The text to find a split point in.
        target_pos: The ideal position to split at.
        max_pos: The maximum allowed position (hard limit).

    Returns:
        The actual split position to use.

    """
    _msg = "_find_split_point starting"
    log.debug(_msg)

    # Ensure target doesn't exceed max
    target = min(target_pos, max_pos)

    # Search backwards from target for paragraph boundary
    search_start = max(0, target - 1000)  # Search up to 1000 chars back
    paragraph_pos = text.rfind(PARAGRAPH_DELIMITER, search_start, target)
    if paragraph_pos != -1:
        # Found paragraph boundary, split after it
        result = paragraph_pos + len(PARAGRAPH_DELIMITER)
        _msg = "_find_split_point returning (paragraph boundary)"
        log.debug(_msg)
        return result

    # Search for sentence boundary
    for delimiter in SENTENCE_DELIMITERS:
        sentence_pos = text.rfind(delimiter, search_start, target)
        if sentence_pos != -1:
            # Found sentence boundary, split after it
            result = sentence_pos + len(delimiter)
            _msg = "_find_split_point returning (sentence boundary)"
            log.debug(_msg)
            return result

    # Fall back to target position
    _msg = "_find_split_point returning (fallback)"
    log.debug(_msg)
    return target


def _normalize_chunking_params(
    max_chunk_chars: int,
    chunk_overlap_chars: int,
) -> tuple[int, int]:
    """Normalize and validate chunking parameters.

    Args:
        max_chunk_chars: Maximum characters per chunk.
        chunk_overlap_chars: Overlap between chunks in characters.

    Returns:
        Tuple of (normalized_max_chars, normalized_overlap_chars).

    """
    try:
        max_chars = int(max_chunk_chars)
        if max_chars <= 0:
            max_chars = DEFAULT_MAX_CHUNK_CHARS
    except (TypeError, ValueError):
        max_chars = DEFAULT_MAX_CHUNK_CHARS

    try:
        overlap_chars = int(chunk_overlap_chars)
        if overlap_chars < 0:
            overlap_chars = DEFAULT_CHUNK_OVERLAP_CHARS
    except (TypeError, ValueError):
        overlap_chars = DEFAULT_CHUNK_OVERLAP_CHARS

    return max_chars, overlap_chars


def _create_single_chunk(content: str) -> list[Chunk]:
    """Create a single chunk from content that fits within limits.

    Args:
        content: The document content.

    Returns:
        List containing a single Chunk.

    """
    chunk = Chunk(
        text=content,
        start_char=0,
        end_char=len(content),
        index=0,
    )
    return [chunk]


def _calculate_next_start(
    actual_end: int,
    overlap_chars: int,
    current_start: int,
) -> int:
    """Calculate the starting position for the next chunk.

    Args:
        actual_end: The end position of the current chunk.
        overlap_chars: The overlap amount in characters.
        current_start: The start position of the current chunk.

    Returns:
        The start position for the next chunk.

    """
    next_start = actual_end - overlap_chars
    if next_start <= current_start:
        next_start = actual_end
    return next_start


def _create_chunks_from_content(
    content: str,
    max_chars: int,
    overlap_chars: int,
) -> list[Chunk]:
    """Create multiple chunks from content using the chunking algorithm.

    Args:
        content: The document content to split.
        max_chars: Maximum characters per chunk.
        overlap_chars: Overlap between chunks in characters.

    Returns:
        List of Chunk objects.

    """
    chunks: list[Chunk] = []
    start_pos = 0
    chunk_index = 0
    content_len = len(content)

    while start_pos < content_len:
        target_end = start_pos + max_chars
        actual_end = _find_split_point(
            content,
            target_end,
            min(start_pos + max_chars, content_len),
        )

        if actual_end <= start_pos:
            actual_end = min(start_pos + max_chars, content_len)

        chunk_text = content[start_pos:actual_end]
        chunk = Chunk(
            text=chunk_text,
            start_char=start_pos,
            end_char=actual_end,
            index=chunk_index,
        )
        chunks.append(chunk)

        if actual_end >= content_len:
            break

        next_start = _calculate_next_start(actual_end, overlap_chars, start_pos)
        start_pos = next_start
        chunk_index += 1

    return chunks


def split_into_chunks(
    content: str,
    max_chunk_chars: int = DEFAULT_MAX_CHUNK_CHARS,
    chunk_overlap_chars: int = DEFAULT_CHUNK_OVERLAP_CHARS,
) -> list[Chunk]:
    """Split document content into overlapping chunks.

    Splits content into chunks that fit within the specified character limit.
    Uses paragraph boundaries when possible, falls back to sentence boundaries,
    then arbitrary character positions. Chunks overlap by the specified amount
    to preserve context at boundaries.

    Args:
        content: The document content to split.
        max_chunk_chars: Maximum characters per chunk (default: 24000).
        chunk_overlap_chars: Overlap between chunks in characters (default: 800).

    Returns:
        List of Chunk objects representing the document chunks.

    Notes:
        - Empty content returns an empty list
        - Content shorter than max_chunk_chars returns a single chunk
        - Overlap ensures context preservation at chunk boundaries

    """
    _msg = "split_into_chunks starting"
    log.debug(_msg)

    # Normalize parameters
    max_chars, overlap_chars = _normalize_chunking_params(
        max_chunk_chars, chunk_overlap_chars
    )

    # Handle empty content
    if not content:
        _msg = "split_into_chunks returning (empty content)"
        log.debug(_msg)
        return []

    # If content fits in a single chunk, return it as-is
    if len(content) <= max_chars:
        _msg = "split_into_chunks returning (single chunk)"
        log.debug(_msg)
        return _create_single_chunk(content)

    # Create multiple chunks
    chunks = _create_chunks_from_content(content, max_chars, overlap_chars)

    _msg = f"split_into_chunks returning ({len(chunks)} chunks)"
    log.debug(_msg)
    return chunks


def should_chunk_document(
    content: str,
    max_chunk_chars: int = DEFAULT_MAX_CHUNK_CHARS,
) -> bool:
    """Determine if a document should be chunked.

    Args:
        content: The document content to check.
        max_chunk_chars: Maximum characters per chunk (default: 24000).

    Returns:
        True if the document exceeds the chunk size limit and should be chunked.

    """
    _msg = "should_chunk_document starting"
    log.debug(_msg)

    # Handle invalid max_chunk_chars (e.g., MagicMock in tests)
    try:
        max_chars = int(max_chunk_chars)
    except (TypeError, ValueError):
        max_chars = DEFAULT_MAX_CHUNK_CHARS

    result = len(content) > max_chars
    _msg = f"should_chunk_document returning (should_chunk={result})"
    log.debug(_msg)
    return result
