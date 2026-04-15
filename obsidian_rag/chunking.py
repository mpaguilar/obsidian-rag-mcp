"""Document chunking module for token-based splitting.

This module provides functionality to split documents into overlapping
chunks based on token counts rather than character counts, using
HuggingFace Tokenizers for accurate tokenization.
"""

import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from obsidian_rag.tokenizer import Tokenizer, TokenizerConfig

log = logging.getLogger(__name__)

# Default chunking parameters (token-based)
DEFAULT_CHUNK_SIZE = 512
DEFAULT_CHUNK_OVERLAP = 50

# Paragraph and sentence delimiters
PARAGRAPH_DELIMITER = "\n\n"
SENTENCE_DELIMITERS = [". ", "! ", "? ", "\n"]

# Task detection pattern
TASK_PATTERN = re.compile(r"^\s*[-*]\s*\[[\s/x-]\]")


class ChunkType(str, Enum):
    """Type of document chunk."""

    CONTENT = "content"
    TASK = "task"


@dataclass
class TokenChunk:
    """Represents a token-based chunk of text.

    Attributes:
        text: The text content of the chunk.
        start_char: Starting character position in the original document.
        end_char: Ending character position in the original document.
        index: Index of this chunk within the document (0-based).
        token_count: Number of tokens in this chunk.
        chunk_type: Type of chunk (content or task).

    """

    text: str
    start_char: int
    end_char: int
    index: int
    token_count: int | None = None
    chunk_type: ChunkType | None = None


# Legacy Chunk dataclass for backward compatibility
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


def _detect_chunk_type(text: str) -> ChunkType:
    """Detect if text represents a task.

    Args:
        text: Text to analyze.

    Returns:
        ChunkType.TASK if text looks like a task, ChunkType.CONTENT otherwise.

    """
    lines = text.strip().split("\n")
    for line in lines[:3]:  # Check first 3 lines
        if TASK_PATTERN.match(line):
            return ChunkType.TASK
    return ChunkType.CONTENT


def _check_boundary_tokens(
    text: str,
    boundary_pos: int,
    delimiter_len: int,
    target_tokens: int,
    tokenizer: "Tokenizer | None",
) -> int | None:
    """Check if splitting at a boundary gives acceptable token count.

    Args:
        text: The text to check.
        boundary_pos: Position of the boundary delimiter.
        delimiter_len: Length of the delimiter.
        target_tokens: Target token count.
        tokenizer: Tokenizer for counting.

    Returns:
        Split position if acceptable, None otherwise.

    """
    from obsidian_rag.tokenizer import count_tokens

    chunk_text = text[:boundary_pos]
    tokens = count_tokens(chunk_text, tokenizer)
    if tokens <= target_tokens + 50:  # Allow small overshoot
        return boundary_pos + delimiter_len
    return None


def _find_paragraph_boundary(
    text: str,
    search_start: int,
    max_chars: int,
    target_tokens: int,
    tokenizer: "Tokenizer | None",
) -> int | None:
    """Find paragraph boundary for splitting.

    Args:
        text: The text to search.
        search_start: Start position for search.
        max_chars: Maximum position to search.
        target_tokens: Target token count.
        tokenizer: Tokenizer for counting.

    Returns:
        Split position if found, None otherwise.

    """
    paragraph_pos = text.rfind(PARAGRAPH_DELIMITER, search_start, max_chars)
    if paragraph_pos != -1:
        result = _check_boundary_tokens(
            text, paragraph_pos, len(PARAGRAPH_DELIMITER), target_tokens, tokenizer
        )
        if result is not None:
            _msg = "_find_split_point returning (paragraph boundary)"
            log.debug(_msg)
            return result
    return None


def _find_sentence_boundary(
    text: str,
    search_start: int,
    max_chars: int,
    target_tokens: int,
    tokenizer: "Tokenizer | None",
) -> int | None:
    """Find sentence boundary for splitting.

    Args:
        text: The text to search.
        search_start: Start position for search.
        max_chars: Maximum position to search.
        target_tokens: Target token count.
        tokenizer: Tokenizer for counting.

    Returns:
        Split position if found, None otherwise.

    """
    for delimiter in SENTENCE_DELIMITERS:
        sentence_pos = text.rfind(delimiter, search_start, max_chars)
        if sentence_pos != -1:
            result = _check_boundary_tokens(
                text, sentence_pos, len(delimiter), target_tokens, tokenizer
            )
            if result is not None:
                _msg = "_find_split_point returning (sentence boundary)"
                log.debug(_msg)
                return result
    return None


def _find_split_point_token_based(
    text: str,
    target_tokens: int,
    tokenizer: "Tokenizer | None",
    max_chars: int,
) -> int:
    """Find the best split point near the target token position.

    Tries to split on paragraph boundaries first, then sentence boundaries,
    then falls back to token count.

    Args:
        text: The text to find a split point in.
        target_tokens: The ideal number of tokens to include.
        tokenizer: Tokenizer instance for counting.
        max_chars: Maximum characters to consider (hard limit).

    Returns:
        The character position to split at.

    """
    _msg = "_find_split_point starting"
    log.debug(_msg)

    # Search backwards from max_chars for paragraph boundary
    search_start = max(0, max_chars - 2000)  # Search up to 2000 chars back

    # Try paragraph boundary first
    para_result = _find_paragraph_boundary(
        text, search_start, max_chars, target_tokens, tokenizer
    )
    if para_result is not None:
        return para_result

    # Try sentence boundaries
    sent_result = _find_sentence_boundary(
        text, search_start, max_chars, target_tokens, tokenizer
    )
    if sent_result is not None:
        return sent_result

    # Fall back to approximate character position
    # Estimate: ~4 chars per token
    fallback_pos = min(target_tokens * 4, max_chars)
    _msg = "_find_split_point returning (fallback)"
    log.debug(_msg)
    return fallback_pos


def _create_token_chunk(
    content: str,
    start_pos: int,
    actual_end: int,
    chunk_index: int,
    tokenizer: "Tokenizer | None",
) -> TokenChunk:
    """Create a TokenChunk from content slice.

    Args:
        content: The document content.
        start_pos: Start character position.
        actual_end: End character position.
        chunk_index: Index of this chunk.
        tokenizer: Tokenizer for counting tokens.

    Returns:
        TokenChunk with metadata.

    """
    from obsidian_rag.tokenizer import count_tokens

    chunk_text = content[start_pos:actual_end]
    token_count = count_tokens(chunk_text, tokenizer)
    chunk_type = _detect_chunk_type(chunk_text)

    return TokenChunk(
        text=chunk_text,
        start_char=start_pos,
        end_char=actual_end,
        index=chunk_index,
        token_count=token_count,
        chunk_type=chunk_type,
    )


def _process_single_chunk(
    content: str,
    start_pos: int,
    content_len: int,
    chunk_index: int,
    config: "TokenizerConfig",
    tokenizer: "Tokenizer | None",
) -> tuple[TokenChunk, int] | None:
    """Process a single chunk from the content.

    Args:
        content: The document content.
        start_pos: Starting position for this chunk.
        content_len: Total content length.
        chunk_index: Index of this chunk.
        config: Tokenizer configuration.
        tokenizer: Tokenizer instance.

    Returns:
        Tuple of (TokenChunk, actual_end) or None if no content.

    """
    target_end = start_pos + (config.chunk_size * 4)
    actual_end = _find_split_point_token_based(
        content,
        config.chunk_size,
        tokenizer,
        min(target_end, content_len),
    )

    if actual_end <= start_pos:
        actual_end = min(start_pos + (config.chunk_size * 4), content_len)

    chunk = _create_token_chunk(content, start_pos, actual_end, chunk_index, tokenizer)
    return chunk, actual_end


def _calculate_next_start(
    actual_end: int,
    chunk_overlap: int,
    current_start: int,
) -> int:
    """Calculate the next chunk start position with overlap.

    Args:
        actual_end: End position of current chunk.
        chunk_overlap: Overlap in tokens.
        current_start: Start position of current chunk.

    Returns:
        Start position for next chunk.

    """
    overlap_chars = chunk_overlap * 4
    next_start = actual_end - overlap_chars
    if next_start <= current_start:
        return actual_end
    return next_start


def split_into_token_chunks(
    content: str,
    config: "TokenizerConfig",
) -> list[TokenChunk]:
    """Split document content into overlapping token-based chunks.

    Args:
        content: The document content to split.
        config: Tokenizer configuration with chunk_size and overlap.

    Returns:
        List of TokenChunk objects.

    Notes:
        - Empty content returns an empty list
        - Chunks overlap by config.chunk_overlap tokens
        - Splits prefer paragraph boundaries, then sentence boundaries

    """
    _msg = "split_into_token_chunks starting"
    log.debug(_msg)

    from obsidian_rag.tokenizer import get_tokenizer

    if not content:
        _msg = "split_into_token_chunks returning (empty content)"
        log.debug(_msg)
        return []

    tokenizer = get_tokenizer(config.model_name)

    chunks: list[TokenChunk] = []
    start_pos = 0
    chunk_index = 0
    content_len = len(content)

    while start_pos < content_len:  # pragma: no branch
        result = _process_single_chunk(
            content, start_pos, content_len, chunk_index, config, tokenizer
        )
        if result is None:
            break  # pragma: no cover
        chunk, actual_end = result
        chunks.append(chunk)

        if actual_end >= content_len:
            break

        start_pos = _calculate_next_start(actual_end, config.chunk_overlap, start_pos)
        chunk_index += 1

    _msg = f"split_into_token_chunks returning ({len(chunks)} chunks)"
    log.debug(_msg)
    return chunks


def should_chunk_document(
    content: str,
    chunk_size: int,
    model_name: str = "gpt2",
) -> bool:
    """Determine if a document should be chunked.

    Args:
        content: The document content to check.
        chunk_size: Maximum tokens per chunk.
        model_name: Name of the tokenizer model.

    Returns:
        True if the document exceeds the chunk size limit.

    """
    _msg = "should_chunk_document starting"
    log.debug(_msg)

    from obsidian_rag.tokenizer import count_tokens, get_tokenizer

    # Validate chunk_size is a number
    if not isinstance(chunk_size, int | float):
        _msg = f"Invalid chunk_size type: {type(chunk_size).__name__}, using default"
        log.warning(_msg)
        chunk_size = DEFAULT_CHUNK_SIZE

    tokenizer = get_tokenizer(model_name)
    token_count = count_tokens(content, tokenizer)

    result = token_count > chunk_size
    _msg = f"should_chunk_document returning: {result} ({token_count} tokens)"
    log.debug(_msg)
    return result


def chunk_document(
    content: str,
    document_id: str,
    chunk_size: int,
    chunk_overlap: int,
    model_name: str = "gpt2",
) -> list[dict]:
    """Chunk a document and prepare for database insertion.

    Args:
        content: Document content.
        document_id: UUID of the parent document.
        chunk_size: Target tokens per chunk.
        chunk_overlap: Tokens to overlap between chunks.
        model_name: Tokenizer model name.

    Returns:
        List of chunk dictionaries ready for database insertion.

    """
    _msg = "chunk_document starting"
    log.debug(_msg)

    from obsidian_rag.tokenizer import TokenizerConfig

    config = TokenizerConfig(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        model_name=model_name,
    )

    chunks = split_into_token_chunks(content, config)

    # Convert to database-ready dictionaries
    result = []
    for chunk in chunks:
        result.append(
            {
                "document_id": document_id,
                "chunk_index": chunk.index,
                "chunk_text": chunk.text,
                "start_char": chunk.start_char,
                "end_char": chunk.end_char,
                "token_count": chunk.token_count,
                "chunk_type": chunk.chunk_type.value if chunk.chunk_type else None,
            }
        )

    _msg = f"chunk_document returning ({len(result)} chunks)"
    log.debug(_msg)
    return result


# Backward compatibility: Keep old function names as aliases
def split_into_chunks(
    content: str,
    max_chunk_chars: int = 24000,
    chunk_overlap_chars: int = 800,
) -> list:
    """Legacy character-based chunking (for backward compatibility).

    Deprecated: Use split_into_token_chunks instead.

    """
    _msg = "split_into_chunks (legacy) starting"
    log.warning(_msg)

    # Validate input types and values
    if not isinstance(max_chunk_chars, int | float):
        _msg = f"Invalid max_chunk_chars type: {type(max_chunk_chars).__name__}, using default"
        log.warning(_msg)
        max_chunk_chars = 24000
    if not isinstance(chunk_overlap_chars, int | float):
        _msg = f"Invalid chunk_overlap_chars type: {type(chunk_overlap_chars).__name__}, using default"
        log.warning(_msg)
        chunk_overlap_chars = 800

    # Convert chars to approximate tokens (~4 chars/token)
    chunk_size = int(max_chunk_chars) // 4
    chunk_overlap = int(chunk_overlap_chars) // 4

    # Ensure minimum chunk size to prevent infinite loops
    if chunk_size <= 0:
        chunk_size = DEFAULT_CHUNK_SIZE
    if chunk_overlap < 0:
        chunk_overlap = DEFAULT_CHUNK_OVERLAP

    from obsidian_rag.tokenizer import TokenizerConfig

    config = TokenizerConfig(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    token_chunks = split_into_token_chunks(content, config)

    # Convert to legacy Chunk format
    return [
        Chunk(text=c.text, start_char=c.start_char, end_char=c.end_char, index=c.index)
        for c in token_chunks
    ]


# Additional backward compatibility wrappers for private functions
def _estimate_tokens(text: str) -> int:
    """Estimate the number of tokens in text (backward compatibility).

    Deprecated: Use tokenizer.count_tokens instead.

    Args:
        text: The text to estimate tokens for.

    Returns:
        Estimated token count.

    """
    _msg = "_estimate_tokens (legacy) starting"
    log.debug(_msg)
    # Rough estimate: ~4 characters per token for English text
    result = len(text) // 4
    _msg = "_estimate_tokens (legacy) returning"
    log.debug(_msg)
    return result


def _normalize_chunking_params(
    max_chunk_chars: int,
    chunk_overlap_chars: int,
) -> tuple[int, int]:
    """Normalize and validate chunking parameters (backward compatibility).

    Args:
        max_chunk_chars: Maximum characters per chunk.
        chunk_overlap_chars: Overlap between chunks in characters.

    Returns:
        Tuple of (normalized_max_chars, normalized_overlap_chars).

    """
    try:
        max_chars = int(max_chunk_chars)
        if max_chars <= 0:
            max_chars = DEFAULT_CHUNK_SIZE * 4  # Convert tokens to chars
    except (TypeError, ValueError):
        max_chars = DEFAULT_CHUNK_SIZE * 4

    try:
        overlap_chars = int(chunk_overlap_chars)
        if overlap_chars < 0:
            overlap_chars = DEFAULT_CHUNK_OVERLAP * 4
    except (TypeError, ValueError):
        overlap_chars = DEFAULT_CHUNK_OVERLAP * 4

    return max_chars, overlap_chars


def _create_single_chunk(content: str) -> list[Chunk]:
    """Create a single chunk from content (backward compatibility).

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


def _create_chunks_from_content(
    content: str,
    max_chars: int,
    overlap_chars: int,
) -> list[Chunk]:
    """Create multiple chunks from content (backward compatibility).

    Args:
        content: The document content to split.
        max_chars: Maximum characters per chunk.
        overlap_chars: Overlap between chunks in characters.

    Returns:
        List of Chunk objects.

    """
    # Convert to token-based parameters and use new implementation
    chunk_size = max_chars // 4
    chunk_overlap = overlap_chars // 4

    from obsidian_rag.tokenizer import TokenizerConfig

    config = TokenizerConfig(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    token_chunks = split_into_token_chunks(content, config)

    # Convert to legacy Chunk format
    return [
        Chunk(text=c.text, start_char=c.start_char, end_char=c.end_char, index=c.index)
        for c in token_chunks
    ]


# Legacy _find_split_point for backward compatibility (old signature)
def _find_split_point_legacy(
    text: str,
    target_pos: int,
    max_pos: int,
) -> int:
    """Find the best split point near the target position (backward compatibility).

    Tries to split on paragraph boundaries first, then sentence boundaries,
    then falls back to the target position.

    Args:
        text: The text to find a split point in.
        target_pos: The ideal position to split at.
        max_pos: The maximum allowed position (hard limit).

    Returns:
        The actual split position to use.

    """
    _msg = "_find_split_point (legacy) starting"
    log.debug(_msg)

    # Ensure target doesn't exceed max
    target = min(target_pos, max_pos)

    # Search backwards from target for paragraph boundary
    search_start = max(0, target - 1000)  # Search up to 1000 chars back
    paragraph_pos = text.rfind(PARAGRAPH_DELIMITER, search_start, target)
    if paragraph_pos != -1:
        # Found paragraph boundary, split after it
        result = paragraph_pos + len(PARAGRAPH_DELIMITER)
        _msg = "_find_split_point (legacy) returning (paragraph boundary)"
        log.debug(_msg)
        return result

    # Search for sentence boundary
    for delimiter in SENTENCE_DELIMITERS:
        sentence_pos = text.rfind(delimiter, search_start, target)
        if sentence_pos != -1:
            # Found sentence boundary, split after it
            result = sentence_pos + len(delimiter)
            _msg = "_find_split_point (legacy) returning (sentence boundary)"
            log.debug(_msg)
            return result

    # Fall back to target position
    _msg = "_find_split_point (legacy) returning (fallback)"
    log.debug(_msg)
    return target
