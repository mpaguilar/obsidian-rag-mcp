"""Tokenizer module for token-based chunking.

This module provides tokenization using HuggingFace Tokenizers library
with fallback to character-based heuristics.
"""

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

log = logging.getLogger(__name__)

if TYPE_CHECKING:
    from tokenizers import Tokenizer

# Default tokenizer model (universal fallback)
DEFAULT_TOKENIZER_MODEL = "gpt2"

# Character heuristic: ~4 characters per token for English text
CHARS_PER_TOKEN = 4

# Global tokenizer cache
_tokenizer_cache: dict[str, "Tokenizer"] = {}


class TokenizerError(Exception):
    """Exception raised for tokenizer-related errors."""

    pass  # noqa: PIE790


@dataclass
class TokenizerConfig:
    """Configuration for token-based chunking.

    Attributes:
        chunk_size: Target number of tokens per chunk (default: 512).
        chunk_overlap: Number of tokens to overlap between chunks (default: 50).
        model_name: Name of the tokenizer model to use (default: gpt2).

    """

    chunk_size: int = 512
    chunk_overlap: int = 50
    model_name: str = DEFAULT_TOKENIZER_MODEL


def initialize_tokenizer(
    model_name: str,
) -> "Tokenizer | None":
    """Initialize a tokenizer from HuggingFace.

    Args:
        model_name: Name of the tokenizer model to load.

    Returns:
        Initialized Tokenizer or None if initialization fails.

    Notes:
        Logs errors but does not raise exceptions on failure.
        Callers should handle None return value.

    """
    _msg = f"initialize_tokenizer starting: {model_name}"
    log.debug(_msg)

    try:
        from tokenizers import Tokenizer

        _msg = f"Loading tokenizer {model_name}"
        log.info(_msg)

        tokenizer = Tokenizer.from_pretrained(model_name)

        _msg = f"initialize_tokenizer returning: {model_name}"
        log.debug(_msg)
        return tokenizer
    except (OSError, ValueError, RuntimeError) as e:
        _msg = f"Failed to load tokenizer {model_name}: {e}"
        log.error(_msg)
        return None


def get_tokenizer(model_name: str) -> "Tokenizer | None":
    """Get or create a tokenizer instance.

    Uses a global cache to avoid reloading tokenizers.

    Args:
        model_name: Name of the tokenizer model.

    Returns:
        Tokenizer instance or None if unavailable.

    """
    _msg = f"get_tokenizer starting: {model_name}"
    log.debug(_msg)

    cache_key = model_name

    if cache_key not in _tokenizer_cache:
        _msg = f"Tokenizer not cached, initializing: {model_name}"
        log.debug(_msg)
        tokenizer = initialize_tokenizer(model_name)
        if tokenizer is not None:  # pragma: no cover
            _tokenizer_cache[cache_key] = tokenizer

    result = _tokenizer_cache.get(cache_key)
    _msg = f"get_tokenizer returning: {result is not None}"
    log.debug(_msg)
    return result


def count_tokens(text: str, tokenizer: "Tokenizer | None") -> int:
    """Count tokens in text.

    Args:
        text: Text to count tokens for.
        tokenizer: Tokenizer instance or None for fallback.

    Returns:
        Estimated token count.

    Notes:
        If tokenizer is None, uses character heuristic (~4 chars/token).

    """
    _msg = "count_tokens starting"
    log.debug(_msg)

    if tokenizer is None:
        # Fallback to character heuristic
        result = len(text) // CHARS_PER_TOKEN
        _msg = f"count_tokens returning (heuristic): {result}"
        log.debug(_msg)
        return result

    try:
        encoding = tokenizer.encode(text)
        result = len(encoding.ids)
        _msg = f"count_tokens returning: {result}"
        log.debug(_msg)
        return result
    except (RuntimeError, ValueError, TypeError) as e:
        _msg = f"Tokenization failed, using heuristic: {e}"
        log.warning(_msg)
        result = len(text) // CHARS_PER_TOKEN
        return result


def clear_tokenizer_cache() -> None:
    """Clear the global tokenizer cache.

    Useful for testing and memory management.

    """
    _msg = "Clearing tokenizer cache"
    log.debug(_msg)
    _tokenizer_cache.clear()
