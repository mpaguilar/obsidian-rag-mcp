"""Re-ranking module for semantic search using flashrank.

This module provides cross-encoder re-ranking capabilities
for improving search result relevance.
"""

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, cast

log = logging.getLogger(__name__)

# Default flashrank model
DEFAULT_RERANK_MODEL = "ms-marco-MiniLM-L-12-v2"
DEFAULT_MAX_LENGTH = 128
DEFAULT_TOP_K = 10

if TYPE_CHECKING:
    from flashrank import Ranker, RerankRequest
else:
    try:
        from flashrank import Ranker, RerankRequest
    except ImportError:
        Ranker = None  # type: ignore[misc,assignment]
        RerankRequest = None  # type: ignore[misc,assignment]


class RerankerProtocol(Protocol):
    """Protocol defining the interface for reranker objects."""

    def rerank(self, request: object) -> list[dict[str, Any]]:
        """Rerank passages based on query.

        Args:
            request: RerankRequest object containing query and passages.

        Returns:
            List of reranked results with scores.

        """
        ...  # noqa: PIE790  # pragma: no cover


class RerankError(Exception):
    """Exception raised for re-ranking errors."""

    pass  # noqa: PIE790


@dataclass
class RerankConfig:
    """Configuration for re-ranking.

    Attributes:
        model: Name of the flashrank model to use.
        max_length: Maximum input length for the model.
        top_k: Number of top results to return after re-ranking.
        enabled: Whether re-ranking is enabled.

    """

    model: str = DEFAULT_RERANK_MODEL
    max_length: int = DEFAULT_MAX_LENGTH
    top_k: int = DEFAULT_TOP_K
    enabled: bool = True


@dataclass
class RerankResult:
    """Result of re-ranking a chunk.

    Attributes:
        chunk_id: Unique identifier of the chunk.
        content: Text content of the chunk.
        score: Re-ranking score (higher is better).
        original_rank: Original rank from vector search.
        new_rank: New rank after re-ranking.

    """

    chunk_id: str
    content: str
    score: float
    original_rank: int
    new_rank: int


def create_reranker(
    model_name: str, max_length: int = DEFAULT_MAX_LENGTH
) -> RerankerProtocol | None:
    """Create a flashrank reranker instance.

    Args:
        model_name: Name of the flashrank model to load.
        max_length: Maximum input length for the model.

    Returns:
        Reranker instance or None if creation fails.

    Notes:
        Logs errors but returns None on failure to allow graceful degradation.

    """
    _msg = f"create_reranker starting: {model_name}"
    log.debug(_msg)

    if Ranker is None:
        _msg = "flashrank not installed, re-ranking unavailable"
        log.warning(_msg)
        return None

    try:
        _msg = f"Loading flashrank model: {model_name}"
        log.info(_msg)

        reranker = Ranker(model_name=model_name, max_length=max_length)

        _msg = f"create_reranker returning: {model_name}"
        log.debug(_msg)
        return cast("RerankerProtocol", reranker)
    except (OSError, ValueError, RuntimeError) as e:
        _msg = f"Failed to create reranker: {e}"
        log.error(_msg)
        return None


def rerank_chunks(
    query: str,
    chunks: list[dict[str, Any]],
    reranker: RerankerProtocol | None,
    top_k: int = DEFAULT_TOP_K,
) -> list[RerankResult]:
    """Re-rank chunks using cross-encoder.

    Args:
        query: The search query.
        chunks: List of chunk dictionaries with 'chunk_id' and 'content' keys.
        reranker: Reranker instance or None.
        top_k: Number of top results to return.

    Returns:
        List of RerankResult ordered by score (highest first).

    Notes:
        Returns empty list if reranker is None or an error occurs.
        This allows graceful degradation to vector-only search.

    """
    _msg = f"rerank_chunks starting: {len(chunks)} chunks"
    log.debug(_msg)

    if not chunks or reranker is None:
        _msg = "rerank_chunks returning (no chunks or no reranker)"
        log.debug(_msg)
        return []

    if RerankRequest is None:
        _msg = "flashrank not installed, cannot perform re-ranking"
        log.warning(_msg)
        return []

    try:
        # Prepare passages for flashrank
        passages = [
            {"id": chunk["chunk_id"], "text": chunk["content"]} for chunk in chunks
        ]

        # Create RerankRequest and perform re-ranking
        rerank_request = RerankRequest(query=query, passages=passages)
        results = reranker.rerank(rerank_request)

        # Convert to RerankResult
        rerank_results = []
        for new_rank, result in enumerate(results[:top_k], start=1):
            # Find original rank
            original_rank = next(
                (i + 1 for i, c in enumerate(chunks) if c["chunk_id"] == result["id"]),
                0,
            )

            rerank_results.append(
                RerankResult(
                    chunk_id=result["id"],
                    content=result["text"],
                    score=result["score"],
                    original_rank=original_rank,
                    new_rank=new_rank,
                )
            )

        _msg = f"rerank_chunks returning: {len(rerank_results)} results"
        log.debug(_msg)
        return rerank_results

    except (OSError, ValueError, RuntimeError) as e:
        _msg = f"Re-ranking failed: {e}"
        log.error(_msg)
        return []
