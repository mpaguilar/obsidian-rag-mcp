"""Chunk-level document query tools for MCP server.

This module provides semantic search at the chunk level with
optional flashrank re-ranking.
"""

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from obsidian_rag.database.models import Document, DocumentChunk, Vault
from obsidian_rag.reranking import create_reranker, rerank_chunks

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

log = logging.getLogger(__name__)


@dataclass
class ChunkQueryResult:
    """Result of a chunk-level query.

    Attributes:
        chunk_id: Unique identifier of the chunk.
        content: Text content of the chunk.
        document_name: Name of the parent document.
        document_path: Path of the parent document.
        vault_name: Name of the vault containing the document.
        chunk_index: Index of this chunk within the document.
        total_chunks: Total number of chunks in the document.
        token_count: Number of tokens in this chunk.
        chunk_type: Type of chunk (content or task).
        similarity_score: Vector similarity score (cosine distance).
        rerank_score: Re-ranking score if re-ranking was applied.

    """

    chunk_id: str
    content: str
    document_name: str
    document_path: str
    vault_name: str
    chunk_index: int
    total_chunks: int
    token_count: int | None
    chunk_type: str | None
    similarity_score: float
    rerank_score: float | None


def query_chunks(
    session: "Session",
    query_embedding: list[float],
    vault_name: str | None = None,
    limit: int = 20,
) -> list[ChunkQueryResult]:
    """Query document chunks using vector similarity.

    Args:
        session: Database session.
        query_embedding: Vector embedding of the query text.
        vault_name: Optional vault name filter.
        limit: Maximum number of results.

    Returns:
        List of ChunkQueryResult ordered by similarity.

    """
    _msg = "query_chunks starting"
    log.debug(_msg)

    from sqlalchemy import func

    # Build base query with document and vault joins
    query = (
        session.query(
            DocumentChunk,
            DocumentChunk.chunk_vector.cosine_distance(query_embedding).label(
                "distance"
            ),
        )
        .join(Document, DocumentChunk.document_id == Document.id)
        .join(Vault, Document.vault_id == Vault.id)
    )

    # Apply vault filter if specified
    if vault_name:
        query = query.filter(Vault.name == vault_name)

    # Order by similarity and limit
    results = query.order_by("distance").limit(limit).all()

    # Convert to ChunkQueryResult
    chunk_results = []
    for chunk, distance in results:
        # Get total chunks for this document
        total_chunks = (
            session.query(func.count(DocumentChunk.id))
            .filter(DocumentChunk.document_id == chunk.document_id)
            .scalar()
        )

        chunk_results.append(
            ChunkQueryResult(
                chunk_id=str(chunk.id),
                content=chunk.chunk_text,
                document_name=chunk.document.file_name,
                document_path=chunk.document.file_path,
                vault_name=chunk.document.vault.name,
                chunk_index=chunk.chunk_index,
                total_chunks=total_chunks,
                token_count=chunk.token_count,
                chunk_type=chunk.chunk_type,
                similarity_score=1.0
                - float(distance),  # Convert distance to similarity
                rerank_score=None,
            )
        )

    _msg = f"query_chunks returning ({len(chunk_results)} results)"
    log.debug(_msg)
    return chunk_results


def rerank_chunk_results(
    query: str,
    chunks: list[ChunkQueryResult],
    rerank_model: str,
    max_length: int,
    top_k: int,
) -> list[ChunkQueryResult]:
    """Re-rank chunk results using flashrank.

    Args:
        query: Original search query.
        chunks: Chunk results from vector search.
        rerank_model: Name of the flashrank model.
        max_length: Maximum input length for the model.
        top_k: Number of top results to return.

    Returns:
        Re-ranked list of ChunkQueryResult.

    """
    _msg = "rerank_chunk_results starting"
    log.debug(_msg)

    if not chunks:
        return []

    # Create reranker
    reranker = create_reranker(rerank_model, max_length)
    if reranker is None:
        _msg = "Reranker unavailable, returning original results"
        log.warning(_msg)
        return chunks

    # Convert to format expected by rerank_chunks
    chunk_dicts = [
        {
            "chunk_id": c.chunk_id,
            "content": c.content,
        }
        for c in chunks
    ]

    # Perform re-ranking
    rerank_results = rerank_chunks(query, chunk_dicts, reranker, top_k)

    # Build result mapping
    chunk_map = {c.chunk_id: c for c in chunks}

    # Create new results with rerank scores
    results = []
    for rr in rerank_results:
        original = chunk_map.get(rr.chunk_id)
        if original:  # pragma: no cover
            results.append(
                ChunkQueryResult(
                    chunk_id=original.chunk_id,
                    content=original.content,
                    document_name=original.document_name,
                    document_path=original.document_path,
                    vault_name=original.vault_name,
                    chunk_index=original.chunk_index,
                    total_chunks=original.total_chunks,
                    token_count=original.token_count,
                    chunk_type=original.chunk_type,
                    similarity_score=original.similarity_score,
                    rerank_score=rr.score,
                )
            )

    _msg = f"rerank_chunk_results returning ({len(results)} results)"
    log.debug(_msg)
    return results
