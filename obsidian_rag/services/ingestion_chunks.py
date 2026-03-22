"""Chunking operations for document ingestion.

This module contains chunking-related methods extracted from ingestion.py
to comply with the 1000 line limit.
"""

import logging
import uuid
from typing import TYPE_CHECKING

from sqlalchemy.orm import Session

from obsidian_rag.chunking import chunk_document
from obsidian_rag.database.models import DocumentChunk

if TYPE_CHECKING:
    from obsidian_rag.llm.base import EmbeddingProvider

log = logging.getLogger(__name__)

# Constants for batch processing
BATCH_SIZE = 10
MAX_RETRIES = 3


def _generate_embedding_with_retry(
    chunk_text: str,
    embedding_provider: "EmbeddingProvider | None",
    max_retries: int = MAX_RETRIES,
) -> list[float] | None:
    """Generate embedding for a chunk with retry logic.

    Args:
        chunk_text: Text to generate embedding for.
        embedding_provider: Provider for generating embeddings.
        max_retries: Maximum number of retry attempts.

    Returns:
        Embedding vector or None if generation fails.

    Notes:
        Retries on transient failures up to max_retries.
        Logs warnings for failed attempts.

    """
    if embedding_provider is None:
        return None

    for attempt in range(max_retries):
        try:
            return embedding_provider.generate_embedding(chunk_text)
        except (OSError, ValueError, RuntimeError) as e:
            _msg = f"Embedding failed (attempt {attempt + 1}): {e}"
            log.warning(_msg)
            if attempt == max_retries - 1:
                _msg = "Max retries reached, skipping chunk"
                log.error(_msg)

    return None


def _create_document_chunk(
    db_session: Session,
    document_id: uuid.UUID,
    chunk_data: dict,
    embedding: list[float],
) -> DocumentChunk:
    """Create a DocumentChunk object.

    Args:
        db_session: Database session for saving chunks.
        document_id: Document UUID.
        chunk_data: Chunk data dictionary with metadata.
        embedding: Embedding vector for the chunk.

    Returns:
        DocumentChunk object ready to be added to session.

    """
    chunk = DocumentChunk(
        document_id=document_id,
        chunk_index=chunk_data["chunk_index"],
        chunk_text=chunk_data["chunk_text"],
        chunk_vector=embedding,
        start_char=chunk_data["start_char"],
        end_char=chunk_data["end_char"],
        token_count=chunk_data.get("token_count"),
        chunk_type=chunk_data.get("chunk_type"),
    )
    db_session.add(chunk)
    return chunk


def _process_chunk_batch(
    db_session: Session,
    document_id: uuid.UUID,
    batch: list[dict],
    embedding_provider: "EmbeddingProvider | None",
) -> int:
    """Process a batch of chunks and create database records.

    Args:
        db_session: Database session for saving chunks.
        document_id: Document UUID.
        batch: List of chunk data dictionaries.
        embedding_provider: Provider for generating embeddings.

    Returns:
        Number of chunks successfully created.

    """
    chunks_created = 0

    for chunk_data in batch:
        embedding = _generate_embedding_with_retry(
            chunk_data["chunk_text"],
            embedding_provider,
        )

        if embedding:
            _create_document_chunk(db_session, document_id, chunk_data, embedding)
            chunks_created += 1

    return chunks_created


def create_chunks_with_embeddings(
    db_session: Session,
    document_id: uuid.UUID,
    content: str,
    embedding_provider: "EmbeddingProvider | None",
    chunk_size: int,
    chunk_overlap: int,
    model_name: str,
) -> int:
    """Create chunks with embeddings for a document.

    Args:
        db_session: Database session for saving chunks.
        document_id: Document UUID.
        content: Document content to chunk.
        embedding_provider: Provider for generating embeddings.
        chunk_size: Target tokens per chunk.
        chunk_overlap: Tokens to overlap between chunks.
        model_name: Tokenizer model name.

    Returns:
        Number of chunks created.

    Notes:
        Uses token-based chunking with settings from config.
        Processes embeddings in batches with retries.
        Creates DocumentChunk objects with token_count and chunk_type.
        Saves chunks directly to database via the provided session.

    """
    _msg = "create_chunks_with_embeddings starting"
    log.debug(_msg)

    if not content.strip():
        _msg = "create_chunks_with_embeddings returning (empty content)"
        log.debug(_msg)
        return 0

    # Chunk the document
    chunks = chunk_document(
        content,
        str(document_id),
        chunk_size,
        chunk_overlap,
        model_name,
    )

    if not chunks:
        return 0

    # Process embeddings in batches
    chunks_created = 0

    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]
        chunks_created += _process_chunk_batch(
            db_session,
            document_id,
            batch,
            embedding_provider,
        )

    _msg = f"create_chunks_with_embeddings returning ({chunks_created} chunks)"
    log.debug(_msg)
    return chunks_created
