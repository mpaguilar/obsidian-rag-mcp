"""Document query tools for MCP server.

All tools in this module are read-only and only use SELECT queries.
"""

import logging
from typing import TYPE_CHECKING

from obsidian_rag.database.models import Document
from obsidian_rag.mcp_server.models import (
    DocumentListResponse,
    create_document_response,
    _validate_limit,
    _validate_offset,
)

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

log = logging.getLogger(__name__)


def query_documents(
    session: "Session",
    query_embedding: list[float],
    limit: int = 20,
    offset: int = 0,
) -> DocumentListResponse:
    """Semantic search over document content using vector similarity.

    Args:
        session: Database session.
        query_embedding: Vector embedding of the query text.
        limit: Maximum number of results (default: 20, max: 100).
        offset: Number of results to skip (default: 0).

    Returns:
        DocumentListResponse with results and pagination info.

    Notes:
        Uses cosine distance for similarity (lower is better).
        Documents without embeddings are excluded.

    """
    _msg = "query_documents starting"
    log.debug(_msg)

    limit = _validate_limit(limit)
    offset = _validate_offset(offset)

    # Build vector similarity query
    # Using cosine distance - lower values indicate higher similarity
    distance_expr = Document.content_vector.cosine_distance(query_embedding)

    query = (
        session.query(Document, distance_expr.label("distance"))
        .filter(Document.content_vector.isnot(None))
        .order_by(distance_expr.asc())
    )

    # Get total count of documents with embeddings
    total_count = query.count()

    # Get paginated results
    results = query.offset(offset).limit(limit).all()

    # Convert to response models
    document_responses = [
        create_document_response(doc, distance) for doc, distance in results
    ]

    # Calculate pagination
    has_more = (offset + limit) < total_count
    next_offset = offset + limit if has_more else None

    _msg = "query_documents returning"
    log.debug(_msg)

    return DocumentListResponse(
        results=document_responses,
        total_count=total_count,
        has_more=has_more,
        next_offset=next_offset,
    )
