"""PostgreSQL-specific document query implementations.

This module contains query implementations optimized for PostgreSQL with pgvector support.
Uses native JSONB operators and vector similarity functions.
"""

import logging
from typing import TYPE_CHECKING

from obsidian_rag.database.models import Document, Vault
from obsidian_rag.mcp_server.models import (
    DocumentListResponse,
    create_document_response,
)
from obsidian_rag.mcp_server.tools.documents_filters import (
    apply_postgresql_property_filter,
    matches_property_filter,
)
from obsidian_rag.mcp_server.tools.documents_params import (
    DocumentQueryParams,
    PropertyQueryParams,
    TagFilterParams,
)
from obsidian_rag.mcp_server.tools.documents_tags import (
    apply_postgresql_tag_filter,
)

if TYPE_CHECKING:
    from sqlalchemy.orm import Query

    from obsidian_rag.mcp_server.models import PropertyFilter

log = logging.getLogger(__name__)


def _extract_document_from_row(row: object) -> Document:
    """Extract Document from query result row.

    Args:
        row: Query result row (could be tuple, object with Document attr, or Document).

    Returns:
        Document instance.

    """
    if hasattr(row, "Document"):
        return getattr(row, "Document")
    if isinstance(row, tuple):
        return row[0]  # type: ignore[return-value]
    return row  # type: ignore[return-value]


def _extract_distance_from_row(row: object) -> float:
    """Extract distance value from query result row.

    Args:
        row: Query result row (could be tuple, object with distance attr, or Document).

    Returns:
        Distance value as float.

    """
    if hasattr(row, "distance"):
        return float(getattr(row, "distance", 0.0))
    if isinstance(row, tuple) and len(row) > 1:
        return float(row[1])
    return 0.0


def _filter_results_by_exclude(
    results: list,
    property_filters_exclude: list["PropertyFilter"] | None,
) -> list:
    """Filter results by exclude property filters.

    Args:
        results: List of query results.
        property_filters_exclude: Property filters to exclude.

    Returns:
        Filtered list of results.

    """
    if not property_filters_exclude:
        return results

    filtered = []
    for row in results:
        doc = (
            getattr(row, "Document", row[0])
            if hasattr(row, "Document")
            else (row[0] if isinstance(row, tuple) else row)
        )
        if not any(matches_property_filter(doc, f) for f in property_filters_exclude):
            filtered.append(row)
    return filtered


def _apply_postgresql_filters(
    query: "Query[Document]",
    property_filters_include: list["PropertyFilter"] | None,
    tag_filter_params: TagFilterParams,
) -> "Query[Document]":
    """Apply PostgreSQL-specific filters to query.

    Args:
        query: SQLAlchemy query object.
        property_filters_include: Property filters to include.
        tag_filter_params: Tag filter parameters.

    Returns:
        Filtered query.

    """
    if property_filters_include:
        for prop_filter in property_filters_include:
            query = apply_postgresql_property_filter(query, prop_filter)

    query = apply_postgresql_tag_filter(query, tag_filter_params.tag_filter)
    return query


def query_documents_postgresql(params: DocumentQueryParams) -> DocumentListResponse:
    """Query documents using PostgreSQL with vector similarity.

    Args:
        params: Document query parameters including session, embedding, filters, and pagination.

    Returns:
        DocumentListResponse with results ordered by similarity.

    """
    _msg = "query_documents_postgresql starting"
    log.debug(_msg)

    session = params.session
    query_embedding = params.query_embedding
    filter_params = params.filter_params
    pagination = params.pagination

    # Query documents with vector similarity
    distance_expr = Document.content_vector.cosine_distance(query_embedding)

    query = session.query(Document, distance_expr.label("distance")).filter(
        Document.content_vector.isnot(None),
    )

    # Apply property and tag filters
    query = _apply_postgresql_filters(
        query,
        filter_params.property_filters.include_filters,
        filter_params.tag_params,
    )

    query = query.order_by(distance_expr.asc())

    # Get total count
    total_count = query.count()

    # Get paginated results
    results = query.offset(pagination.offset).limit(pagination.limit).all()

    # Apply exclude filters
    if filter_params.property_filters.exclude_filters:
        results = _filter_results_by_exclude(
            results, filter_params.property_filters.exclude_filters
        )
        total_count = len(results)

    document_responses = []
    for row in results:
        doc = _extract_document_from_row(row)
        distance = _extract_distance_from_row(row)
        document_responses.append(create_document_response(doc, distance))

    has_more = (pagination.offset + pagination.limit) < total_count
    next_offset = pagination.offset + pagination.limit if has_more else None

    _msg = "query_documents_postgresql returning"
    log.debug(_msg)

    return DocumentListResponse(
        results=document_responses,
        total_count=total_count,
        has_more=has_more,
        next_offset=next_offset,
    )


def get_documents_by_property_postgresql(
    params: PropertyQueryParams,
) -> tuple[list[Document], int]:
    """Query documents using PostgreSQL with property filters.

    Args:
        params: Property query parameters including session, filters, and pagination.

    Returns:
        Tuple of (results list, total count).

    """
    _msg = "get_documents_by_property_postgresql starting"
    log.debug(_msg)

    session = params.session
    property_filters = params.property_filters
    tag_params = params.tag_params
    vault_name = params.vault_name
    pagination = params.pagination

    # Join with Vault if filtering by vault_name
    if vault_name is not None:
        query = session.query(Document).join(Vault)
        query = query.filter(Vault.name == vault_name)
    else:
        query = session.query(Document)

    if property_filters.include_filters:
        for prop_filter in property_filters.include_filters:
            query = apply_postgresql_property_filter(query, prop_filter)

    query = apply_postgresql_tag_filter(query, tag_params.tag_filter)
    query = query.order_by(Document.file_name)

    total_count = query.count()
    results = query.offset(pagination.offset).limit(pagination.limit).all()

    if property_filters.exclude_filters:
        results = [
            doc
            for doc in results
            if not any(
                matches_property_filter(doc, f)
                for f in property_filters.exclude_filters
            )
        ]
        total_count = len(results)

    _msg = "get_documents_by_property_postgresql returning"
    log.debug(_msg)

    return results, total_count
