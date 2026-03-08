"""SQLite-specific document query implementations.

This module contains query implementations for SQLite databases.
Uses Python-based filtering since SQLite doesn't support JSONB or vector operations natively.
"""

import logging

from obsidian_rag.database.models import Document, Vault
from obsidian_rag.mcp_server.models import (
    DocumentListResponse,
    create_document_response,
)
from obsidian_rag.mcp_server.tools.documents_filters import matches_property_filters
from obsidian_rag.mcp_server.tools.documents_params import (
    DocumentQueryParams,
    PropertyQueryParams,
)
from obsidian_rag.mcp_server.tools.documents_tags import matches_tag_filter

log = logging.getLogger(__name__)


def query_documents_sqlite(params: DocumentQueryParams) -> DocumentListResponse:
    """Query documents using SQLite (Python-based filtering).

    Args:
        params: Document query parameters including session, embedding, filters, and pagination.

    Returns:
        DocumentListResponse with filtered results.

    """
    _msg = "query_documents_sqlite starting"
    log.debug(_msg)

    session = params.session
    filter_params = params.filter_params
    pagination = params.pagination

    query = session.query(Document)
    all_docs = query.all()

    filtered_docs = [
        doc
        for doc in all_docs
        if matches_property_filters(
            doc,
            filter_params.property_filters.include_filters,
            filter_params.property_filters.exclude_filters,
        )
        and matches_tag_filter(doc, filter_params.tag_params.tag_filter)
    ]

    total_count = len(filtered_docs)
    results = filtered_docs[pagination.offset : pagination.offset + pagination.limit]

    document_responses = [create_document_response(doc, 0.0) for doc in results]

    has_more = (pagination.offset + pagination.limit) < total_count
    next_offset = pagination.offset + pagination.limit if has_more else None

    _msg = "query_documents_sqlite returning"
    log.debug(_msg)

    return DocumentListResponse(
        results=document_responses,
        total_count=total_count,
        has_more=has_more,
        next_offset=next_offset,
    )


def get_documents_by_property_sqlite(
    params: PropertyQueryParams,
) -> tuple[list[Document], int]:
    """Query documents using SQLite with Python filtering.

    Args:
        params: Property query parameters including session, filters, and pagination.

    Returns:
        Tuple of (results list, total count).

    """
    _msg = "get_documents_by_property_sqlite starting"
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

    query = query.order_by(Document.file_name)
    all_docs = query.all()

    filtered_docs = [
        doc
        for doc in all_docs
        if matches_property_filters(
            doc,
            property_filters.include_filters,
            property_filters.exclude_filters,
        )
        and matches_tag_filter(doc, tag_params.tag_filter)
    ]

    total_count = len(filtered_docs)
    results = filtered_docs[pagination.offset : pagination.offset + pagination.limit]

    _msg = "get_documents_by_property_sqlite returning"
    log.debug(_msg)

    return results, total_count
