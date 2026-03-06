"""Document query tools for MCP server.

All tools in this module are read-only and only use SELECT queries.

This module serves as the public API that re-exports functions from
dedicated submodules for PostgreSQL, SQLite, filters, and tags.
"""

import logging
from typing import TYPE_CHECKING

from obsidian_rag.database.models import Document
from obsidian_rag.mcp_server.models import (
    DocumentListResponse,
    TagFilter,
    TagListResponse,
    _validate_limit,
    _validate_offset,
    create_document_response,
)
from obsidian_rag.mcp_server.tools.documents_filters import validate_property_filters
from obsidian_rag.mcp_server.tools.documents_params import (
    DocumentQueryParams,
    PaginationParams,
    PropertyFilterParams,
    PropertyQueryParams,
    QueryFilterParams,
    TagFilterParams,
)
from obsidian_rag.mcp_server.tools.documents_postgres import (
    get_documents_by_property_postgresql,
    query_documents_postgresql,
)
from obsidian_rag.mcp_server.tools.documents_sqlite import (
    get_documents_by_property_sqlite,
    query_documents_sqlite,
)
from obsidian_rag.mcp_server.tools.documents_tags import validate_tag_filter

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

log = logging.getLogger(__name__)

# Maximum query complexity limits (re-exported for backward compatibility)
MAX_PROPERTY_FILTERS = 10
MAX_TAGS_PER_QUERY = 50


def _get_relative_path(file_path: str, vault_root: str | None) -> str:
    """Calculate relative path from vault root.

    Args:
        file_path: Absolute file path.
        vault_root: Vault root path (or None if not set).

    Returns:
        Relative path starting with ./ if vault_root is set,
        otherwise returns the original file_path.

    """
    if vault_root is None:
        return file_path

    prefix = vault_root.rstrip("/") + "/"
    if file_path.startswith(prefix):
        relative = file_path[len(prefix) :]
        if not relative.startswith("./"):
            relative = "./" + relative
        return relative
    return file_path


def _glob_to_like(pattern: str) -> str:
    """Convert glob pattern to SQL LIKE pattern.

    Args:
        pattern: Glob pattern with *, ?, [abc] wildcards.

    Returns:
        SQL LIKE pattern with %, _, and character classes.

    Notes:
        - * becomes %
        - ? becomes _
        - [abc] character classes are preserved for SQL
        - % and _ are escaped with backslash

    """
    # Escape SQL special characters first
    result = pattern.replace("\\", "\\\\").replace("%", r"\%").replace("_", r"\_")
    # Convert glob wildcards to SQL LIKE
    result = result.replace("*", "%").replace("?", "_")
    return result


def _build_document_list_response(
    results: list[Document],
    total_count: int,
    offset: int,
    limit: int,
) -> DocumentListResponse:
    """Build DocumentListResponse from query results.

    Args:
        results: List of Document objects.
        total_count: Total number of matching documents.
        offset: Current offset.
        limit: Current limit.

    Returns:
        DocumentListResponse with results and pagination info.

    """
    document_responses = []
    for doc in results:
        relative_path = _get_relative_path(doc.file_path, doc.vault_root)
        doc_response = create_document_response(doc, 0.0)
        doc_response.file_path = relative_path
        document_responses.append(doc_response)

    has_more = (offset + limit) < total_count
    next_offset = offset + limit if has_more else None

    return DocumentListResponse(
        results=document_responses,
        total_count=total_count,
        has_more=has_more,
        next_offset=next_offset,
    )


def query_documents(
    session: "Session",
    query_embedding: list[float],
    filter_params: PropertyFilterParams | None = None,
    tag_filter: TagFilter | None = None,
    pagination: PaginationParams | None = None,
) -> DocumentListResponse:
    """Semantic search over document content with optional property and tag filters.

    Args:
        session: Database session.
        query_embedding: Vector embedding of the query text.
        filter_params: Property filter parameters with include/exclude lists.
        tag_filter: Tag filter with include/exclude lists.
        pagination: Pagination parameters (limit/offset).

    Returns:
        DocumentListResponse with results and pagination info.

    Raises:
        ValueError: If filter validation fails.

    Notes:
        Uses cosine distance for similarity (lower is better).
        Documents without embeddings are excluded.
        For SQLite databases, filtering is done in Python.

    """
    _msg = "query_documents starting"
    log.debug(_msg)

    # Use default pagination if not provided
    pagination = pagination or PaginationParams(limit=20, offset=0)
    limit = _validate_limit(pagination.limit)
    offset = pagination.offset

    # Extract include/exclude filters from filter_params
    property_filters_include = filter_params.include_filters if filter_params else None
    property_filters_exclude = filter_params.exclude_filters if filter_params else None

    validate_property_filters(property_filters_include)
    validate_property_filters(property_filters_exclude)
    validate_tag_filter(tag_filter)

    dialect = session.bind.dialect.name if session.bind else "unknown"
    is_postgresql = dialect == "postgresql"

    # Build filter parameters
    property_filter_params = PropertyFilterParams(
        include_filters=property_filters_include,
        exclude_filters=property_filters_exclude,
    )
    tag_filter_params = TagFilterParams(tag_filter=tag_filter)
    query_filter_params = QueryFilterParams(
        property_filters=property_filter_params,
        tag_params=tag_filter_params,
    )
    pagination = PaginationParams(limit=limit, offset=offset)
    query_params = DocumentQueryParams(
        session=session,
        query_embedding=query_embedding,
        filter_params=query_filter_params,
        pagination=pagination,
    )

    if not is_postgresql:
        result = query_documents_sqlite(query_params)
        _msg = "query_documents returning (SQLite path)"
        log.debug(_msg)
        return result

    result = query_documents_postgresql(query_params)
    _msg = "query_documents returning"
    log.debug(_msg)
    return result


def get_documents_by_tag(
    session: "Session",
    tag_filter: TagFilter,
    vault_root: str | None = None,
    limit: int = 20,
    offset: int = 0,
) -> DocumentListResponse:
    """Query documents filtered by tags with include/exclude semantics.

    Args:
        session: Database session.
        tag_filter: Tag filter with include/exclude lists.
        vault_root: Filter by specific vault root path (optional).
        limit: Maximum number of results (default: 20, max: 100).
        offset: Number of results to skip (default: 0).

    Returns:
        DocumentListResponse with results and pagination info.

    Raises:
        ValueError: If tag filter validation fails.

    Notes:
        Tag matching is case-insensitive substring match.
        File paths are returned relative to vault_root if available.
        Relative paths always start with ./ (e.g., ./foo/bar.md).
        For SQLite databases, filtering is done in Python.

    """
    _msg = "get_documents_by_tag starting"
    log.debug(_msg)

    limit = _validate_limit(limit)
    offset = _validate_offset(offset)

    # Validate tag filter
    validate_tag_filter(tag_filter)

    dialect = session.bind.dialect.name if session.bind else "unknown"
    is_postgresql = dialect == "postgresql"

    query = session.query(Document)

    if vault_root is not None:
        query = query.filter(Document.vault_root == vault_root)

    if is_postgresql:
        from obsidian_rag.mcp_server.tools.documents_tags import (
            apply_postgresql_tag_filter,
        )

        query = apply_postgresql_tag_filter(query, tag_filter)
        query = query.order_by(Document.file_name)
        total_count = query.count()
        results = query.offset(offset).limit(limit).all()
    else:
        from obsidian_rag.mcp_server.tools.documents_tags import matches_tag_filter

        query = query.order_by(Document.file_name)
        all_docs = query.all()
        filtered_docs = [doc for doc in all_docs if matches_tag_filter(doc, tag_filter)]
        total_count = len(filtered_docs)
        results = filtered_docs[offset : offset + limit]

    document_responses = []
    for doc in results:
        relative_path = _get_relative_path(doc.file_path, doc.vault_root)
        doc_response = create_document_response(doc, 0.0)
        doc_response.file_path = relative_path
        document_responses.append(doc_response)

    has_more = (offset + limit) < total_count
    next_offset = offset + limit if has_more else None

    _msg = "get_documents_by_tag returning"
    log.debug(_msg)

    return DocumentListResponse(
        results=document_responses,
        total_count=total_count,
        has_more=has_more,
        next_offset=next_offset,
    )


def get_documents_by_property(
    session: "Session",
    property_filters: PropertyFilterParams | None = None,
    tag_filter: TagFilter | None = None,
    vault_root: str | None = None,
    pagination: PaginationParams | None = None,
) -> DocumentListResponse:
    """Query documents filtered by frontmatter properties.

    Args:
        session: Database session.
        property_filters: Property filter parameters with include/exclude lists.
        tag_filter: Optional tag filter to also apply.
        vault_root: Filter by specific vault root path (optional).
        pagination: Pagination parameters (limit/offset).

    Returns:
        DocumentListResponse with results and pagination info.

    Raises:
        ValueError: If property filter validation fails.

    Notes:
        Property paths use dot notation (e.g., "author.name").
        Supported operators: equals, contains, exists, in, starts_with, regex.
        For SQLite databases, filtering is done in Python.

    """
    _msg = "get_documents_by_property starting"
    log.debug(_msg)

    # Use default pagination if not provided
    pagination = pagination or PaginationParams(limit=20, offset=0)
    limit = _validate_limit(pagination.limit)
    offset = pagination.offset

    # Extract include/exclude filters from property_filters
    include_properties = property_filters.include_filters if property_filters else None
    exclude_properties = property_filters.exclude_filters if property_filters else None

    validate_property_filters(include_properties)
    validate_property_filters(exclude_properties)
    validate_tag_filter(tag_filter)

    dialect = session.bind.dialect.name if session.bind else "unknown"
    is_postgresql = dialect == "postgresql"

    # Build query parameters
    filter_params = PropertyFilterParams(
        include_filters=include_properties,
        exclude_filters=exclude_properties,
    )
    tag_params = TagFilterParams(tag_filter=tag_filter)
    pagination = PaginationParams(limit=limit, offset=offset)
    query_params = PropertyQueryParams(
        session=session,
        property_filters=filter_params,
        tag_params=tag_params,
        vault_root=vault_root,
        pagination=pagination,
    )

    if is_postgresql:
        results, total_count = get_documents_by_property_postgresql(query_params)
    else:
        results, total_count = get_documents_by_property_sqlite(query_params)

    _msg = "get_documents_by_property returning"
    log.debug(_msg)

    return _build_document_list_response(results, total_count, offset, limit)


def _extract_tags_postgresql(session: "Session", pattern: str | None) -> list[str]:
    """Extract tags using PostgreSQL UNNEST function.

    Args:
        session: Database session.
        pattern: Optional glob pattern to filter tags.

    Returns:
        Sorted list of unique tags.

    """
    from sqlalchemy import func

    tags_query = session.query(
        func.distinct(func.unnest(Document.tags)).label("tag"),
    ).filter(Document.tags.isnot(None))

    if pattern is not None:
        like_pattern = _glob_to_like(pattern)
        tags_query = tags_query.filter(
            func.lower(func.unnest(Document.tags)).ilike(func.lower(like_pattern)),
        )

    tags_query = tags_query.order_by("tag")
    return [row.tag for row in tags_query.all() if row.tag is not None]


def _extract_tags_sqlite(session: "Session", pattern: str | None) -> list[str]:
    """Extract tags using Python for SQLite.

    Args:
        session: Database session.
        pattern: Optional glob pattern to filter tags.

    Returns:
        Sorted list of unique tags.

    """
    docs_with_tags = session.query(Document).filter(Document.tags.isnot(None)).all()

    unique_tags = set()
    for doc in docs_with_tags:
        if doc.tags:
            for tag in doc.tags:
                unique_tags.add(tag)

    if pattern is not None:
        from obsidian_rag.mcp_server.tools.documents_tags import _matches_glob

        unique_tags = {t for t in unique_tags if _matches_glob(t, pattern)}

    return sorted(unique_tags)


def get_all_tags(
    session: "Session",
    pattern: str | None,
    limit: int,
    offset: int,
) -> TagListResponse:
    """Query all unique document tags with optional pattern filtering.

    Args:
        session: Database session.
        pattern: Glob pattern for filtering tags (optional).
            Supports * (any chars), ? (single char), [abc] (char class).
        limit: Maximum number of results (default: 20, max: 100).
        offset: Number of results to skip (default: 0).

    Returns:
        TagListResponse with tags and pagination info.

    Notes:
        Tags are extracted from documents.tags column.
        Duplicate tags across documents are deduplicated.
        Pattern matching is case-insensitive.
        For SQLite databases, extraction is done in Python.

    """
    _msg = "get_all_tags starting"
    log.debug(_msg)

    limit = _validate_limit(limit)
    offset = _validate_offset(offset)

    dialect = session.bind.dialect.name if session.bind else "unknown"
    is_postgresql = dialect == "postgresql"

    if is_postgresql:
        all_tags = _extract_tags_postgresql(session, pattern)
    else:
        all_tags = _extract_tags_sqlite(session, pattern)

    total_count = len(all_tags)
    paginated_tags = all_tags[offset : offset + limit]

    has_more = (offset + limit) < total_count
    next_offset = offset + limit if has_more else None

    _msg = "get_all_tags returning"
    log.debug(_msg)

    return TagListResponse(
        tags=paginated_tags,
        total_count=total_count,
        has_more=has_more,
        next_offset=next_offset,
    )
