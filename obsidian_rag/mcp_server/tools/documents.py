"""Document query tools for MCP server.

All tools in this module are read-only and only use SELECT queries.
"""

import logging
import re
from typing import TYPE_CHECKING

from sqlalchemy import func, or_

from obsidian_rag.database.models import Document
from obsidian_rag.mcp_server.models import (
    DocumentListResponse,
    TagListResponse,
    create_document_response,
    _validate_limit,
    _validate_offset,
)

if TYPE_CHECKING:
    from sqlalchemy.orm import Query, Session

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
        For SQLite databases (used in testing), returns empty results since
        pg_vector's cosine distance operator is not available.

    """
    _msg = "query_documents starting"
    log.debug(_msg)

    limit = _validate_limit(limit)
    offset = _validate_offset(offset)

    # Detect database dialect - PostgreSQL is required for vector operations
    dialect = session.bind.dialect.name if session.bind else "unknown"

    if dialect != "postgresql":
        # For SQLite and other databases without pg_vector, return empty results
        # since content_vector won't be populated with proper vector data
        _msg = f"Vector similarity not supported for dialect: {dialect}"
        log.debug(_msg)
        return DocumentListResponse(
            results=[],
            total_count=0,
            has_more=False,
            next_offset=None,
        )

    # Build vector similarity query for PostgreSQL
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


def _has_tags(doc: Document, search_tag: str) -> bool:
    """Check if document has a matching tag (case-insensitive substring).

    Args:
        doc: Document to check.
        search_tag: Tag to search for (lowercase).

    Returns:
        True if document has a matching tag.

    """
    if doc.tags is None:
        return False
    for t in doc.tags:
        if search_tag in t.lower():
            return True
    return False


def _is_untagged(doc: Document) -> bool:
    """Check if document has no tags.

    Args:
        doc: Document to check.

    Returns:
        True if document has no tags or empty tags.

    """
    return doc.tags is None or len(doc.tags) == 0


def _matches_glob(tag: str, pattern: str) -> bool:
    """Check if tag matches glob pattern.

    Args:
        tag: Tag to check.
        pattern: Glob pattern (supports *, ?, [abc]).

    Returns:
        True if tag matches pattern.

    Notes:
        Uses fnmatch for glob pattern matching.

    """
    import fnmatch

    return fnmatch.fnmatch(tag.lower(), pattern.lower())


def _apply_postgresql_tag_filter(
    query: "Query[Document]",
    tag: str | None,
    include_untagged: bool,
):
    """Apply PostgreSQL-specific tag filtering.

    Args:
        query: SQLAlchemy query object.
        tag: Tag to filter by.
        include_untagged: Whether to include untagged documents.

    Returns:
        Filtered query.

    """
    if tag is None and not include_untagged:
        return query

    if tag is not None:
        search_tag = tag.lower()
        tag_filter = func.lower(Document.tags).contains(search_tag)
        untagged_filter = or_(
            Document.tags.is_(None),
            func.array_length(Document.tags, 1) == 0,
        )
        query = query.filter(
            or_(tag_filter, untagged_filter) if include_untagged else tag_filter
        )
    elif include_untagged:
        query = query.filter(
            or_(
                Document.tags.is_(None),
                func.array_length(Document.tags, 1) == 0,
            )
        )

    return query


def _should_include_doc(doc: Document, tag: str | None, include_untagged: bool) -> bool:
    """Check if document should be included based on tag filters.

    Args:
        doc: Document to check.
        tag: Tag to filter by.
        include_untagged: Whether to include untagged documents.

    Returns:
        True if document should be included.

    """
    if tag is None:
        return include_untagged and _is_untagged(doc)

    has_tag = _has_tags(doc, tag.lower())
    if include_untagged:
        return has_tag or _is_untagged(doc)
    return has_tag


def _filter_docs_python(
    all_docs: list[Document],
    tag: str | None,
    include_untagged: bool,
) -> list[Document]:
    """Filter documents using Python logic.

    Args:
        all_docs: List of documents to filter.
        tag: Tag to filter by.
        include_untagged: Whether to include untagged documents.

    Returns:
        Filtered list of documents.

    """
    if tag is None and not include_untagged:
        return all_docs

    return [doc for doc in all_docs if _should_include_doc(doc, tag, include_untagged)]


def get_documents_by_tag(  # noqa: PLR0913
    session: "Session",
    tag: str | None,
    vault_root: str | None,
    include_untagged: bool,
    limit: int,
    offset: int,
) -> DocumentListResponse:
    """Query documents filtered by tag.

    Args:
        session: Database session.
        tag: Tag to filter by (optional, case-insensitive substring match).
        vault_root: Filter by specific vault root path (optional).
        include_untagged: Include documents with no tags when True.
        limit: Maximum number of results (default: 20, max: 100).
        offset: Number of results to skip (default: 0).

    Returns:
        DocumentListResponse with results and pagination info.

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

    dialect = session.bind.dialect.name if session.bind else "unknown"
    is_postgresql = dialect == "postgresql"

    query = session.query(Document)

    if vault_root is not None:
        query = query.filter(Document.vault_root == vault_root)

    if is_postgresql:
        query = _apply_postgresql_tag_filter(query, tag, include_untagged)
        query = query.order_by(Document.file_name)
        total_count = query.count()
        results = query.offset(offset).limit(limit).all()
    else:
        query = query.order_by(Document.file_name)
        all_docs = query.all()
        filtered_docs = _filter_docs_python(all_docs, tag, include_untagged)
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


def _extract_tags_postgresql(session: "Session", pattern: str | None) -> list[str]:
    """Extract tags using PostgreSQL UNNEST function.

    Args:
        session: Database session.
        pattern: Optional glob pattern to filter tags.

    Returns:
        Sorted list of unique tags.

    """
    tags_query = session.query(
        func.distinct(func.unnest(Document.tags)).label("tag")
    ).filter(Document.tags.isnot(None))

    if pattern is not None:
        like_pattern = _glob_to_like(pattern)
        tags_query = tags_query.filter(
            func.lower(func.unnest(Document.tags)).ilike(func.lower(like_pattern))
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
