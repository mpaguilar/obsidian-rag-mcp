"""Tag filtering utilities for document queries.

This module contains tag filtering logic for both PostgreSQL and SQLite databases.
All functions are pure logic functions that can be tested independently.
"""

import fnmatch
import logging
from typing import TYPE_CHECKING, Optional

from sqlalchemy import func, or_, text

from obsidian_rag.database.models import Document

if TYPE_CHECKING:
    from sqlalchemy.orm import Query, Session

    from obsidian_rag.mcp_server.models import TagFilter

log = logging.getLogger(__name__)

# Maximum query complexity limits
MAX_TAGS_PER_QUERY = 50


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
    return fnmatch.fnmatch(tag.lower(), pattern.lower())


def _tag_in_doc_tags(tag: str, doc_tags_lower: set[str]) -> bool:
    """Check if a tag is present in document tags (substring match).

    Args:
        tag: Tag to search for (lowercase).
        doc_tags_lower: Set of document tags (lowercase).

    Returns:
        True if tag is found in any document tag.

    """
    tag_lower = tag.lower()
    for doc_tag in doc_tags_lower:
        if tag_lower in doc_tag:
            return True
    return False


def _matches_all_tags(doc: Document, tags: list[str]) -> bool:
    """Check if document has ALL of the specified tags.

    Args:
        doc: Document to check.
        tags: List of tags to match (case-insensitive).

    Returns:
        True if document has all tags.

    """
    if not tags:
        return True
    if doc.tags is None:
        return False
    doc_tags_lower = {t.lower() for t in doc.tags}
    return all(_tag_in_doc_tags(tag, doc_tags_lower) for tag in tags)


def _matches_any_tags(doc: Document, tags: list[str]) -> bool:
    """Check if document has ANY of the specified tags.

    Args:
        doc: Document to check.
        tags: List of tags to match (case-insensitive).

    Returns:
        True if document has at least one tag.

    """
    if not tags:
        return True
    if doc.tags is None:
        return False
    doc_tags_lower = {t.lower() for t in doc.tags}
    return any(_tag_in_doc_tags(tag, doc_tags_lower) for tag in tags)


def _has_any_excluded_tags(doc: Document, tags: list[str]) -> bool:
    """Check if document has ANY of the excluded tags.

    Args:
        doc: Document to check.
        tags: List of tags to exclude (case-insensitive).

    Returns:
        True if document has any excluded tag.

    """
    if not tags or doc.tags is None:
        return False
    doc_tags_lower = {t.lower() for t in doc.tags}
    return any(_tag_in_doc_tags(tag, doc_tags_lower) for tag in tags)


def _check_tag_count(tag_filter: "TagFilter") -> None:
    """Check if tag counts are within limits.

    Args:
        tag_filter: Tag filter to validate.

    Raises:
        ValueError: If tag count exceeds maximum.

    """
    if len(tag_filter.include_tags) > MAX_TAGS_PER_QUERY:
        msg = f"Maximum {MAX_TAGS_PER_QUERY} include_tags allowed"
        raise ValueError(msg)
    if len(tag_filter.exclude_tags) > MAX_TAGS_PER_QUERY:
        msg = f"Maximum {MAX_TAGS_PER_QUERY} exclude_tags allowed"
        raise ValueError(msg)


def _check_tag_conflicts(tag_filter: "TagFilter") -> None:
    """Check for conflicting tags in include and exclude lists.

    Args:
        tag_filter: Tag filter to validate.

    Raises:
        ValueError: If conflicting tags found.

    """
    if not tag_filter.include_tags or not tag_filter.exclude_tags:
        return
    include_set = {t.lower() for t in tag_filter.include_tags}
    exclude_set = {t.lower() for t in tag_filter.exclude_tags}
    conflicts = include_set & exclude_set
    if conflicts:
        msg = f"Conflicting tags: {sorted(conflicts)}. Tags cannot appear in both include and exclude lists."
        raise ValueError(msg)


def validate_tag_filter(tag_filter: Optional["TagFilter"]) -> None:
    """Validate tag filter parameters.

    Args:
        tag_filter: Tag filter to validate.

    Raises:
        ValueError: If validation fails.

    """
    if tag_filter is None:
        return
    _check_tag_count(tag_filter)
    _check_tag_conflicts(tag_filter)


def _check_include_tags(doc: Document, tag_filter: "TagFilter") -> bool:
    """Check if document passes include tag filter.

    Args:
        doc: Document to check.
        tag_filter: Tag filter with include tags.

    Returns:
        True if document passes include filter.

    """
    if not tag_filter.include_tags:
        return True
    if tag_filter.match_mode == "all":
        return _matches_all_tags(doc, tag_filter.include_tags)
    return _matches_any_tags(doc, tag_filter.include_tags)


def _check_exclude_tags(doc: Document, tag_filter: "TagFilter") -> bool:
    """Check if document passes exclude tag filter.

    Args:
        doc: Document to check.
        tag_filter: Tag filter with exclude tags.

    Returns:
        True if document passes exclude filter (no excluded tags present).

    """
    if not tag_filter.exclude_tags:
        return True
    return not _has_any_excluded_tags(doc, tag_filter.exclude_tags)


def matches_tag_filter(doc: Document, tag_filter: Optional["TagFilter"]) -> bool:
    """Check if document matches tag filter.

    Args:
        doc: Document to check.
        tag_filter: Tag filter to apply.

    Returns:
        True if document matches the filter.

    """
    if tag_filter is None:
        return True
    return _check_include_tags(doc, tag_filter) and _check_exclude_tags(doc, tag_filter)


def apply_postgresql_include_tags(
    query: "Query[Document]",
    tag_filter: "TagFilter",
) -> "Query[Document]":
    """Apply PostgreSQL include tag filter.

    Args:
        query: SQLAlchemy query object.
        tag_filter: Tag filter with include tags.

    Returns:
        Filtered query.

    """
    if not tag_filter.include_tags:
        return query

    include_lower = [t.lower() for t in tag_filter.include_tags]
    if tag_filter.match_mode == "all":
        # Document must have ALL include tags
        for tag in include_lower:
            query = query.filter(
                func.lower(func.array_to_string(Document.tags, ",")).contains(tag)
            )
    else:  # "any"
        # Document must have ANY of the include tags
        conditions = [
            func.lower(func.array_to_string(Document.tags, ",")).contains(tag)
            for tag in include_lower
        ]
        query = query.filter(or_(*conditions))
    return query


def apply_postgresql_exclude_tags(
    query: "Query[Document]",
    tag_filter: "TagFilter",
) -> "Query[Document]":
    """Apply PostgreSQL exclude tag filter.

    Args:
        query: SQLAlchemy query object.
        tag_filter: Tag filter with exclude tags.

    Returns:
        Filtered query.

    """
    if not tag_filter.exclude_tags:
        return query

    exclude_lower = [t.lower() for t in tag_filter.exclude_tags]
    # Document must NOT have ANY exclude tags
    exclude_conditions = [
        func.lower(func.array_to_string(Document.tags, ",")).contains(tag)
        for tag in exclude_lower
    ]
    query = query.filter(text("NOT (") + or_(*exclude_conditions) + text(")"))
    return query


def apply_postgresql_tag_filter(
    query: "Query[Document]",
    tag_filter: Optional["TagFilter"],
) -> "Query[Document]":
    """Apply PostgreSQL tag filtering with include/exclude lists.

    Args:
        query: SQLAlchemy query object.
        tag_filter: Tag filter to apply.

    Returns:
        Filtered query.

    """
    if tag_filter is None:
        return query

    query = apply_postgresql_include_tags(query, tag_filter)
    query = apply_postgresql_exclude_tags(query, tag_filter)
    return query
