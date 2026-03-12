"""Task query tools for MCP server.

All tools in this module are read-only and only use SELECT queries.
"""

import logging
from datetime import date
from typing import TYPE_CHECKING, Any

from sqlalchemy import func, or_

from obsidian_rag.database.models import Document, Task, TaskStatus
from obsidian_rag.mcp_server.models import (
    TaskListResponse,
    _validate_limit,
    _validate_offset,
    create_task_response,
)
from obsidian_rag.mcp_server.tools.tasks_params import GetTasksFilterParams

if TYPE_CHECKING:
    from sqlalchemy.orm import Query, Session

log = logging.getLogger(__name__)


def _apply_status_filter(query: "Query[Any]", status: list[str] | None):
    """Apply status filter to query.

    Args:
        query: The base query to filter.
        status: List of statuses to filter by.

    Returns:
        Query with status filter applied.

    """
    if status:
        return query.filter(Task.status.in_(status))
    return query


def _apply_due_date_filters(
    query: "Query[Any]",
    due_before: date | None,
    due_after: date | None,
):
    """Apply due date filters to query.

    Args:
        query: The base query to filter.
        due_before: Filter tasks due on or before this date.
        due_after: Filter tasks due on or after this date.

    Returns:
        Query with due date filters applied.

    """
    if due_before is not None:
        query = query.filter(Task.due <= due_before)
    if due_after is not None:
        query = query.filter(Task.due >= due_after)
    return query


def _apply_scheduled_date_filters(
    query: "Query[Any]",
    scheduled_before: date | None,
    scheduled_after: date | None,
):
    """Apply scheduled date filters to query.

    Args:
        query: The base query to filter.
        scheduled_before: Filter tasks scheduled on or before this date.
        scheduled_after: Filter tasks scheduled on or after this date.

    Returns:
        Query with scheduled date filters applied.

    """
    if scheduled_before is not None:
        query = query.filter(Task.scheduled <= scheduled_before)
    if scheduled_after is not None:
        query = query.filter(Task.scheduled >= scheduled_after)
    return query


def _apply_completion_date_filters(
    query: "Query[Any]",
    completion_before: date | None,
    completion_after: date | None,
):
    """Apply completion date filters to query.

    Args:
        query: The base query to filter.
        completion_before: Filter tasks completed on or before this date.
        completion_after: Filter tasks completed on or after this date.

    Returns:
        Query with completion date filters applied.

    """
    if completion_before is not None:
        query = query.filter(Task.completion <= completion_before)
    if completion_after is not None:
        query = query.filter(Task.completion >= completion_after)
    return query


def _apply_tag_filters(
    query: "Query[Any]",
    tags: list[str] | None,
    *,
    is_postgresql: bool = True,
):
    """Apply tag filters to query.

    Args:
        query: The base query to filter.
        tags: List of tags that tasks must have.
        is_postgresql: Whether the database is PostgreSQL (uses array functions).
            If False (SQLite), tag filtering is skipped here and done in Python.

    Returns:
        Query with tag filters applied (for PostgreSQL) or unchanged (for SQLite).

    """
    if not tags:
        return query

    if not is_postgresql:
        # For SQLite, skip SQL-level tag filtering
        # Tags will be filtered in Python after fetching results
        return query

    # For PostgreSQL, use array_to_string for proper array filtering
    for tag in tags:
        tag_lower = tag.lower()
        query = query.filter(
            or_(
                func.lower(func.array_to_string(Task.tags, ",")).contains(tag_lower),
                func.lower(func.array_to_string(Document.tags, ",")).contains(
                    tag_lower
                ),
            ),
        )
    return query


def _tags_contain_substring(tags: list[str] | None, substring: str) -> bool:
    """Check if any tag contains the substring (case-insensitive).

    Args:
        tags: List of tags to check.
        substring: Substring to search for (already lowercased).

    Returns:
        True if any tag contains the substring.

    """
    if not tags:
        return False
    return any(substring in t.lower() for t in tags)


def _task_has_tag(task: Task, doc: Document, search_tag: str) -> bool:
    """Check if task or document has a matching tag (case-insensitive substring).

    Args:
        task: Task to check.
        doc: Document to check.
        search_tag: Tag to search for.

    Returns:
        True if task or document has a matching tag.

    """
    tag_lower = search_tag.lower()
    return _tags_contain_substring(task.tags, tag_lower) or _tags_contain_substring(
        doc.tags, tag_lower
    )


def _filter_by_tags_python(
    results: list[tuple[Task, Document]],
    tags: list[str],
) -> list[tuple[Task, Document]]:
    """Filter results by tags in Python (for SQLite).

    Args:
        results: List of (task, document) tuples from query.
        tags: List of tags that tasks must have (AND logic).

    Returns:
        Filtered list of results where task/document has all tags.

    """
    filtered = []
    for task, doc in results:
        # Task must have ALL tags (AND logic)
        if all(_task_has_tag(task, doc, tag) for tag in tags):
            filtered.append((task, doc))
    return filtered


def _apply_priority_filter(query: "Query[Any]", priority: list[str] | None):
    """Apply priority filter to query.

    Args:
        query: The base query to filter.
        priority: List of priorities to filter by.

    Returns:
        Query with priority filter applied.

    """
    if priority:
        return query.filter(Task.priority.in_(priority))
    return query


def _apply_status_exclusion_filters(
    query: "Query[Any]",
    *,
    include_completed: bool,
    include_cancelled: bool,
):
    """Apply status exclusion filters to query.

    Args:
        query: The base query to filter.
        include_completed: Whether to include completed tasks.
        include_cancelled: Whether to include cancelled tasks.

    Returns:
        Query with status exclusion filters applied.

    """
    if not include_completed:
        query = query.filter(Task.status != TaskStatus.COMPLETED.value)
    if not include_cancelled:
        query = query.filter(Task.status != TaskStatus.CANCELLED.value)
    return query


def get_tasks(
    session: "Session",
    filters: GetTasksFilterParams | None = None,
) -> TaskListResponse:
    """Query tasks with comprehensive filtering.

    This is a generic task query tool that supports filtering by status,
    date ranges, tags, and priority. All filters are optional and combined
    with AND logic.

    Args:
        session: Database session.
        filters: Filter parameters including status, date ranges, tags,
            priority, and pagination options. If None, returns all tasks.

    Returns:
        TaskListResponse with results and pagination info.

    Notes:
        Date comparisons are inclusive (>= for after, <= for before).
        Tasks without dates are excluded from date filter comparisons.
        Multiple status filters use OR logic (task matches any status).
        Multiple tag filters use AND logic (task must have all tags).
        Multiple priority filters use OR logic (task matches any priority).

    """
    _msg = "get_tasks starting"
    log.debug(_msg)

    # Use default filters if none provided
    filters = filters or GetTasksFilterParams()

    # Validate pagination
    limit = _validate_limit(filters.limit)
    offset = _validate_offset(filters.offset)

    # Detect database dialect
    dialect = session.bind.dialect.name if session.bind else "unknown"
    is_postgresql = dialect == "postgresql"

    # Build base query
    query = session.query(Task, Document).join(
        Document, Task.document_id == Document.id
    )

    # Apply all filters
    query = _apply_status_filter(query, filters.status)
    query = _apply_due_date_filters(query, filters.due_before, filters.due_after)
    query = _apply_scheduled_date_filters(
        query, filters.scheduled_before, filters.scheduled_after
    )
    query = _apply_completion_date_filters(
        query, filters.completion_before, filters.completion_after
    )
    query = _apply_tag_filters(query, filters.tags, is_postgresql=is_postgresql)
    query = _apply_priority_filter(query, filters.priority)
    query = _apply_status_exclusion_filters(
        query,
        include_completed=filters.include_completed,
        include_cancelled=filters.include_cancelled,
    )

    # Order by priority and due date
    query = query.order_by(Task.priority, Task.due)

    # Get total count (before pagination)
    total_count = query.count()

    # Get paginated results
    results = query.offset(offset).limit(limit).all()

    # For SQLite, apply tag filtering in Python if needed
    if not is_postgresql and filters.tags:
        results = _filter_by_tags_python(results, filters.tags)
        total_count = len(results)

    # Convert to response models
    task_responses = [create_task_response(task, doc) for task, doc in results]

    # Calculate pagination
    has_more = (offset + limit) < total_count
    next_offset = offset + limit if has_more else None

    _msg = "get_tasks returning"
    log.debug(_msg)

    return TaskListResponse(
        results=task_responses,
        total_count=total_count,
        has_more=has_more,
        next_offset=next_offset,
    )
