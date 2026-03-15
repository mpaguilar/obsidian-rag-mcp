"""Task query tools for MCP server.

All tools in this module are read-only and only use SELECT queries.
"""

import logging
from datetime import date
from typing import TYPE_CHECKING, Any, TypeVar

from sqlalchemy import and_, func, or_
from sqlalchemy.sql.elements import ColumnElement

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

# Type variable for query types to preserve type through filter operations
_T = TypeVar("_T")

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


def _build_due_conditions(
    due_before: date | None,
    due_after: date | None,
) -> list[ColumnElement[bool]]:
    """Build SQL conditions for due date filtering.

    Args:
        due_before: Filter tasks due on or before this date.
        due_after: Filter tasks due on or after this date.

    Returns:
        List of SQLAlchemy conditions for due date.

    """
    conditions: list[ColumnElement[bool]] = []
    if due_before is not None:
        conditions.append(Task.due <= due_before)
    if due_after is not None:
        conditions.append(Task.due >= due_after)
    return conditions


def _build_scheduled_conditions(
    scheduled_before: date | None,
    scheduled_after: date | None,
) -> list[ColumnElement[bool]]:
    """Build SQL conditions for scheduled date filtering.

    Args:
        scheduled_before: Filter tasks scheduled on or before this date.
        scheduled_after: Filter tasks scheduled on or after this date.

    Returns:
        List of SQLAlchemy conditions for scheduled date.

    """
    conditions: list[ColumnElement[bool]] = []
    if scheduled_before is not None:
        conditions.append(Task.scheduled <= scheduled_before)
    if scheduled_after is not None:
        conditions.append(Task.scheduled >= scheduled_after)
    return conditions


def _build_completion_conditions(
    completion_before: date | None,
    completion_after: date | None,
) -> list[ColumnElement[bool]]:
    """Build SQL conditions for completion date filtering.

    Args:
        completion_before: Filter tasks completed on or before this date.
        completion_after: Filter tasks completed on or after this date.

    Returns:
        List of SQLAlchemy conditions for completion date.

    """
    conditions: list[ColumnElement[bool]] = []
    if completion_before is not None:
        conditions.append(Task.completion <= completion_before)
    if completion_after is not None:
        conditions.append(Task.completion >= completion_after)
    return conditions


def _combine_conditions_if_any(
    conditions: list[ColumnElement[bool]],
) -> ColumnElement[bool] | None:
    """Combine multiple conditions with AND if there are any.

    Args:
        conditions: List of SQLAlchemy conditions.

    Returns:
        Combined condition with and_() if multiple, single condition if one,
        or None if empty.

    """
    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return and_(*conditions)


def _apply_date_type_conditions(
    query: "Query[_T]",
    date_type_conditions: list[ColumnElement[bool]],
    *,
    date_match_mode: str,
) -> "Query[_T]":
    """Apply date type conditions to query with specified match mode.

    Args:
        query: The base query to filter.
        date_type_conditions: List of combined conditions for each date type.
        date_match_mode: How to combine date filters - "all" for AND logic,
            "any" for OR logic across all date conditions.

    Returns:
        Query with date type conditions applied.

    """
    if date_match_mode == "any":
        # OR logic: task matches if ANY date type condition is satisfied
        return query.filter(or_(*date_type_conditions))

    # AND logic (default): task matches if ALL date type conditions are satisfied
    for condition in date_type_conditions:
        query = query.filter(condition)
    return query


def _apply_date_filters(
    query: "Query[_T]",
    filters: GetTasksFilterParams,
) -> "Query[_T]":
    """Apply date filters to query with specified match mode.

    Args:
        query: The base query to filter.
        filters: Filter parameters including date ranges and match mode.

    Returns:
        Query with date filters applied.

    """
    # Build conditions grouped by date type
    due_conditions = _build_due_conditions(filters.due_before, filters.due_after)
    scheduled_conditions = _build_scheduled_conditions(
        filters.scheduled_before, filters.scheduled_after
    )
    completion_conditions = _build_completion_conditions(
        filters.completion_before, filters.completion_after
    )

    # Build date type conditions (AND within each type)
    date_type_conditions = []
    for conditions in [due_conditions, scheduled_conditions, completion_conditions]:
        combined = _combine_conditions_if_any(conditions)
        if combined is not None:
            date_type_conditions.append(combined)

    if not date_type_conditions:
        return query

    return _apply_date_type_conditions(
        query, date_type_conditions, date_match_mode=filters.date_match_mode
    )


def _check_due_conditions(
    task: Task,
    due_before: date | None,
    due_after: date | None,
) -> list[bool]:
    """Check due date conditions for a task.

    Args:
        task: Task to check.
        due_before: Filter tasks due on or before this date.
        due_after: Filter tasks due on or after this date.

    Returns:
        List of condition results.

    """
    conditions = []
    if due_before is not None:
        conditions.append(task.due is not None and task.due <= due_before)
    if due_after is not None:
        conditions.append(task.due is not None and task.due >= due_after)
    return conditions


def _check_scheduled_conditions(
    task: Task,
    scheduled_before: date | None,
    scheduled_after: date | None,
) -> list[bool]:
    """Check scheduled date conditions for a task.

    Args:
        task: Task to check.
        scheduled_before: Filter tasks scheduled on or before this date.
        scheduled_after: Filter tasks scheduled on or after this date.

    Returns:
        List of condition results.

    """
    conditions = []
    if scheduled_before is not None:
        conditions.append(
            task.scheduled is not None and task.scheduled <= scheduled_before
        )
    if scheduled_after is not None:
        conditions.append(
            task.scheduled is not None and task.scheduled >= scheduled_after
        )
    return conditions


def _check_completion_conditions(
    task: Task,
    completion_before: date | None,
    completion_after: date | None,
) -> list[bool]:
    """Check completion date conditions for a task.

    Args:
        task: Task to check.
        completion_before: Filter tasks completed on or before this date.
        completion_after: Filter tasks completed on or after this date.

    Returns:
        List of condition results.

    """
    conditions = []
    if completion_before is not None:
        conditions.append(
            task.completion is not None and task.completion <= completion_before
        )
    if completion_after is not None:
        conditions.append(
            task.completion is not None and task.completion >= completion_after
        )
    return conditions


def _evaluate_date_type_conditions(
    date_type_conditions_met: list[bool],
    *,
    date_match_mode: str,
) -> bool:
    """Evaluate date type conditions based on match mode.

    Args:
        date_type_conditions_met: List of boolean results for each date type.
        date_match_mode: How to combine date filters - "all" for AND logic,
            "any" for OR logic across all date conditions.

    Returns:
        True if task matches based on the match mode.

    """
    if date_match_mode == "any":
        # OR logic: task matches if ANY date type condition is satisfied
        return any(date_type_conditions_met)
    # AND logic: task matches if ALL date type conditions are satisfied
    return all(date_type_conditions_met)


def _task_matches_date_filters(
    task: Task,
    filters: GetTasksFilterParams,
) -> bool:
    """Check if a task matches date filters in Python (for SQLite).

    Args:
        task: Task to check.
        filters: Filter parameters including date ranges and match mode.

    Returns:
        True if task matches the date filters.

    """
    # Group conditions by date type (AND within each type)
    date_type_conditions_met = []

    due_conditions = _check_due_conditions(task, filters.due_before, filters.due_after)
    if due_conditions:
        date_type_conditions_met.append(all(due_conditions))

    scheduled_conditions = _check_scheduled_conditions(
        task, filters.scheduled_before, filters.scheduled_after
    )
    if scheduled_conditions:
        date_type_conditions_met.append(all(scheduled_conditions))

    completion_conditions = _check_completion_conditions(
        task, filters.completion_before, filters.completion_after
    )
    if completion_conditions:
        date_type_conditions_met.append(all(completion_conditions))

    if not date_type_conditions_met:
        return True  # No date filters, task passes

    return _evaluate_date_type_conditions(
        date_type_conditions_met, date_match_mode=filters.date_match_mode
    )


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


def _filter_by_date_python(
    results: list[tuple[Task, Document]],
    filters: GetTasksFilterParams,
) -> list[tuple[Task, Document]]:
    """Filter results by date filters in Python (for SQLite).

    Args:
        results: List of (task, document) tuples from query.
        filters: Filter parameters including date ranges and match mode.

    Returns:
        Filtered list of results matching date filters.

    """
    filtered = []
    for task, doc in results:
        if _task_matches_date_filters(task, filters):
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
    with AND logic by default. Use date_match_mode="any" for OR logic across
    date conditions.

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
        Date filters use AND logic by default (all must match), or OR logic
        when date_match_mode="any" (any one can match).

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

    # Apply date filters (PostgreSQL can do this in SQL, SQLite needs Python)
    if is_postgresql:
        query = _apply_date_filters(query, filters)  # type: ignore[assignment]

    query = _apply_tag_filters(query, filters.tags, is_postgresql=is_postgresql)
    query = _apply_priority_filter(query, filters.priority)
    query = _apply_status_exclusion_filters(
        query,
        include_completed=filters.include_completed,
        include_cancelled=filters.include_cancelled,
    )

    # Order by priority and due date
    query = query.order_by(Task.priority, Task.due)

    if is_postgresql:
        # Get total count (before pagination)
        total_count = query.count()

        # Get paginated results
        results = query.offset(offset).limit(limit).all()
    else:
        # For SQLite, fetch all results first, then apply Python filtering and pagination
        all_results = query.all()

        # Apply tag filtering in Python if needed
        if filters.tags:
            all_results = _filter_by_tags_python(all_results, filters.tags)

        # Apply date filtering in Python
        if (
            filters.due_before is not None
            or filters.due_after is not None
            or filters.scheduled_before is not None
            or filters.scheduled_after is not None
            or filters.completion_before is not None
            or filters.completion_after is not None
        ):
            all_results = _filter_by_date_python(all_results, filters)

        # Get total count after filtering
        total_count = len(all_results)

        # Apply pagination in Python
        results = all_results[offset : offset + limit]

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
