"""Task query tools for MCP server.

All tools in this module are read-only and only use SELECT queries.
"""

import logging
from datetime import date
from typing import TYPE_CHECKING, Any, TypeVar

from sqlalchemy import and_, func, or_
from sqlalchemy.sql.elements import ColumnElement

from obsidian_rag.database.models import Document, Task
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


def _apply_tag_filters(
    query: "Query[Any]",
    tags: list[str] | None,
):
    """Apply tag filters to query.

    Args:
        query: The base query to filter.
        tags: List of tags that tasks must have.

    Returns:
        Query with tag filters applied.

    """
    if not tags:
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


def get_tasks(
    session: "Session",
    filters: GetTasksFilterParams | None = None,
) -> TaskListResponse:
    """Query tasks with comprehensive filtering.

    This is a generic task query tool that supports filtering by status,
    date ranges, tags, and priority. All filters are optional and combined
    with AND logic by default. Use date_match_mode="any" for OR logic across
    date conditions.

    Valid Status Values:
        - "not_completed": Tasks that are not yet completed
        - "completed": Tasks that have been completed
        - "in_progress": Tasks currently being worked on
        - "cancelled": Tasks that have been cancelled

    Valid Priority Values:
        - "highest": Critical priority tasks
        - "high": High priority tasks
        - "normal": Normal priority tasks (default)
        - "low": Low priority tasks
        - "lowest": Lowest priority tasks

    Filter Logic:
        - Multiple status values: OR logic (task matches ANY status)
        - Multiple priority values: OR logic (task matches ANY priority)
        - Multiple tags: AND logic (task must have ALL tags)
        - Date filters: Configurable via date_match_mode
            - "all" (default): AND logic across all date conditions
            - "any": OR logic across all date conditions
        - Different filter types (status, tags, priority, dates): AND logic

    Args:
        session: Database session.
        filters: Filter parameters including status, date ranges, tags,
            priority, and pagination options. If None, returns all tasks.

    Returns:
        TaskListResponse with results and pagination info.

    Notes:
        Date comparisons are inclusive (>= for after, <= for before).
        Tasks without dates are excluded from date filter comparisons.

    """
    _msg = "get_tasks starting"
    log.debug(_msg)

    # Use default filters if none provided
    filters = filters or GetTasksFilterParams()

    # Validate pagination
    limit = _validate_limit(filters.limit)
    offset = _validate_offset(filters.offset)

    # Build base query
    query = session.query(Task, Document).join(
        Document, Task.document_id == Document.id
    )

    # Apply all filters
    query = _apply_status_filter(query, filters.status)
    query = _apply_date_filters(query, filters)  # type: ignore[assignment]
    query = _apply_tag_filters(query, filters.tags)
    query = _apply_priority_filter(query, filters.priority)

    # Order by priority and due date
    query = query.order_by(Task.priority, Task.due)

    # Get total count (before pagination)
    total_count = query.count()

    # Get paginated results
    results = query.offset(offset).limit(limit).all()

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
