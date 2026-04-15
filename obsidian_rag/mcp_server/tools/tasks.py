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


def _strip_tag_prefix(tag: str) -> str:
    """Strip leading '#' characters from a tag string.

    Tags in the database are stored without the '#' prefix, but MCP clients
    may include it when passing filter values. This function normalizes
    tag filter values by removing the '#' prefix before building SQL conditions.

    Args:
        tag: Tag string that may include a leading '#' prefix.

    Returns:
        Tag string with leading '#' characters removed. Returns empty string
        if the tag consists only of '#' characters.

    """
    _msg = "_strip_tag_prefix starting"
    log.debug(_msg)
    result = tag.lstrip("#")
    _msg = "_strip_tag_prefix returning"
    log.debug(_msg)
    return result


def _strip_tag_list(tags: list[str]) -> list[str]:
    """Strip '#' prefix from all tags in a list, removing empty results.

    Args:
        tags: List of tag strings that may include '#' prefixes.

    Returns:
        List of tag strings with '#' prefixes stripped, excluding any
        tags that become empty after stripping.

    """
    _msg = "_strip_tag_list starting"
    log.debug(_msg)
    result = [stripped for tag in tags if (stripped := _strip_tag_prefix(tag))]
    _msg = "_strip_tag_list returning"
    log.debug(_msg)
    return result


def _validate_tag_filters(
    include_tags: list[str] | None,
    exclude_tags: list[str] | None,
) -> None:
    """Validate tag filter parameters.

    Checks for conflicting tags (same tag in both include and exclude lists).
    Tags are compared case-insensitively.

    Args:
        include_tags: List of tags that tasks must have.
        exclude_tags: List of tags that tasks must NOT have.

    Raises:
        ValueError: If conflicting tags are found.

    Notes:
        This validation prevents logical contradictions in filter parameters.
        Case-insensitive comparison ensures "Work" and "work" are treated as conflicts.

    """
    _msg = "_validate_tag_filters starting"
    log.debug(_msg)

    if not include_tags or not exclude_tags:
        _msg = "_validate_tag_filters returning - no validation needed"
        log.debug(_msg)
        return

    # Strip '#' prefix before comparing for conflicts
    include_stripped = _strip_tag_list(include_tags)
    exclude_stripped = _strip_tag_list(exclude_tags)

    include_set = {tag.lower() for tag in include_stripped}
    exclude_set = {tag.lower() for tag in exclude_stripped}
    conflicts = include_set & exclude_set

    if conflicts:
        conflict_list = sorted(conflicts)
        _msg = f"Conflicting tags found: {conflict_list}. Tags cannot appear in both include and exclude lists."
        log.error(_msg)
        raise ValueError(_msg)

    _msg = "_validate_tag_filters returning - validation passed"
    log.debug(_msg)


def _apply_status_filter(query: "Query[Any]", status: list[str] | None) -> "Query[Any]":
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


def _build_tag_condition(tag: str) -> "ColumnElement[bool]":
    """Build SQL condition for a single tag match.

    Args:
        tag: Tag string to match (should already be lowercased).

    Returns:
        SQLAlchemy condition for tag matching.

    """
    return or_(
        func.lower(func.array_to_string(Task.tags, ",")).contains(tag),
        func.lower(func.array_to_string(Document.tags, ",")).contains(tag),
    )


def _apply_include_tags_any(
    query: "Query[Any]", include_tags: list[str]
) -> "Query[Any]":
    """Apply include_tags with OR logic (any match).

    Args:
        query: The base query to filter.
        include_tags: List of tags that tasks must have (any of them).

    Returns:
        Query with include tag filters applied.

    """
    stripped = _strip_tag_list(include_tags)
    if not stripped:
        return query
    include_lower = [t.lower() for t in stripped]
    include_conditions = [_build_tag_condition(tag) for tag in include_lower]
    return query.filter(or_(*include_conditions))


def _apply_include_tags_all(
    query: "Query[Any]", include_tags: list[str]
) -> "Query[Any]":
    """Apply include_tags with AND logic (all match).

    Args:
        query: The base query to filter.
        include_tags: List of tags that tasks must have (all of them).

    Returns:
        Query with include tag filters applied.

    """
    stripped = _strip_tag_list(include_tags)
    if not stripped:
        return query
    include_lower = [t.lower() for t in stripped]
    for tag in include_lower:
        query = query.filter(_build_tag_condition(tag))
    return query


def _apply_exclude_tags(query: "Query[Any]", exclude_tags: list[str]) -> "Query[Any]":
    """Apply exclude_tags with OR logic (any match excludes).

    Args:
        query: The base query to filter.
        exclude_tags: List of tags that tasks must NOT have.

    Returns:
        Query with exclude tag filters applied.

    """
    stripped = _strip_tag_list(exclude_tags)
    if not stripped:
        return query
    exclude_lower = [t.lower() for t in stripped]
    exclude_conditions = [_build_tag_condition(tag) for tag in exclude_lower]
    return query.filter(~or_(*exclude_conditions))


def _apply_tag_filters(
    query: "Query[Any]",
    filters: GetTasksFilterParams,
) -> "Query[Any]":
    """Apply tag filters to query with support for include/exclude and match modes.

    Supports 'include_tags' parameter with configurable match_mode ("all" for
    AND, "any" for OR). Also supports 'exclude_tags' for exclusion filtering.

    Args:
        query: The base query to filter.
        filters: Filter parameters including include_tags, exclude_tags,
            and tag_match_mode.

    Returns:
        Query with tag filters applied.

    Notes:
        'include_tags' with tag_match_mode="all" uses AND logic.
        'include_tags' with tag_match_mode="any" uses OR logic.
        'exclude_tags' always uses OR logic (any excluded tag disqualifies).
        Tag matching is case-insensitive.

    """
    _msg = "_apply_tag_filters starting"
    log.debug(_msg)

    # Handle 'include_tags' parameter with match_mode
    if filters.include_tags:
        if filters.tag_match_mode == "any":
            query = _apply_include_tags_any(query, filters.include_tags)
        else:
            query = _apply_include_tags_all(query, filters.include_tags)

    # Handle 'exclude_tags' parameter (always OR logic)
    if filters.exclude_tags:
        query = _apply_exclude_tags(query, filters.exclude_tags)

    _msg = "_apply_tag_filters returning"
    log.debug(_msg)
    return query


def _apply_priority_filter(
    query: "Query[Any]", priority: list[str] | None
) -> "Query[Any]":
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
    date conditions. Use tag_match_mode="any" for OR logic across include_tags.

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
        - include_tags with tag_match_mode="all" (default): AND logic
        - include_tags with tag_match_mode="any": OR logic
        - exclude_tags: OR logic (task excluded if it has ANY excluded tag)
        - Date filters: Configurable via date_match_mode
            - "all" (default): AND logic across all date conditions
            - "any": OR logic across all date conditions
        - Different filter types (status, tags, priority, dates): AND logic

    Tag Filtering:
        Tags should NOT include the '#' prefix. Use plain tag names like
        "personal/expenses" or "business/iConnections" instead of
        "#personal/expenses" or "#business/iConnections".

    Tag Filtering Examples:
        - include_tags=["work", "urgent"], tag_match_mode="all": Task must have BOTH
        - include_tags=["work", "personal"], tag_match_mode="any": Task has EITHER
        - exclude_tags=["blocked"]: Task must NOT have "blocked" tag
        - include_tags=["work"], exclude_tags=["blocked"]: Task has "work" but NOT "blocked"

    Args:
        session: Database session.
        filters: Filter parameters including status, date ranges, tags,
            priority, and pagination options. If None, returns all tasks.

    Returns:
        TaskListResponse with results and pagination info.

    Raises:
        ValueError: If tag filter validation fails (conflicting tags).

    Notes:
        Date comparisons are inclusive (>= for after, <= for before).
        Tasks without dates are excluded from date filter comparisons.
        Tag validation prevents conflicting tags in include/exclude lists.

    """
    _msg = "get_tasks starting"
    log.debug(_msg)

    # Use default filters if none provided
    filters = filters or GetTasksFilterParams()

    # Validate tag filters (checks for conflicting tags)
    _validate_tag_filters(filters.include_tags, filters.exclude_tags)

    # Validate pagination
    limit = _validate_limit(filters.limit)
    offset = _validate_offset(filters.offset)

    # Build base query
    query = session.query(Task, Document).join(
        Document, Task.document_id == Document.id
    )

    # Apply all filters
    query = _apply_status_filter(query, filters.status)  # type: ignore[assignment]
    query = _apply_date_filters(query, filters)  # type: ignore[assignment]
    query = _apply_tag_filters(query, filters)  # type: ignore[assignment]
    query = _apply_priority_filter(query, filters.priority)  # type: ignore[assignment]

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
