"""Task query tools for MCP server.

All tools in this module are read-only and only use SELECT queries.
"""

import logging
from datetime import date, timedelta
from typing import TYPE_CHECKING

from sqlalchemy import func, or_

from obsidian_rag.database.models import Document, Task, TaskStatus
from obsidian_rag.mcp_server.models import (
    TaskListResponse,
    _validate_limit,
    _validate_offset,
    create_task_response,
)

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

log = logging.getLogger(__name__)


def get_incomplete_tasks(
    session: "Session",
    limit: int = 20,
    offset: int = 0,
    *,
    include_cancelled: bool = False,
) -> TaskListResponse:
    """Query tasks that are not completed.

    Args:
        session: Database session.
        limit: Maximum number of results (default: 20, max: 100).
        offset: Number of results to skip (default: 0).
        include_cancelled: Whether to include cancelled tasks.

    Returns:
        TaskListResponse with results and pagination info.

    """
    _msg = "get_incomplete_tasks starting"
    log.debug(_msg)

    limit = _validate_limit(limit)
    offset = _validate_offset(offset)

    # Build query for incomplete tasks
    statuses = [TaskStatus.NOT_COMPLETED.value, TaskStatus.IN_PROGRESS.value]
    if include_cancelled:
        statuses.append(TaskStatus.CANCELLED.value)

    query = (
        session.query(Task, Document)
        .join(Document, Task.document_id == Document.id)
        .filter(Task.status.in_(statuses))
        .order_by(Task.due.asc())
    )

    # Get total count
    total_count = query.count()

    # Get paginated results
    results = query.offset(offset).limit(limit).all()

    # Convert to response models
    task_responses = [create_task_response(task, doc) for task, doc in results]

    # Calculate pagination
    has_more = (offset + limit) < total_count
    next_offset = offset + limit if has_more else None

    _msg = "get_incomplete_tasks returning"
    log.debug(_msg)

    return TaskListResponse(
        results=task_responses,
        total_count=total_count,
        has_more=has_more,
        next_offset=next_offset,
    )


def get_tasks_due_this_week(
    session: "Session",
    limit: int = 20,
    offset: int = 0,
    *,
    include_completed: bool = True,
) -> TaskListResponse:
    """Query tasks due within the next 7 days.

    Args:
        session: Database session.
        limit: Maximum number of results (default: 20, max: 100).
        offset: Number of results to skip (default: 0).
        include_completed: Whether to include completed tasks.

    Returns:
        TaskListResponse with results and pagination info.

    """
    _msg = "get_tasks_due_this_week starting"
    log.debug(_msg)

    limit = _validate_limit(limit)
    offset = _validate_offset(offset)

    # Calculate date range (today to 7 days from now)
    today = date.today()
    week_from_now = today + timedelta(days=7)

    # Build query
    query = (
        session.query(Task, Document)
        .join(Document, Task.document_id == Document.id)
        .filter(Task.due.isnot(None))
        .filter(Task.due >= today)
        .filter(Task.due <= week_from_now)
    )

    if not include_completed:
        query = query.filter(Task.status != TaskStatus.COMPLETED.value)

    query = query.order_by(Task.due.asc())

    # Get total count
    total_count = query.count()

    # Get paginated results
    results = query.offset(offset).limit(limit).all()

    # Convert to response models
    task_responses = [create_task_response(task, doc) for task, doc in results]

    # Calculate pagination
    has_more = (offset + limit) < total_count
    next_offset = offset + limit if has_more else None

    _msg = "get_tasks_due_this_week returning"
    log.debug(_msg)

    return TaskListResponse(
        results=task_responses,
        total_count=total_count,
        has_more=has_more,
        next_offset=next_offset,
    )


def _task_has_tag(task: Task, doc: Document, search_tag: str) -> bool:
    """Check if task or document has a matching tag (case-insensitive substring).

    Args:
        task: Task to check.
        doc: Document to check.
        search_tag: Tag to search for (lowercase).

    Returns:
        True if task or document has a matching tag.

    """
    # Check task tags
    if task.tags is not None:
        if any(search_tag in t.lower() for t in task.tags):
            return True

    # Check document tags
    if doc.tags is not None:
        if any(search_tag in t.lower() for t in doc.tags):
            return True

    return False


def get_tasks_by_tag(
    session: "Session",
    tag: str,
    limit: int = 20,
    offset: int = 0,
) -> TaskListResponse:
    """Query tasks by tag (matches task or document level).

    Args:
        session: Database session.
        tag: Tag to search for (case-insensitive).
        limit: Maximum number of results (default: 20, max: 100).
        offset: Number of results to skip (default: 0).

    Returns:
        TaskListResponse with results and pagination info.

    Notes:
        For PostgreSQL: Uses array_to_string for array filtering.
        For SQLite: Filters in Python due to lack of array support.

    """
    _msg = f"get_tasks_by_tag starting with tag: {tag}"
    log.debug(_msg)

    limit = _validate_limit(limit)
    offset = _validate_offset(offset)

    # Normalize tag for case-insensitive search
    search_tag = tag.lower()

    # Check database dialect
    dialect = session.bind.dialect.name if session.bind else "unknown"
    is_postgresql = dialect == "postgresql"

    if is_postgresql:  # pragma: no cover (PostgreSQL-only)
        # PostgreSQL: Use array_to_string for array filtering
        query = (
            session.query(Task, Document)
            .join(Document, Task.document_id == Document.id)
            .filter(
                or_(
                    func.lower(func.array_to_string(Task.tags, ",")).contains(
                        search_tag,
                    ),
                    func.lower(func.array_to_string(Document.tags, ",")).contains(
                        search_tag,
                    ),
                ),
            )
            .order_by(Task.due.asc())
        )

        # Get total count
        total_count = query.count()

        # Get paginated results
        results = query.offset(offset).limit(limit).all()
    else:
        # SQLite: Filter in Python due to lack of array support
        query = (
            session.query(Task, Document)
            .join(Document, Task.document_id == Document.id)
            .order_by(Task.due.asc())
        )

        # Get all results and filter in Python
        all_results = query.all()
        filtered_results = [
            (task, doc)
            for task, doc in all_results
            if _task_has_tag(task, doc, search_tag)
        ]

        total_count = len(filtered_results)
        results = filtered_results[offset : offset + limit]

    # Convert to response models
    task_responses = [create_task_response(task, doc) for task, doc in results]

    # Calculate pagination
    has_more = (offset + limit) < total_count
    next_offset = offset + limit if has_more else None

    _msg = "get_tasks_by_tag returning"
    log.debug(_msg)

    return TaskListResponse(
        results=task_responses,
        total_count=total_count,
        has_more=has_more,
        next_offset=next_offset,
    )


def get_completed_tasks(
    session: "Session",
    limit: int = 20,
    offset: int = 0,
    completed_since: date | None = None,
) -> TaskListResponse:
    """Query completed tasks with optional date filter.

    Args:
        session: Database session.
        limit: Maximum number of results (default: 20, max: 100).
        offset: Number of results to skip (default: 0).
        completed_since: Filter tasks completed after this date.

    Returns:
        TaskListResponse with results and pagination info.

    """
    _msg = "get_completed_tasks starting"
    log.debug(_msg)

    limit = _validate_limit(limit)
    offset = _validate_offset(offset)

    # Build query for completed tasks
    query = (
        session.query(Task, Document)
        .join(Document, Task.document_id == Document.id)
        .filter(Task.status == TaskStatus.COMPLETED.value)
    )

    # Apply date filter if provided
    if completed_since is not None:
        query = query.filter(Task.completion >= completed_since)

    query = query.order_by(Task.completion.desc())

    # Get total count
    total_count = query.count()

    # Get paginated results
    results = query.offset(offset).limit(limit).all()

    # Convert to response models
    task_responses = [create_task_response(task, doc) for task, doc in results]

    # Calculate pagination
    has_more = (offset + limit) < total_count
    next_offset = offset + limit if has_more else None

    _msg = "get_completed_tasks returning"
    log.debug(_msg)

    return TaskListResponse(
        results=task_responses,
        total_count=total_count,
        has_more=has_more,
        next_offset=next_offset,
    )
