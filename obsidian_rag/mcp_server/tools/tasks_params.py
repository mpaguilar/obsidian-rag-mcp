"""Parameter dataclasses for task query tools."""

from dataclasses import dataclass
from datetime import date


@dataclass
class GetTasksFilterParams:
    """Parameters for the generic get_tasks tool.

    All filters are optional and combined with AND logic.
    Date comparisons are inclusive (>= for after, <= for before).

    Attributes:
        status: List of statuses to filter by (e.g., ['not_completed', 'in_progress']).
        due_after: Filter tasks due on or after this date.
        due_before: Filter tasks due on or before this date.
        scheduled_after: Filter tasks scheduled on or after this date.
        scheduled_before: Filter tasks scheduled on or before this date.
        completion_after: Filter tasks completed on or after this date.
        completion_before: Filter tasks completed on or before this date.
        tags: List of tags that tasks must have (all tags required).
        priority: List of priorities to filter by.
        include_completed: Whether to include completed tasks (default: True).
        include_cancelled: Whether to include cancelled tasks (default: False).
        limit: Maximum number of results (default: 20, max: 100).
        offset: Number of results to skip (default: 0).

    """

    status: list[str] | None = None
    due_after: date | None = None
    due_before: date | None = None
    scheduled_after: date | None = None
    scheduled_before: date | None = None
    completion_after: date | None = None
    completion_before: date | None = None
    tags: list[str] | None = None
    priority: list[str] | None = None
    include_completed: bool = True
    include_cancelled: bool = False
    limit: int = 20
    offset: int = 0
