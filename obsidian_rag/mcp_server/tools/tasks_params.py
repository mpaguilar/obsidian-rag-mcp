"""Parameter dataclasses for task query tools."""

from dataclasses import dataclass
from datetime import date
from typing import Literal


@dataclass
class GetTasksFilterParams:
    """Parameters for the generic get_tasks tool.

    All filters are optional and combined with AND logic by default.
    Use date_match_mode="any" for OR logic across date conditions.
    Date comparisons are inclusive (>= for after, <= for before).

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

    Attributes:
        status: List of statuses to filter by.
            Valid values: "not_completed", "completed", "in_progress", "cancelled".
            Multiple values use OR logic (task matches any).
        due_after: Filter tasks due on or after this date (inclusive).
        due_before: Filter tasks due on or before this date (inclusive).
        scheduled_after: Filter tasks scheduled on or after this date (inclusive).
        scheduled_before: Filter tasks scheduled on or before this date (inclusive).
        completion_after: Filter tasks completed on or after this date (inclusive).
        completion_before: Filter tasks completed on or before this date (inclusive).
        tags: List of tags that tasks must have (all tags required, AND logic).
        priority: List of priorities to filter by.
            Valid values: "highest", "high", "normal", "low", "lowest".
            Multiple values use OR logic (task matches any).
        date_match_mode: How to combine date filters - "all" for AND logic (default),
            "any" for OR logic across all date conditions.
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
    date_match_mode: Literal["all", "any"] = "all"
    limit: int = 20
    offset: int = 0
