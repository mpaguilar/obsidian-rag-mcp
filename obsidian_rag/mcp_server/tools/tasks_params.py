"""Parameter dataclasses for task query tools."""

from dataclasses import dataclass
from datetime import date
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from obsidian_rag.mcp_server.handlers import TagFilterStrings, TaskDateFilterStrings


@dataclass
class GetTasksFilterParams:
    """Parameters for the generic get_tasks tool.

    All filters are optional and combined with AND logic by default.
    Use date_match_mode="any" for OR logic across date conditions.
    Use tag_match_mode="any" for OR logic across include_tags.
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
        - Legacy tags parameter: AND logic (task must have ALL tags)
        - include_tags with tag_match_mode="all" (default): AND logic
        - include_tags with tag_match_mode="any": OR logic
        - exclude_tags: OR logic (task excluded if it has ANY excluded tag)
        - Date filters: Configurable via date_match_mode
            - "all" (default): AND logic across all date conditions
            - "any": OR logic across all date conditions
        - Different filter types (status, tags, priority, dates): AND logic

    Tag Filtering Examples:
        - include_tags=["work", "urgent"], tag_match_mode="all": Task must have BOTH
        - include_tags=["work", "personal"], tag_match_mode="any": Task has EITHER
        - exclude_tags=["blocked"]: Task must NOT have "blocked" tag
        - include_tags=["work"], exclude_tags=["blocked"]: Task has "work" but NOT "blocked"

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
        tags: Legacy parameter - list of tags that tasks must have (all required, AND logic).
            Deprecated: Use include_tags instead.
        include_tags: List of tags that tasks must have.
            Use tag_match_mode to control AND vs OR logic.
        exclude_tags: List of tags that tasks must NOT have (OR logic - any match excludes).
        tag_match_mode: How to combine include_tags - "all" for AND logic (default),
            "any" for OR logic (task matches if it has ANY of the include_tags).
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
    tags: list[str] | None = None  # Legacy parameter
    include_tags: list[str] | None = None
    exclude_tags: list[str] | None = None
    tag_match_mode: Literal["all", "any"] = "all"
    priority: list[str] | None = None
    date_match_mode: Literal["all", "any"] = "all"
    limit: int = 20
    offset: int = 0


@dataclass
class GetTasksToolParams:
    """Complete parameters for get_tasks tool function.

    This dataclass bundles all parameters for the get_tasks tool
    to comply with the 5 argument limit per function (PLR0913).

    Attributes:
        db_manager: Database manager for session management.
        filters: All filter parameters bundled into GetTasksFilterParams.

    """

    db_manager: object
    filters: GetTasksFilterParams


@dataclass
class GetTasksRequest:
    """Request parameters for get_tasks handler.

    This dataclass bundles all filter parameters (excluding db_manager)
    to comply with the 5 argument limit per function (PLR0913).

    Attributes:
        status: List of statuses to filter by.
        tag_filters: Tag filter parameters with include/exclude lists and match mode.
        date_filters: Date filter parameters with ISO date strings.
        tags: Legacy parameter - list of tags to filter by (all required).
        priority: List of priorities to filter by.
        limit: Maximum number of results.
        offset: Number of results to skip.

    """

    status: list[str] | None = None
    tag_filters: "TagFilterStrings | None" = None
    date_filters: "TaskDateFilterStrings | None" = None
    tags: list[str] | None = None
    priority: list[str] | None = None
    limit: int = 20
    offset: int = 0
