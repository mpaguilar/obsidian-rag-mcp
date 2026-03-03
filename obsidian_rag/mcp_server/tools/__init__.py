"""MCP tools package."""

from obsidian_rag.mcp_server.tools.tasks import (
    get_completed_tasks,
    get_incomplete_tasks,
    get_tasks_by_tag,
    get_tasks_due_this_week,
)

__all__ = [
    "get_completed_tasks",
    "get_incomplete_tasks",
    "get_tasks_by_tag",
    "get_tasks_due_this_week",
]
