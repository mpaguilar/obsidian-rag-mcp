"""Tests for inline_filters propagation through get_tasks_tool()."""

from unittest.mock import MagicMock, patch

from obsidian_rag.mcp_server.models import PropertyFilter
from obsidian_rag.mcp_server.tool_definitions import get_tasks_tool
from obsidian_rag.mcp_server.tools.tasks_params import GetTasksRequest


def test_get_tasks_tool_propagates_inline_filters():
    """Test that get_tasks_tool passes inline_filters through to handler."""
    mock_db_manager = MagicMock()
    inline_filters = [
        PropertyFilter(path="priority", operator="equals", value="high"),
        PropertyFilter(path="status", operator="in", value=["urgent"]),
    ]
    request = GetTasksRequest(
        status=["not_completed"],
        inline_filters=inline_filters,
    )

    with patch(
        "obsidian_rag.mcp_server.tool_definitions._get_tasks_handler"
    ) as mock_handler:
        mock_handler.return_value = {
            "results": [],
            "total_count": 0,
            "has_more": False,
            "next_offset": None,
        }

        get_tasks_tool(db_manager=mock_db_manager, request=request)

        mock_handler.assert_called_once()
        passed_request = mock_handler.call_args.args[1]
        assert passed_request.inline_filters is not None
        assert len(passed_request.inline_filters) == 2
        assert passed_request.inline_filters[0].path == "priority"
        assert passed_request.inline_filters[1].path == "status"


def test_get_tasks_tool_inline_filters_none():
    """Test that get_tasks_tool passes None inline_filters correctly."""
    mock_db_manager = MagicMock()
    request = GetTasksRequest(
        status=["completed"],
        inline_filters=None,
    )

    with patch(
        "obsidian_rag.mcp_server.tool_definitions._get_tasks_handler"
    ) as mock_handler:
        mock_handler.return_value = {
            "results": [],
            "total_count": 0,
            "has_more": False,
            "next_offset": None,
        }

        get_tasks_tool(db_manager=mock_db_manager, request=request)

        mock_handler.assert_called_once()
        passed_request = mock_handler.call_args.args[1]
        assert passed_request.inline_filters is None
