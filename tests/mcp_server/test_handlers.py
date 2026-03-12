"""Tests for _get_tasks_handler function."""

from unittest.mock import MagicMock, patch

import pytest

from obsidian_rag.mcp_server.handlers import TaskDateFilterStrings, _get_tasks_handler


class TestGetTasksHandler:
    """Tests for _get_tasks_handler function."""

    @patch("obsidian_rag.mcp_server.handlers.get_tasks_tool")
    @patch("obsidian_rag.mcp_server.handlers.parse_iso_date")
    def test_handler_parses_dates(self, mock_parse_date, mock_get_tasks):
        """Test that handler parses date parameters."""
        from datetime import date

        mock_db_manager = MagicMock()
        mock_session = MagicMock()
        mock_db_manager.get_session.return_value.__enter__ = MagicMock(
            return_value=mock_session
        )
        mock_db_manager.get_session.return_value.__exit__ = MagicMock(
            return_value=False
        )

        mock_parse_date.side_effect = [
            date(2026, 1, 1),  # due_after
            date(2026, 12, 31),  # due_before
            None,  # scheduled_after
            None,  # scheduled_before
            None,  # completion_after
            None,  # completion_before
        ]

        mock_get_tasks.return_value = MagicMock(model_dump=lambda: {"results": []})

        date_filters = TaskDateFilterStrings(
            due_after="2026-01-01",
            due_before="2026-12-31",
        )

        _get_tasks_handler(
            db_manager=mock_db_manager,
            date_filters=date_filters,
        )

        assert mock_parse_date.call_count == 6
        mock_get_tasks.assert_called_once()

    @patch("obsidian_rag.mcp_server.handlers.get_tasks_tool")
    def test_handler_with_no_dates(self, mock_get_tasks):
        """Test handler with no date parameters."""
        mock_db_manager = MagicMock()
        mock_session = MagicMock()
        mock_db_manager.get_session.return_value.__enter__ = MagicMock(
            return_value=mock_session
        )
        mock_db_manager.get_session.return_value.__exit__ = MagicMock(
            return_value=False
        )

        mock_get_tasks.return_value = MagicMock(model_dump=lambda: {"results": []})

        _get_tasks_handler(
            db_manager=mock_db_manager,
            status=["not_completed"],
            limit=10,
        )

        mock_get_tasks.assert_called_once()
        call_args = mock_get_tasks.call_args
        assert call_args.kwargs["filters"].status == ["not_completed"]
        assert call_args.kwargs["filters"].limit == 10
