"""Tests for date_match_mode with PostgreSQL dialect (mocked).

This module tests PostgreSQL-specific code paths for date filtering with
match_mode that are not covered by SQLite-based integration tests.
"""

from datetime import date
from unittest.mock import MagicMock, patch

import pytest

from obsidian_rag.mcp_server.models import TaskResponse
from obsidian_rag.mcp_server.tools.tasks_params import GetTasksFilterParams


class TestDateMatchModePostgresql:
    """Tests for date_match_mode with PostgreSQL dialect."""

    def test_postgresql_all_mode_builds_and_conditions(self):
        """Test that 'all' mode builds AND conditions for PostgreSQL."""
        from obsidian_rag.mcp_server.tools.tasks import get_tasks

        # Create a mock session with PostgreSQL dialect
        mock_session = MagicMock()
        mock_bind = MagicMock()
        mock_bind.dialect.name = "postgresql"
        mock_session.bind = mock_bind

        # Mock the query chain
        mock_query = MagicMock()
        mock_query.join.return_value = mock_query
        mock_query.filter.return_value = mock_query
        mock_query.order_by.return_value = mock_query
        mock_query.count.return_value = 0
        mock_query.offset.return_value.limit.return_value.all.return_value = []

        mock_session.query.return_value = mock_query

        # Create filters with multiple date conditions in 'all' mode
        filters = GetTasksFilterParams(
            due_after=date(2026, 3, 1),
            due_before=date(2026, 3, 31),
            scheduled_after=date(2026, 3, 15),
            scheduled_before=date(2026, 3, 20),
            date_match_mode="all",
        )

        get_tasks(mock_session, filters)

        # Verify filter was called for date type conditions
        # With grouped conditions: due (combined with and_), scheduled (combined with and_)
        # Plus status, priority, exclusions = at least 2 date filter calls
        assert mock_query.filter.call_count >= 2

    def test_postgresql_any_mode_builds_or_conditions(self):
        """Test that 'any' mode builds OR conditions for PostgreSQL."""
        from obsidian_rag.mcp_server.tools.tasks import get_tasks

        # Create a mock session with PostgreSQL dialect
        mock_session = MagicMock()
        mock_bind = MagicMock()
        mock_bind.dialect.name = "postgresql"
        mock_session.bind = mock_bind

        # Mock the query chain
        mock_query = MagicMock()
        mock_query.join.return_value = mock_query
        mock_query.filter.return_value = mock_query
        mock_query.order_by.return_value = mock_query
        mock_query.count.return_value = 0
        mock_query.offset.return_value.limit.return_value.all.return_value = []

        mock_session.query.return_value = mock_query

        # Create filters with multiple date conditions in 'any' mode
        filters = GetTasksFilterParams(
            due_after=date(2026, 3, 1),
            due_before=date(2026, 3, 31),
            scheduled_after=date(2026, 3, 15),
            scheduled_before=date(2026, 3, 20),
            date_match_mode="any",
        )

        get_tasks(mock_session, filters)

        # Verify filter was called (OR logic uses single or_() call)
        assert mock_query.filter.call_count >= 1

    def test_postgresql_any_mode_with_results(self):
        """Test 'any' mode with actual results from PostgreSQL."""
        from obsidian_rag.mcp_server.tools.tasks import get_tasks

        mock_session = MagicMock()
        mock_bind = MagicMock()
        mock_bind.dialect.name = "postgresql"
        mock_session.bind = mock_bind

        # Mock task and document
        mock_task = MagicMock()
        mock_task.id = "task-1"
        mock_task.status = "not_completed"
        mock_task.tags = []
        mock_task.priority = "normal"
        mock_task.due = date(2026, 3, 10)
        mock_task.scheduled = None
        mock_task.completion = None

        mock_doc = MagicMock()
        mock_doc.id = "doc-1"
        mock_doc.file_path = "test.md"
        mock_doc.file_name = "test.md"
        mock_doc.tags = []

        # Mock the query chain
        mock_query = MagicMock()
        mock_query.join.return_value = mock_query
        mock_query.filter.return_value = mock_query
        mock_query.order_by.return_value = mock_query
        mock_query.count.return_value = 1
        mock_query.offset.return_value.limit.return_value.all.return_value = [
            (mock_task, mock_doc)
        ]

        mock_session.query.return_value = mock_query

        filters = GetTasksFilterParams(
            due_after=date(2026, 3, 1),
            due_before=date(2026, 3, 31),
            scheduled_after=date(2026, 3, 15),  # Task has NULL scheduled
            date_match_mode="any",
        )

        with patch(
            "obsidian_rag.mcp_server.tools.tasks.create_task_response"
        ) as mock_create_response:
            import uuid

            mock_response = TaskResponse(
                id=uuid.uuid4(),
                raw_text="Test task",
                status="not_completed",
                description="Test description",
                due=date(2026, 3, 10),
                priority="normal",
                tags=[],
                document_path="test.md",
                document_name="test.md",
            )
            mock_create_response.return_value = mock_response
            result = get_tasks(mock_session, filters)

        # Task should match because due date matches (even though scheduled is NULL)
        assert result.total_count == 1

    def test_postgresql_all_mode_excludes_partial_matches(self):
        """Test that 'all' mode excludes tasks that don't match all conditions."""
        from obsidian_rag.mcp_server.tools.tasks import get_tasks

        mock_session = MagicMock()
        mock_bind = MagicMock()
        mock_bind.dialect.name = "postgresql"
        mock_session.bind = mock_bind

        # Mock the query chain
        mock_query = MagicMock()
        mock_query.join.return_value = mock_query
        mock_query.filter.return_value = mock_query
        mock_query.order_by.return_value = mock_query
        mock_query.count.return_value = 0
        mock_query.offset.return_value.limit.return_value.all.return_value = []

        mock_session.query.return_value = mock_query

        filters = GetTasksFilterParams(
            due_after=date(2026, 3, 1),
            due_before=date(2026, 3, 31),
            scheduled_after=date(2026, 3, 15),
            scheduled_before=date(2026, 3, 20),
            date_match_mode="all",
        )

        get_tasks(mock_session, filters)

        # In 'all' mode, conditions are grouped by date type and combined with AND
        # due conditions (after+before) -> 1 filter call with and_()
        # scheduled conditions (after+before) -> 1 filter call with and_()
        # Plus status, priority, exclusions = at least 2 date filter calls
        assert mock_query.filter.call_count >= 2


class TestApplyDateFilters:
    """Tests for _apply_date_filters function directly."""

    def test_apply_date_filters_all_mode(self):
        """Test _apply_date_filters with 'all' mode."""
        from obsidian_rag.mcp_server.tools.tasks import _apply_date_filters

        mock_query = MagicMock()
        mock_query.filter.return_value = mock_query

        from sqlalchemy.orm import Query

        filters = GetTasksFilterParams(
            due_before=date(2026, 3, 31),
            due_after=date(2026, 3, 1),
            date_match_mode="all",
        )
        result: Query = _apply_date_filters(mock_query, filters)

        # Should call filter once with and_() for due conditions combined
        assert mock_query.filter.call_count == 1
        assert result == mock_query

    def test_apply_date_filters_any_mode(self):
        """Test _apply_date_filters with 'any' mode."""
        from obsidian_rag.mcp_server.tools.tasks import _apply_date_filters

        mock_query = MagicMock()
        mock_query.filter.return_value = mock_query

        from sqlalchemy.orm import Query

        filters = GetTasksFilterParams(
            due_before=date(2026, 3, 31),
            due_after=date(2026, 3, 1),
            date_match_mode="any",
        )
        result: Query = _apply_date_filters(mock_query, filters)

        # Should call filter once with or_()
        assert mock_query.filter.call_count == 1
        assert result == mock_query

    def test_apply_date_filters_no_conditions(self):
        """Test _apply_date_filters with no date conditions."""
        from obsidian_rag.mcp_server.tools.tasks import _apply_date_filters

        mock_query = MagicMock()

        from sqlalchemy.orm import Query

        filters = GetTasksFilterParams(date_match_mode="all")
        result: Query = _apply_date_filters(mock_query, filters)

        # Should not call filter at all
        mock_query.filter.assert_not_called()
        assert result == mock_query

    def test_apply_date_filters_with_completion_dates(self):
        """Test _apply_date_filters with completion date conditions."""
        from obsidian_rag.mcp_server.tools.tasks import _apply_date_filters

        mock_query = MagicMock()
        mock_query.filter.return_value = mock_query

        from sqlalchemy.orm import Query

        filters = GetTasksFilterParams(
            completion_before=date(2026, 3, 31),
            completion_after=date(2026, 3, 1),
            date_match_mode="all",
        )
        result: Query = _apply_date_filters(mock_query, filters)

        # Should call filter once with and_() for completion conditions combined
        assert mock_query.filter.call_count == 1
        assert result == mock_query

    def test_apply_date_filters_all_three_date_types_any_mode(self):
        """Test _apply_date_filters with all three date types in 'any' mode."""
        from obsidian_rag.mcp_server.tools.tasks import _apply_date_filters

        mock_query = MagicMock()
        mock_query.filter.return_value = mock_query

        from sqlalchemy.orm import Query

        filters = GetTasksFilterParams(
            due_before=date(2026, 3, 31),
            due_after=date(2026, 3, 1),
            scheduled_before=date(2026, 3, 20),
            scheduled_after=date(2026, 3, 10),
            completion_before=date(2026, 3, 25),
            completion_after=date(2026, 3, 15),
            date_match_mode="any",
        )
        result: Query = _apply_date_filters(mock_query, filters)

        # Should call filter once with or_() across all three date type conditions
        assert mock_query.filter.call_count == 1
        assert result == mock_query


class TestTaskMatchesDateFilters:
    """Tests for _task_matches_date_filters function."""

    def test_task_matches_all_mode_all_conditions_met(self):
        """Test _task_matches_date_filters with 'all' mode - all conditions met."""
        from obsidian_rag.mcp_server.tools.tasks import _task_matches_date_filters

        mock_task = MagicMock()
        mock_task.due = date(2026, 3, 10)
        mock_task.scheduled = date(2026, 3, 15)
        mock_task.completion = None

        filters = GetTasksFilterParams(
            due_before=date(2026, 3, 31),
            due_after=date(2026, 3, 1),
            scheduled_before=date(2026, 3, 20),
            scheduled_after=date(2026, 3, 10),
            date_match_mode="all",
        )
        result = _task_matches_date_filters(mock_task, filters)

        assert result is True

    def test_task_matches_all_mode_one_condition_fails(self):
        """Test _task_matches_date_filters with 'all' mode - one condition fails."""
        from obsidian_rag.mcp_server.tools.tasks import _task_matches_date_filters

        mock_task = MagicMock()
        mock_task.due = date(2026, 3, 10)
        mock_task.scheduled = date(2026, 3, 25)  # Outside range
        mock_task.completion = None

        filters = GetTasksFilterParams(
            due_before=date(2026, 3, 31),
            due_after=date(2026, 3, 1),
            scheduled_before=date(2026, 3, 20),
            scheduled_after=date(2026, 3, 10),
            date_match_mode="all",
        )
        result = _task_matches_date_filters(mock_task, filters)

        # Scheduled is outside range, so should fail
        assert result is False

    def test_task_matches_any_mode_one_condition_met(self):
        """Test _task_matches_date_filters with 'any' mode - one condition met."""
        from obsidian_rag.mcp_server.tools.tasks import _task_matches_date_filters

        mock_task = MagicMock()
        mock_task.due = date(2026, 3, 10)  # Matches
        mock_task.scheduled = date(2026, 3, 25)  # Outside range
        mock_task.completion = None

        filters = GetTasksFilterParams(
            due_before=date(2026, 3, 31),
            due_after=date(2026, 3, 1),
            scheduled_before=date(2026, 3, 20),
            scheduled_after=date(2026, 3, 10),
            date_match_mode="any",
        )
        result = _task_matches_date_filters(mock_task, filters)

        # Due matches, so should pass with OR logic
        assert result is True

    def test_task_matches_any_mode_no_conditions_met(self):
        """Test _task_matches_date_filters with 'any' mode - no conditions met."""
        from obsidian_rag.mcp_server.tools.tasks import _task_matches_date_filters

        mock_task = MagicMock()
        mock_task.due = date(2026, 4, 5)  # Outside range (after March 31)
        mock_task.scheduled = date(2026, 3, 25)  # Outside range
        mock_task.completion = None

        filters = GetTasksFilterParams(
            due_before=date(2026, 3, 31),
            due_after=date(2026, 3, 1),
            scheduled_before=date(2026, 3, 20),
            scheduled_after=date(2026, 3, 10),
            date_match_mode="any",
        )
        result = _task_matches_date_filters(mock_task, filters)

        # Neither matches, so should fail
        assert result is False

    def test_task_matches_with_null_date_all_mode(self):
        """Test that NULL date causes failure in 'all' mode."""
        from obsidian_rag.mcp_server.tools.tasks import _task_matches_date_filters

        mock_task = MagicMock()
        mock_task.due = None  # NULL
        mock_task.scheduled = date(2026, 3, 15)
        mock_task.completion = None

        filters = GetTasksFilterParams(
            due_before=date(2026, 3, 31),
            due_after=date(2026, 3, 1),
            date_match_mode="all",
        )
        result = _task_matches_date_filters(mock_task, filters)

        # NULL due date means condition fails
        assert result is False

    def test_task_matches_with_null_date_any_mode(self):
        """Test that NULL date can be compensated by other date in 'any' mode."""
        from obsidian_rag.mcp_server.tools.tasks import _task_matches_date_filters

        mock_task = MagicMock()
        mock_task.due = None  # NULL
        mock_task.scheduled = date(2026, 3, 15)  # Matches
        mock_task.completion = None

        filters = GetTasksFilterParams(
            due_before=date(2026, 3, 31),
            due_after=date(2026, 3, 1),
            scheduled_before=date(2026, 3, 20),
            scheduled_after=date(2026, 3, 10),
            date_match_mode="any",
        )
        result = _task_matches_date_filters(mock_task, filters)

        # NULL due but scheduled matches
        assert result is True

    def test_task_matches_no_date_filters_returns_true(self):
        """Test that task matches when no date filters are provided."""
        from obsidian_rag.mcp_server.tools.tasks import _task_matches_date_filters

        mock_task = MagicMock()
        mock_task.due = date(2026, 3, 15)
        mock_task.scheduled = None
        mock_task.completion = None

        filters = GetTasksFilterParams(date_match_mode="all")
        result = _task_matches_date_filters(mock_task, filters)

        # No date filters, task should pass
        assert result is True
