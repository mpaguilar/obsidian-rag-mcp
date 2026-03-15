"""Tests for MCP task tools PostgreSQL path.

This module tests PostgreSQL-specific code paths in tasks.py.
"""

from unittest.mock import MagicMock, patch

import pytest

from obsidian_rag.mcp_server.models import TaskResponse
from obsidian_rag.mcp_server.tools.tasks_params import GetTasksFilterParams


class TestGetTasksPostgresql:
    """Tests for get_tasks with PostgreSQL dialect."""

    def test_get_tasks_postgresql_with_tag_filter(self):
        """Test get_tasks with PostgreSQL dialect and tag filtering."""
        from obsidian_rag.mcp_server.tools.tasks import get_tasks

        # Create a mock session with PostgreSQL dialect
        mock_session = MagicMock()
        mock_bind = MagicMock()
        mock_bind.dialect.name = "postgresql"
        mock_session.bind = mock_bind

        # Mock the query chain - need to include join() in the chain
        mock_query = MagicMock()
        mock_query.join.return_value = mock_query
        mock_query.filter.return_value = mock_query
        mock_query.order_by.return_value = mock_query
        mock_query.count.return_value = 0
        mock_query.offset.return_value.limit.return_value.all.return_value = []

        mock_session.query.return_value = mock_query

        # Create filters with tags
        filters = GetTasksFilterParams(
            tags=["work", "urgent"],
            limit=20,
            offset=0,
        )

        result = get_tasks(mock_session, filters)

        assert result.total_count == 0
        assert len(result.results) == 0
        # Verify that filter was called
        assert mock_query.filter.call_count >= 1

    def test_get_tasks_postgresql_no_tags(self):
        """Test get_tasks with PostgreSQL dialect but no tag filters."""
        from obsidian_rag.mcp_server.tools.tasks import get_tasks

        # Create a mock session with PostgreSQL dialect
        mock_session = MagicMock()
        mock_bind = MagicMock()
        mock_bind.dialect.name = "postgresql"
        mock_session.bind = mock_bind

        # Mock the query chain - need to include join() in the chain
        mock_query = MagicMock()
        mock_query.join.return_value = mock_query
        mock_query.filter.return_value = mock_query
        mock_query.order_by.return_value = mock_query
        mock_query.count.return_value = 0
        mock_query.offset.return_value.limit.return_value.all.return_value = []

        mock_session.query.return_value = mock_query

        # Create filters without tags
        filters = GetTasksFilterParams(
            status=["not_completed"],
            limit=20,
            offset=0,
        )

        result = get_tasks(mock_session, filters)

        assert result.total_count == 0
        assert len(result.results) == 0

    def test_get_tasks_postgresql_multiple_tags(self):
        """Test get_tasks with PostgreSQL and multiple tags (AND logic)."""
        from obsidian_rag.mcp_server.tools.tasks import get_tasks

        # Create a mock session with PostgreSQL dialect
        mock_session = MagicMock()
        mock_bind = MagicMock()
        mock_bind.dialect.name = "postgresql"
        mock_session.bind = mock_bind

        # Mock the query chain - need to include join() in the chain
        mock_query = MagicMock()
        mock_query.join.return_value = mock_query
        mock_query.filter.return_value = mock_query
        mock_query.order_by.return_value = mock_query
        mock_query.count.return_value = 1

        # Mock task and document
        mock_task = MagicMock()
        mock_task.id = "task-1"
        mock_task.status = "not_completed"
        mock_task.tags = ["work", "urgent"]
        mock_task.priority = "high"

        mock_doc = MagicMock()
        mock_doc.id = "doc-1"
        mock_doc.file_path = "test.md"
        mock_doc.file_name = "test.md"
        mock_doc.tags = ["personal"]

        mock_query.offset.return_value.limit.return_value.all.return_value = [
            (mock_task, mock_doc)
        ]

        mock_session.query.return_value = mock_query

        # Create filters with multiple tags
        filters = GetTasksFilterParams(
            tags=["work", "urgent"],
            limit=20,
            offset=0,
        )

        with patch(
            "obsidian_rag.mcp_server.tools.tasks.create_task_response"
        ) as mock_create_response:
            # Create a proper TaskResponse mock that Pydantic will accept
            import uuid
            from datetime import date

            mock_response = TaskResponse(
                id=uuid.uuid4(),
                raw_text="Test task",
                status="not_completed",
                description="Test description",
                due=date(2026, 3, 11),
                priority="high",
                tags=["work", "urgent"],
                document_path="test.md",
                document_name="test.md",
            )
            mock_create_response.return_value = mock_response
            result = get_tasks(mock_session, filters)

        assert result.total_count == 1
        # Filter should be called multiple times (once per tag + other filters)
        assert mock_query.filter.call_count >= 2


class TestApplyTagFiltersPostgresql:
    """Tests for _apply_tag_filters with PostgreSQL dialect."""

    def test_apply_tag_filters_postgresql_single_tag(self):
        """Test _apply_tag_filters with single tag on PostgreSQL."""
        from obsidian_rag.mcp_server.tools.tasks import _apply_tag_filters

        mock_query = MagicMock()
        mock_query.filter.return_value = mock_query

        tags = ["work"]
        result = _apply_tag_filters(mock_query, tags)

        # Should call filter once for the tag
        assert mock_query.filter.call_count == 1
        assert result == mock_query

    def test_apply_tag_filters_postgresql_multiple_tags(self):
        """Test _apply_tag_filters with multiple tags on PostgreSQL."""
        from obsidian_rag.mcp_server.tools.tasks import _apply_tag_filters

        mock_query = MagicMock()
        mock_query.filter.return_value = mock_query

        tags = ["work", "urgent", "personal"]
        result = _apply_tag_filters(mock_query, tags)

        # Should call filter once per tag
        assert mock_query.filter.call_count == 3
        assert result == mock_query

    def test_apply_tag_filters_postgresql_empty_tags(self):
        """Test _apply_tag_filters with empty tags list on PostgreSQL."""
        from obsidian_rag.mcp_server.tools.tasks import _apply_tag_filters

        mock_query = MagicMock()

        tags: list[str] = []
        result = _apply_tag_filters(mock_query, tags)

        # Should not call filter
        mock_query.filter.assert_not_called()
        assert result == mock_query

    def test_apply_tag_filters_postgresql_none_tags(self):
        """Test _apply_tag_filters with None tags on PostgreSQL."""
        from obsidian_rag.mcp_server.tools.tasks import _apply_tag_filters

        mock_query = MagicMock()

        tags = None
        result = _apply_tag_filters(mock_query, tags)

        # Should not call filter
        mock_query.filter.assert_not_called()
        assert result == mock_query
