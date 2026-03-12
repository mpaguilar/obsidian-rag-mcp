"""Tests for MCP task tools PostgreSQL path.

This module tests PostgreSQL-specific code paths in tasks.py that are not
covered by SQLite-based integration tests.
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
        result = _apply_tag_filters(mock_query, tags, is_postgresql=True)

        # Should call filter once for the tag
        assert mock_query.filter.call_count == 1
        assert result == mock_query

    def test_apply_tag_filters_postgresql_multiple_tags(self):
        """Test _apply_tag_filters with multiple tags on PostgreSQL."""
        from obsidian_rag.mcp_server.tools.tasks import _apply_tag_filters

        mock_query = MagicMock()
        mock_query.filter.return_value = mock_query

        tags = ["work", "urgent", "personal"]
        result = _apply_tag_filters(mock_query, tags, is_postgresql=True)

        # Should call filter once per tag
        assert mock_query.filter.call_count == 3
        assert result == mock_query

    def test_apply_tag_filters_postgresql_empty_tags(self):
        """Test _apply_tag_filters with empty tags list on PostgreSQL."""
        from obsidian_rag.mcp_server.tools.tasks import _apply_tag_filters

        mock_query = MagicMock()

        tags: list[str] = []
        result = _apply_tag_filters(mock_query, tags, is_postgresql=True)

        # Should not call filter
        mock_query.filter.assert_not_called()
        assert result == mock_query

    def test_apply_tag_filters_postgresql_none_tags(self):
        """Test _apply_tag_filters with None tags on PostgreSQL."""
        from obsidian_rag.mcp_server.tools.tasks import _apply_tag_filters

        mock_query = MagicMock()

        tags = None
        result = _apply_tag_filters(mock_query, tags, is_postgresql=True)

        # Should not call filter
        mock_query.filter.assert_not_called()
        assert result == mock_query

    def test_apply_tag_filters_sqlite_skips_sql_filtering(self):
        """Test _apply_tag_filters skips SQL filtering for SQLite."""
        from obsidian_rag.mcp_server.tools.tasks import _apply_tag_filters

        mock_query = MagicMock()

        tags = ["work", "urgent"]
        result = _apply_tag_filters(mock_query, tags, is_postgresql=False)

        # Should not call filter for SQLite
        mock_query.filter.assert_not_called()
        assert result == mock_query


class TestFilterByTagsPython:
    """Tests for _filter_by_tags_python helper function."""

    def test_filter_by_tags_python_single_tag(self):
        """Test filtering by single tag in Python."""
        from obsidian_rag.mcp_server.tools.tasks import _filter_by_tags_python

        # Create mock tasks and documents
        task1 = MagicMock()
        task1.tags = ["work", "urgent"]

        doc1 = MagicMock()
        doc1.tags = ["personal"]

        task2 = MagicMock()
        task2.tags = ["home"]

        doc2 = MagicMock()
        doc2.tags = []

        results = [(task1, doc1), (task2, doc2)]

        # Filter by "work" tag
        filtered = _filter_by_tags_python(results, ["work"])  # type: ignore[arg-type]

        assert len(filtered) == 1
        assert filtered[0][0] == task1

    def test_filter_by_tags_python_multiple_tags_and_logic(self):
        """Test filtering by multiple tags uses AND logic."""
        from obsidian_rag.mcp_server.tools.tasks import _filter_by_tags_python

        # Task with both tags
        task1 = MagicMock()
        task1.tags = ["work", "urgent"]

        doc1 = MagicMock()
        doc1.tags = []

        # Task with only one tag
        task2 = MagicMock()
        task2.tags = ["work"]

        doc2 = MagicMock()
        doc2.tags = []

        results = [(task1, doc1), (task2, doc2)]

        # Filter by both "work" AND "urgent"
        filtered = _filter_by_tags_python(results, ["work", "urgent"])  # type: ignore[arg-type]

        # Only task1 has both tags
        assert len(filtered) == 1
        assert filtered[0][0] == task1

    def test_filter_by_tags_python_document_tag_match(self):
        """Test that document tags are also checked."""
        from obsidian_rag.mcp_server.tools.tasks import _filter_by_tags_python

        # Task with no tags, but document has the tag
        task1 = MagicMock()
        task1.tags = []

        doc1 = MagicMock()
        doc1.tags = ["work"]

        # Task with no matching tags
        task2 = MagicMock()
        task2.tags = []

        doc2 = MagicMock()
        doc2.tags = ["home"]

        results = [(task1, doc1), (task2, doc2)]

        # Filter by "work" tag
        filtered = _filter_by_tags_python(results, ["work"])  # type: ignore[arg-type]

        # Task1 matches because its document has the tag
        assert len(filtered) == 1
        assert filtered[0][0] == task1

    def test_filter_by_tags_python_case_insensitive(self):
        """Test that tag filtering is case-insensitive."""
        from obsidian_rag.mcp_server.tools.tasks import _filter_by_tags_python

        task1 = MagicMock()
        task1.tags = ["WORK", "URGENT"]

        doc1 = MagicMock()
        doc1.tags = []

        results = [(task1, doc1)]

        # Filter by lowercase tag
        filtered = _filter_by_tags_python(results, ["work"])  # type: ignore[arg-type]

        assert len(filtered) == 1

    def test_filter_by_tags_python_no_matches(self):
        """Test filtering when no tasks match."""
        from obsidian_rag.mcp_server.tools.tasks import _filter_by_tags_python

        task1 = MagicMock()
        task1.tags = ["home"]

        doc1 = MagicMock()
        doc1.tags = ["personal"]

        results = [(task1, doc1)]

        # Filter by non-existent tag
        filtered = _filter_by_tags_python(results, ["work"])  # type: ignore[arg-type]

        assert len(filtered) == 0

    def test_filter_by_tags_python_empty_tags_list(self):
        """Test filtering with empty tags list returns all results."""
        from obsidian_rag.mcp_server.tools.tasks import _filter_by_tags_python

        task1 = MagicMock()
        doc1 = MagicMock()

        task2 = MagicMock()
        doc2 = MagicMock()

        results = [(task1, doc1), (task2, doc2)]

        # Filter with empty tags
        filtered = _filter_by_tags_python(results, [])  # type: ignore[arg-type]

        # Should return all results
        assert len(filtered) == 2

    def test_filter_by_tags_python_none_tags(self):
        """Test filtering with None task tags."""
        from obsidian_rag.mcp_server.tools.tasks import _filter_by_tags_python

        task1 = MagicMock()
        task1.tags = None

        doc1 = MagicMock()
        doc1.tags = ["work"]

        results = [(task1, doc1)]

        # Filter by tag that exists in document
        filtered = _filter_by_tags_python(results, ["work"])  # type: ignore[arg-type]

        assert len(filtered) == 1

    def test_filter_by_tags_python_substring_match(self):
        """Test that tag filtering uses substring matching."""
        from obsidian_rag.mcp_server.tools.tasks import _filter_by_tags_python

        task1 = MagicMock()
        task1.tags = ["personal/bills", "personal/expenses"]

        doc1 = MagicMock()
        doc1.tags = []

        results = [(task1, doc1)]

        # Filter by partial tag match
        filtered = _filter_by_tags_python(results, ["personal"])  # type: ignore[arg-type]

        assert len(filtered) == 1


class TestTaskHasTag:
    """Tests for _task_has_tag helper function."""

    def test_task_has_tag_in_task_tags(self):
        """Test finding tag in task tags."""
        from obsidian_rag.mcp_server.tools.tasks import _task_has_tag

        task = MagicMock()
        task.tags = ["work", "urgent"]

        doc = MagicMock()
        doc.tags = []

        assert _task_has_tag(task, doc, "work") is True  # type: ignore[arg-type]
        assert _task_has_tag(task, doc, "urgent") is True  # type: ignore[arg-type]
        assert _task_has_tag(task, doc, "personal") is False  # type: ignore[arg-type]

    def test_task_has_tag_in_document_tags(self):
        """Test finding tag in document tags."""
        from obsidian_rag.mcp_server.tools.tasks import _task_has_tag

        task = MagicMock()
        task.tags = []

        doc = MagicMock()
        doc.tags = ["work", "personal"]

        assert _task_has_tag(task, doc, "work") is True  # type: ignore[arg-type]
        assert _task_has_tag(task, doc, "personal") is True  # type: ignore[arg-type]
        assert _task_has_tag(task, doc, "urgent") is False  # type: ignore[arg-type]

    def test_task_has_tag_case_insensitive(self):
        """Test tag matching is case-insensitive."""
        from obsidian_rag.mcp_server.tools.tasks import _task_has_tag

        task = MagicMock()
        task.tags = ["WORK", "URGENT"]

        doc = MagicMock()
        doc.tags = []

        assert _task_has_tag(task, doc, "work") is True  # type: ignore[arg-type]
        assert _task_has_tag(task, doc, "urgent") is True  # type: ignore[arg-type]

    def test_task_has_tag_none_tags(self):
        """Test with None tags."""
        from obsidian_rag.mcp_server.tools.tasks import _task_has_tag

        task = MagicMock()
        task.tags = None

        doc = MagicMock()
        doc.tags = None

        assert _task_has_tag(task, doc, "work") is False  # type: ignore[arg-type]

    def test_task_has_tag_substring_match(self):
        """Test substring matching in tags."""
        from obsidian_rag.mcp_server.tools.tasks import _task_has_tag

        task = MagicMock()
        task.tags = ["personal/bills", "work/project"]

        doc = MagicMock()
        doc.tags = []

        assert _task_has_tag(task, doc, "personal") is True  # type: ignore[arg-type]
        assert _task_has_tag(task, doc, "bills") is True  # type: ignore[arg-type]
        assert _task_has_tag(task, doc, "project") is True  # type: ignore[arg-type]
