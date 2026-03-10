"""Tests for tasks.py PostgreSQL-specific branch.

Tests for get_tasks_by_tag PostgreSQL path that uses array_to_string.
"""

import uuid
from datetime import date
from unittest.mock import MagicMock, patch

import pytest

from obsidian_rag.database.models import TaskStatus, TaskPriority


def create_mock_session():
    """Create a mock session with PostgreSQL dialect."""
    mock_session = MagicMock()
    mock_bind = MagicMock()
    mock_bind.dialect.name = "postgresql"
    mock_session.bind = mock_bind
    return mock_session


def create_mock_task():
    """Create a mock task for testing."""
    mock_task = MagicMock()
    mock_task.id = uuid.uuid4()
    mock_task.description = "Test task"
    mock_task.status = TaskStatus.NOT_COMPLETED.value
    mock_task.priority = TaskPriority.NORMAL.value
    mock_task.tags = ["work", "urgent"]
    mock_task.due = date.today()
    mock_task.scheduled = None
    mock_task.repeat = None
    mock_task.completion = None
    mock_task.raw_text = "- [ ] Test task"
    mock_task.line_number = 1
    return mock_task


def create_mock_document():
    """Create a mock document for testing."""
    mock_doc = MagicMock()
    mock_doc.id = uuid.uuid4()
    mock_doc.file_path = "/test/doc.md"
    mock_doc.file_name = "doc.md"
    mock_doc.tags = ["work"]
    return mock_doc


class TestGetTasksByTagPostgresql:
    """Tests for get_tasks_by_tag PostgreSQL branch (lines 212-227)."""

    def test_get_tasks_by_tag_postgresql_branch(self):
        """Test get_tasks_by_tag with PostgreSQL dialect (lines 212-227)."""
        from obsidian_rag.mcp_server.tools.tasks import get_tasks_by_tag

        mock_session = create_mock_session()
        mock_task = create_mock_task()
        mock_doc = create_mock_document()

        # Create a proper mock that supports count() returning int
        class MockQuery:
            def __init__(self):
                self._count = 1
                self._results = [(mock_task, mock_doc)]

            def join(self, *args, **kwargs):
                return self

            def filter(self, *args, **kwargs):
                return self

            def order_by(self, *args, **kwargs):
                return self

            def count(self):
                return self._count

            def offset(self, n):
                return self

            def limit(self, n):
                return self

            def all(self):
                return self._results

        mock_query = MockQuery()
        mock_session.query.return_value = mock_query

        result = get_tasks_by_tag(mock_session, tag="work", limit=20, offset=0)

        assert result.total_count == 1
        assert len(result.results) == 1
        assert result.results[0].description == "Test task"

    def test_get_tasks_by_tag_postgresql_multiple_results(self):
        """Test get_tasks_by_tag PostgreSQL with multiple results."""
        from obsidian_rag.mcp_server.tools.tasks import get_tasks_by_tag

        mock_session = create_mock_session()

        mock_task1 = create_mock_task()
        mock_task1.description = "Task 1"

        mock_task2 = create_mock_task()
        mock_task2.description = "Task 2"

        mock_doc = create_mock_document()

        class MockQuery:
            def __init__(self):
                self._count = 2
                self._results = [(mock_task1, mock_doc), (mock_task2, mock_doc)]

            def join(self, *args, **kwargs):
                return self

            def filter(self, *args, **kwargs):
                return self

            def order_by(self, *args, **kwargs):
                return self

            def count(self):
                return self._count

            def offset(self, n):
                return self

            def limit(self, n):
                return self

            def all(self):
                return self._results

        mock_query = MockQuery()
        mock_session.query.return_value = mock_query

        result = get_tasks_by_tag(mock_session, tag="work", limit=20, offset=0)

        assert result.total_count == 2
        assert len(result.results) == 2

    def test_get_tasks_by_tag_postgresql_empty_results(self):
        """Test get_tasks_by_tag PostgreSQL with empty results."""
        from obsidian_rag.mcp_server.tools.tasks import get_tasks_by_tag

        mock_session = create_mock_session()

        class MockQuery:
            def __init__(self):
                self._count = 0
                self._results = []

            def join(self, *args, **kwargs):
                return self

            def filter(self, *args, **kwargs):
                return self

            def order_by(self, *args, **kwargs):
                return self

            def count(self):
                return self._count

            def offset(self, n):
                return self

            def limit(self, n):
                return self

            def all(self):
                return self._results

        mock_query = MockQuery()
        mock_session.query.return_value = mock_query

        result = get_tasks_by_tag(mock_session, tag="nonexistent", limit=20, offset=0)

        assert result.total_count == 0
        assert len(result.results) == 0
        assert result.has_more is False

    def test_get_tasks_by_tag_postgresql_pagination(self):
        """Test get_tasks_by_tag PostgreSQL with pagination."""
        from obsidian_rag.mcp_server.tools.tasks import get_tasks_by_tag

        mock_session = create_mock_session()
        mock_task = create_mock_task()
        mock_doc = create_mock_document()

        class MockQuery:
            def __init__(self):
                self._count = 10  # Total 10 tasks
                self._results = [(mock_task, mock_doc)]

            def join(self, *args, **kwargs):
                return self

            def filter(self, *args, **kwargs):
                return self

            def order_by(self, *args, **kwargs):
                return self

            def count(self):
                return self._count

            def offset(self, n):
                return self

            def limit(self, n):
                return self

            def all(self):
                return self._results

        mock_query = MockQuery()
        mock_session.query.return_value = mock_query

        result = get_tasks_by_tag(mock_session, tag="work", limit=5, offset=0)

        assert result.has_more is True
        assert result.next_offset == 5

    def test_get_tasks_by_tag_postgresql_no_match_in_doc_tags(self):
        """Test get_tasks_by_tag PostgreSQL when tag matches task but not doc."""
        from obsidian_rag.mcp_server.tools.tasks import get_tasks_by_tag

        mock_session = create_mock_session()
        mock_task = create_mock_task()
        mock_task.tags = ["urgent"]  # Has urgent tag
        mock_doc = create_mock_document()
        mock_doc.tags = ["other"]  # Doc has different tag

        class MockQuery:
            def __init__(self):
                self._count = 1
                self._results = [(mock_task, mock_doc)]

            def join(self, *args, **kwargs):
                return self

            def filter(self, *args, **kwargs):
                return self

            def order_by(self, *args, **kwargs):
                return self

            def count(self):
                return self._count

            def offset(self, n):
                return self

            def limit(self, n):
                return self

            def all(self):
                return self._results

        mock_query = MockQuery()
        mock_session.query.return_value = mock_query

        # Tag search should still find it via task tags
        result = get_tasks_by_tag(mock_session, tag="urgent", limit=20, offset=0)

        assert result.total_count == 1


class TestGetTasksByTagPostgresqlEdgeCases:
    """Edge case tests for get_tasks_by_tag PostgreSQL branch."""

    def test_get_tasks_by_tag_postgresql_case_insensitive(self):
        """Test get_tasks_by_tag PostgreSQL is case-insensitive."""
        from obsidian_rag.mcp_server.tools.tasks import get_tasks_by_tag

        mock_session = create_mock_session()
        mock_task = create_mock_task()
        mock_doc = create_mock_document()

        class MockQuery:
            def __init__(self):
                self._count = 1
                self._results = [(mock_task, mock_doc)]

            def join(self, *args, **kwargs):
                return self

            def filter(self, *args, **kwargs):
                return self

            def order_by(self, *args, **kwargs):
                return self

            def count(self):
                return self._count

            def offset(self, n):
                return self

            def limit(self, n):
                return self

            def all(self):
                return self._results

        mock_query = MockQuery()
        mock_session.query.return_value = mock_query

        # Search with uppercase should find lowercase tags
        result = get_tasks_by_tag(mock_session, tag="WORK", limit=20, offset=0)

        assert result.total_count == 1
