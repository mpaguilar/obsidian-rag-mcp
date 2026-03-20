"""Integration tests for get_tasks MCP tool."""

import uuid
from datetime import date, timedelta
from unittest.mock import MagicMock

import pytest

from obsidian_rag.database.models import Document, Task, TaskPriority, TaskStatus, Vault
from obsidian_rag.mcp_server.handlers import (
    GetTasksRequest,
    TagFilterStrings,
    _get_tasks_handler,
)
from obsidian_rag.mcp_server.tools.tasks import get_tasks
from obsidian_rag.mcp_server.tools.tasks_params import GetTasksFilterParams


@pytest.fixture
def sample_data(db_session):
    """Create sample vault, document, and tasks for testing."""
    from datetime import datetime

    vault = Vault(
        id=uuid.uuid4(),
        name="test_vault",
        container_path="/test",
        host_path="/test",
    )

    doc = Document(
        id=uuid.uuid4(),
        vault_id=vault.id,
        file_path="/test/doc.md",
        file_name="doc.md",
        content="# Test",
        checksum_md5="abc123",
        created_at_fs=datetime.now(),
        modified_at_fs=datetime.now(),
        frontmatter_json={},
        tags=["work", "urgent"],
    )

    today = date.today()

    # Create diverse tasks
    tasks = [
        Task(
            id=uuid.uuid4(),
            document_id=doc.id,
            line_number=1,
            raw_text="- [ ] High priority due soon",
            status=TaskStatus.NOT_COMPLETED.value,
            description="High priority due soon",
            due=today + timedelta(days=2),
            priority=TaskPriority.HIGH.value,
            tags=["work"],
        ),
        Task(
            id=uuid.uuid4(),
            document_id=doc.id,
            line_number=2,
            raw_text="- [ ] Normal priority due later",
            status=TaskStatus.NOT_COMPLETED.value,
            description="Normal priority due later",
            due=today + timedelta(days=10),
            priority=TaskPriority.NORMAL.value,
            tags=["work"],
        ),
        Task(
            id=uuid.uuid4(),
            document_id=doc.id,
            line_number=3,
            raw_text="- [x] Completed yesterday",
            status=TaskStatus.COMPLETED.value,
            description="Completed yesterday",
            completion=today - timedelta(days=1),
            priority=TaskPriority.NORMAL.value,
        ),
        Task(
            id=uuid.uuid4(),
            document_id=doc.id,
            line_number=4,
            raw_text="- [-] Cancelled task",
            status=TaskStatus.CANCELLED.value,
            description="Cancelled task",
            priority=TaskPriority.LOW.value,
        ),
        Task(
            id=uuid.uuid4(),
            document_id=doc.id,
            line_number=5,
            raw_text="- [/] In progress",
            status=TaskStatus.IN_PROGRESS.value,
            description="In progress",
            due=today + timedelta(days=5),
            priority=TaskPriority.HIGH.value,
            tags=["urgent"],
        ),
    ]

    return vault, doc, tasks


def _configure_mock_for_tasks(db_session, tasks, doc, total_count=None):
    """Configure mock session to return tasks with their documents."""
    if total_count is None:
        total_count = len(tasks)

    # Create (Task, Document) tuples as expected by get_tasks
    task_doc_pairs = [(task, doc) for task in tasks]

    # Configure the mock query chain
    query_mock = MagicMock()
    query_mock.all.return_value = task_doc_pairs
    query_mock.count.return_value = total_count
    query_mock.filter.return_value = query_mock
    query_mock.order_by.return_value = query_mock
    query_mock.offset.return_value = query_mock
    query_mock.limit.return_value = query_mock
    query_mock.join.return_value = query_mock

    db_session.query.return_value = query_mock


class TestGetTasksIntegration:
    """Integration tests for get_tasks tool."""

    def test_get_tasks_no_filters(self, db_session, sample_data):
        """Test get_tasks with no filters returns all tasks."""
        vault, doc, tasks = sample_data

        # Use all tasks (no filtering)
        _configure_mock_for_tasks(db_session, tasks, doc, total_count=5)

        filters = GetTasksFilterParams()
        result = get_tasks(db_session, filters)

        # Without filters, we get all 5 tasks
        assert result.total_count == 5
        assert len(result.results) == 5

    def test_get_tasks_status_filter(self, db_session, sample_data):
        """Test filtering by multiple statuses."""
        vault, doc, tasks = sample_data

        # Filter for not_completed and in_progress
        filtered = [
            t
            for t in tasks
            if t.status
            in [TaskStatus.NOT_COMPLETED.value, TaskStatus.IN_PROGRESS.value]
        ]
        _configure_mock_for_tasks(db_session, filtered, doc, total_count=3)

        filters = GetTasksFilterParams(status=["not_completed", "in_progress"])
        result = get_tasks(db_session, filters)

        assert result.total_count == 3  # 2 not_completed + 1 in_progress

    def test_get_tasks_due_date_range(self, db_session, sample_data):
        """Test filtering by due date range."""
        vault, doc, tasks = sample_data
        today = date.today()

        # Tasks due within next 7 days
        filtered = [
            t for t in tasks if t.due and today <= t.due <= today + timedelta(days=7)
        ]
        _configure_mock_for_tasks(db_session, filtered, doc, total_count=len(filtered))

        filters = GetTasksFilterParams(
            due_after=today,
            due_before=today + timedelta(days=7),
        )
        result = get_tasks(db_session, filters)

        # Should find tasks due within next 7 days
        assert result.total_count >= 1

    def test_get_tasks_priority_filter(self, db_session, sample_data):
        """Test filtering by priority."""
        vault, doc, tasks = sample_data

        # High priority incomplete tasks
        filtered = [
            t
            for t in tasks
            if t.priority == TaskPriority.HIGH.value
            and t.status != TaskStatus.COMPLETED.value
        ]
        _configure_mock_for_tasks(db_session, filtered, doc, total_count=len(filtered))

        filters = GetTasksFilterParams(
            priority=["high"],
            status=["not_completed", "in_progress", "cancelled"],
        )
        result = get_tasks(db_session, filters)

        # Should find high priority incomplete tasks
        assert result.total_count >= 1
        for task in result.results:
            assert task.priority == "high"

    def test_get_tasks_exclude_completed(self, db_session, sample_data):
        """Test excluding completed tasks using status filter."""
        vault, doc, tasks = sample_data

        # Filter to get all non-completed tasks
        filtered = [t for t in tasks if t.status != TaskStatus.COMPLETED.value]
        _configure_mock_for_tasks(db_session, filtered, doc, total_count=4)

        filters = GetTasksFilterParams(
            status=["not_completed", "in_progress", "cancelled"],
        )
        result = get_tasks(db_session, filters)

        # Should get 4 tasks: 2 not_completed + 1 in_progress + 1 cancelled
        assert result.total_count == 4
        for task in result.results:
            assert task.status != "completed"

    def test_get_tasks_include_cancelled(self, db_session, sample_data):
        """Test including cancelled tasks using status filter."""
        vault, doc, tasks = sample_data

        # Non-completed tasks including cancelled
        filtered = [t for t in tasks if t.status != TaskStatus.COMPLETED.value]
        _configure_mock_for_tasks(db_session, filtered, doc, total_count=4)

        filters = GetTasksFilterParams(
            status=["not_completed", "in_progress", "cancelled"],
        )
        result = get_tasks(db_session, filters)

        # Should include cancelled
        cancelled_count = sum(1 for t in result.results if t.status == "cancelled")
        assert cancelled_count == 1

    def test_get_tasks_tag_filter(self, db_session, sample_data):
        """Test filtering by tags."""
        vault, doc, tasks = sample_data

        # Tasks with urgent tag
        filtered = [t for t in tasks if t.tags and "urgent" in t.tags]
        _configure_mock_for_tasks(db_session, filtered, doc, total_count=len(filtered))

        filters = GetTasksFilterParams(tags=["urgent"])
        result = get_tasks(db_session, filters)

        # Should find task with urgent tag
        assert result.total_count >= 1

    def test_get_tasks_combined_filters(self, db_session, sample_data):
        """Test combining multiple filters."""
        vault, doc, tasks = sample_data
        today = date.today()

        # Tasks matching all criteria
        filtered = [
            t
            for t in tasks
            if t.status
            in [TaskStatus.NOT_COMPLETED.value, TaskStatus.IN_PROGRESS.value]
            and t.due
            and today <= t.due <= today + timedelta(days=7)
            and t.priority == TaskPriority.HIGH.value
            and t.status != TaskStatus.COMPLETED.value
        ]
        _configure_mock_for_tasks(db_session, filtered, doc, total_count=len(filtered))

        filters = GetTasksFilterParams(
            status=["not_completed", "in_progress"],
            due_after=today,
            due_before=today + timedelta(days=7),
            priority=["high"],
        )
        result = get_tasks(db_session, filters)

        # All returned tasks should match all criteria
        for task in result.results:
            assert task.status in ["not_completed", "in_progress"]
            assert task.priority == "high"

    def test_get_tasks_pagination(self, db_session, sample_data):
        """Test pagination with limit and offset."""
        vault, doc, tasks = sample_data

        # All tasks for pagination
        all_tasks = tasks

        # Get first 2 results
        _configure_mock_for_tasks(db_session, all_tasks[:2], doc, total_count=5)
        filters = GetTasksFilterParams(limit=2, offset=0)
        result1 = get_tasks(db_session, filters)

        assert len(result1.results) == 2
        assert result1.has_more is True
        assert result1.next_offset == 2

        # Get next 2 results
        _configure_mock_for_tasks(db_session, all_tasks[2:4], doc, total_count=5)
        filters = GetTasksFilterParams(limit=2, offset=2)
        result2 = get_tasks(db_session, filters)

        assert len(result2.results) == 2

    def test_get_tasks_handler_integration(self, db_session, sample_data):
        """Test _get_tasks_handler with database session."""
        from obsidian_rag.mcp_server.handlers import TaskDateFilterStrings

        vault, doc, tasks = sample_data
        today = date.today()

        # Create a mock db_manager that returns our test session
        db_manager = MagicMock()
        db_manager.get_session.return_value.__enter__ = MagicMock(
            return_value=db_session
        )
        db_manager.get_session.return_value.__exit__ = MagicMock(return_value=False)

        # Configure mock for not_completed tasks
        not_completed = [t for t in tasks if t.status == TaskStatus.NOT_COMPLETED.value]
        _configure_mock_for_tasks(
            db_session, not_completed, doc, total_count=len(not_completed)
        )

        from obsidian_rag.mcp_server.handlers import GetTasksRequest

        date_filters = TaskDateFilterStrings(
            due_after=today.isoformat(),
            due_before=(today + timedelta(days=30)).isoformat(),
        )

        request = GetTasksRequest(
            status=["not_completed"],
            date_filters=date_filters,
            limit=10,
        )

        result = _get_tasks_handler(
            db_manager=db_manager,
            request=request,
        )

        assert "results" in result
        assert "total_count" in result
        total_count = result["total_count"]
        assert isinstance(total_count, int) and total_count >= 1


class TestGetTasksEdgeCases:
    """Edge case tests for get_tasks tool."""

    def test_get_tasks_empty_database(self, db_session):
        """Test with no tasks in database."""
        _configure_mock_for_tasks(db_session, [], None, total_count=0)

        filters = GetTasksFilterParams()
        result = get_tasks(db_session, filters)

        assert result.total_count == 0
        assert result.results == []
        assert result.has_more is False

    def test_get_tasks_no_matching_filters(self, db_session, sample_data):
        """Test when filters match no tasks."""
        vault, doc, tasks = sample_data

        # No tasks should match nonexistent tag
        _configure_mock_for_tasks(db_session, [], doc, total_count=0)

        filters = GetTasksFilterParams(
            status=["completed"],
            tags=["nonexistent_tag"],
        )
        result = get_tasks(db_session, filters)

        assert result.total_count == 0
        assert result.results == []

    def test_get_tasks_invalid_date_range(self, db_session, sample_data):
        """Test with impossible date range (after > before)."""
        vault, doc, tasks = sample_data
        today = date.today()

        # No tasks can satisfy impossible date range
        _configure_mock_for_tasks(db_session, [], doc, total_count=0)

        # This is valid - just returns no results
        filters = GetTasksFilterParams(
            due_after=today + timedelta(days=30),
            due_before=today,
        )
        result = get_tasks(db_session, filters)

        # Should return empty (no tasks can satisfy this)
        assert result.total_count == 0


class TestGetTasksTagFiltering:
    """Integration tests for tag filtering features."""

    def test_get_tasks_integration_include_tags_all_mode(self, db_session, sample_data):
        """Integration test for include_tags with 'all' match mode."""
        from unittest.mock import patch

        vault, doc, tasks = sample_data

        # Create a mock db_manager that returns our test session
        db_manager = MagicMock()
        db_manager.get_session.return_value.__enter__ = MagicMock(
            return_value=db_session
        )
        db_manager.get_session.return_value.__exit__ = MagicMock(return_value=False)

        with patch("obsidian_rag.mcp_server.handlers.get_tasks_tool") as mock_get_tasks:
            mock_get_tasks.return_value.model_dump.return_value = {
                "results": [
                    {
                        "id": "task-1",
                        "description": "Work urgent task",
                        "tags": ["work", "urgent"],
                        "status": "not_completed",
                    }
                ],
                "total_count": 1,
                "has_more": False,
                "next_offset": None,
            }

            request = GetTasksRequest(
                tag_filters=TagFilterStrings(
                    include_tags=["work", "urgent"],
                    match_mode="all",
                ),
            )

            result = _get_tasks_handler(
                db_manager=db_manager,
                request=request,
            )

            assert result["total_count"] == 1
            results_list = result["results"]
            assert isinstance(results_list, list)
            assert len(results_list) == 1

            # Verify the filter params passed to get_tasks_tool
            call_args = mock_get_tasks.call_args
            filter_params = call_args.kwargs["filters"]
            assert filter_params.include_tags == ["work", "urgent"]
            assert filter_params.tag_match_mode == "all"

    def test_get_tasks_integration_include_tags_any_mode(self, db_session, sample_data):
        """Integration test for include_tags with 'any' match mode."""
        from unittest.mock import patch

        vault, doc, tasks = sample_data

        # Create a mock db_manager that returns our test session
        db_manager = MagicMock()
        db_manager.get_session.return_value.__enter__ = MagicMock(
            return_value=db_session
        )
        db_manager.get_session.return_value.__exit__ = MagicMock(return_value=False)

        with patch("obsidian_rag.mcp_server.handlers.get_tasks_tool") as mock_get_tasks:
            mock_get_tasks.return_value.model_dump.return_value = {
                "results": [
                    {
                        "id": "task-1",
                        "description": "Work task",
                        "tags": ["work"],
                        "status": "not_completed",
                    },
                    {
                        "id": "task-2",
                        "description": "Personal task",
                        "tags": ["personal"],
                        "status": "not_completed",
                    },
                ],
                "total_count": 2,
                "has_more": False,
                "next_offset": None,
            }

            request = GetTasksRequest(
                tag_filters=TagFilterStrings(
                    include_tags=["work", "personal"],
                    match_mode="any",
                ),
            )

            result = _get_tasks_handler(
                db_manager=db_manager,
                request=request,
            )

            assert result["total_count"] == 2
            results_list = result["results"]
            assert isinstance(results_list, list)
            assert len(results_list) == 2

            # Verify the filter params passed to get_tasks_tool
            call_args = mock_get_tasks.call_args
            filter_params = call_args.kwargs["filters"]
            assert filter_params.include_tags == ["work", "personal"]
            assert filter_params.tag_match_mode == "any"

    def test_get_tasks_integration_exclude_tags(self, db_session, sample_data):
        """Integration test for exclude_tags filtering."""
        from unittest.mock import patch

        vault, doc, tasks = sample_data

        # Create a mock db_manager that returns our test session
        db_manager = MagicMock()
        db_manager.get_session.return_value.__enter__ = MagicMock(
            return_value=db_session
        )
        db_manager.get_session.return_value.__exit__ = MagicMock(return_value=False)

        with patch("obsidian_rag.mcp_server.handlers.get_tasks_tool") as mock_get_tasks:
            mock_get_tasks.return_value.model_dump.return_value = {
                "results": [
                    {
                        "id": "task-1",
                        "description": "Active task",
                        "tags": ["work"],
                        "status": "not_completed",
                    }
                ],
                "total_count": 1,
                "has_more": False,
                "next_offset": None,
            }

            request = GetTasksRequest(
                tag_filters=TagFilterStrings(
                    exclude_tags=["blocked", "waiting"],
                ),
            )

            result = _get_tasks_handler(
                db_manager=db_manager,
                request=request,
            )

            assert result["total_count"] == 1

            # Verify exclude_tags is passed to get_tasks_tool
            call_args = mock_get_tasks.call_args
            filter_params = call_args.kwargs["filters"]
            assert filter_params.exclude_tags == ["blocked", "waiting"]

    def test_get_tasks_integration_combined_include_and_exclude(
        self, db_session, sample_data
    ):
        """Integration test for combined include_tags and exclude_tags."""
        from unittest.mock import patch

        vault, doc, tasks = sample_data

        # Create a mock db_manager that returns our test session
        db_manager = MagicMock()
        db_manager.get_session.return_value.__enter__ = MagicMock(
            return_value=db_session
        )
        db_manager.get_session.return_value.__exit__ = MagicMock(return_value=False)

        with patch("obsidian_rag.mcp_server.handlers.get_tasks_tool") as mock_get_tasks:
            mock_get_tasks.return_value.model_dump.return_value = {
                "results": [
                    {
                        "id": "task-1",
                        "description": "Work task not blocked",
                        "tags": ["work"],
                        "status": "not_completed",
                    }
                ],
                "total_count": 1,
                "has_more": False,
                "next_offset": None,
            }

            request = GetTasksRequest(
                tag_filters=TagFilterStrings(
                    include_tags=["work"],
                    exclude_tags=["blocked"],
                    match_mode="all",
                ),
            )

            result = _get_tasks_handler(
                db_manager=db_manager,
                request=request,
            )

            assert result["total_count"] == 1

            # Verify both tag filters are passed
            call_args = mock_get_tasks.call_args
            filter_params = call_args.kwargs["filters"]
            assert filter_params.include_tags == ["work"]
            assert filter_params.exclude_tags == ["blocked"]
            assert filter_params.tag_match_mode == "all"

    def test_get_tasks_integration_conflicting_tags_raises_error(
        self, db_session, sample_data
    ):
        """Integration test that conflicting tags raise ValueError."""

        vault, doc, tasks = sample_data

        # Create a mock db_manager that returns our test session
        db_manager = MagicMock()
        db_manager.get_session.return_value.__enter__ = MagicMock(
            return_value=db_session
        )
        db_manager.get_session.return_value.__exit__ = MagicMock(return_value=False)

        with pytest.raises(ValueError, match="Conflicting tags found"):
            request = GetTasksRequest(
                tag_filters=TagFilterStrings(
                    include_tags=["work"],
                    exclude_tags=["work"],  # Same tag in both lists
                ),
            )

            _get_tasks_handler(
                db_manager=db_manager,
                request=request,
            )

    def test_get_tasks_integration_case_insensitive_conflict(
        self, db_session, sample_data
    ):
        """Integration test for case-insensitive conflict detection."""

        vault, doc, tasks = sample_data

        # Create a mock db_manager that returns our test session
        db_manager = MagicMock()
        db_manager.get_session.return_value.__enter__ = MagicMock(
            return_value=db_session
        )
        db_manager.get_session.return_value.__exit__ = MagicMock(return_value=False)

        with pytest.raises(ValueError, match="Conflicting tags found"):
            request = GetTasksRequest(
                tag_filters=TagFilterStrings(
                    include_tags=["Work"],  # Capitalized
                    exclude_tags=["work"],  # Lowercase - should still conflict
                ),
            )

            _get_tasks_handler(
                db_manager=db_manager,
                request=request,
            )

    def test_get_tasks_integration_backward_compatibility_legacy_tags(
        self, db_session, sample_data
    ):
        """Integration test for backward compatibility with legacy tags parameter."""
        from unittest.mock import patch

        vault, doc, tasks = sample_data

        # Create a mock db_manager that returns our test session
        db_manager = MagicMock()
        db_manager.get_session.return_value.__enter__ = MagicMock(
            return_value=db_session
        )
        db_manager.get_session.return_value.__exit__ = MagicMock(return_value=False)

        with patch("obsidian_rag.mcp_server.handlers.get_tasks_tool") as mock_get_tasks:
            mock_get_tasks.return_value.model_dump.return_value = {
                "results": [
                    {
                        "id": "task-1",
                        "description": "Legacy tagged task",
                        "tags": ["work", "urgent"],
                        "status": "not_completed",
                    }
                ],
                "total_count": 1,
                "has_more": False,
                "next_offset": None,
            }

            # Use legacy 'tags' parameter
            request = GetTasksRequest(
                tags=["work", "urgent"],
            )

            result = _get_tasks_handler(
                db_manager=db_manager,
                request=request,
            )

            assert result["total_count"] == 1

            # Verify legacy tags parameter is passed
            call_args = mock_get_tasks.call_args
            filter_params = call_args.kwargs["filters"]
            assert filter_params.tags == ["work", "urgent"]
            assert filter_params.include_tags is None
            assert filter_params.exclude_tags is None
