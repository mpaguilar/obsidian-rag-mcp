"""Tests for get_tasks function."""

import uuid
from datetime import date, timedelta
from unittest.mock import MagicMock, patch

import pytest

from obsidian_rag.database.models import Document, Task, TaskStatus
from obsidian_rag.mcp_server.tools.tasks import get_tasks
from obsidian_rag.mcp_server.tools.tasks_params import GetTasksFilterParams


class TestGetTasks:
    """Tests for get_tasks function."""

    def _create_mock_session_with_tasks(self, tasks, documents):
        """Create a mock session with tasks and documents."""
        mock_session = MagicMock()
        mock_session.bind.dialect.name = "postgresql"

        # Create mock query chain
        mock_query = MagicMock()
        mock_query.join.return_value = mock_query
        mock_query.filter.return_value = mock_query
        mock_query.order_by.return_value = mock_query
        mock_query.limit.return_value = mock_query
        mock_query.offset.return_value = mock_query
        mock_query.count.return_value = len(tasks)

        # Create result tuples (task, document)
        results = []
        for i, task in enumerate(tasks):
            doc = documents[i] if i < len(documents) else MagicMock(spec=Document)
            results.append((task, doc))

        mock_query.all.return_value = results
        mock_session.query.return_value = mock_query

        return mock_session

    def test_empty_result(self):
        """Test with no tasks in database."""
        mock_session = MagicMock()
        mock_session.bind.dialect.name = "postgresql"

        mock_query = MagicMock()
        mock_query.join.return_value = mock_query
        mock_query.filter.return_value = mock_query
        mock_query.order_by.return_value = mock_query
        mock_query.limit.return_value = mock_query
        mock_query.offset.return_value = mock_query
        mock_query.count.return_value = 0
        mock_query.all.return_value = []

        mock_session.query.return_value = mock_query

        filters = GetTasksFilterParams()
        result = get_tasks(mock_session, filters)

        assert result.results == []
        assert result.total_count == 0
        assert result.has_more is False

    def test_no_filters_returns_all(self):
        """Test that no filters returns all tasks."""
        task = MagicMock(spec=Task)
        task.id = uuid.uuid4()
        task.status = TaskStatus.NOT_COMPLETED.value
        task.description = "Test task"
        task.tags = []
        task.priority = "normal"
        task.due = None
        task.scheduled = None
        task.completion = None

        doc = MagicMock(spec=Document)
        doc.id = uuid.uuid4()
        doc.file_path = "/test/doc.md"
        doc.file_name = "doc.md"
        doc.tags = []

        mock_session = self._create_mock_session_with_tasks([task], [doc])

        with patch(
            "obsidian_rag.mcp_server.tools.tasks.create_task_response"
        ) as mock_create_response:
            from obsidian_rag.mcp_server.models import TaskResponse

            mock_response = TaskResponse(
                id=uuid.uuid4(),
                raw_text="Test task",
                status="not_completed",
                description="Test task",
                due=None,
                priority="normal",
                tags=[],
                document_path="/test/doc.md",
                document_name="doc.md",
            )
            mock_create_response.return_value = mock_response

            filters = GetTasksFilterParams()
            result = get_tasks(mock_session, filters)

        assert result.total_count == 1
        assert len(result.results) == 1

    def test_status_filter(self):
        """Test filtering by status."""
        task = MagicMock(spec=Task)
        task.id = uuid.uuid4()
        task.status = TaskStatus.NOT_COMPLETED.value
        task.description = "Incomplete task"
        task.tags = []
        task.priority = "normal"
        task.due = None
        task.scheduled = None
        task.completion = None

        doc = MagicMock(spec=Document)
        doc.id = uuid.uuid4()
        doc.file_path = "/test/doc.md"
        doc.file_name = "doc.md"
        doc.tags = []

        mock_session = self._create_mock_session_with_tasks([task], [doc])

        with patch(
            "obsidian_rag.mcp_server.tools.tasks.create_task_response"
        ) as mock_create_response:
            from obsidian_rag.mcp_server.models import TaskResponse

            mock_response = TaskResponse(
                id=uuid.uuid4(),
                raw_text="- [ ] Incomplete",
                status="not_completed",
                description="Incomplete task",
                due=None,
                priority="normal",
                tags=[],
                document_path="/test/doc.md",
                document_name="doc.md",
            )
            mock_create_response.return_value = mock_response

            filters = GetTasksFilterParams(status=["not_completed"])
            result = get_tasks(mock_session, filters)

        assert result.total_count == 1
        assert result.results[0].status == "not_completed"

    def test_due_date_range_filter(self):
        """Test filtering by due date range."""
        today = date.today()

        task = MagicMock(spec=Task)
        task.id = uuid.uuid4()
        task.status = TaskStatus.NOT_COMPLETED.value
        task.description = "Due in range"
        task.tags = []
        task.priority = "normal"
        task.due = today + timedelta(days=5)
        task.scheduled = None
        task.completion = None

        doc = MagicMock(spec=Document)
        doc.id = uuid.uuid4()
        doc.file_path = "/test/doc.md"
        doc.file_name = "doc.md"
        doc.tags = []

        mock_session = self._create_mock_session_with_tasks([task], [doc])

        with patch(
            "obsidian_rag.mcp_server.tools.tasks.create_task_response"
        ) as mock_create_response:
            from obsidian_rag.mcp_server.models import TaskResponse

            mock_response = TaskResponse(
                id=uuid.uuid4(),
                raw_text="- [ ] Due in range",
                status="not_completed",
                description="Due in range",
                due=today + timedelta(days=5),
                priority="normal",
                tags=[],
                document_path="/test/doc.md",
                document_name="doc.md",
            )
            mock_create_response.return_value = mock_response

            filters = GetTasksFilterParams(
                due_after=today,
                due_before=today + timedelta(days=10),
            )
            result = get_tasks(mock_session, filters)

        assert result.total_count == 1
        assert result.results[0].description == "Due in range"

    def test_scheduled_date_filter(self):
        """Test filtering by scheduled date."""
        today = date.today()

        task = MagicMock(spec=Task)
        task.id = uuid.uuid4()
        task.status = TaskStatus.NOT_COMPLETED.value
        task.description = "Scheduled task"
        task.tags = []
        task.priority = "normal"
        task.due = None
        task.scheduled = today + timedelta(days=3)
        task.completion = None

        doc = MagicMock(spec=Document)
        doc.id = uuid.uuid4()
        doc.file_path = "/test/doc.md"
        doc.file_name = "doc.md"
        doc.tags = []

        mock_session = self._create_mock_session_with_tasks([task], [doc])

        with patch(
            "obsidian_rag.mcp_server.tools.tasks.create_task_response"
        ) as mock_create_response:
            from obsidian_rag.mcp_server.models import TaskResponse

            mock_response = TaskResponse(
                id=uuid.uuid4(),
                raw_text="- [ ] Scheduled task",
                status="not_completed",
                description="Scheduled task",
                due=None,
                priority="normal",
                tags=[],
                document_path="/test/doc.md",
                document_name="doc.md",
            )
            mock_create_response.return_value = mock_response

            filters = GetTasksFilterParams(
                scheduled_after=today,
                scheduled_before=today + timedelta(days=7),
            )
            result = get_tasks(mock_session, filters)

        assert result.total_count == 1

    def test_completion_date_filter(self):
        """Test filtering by completion date."""
        today = date.today()

        task = MagicMock(spec=Task)
        task.id = uuid.uuid4()
        task.status = TaskStatus.COMPLETED.value
        task.description = "Completed task"
        task.tags = []
        task.priority = "normal"
        task.due = None
        task.scheduled = None
        task.completion = today - timedelta(days=2)

        doc = MagicMock(spec=Document)
        doc.id = uuid.uuid4()
        doc.file_path = "/test/doc.md"
        doc.file_name = "doc.md"
        doc.tags = []

        mock_session = self._create_mock_session_with_tasks([task], [doc])

        with patch(
            "obsidian_rag.mcp_server.tools.tasks.create_task_response"
        ) as mock_create_response:
            from obsidian_rag.mcp_server.models import TaskResponse

            mock_response = TaskResponse(
                id=uuid.uuid4(),
                raw_text="- [x] Completed task",
                status="completed",
                description="Completed task",
                due=None,
                priority="normal",
                tags=[],
                document_path="/test/doc.md",
                document_name="doc.md",
            )
            mock_create_response.return_value = mock_response

            filters = GetTasksFilterParams(
                completion_after=today - timedelta(days=7),
                completion_before=today,
            )
            result = get_tasks(mock_session, filters)

        assert result.total_count == 1

    def test_priority_filter(self):
        """Test filtering by priority."""
        task = MagicMock(spec=Task)
        task.id = uuid.uuid4()
        task.status = TaskStatus.NOT_COMPLETED.value
        task.description = "High priority"
        task.tags = []
        task.priority = "high"
        task.due = None
        task.scheduled = None
        task.completion = None

        doc = MagicMock(spec=Document)
        doc.id = uuid.uuid4()
        doc.file_path = "/test/doc.md"
        doc.file_name = "doc.md"
        doc.tags = []

        mock_session = self._create_mock_session_with_tasks([task], [doc])

        with patch(
            "obsidian_rag.mcp_server.tools.tasks.create_task_response"
        ) as mock_create_response:
            from obsidian_rag.mcp_server.models import TaskResponse

            mock_response = TaskResponse(
                id=uuid.uuid4(),
                raw_text="- [ ] High priority",
                status="not_completed",
                description="High priority",
                due=None,
                priority="high",
                tags=[],
                document_path="/test/doc.md",
                document_name="doc.md",
            )
            mock_create_response.return_value = mock_response

            filters = GetTasksFilterParams(priority=["high", "highest"])
            result = get_tasks(mock_session, filters)

        assert result.total_count == 1
        assert result.results[0].priority == "high"

    def test_pagination(self):
        """Test pagination with limit and offset."""
        tasks = []
        docs = []
        for i in range(5):
            task = MagicMock(spec=Task)
            task.id = uuid.uuid4()
            task.status = TaskStatus.NOT_COMPLETED.value
            task.description = f"Task {i}"
            task.tags = []
            task.priority = "normal"
            task.due = None
            task.scheduled = None
            task.completion = None
            tasks.append(task)

            doc = MagicMock(spec=Document)
            doc.id = uuid.uuid4()
            doc.file_path = f"/test/task{i}.md"
            doc.file_name = f"task{i}.md"
            doc.tags = []
            docs.append(doc)

        mock_session = MagicMock()
        mock_session.bind.dialect.name = "postgresql"

        mock_query = MagicMock()
        mock_query.join.return_value = mock_query
        mock_query.filter.return_value = mock_query
        mock_query.order_by.return_value = mock_query
        mock_query.limit.return_value = mock_query
        mock_query.offset.return_value = mock_query
        mock_query.count.return_value = 5
        mock_query.all.return_value = list(zip(tasks[:2], docs[:2]))

        mock_session.query.return_value = mock_query

        with patch(
            "obsidian_rag.mcp_server.tools.tasks.create_task_response"
        ) as mock_create_response:
            from obsidian_rag.mcp_server.models import TaskResponse

            mock_response = TaskResponse(
                id=uuid.uuid4(),
                raw_text="Task",
                status="not_completed",
                description="Task",
                due=None,
                priority="normal",
                tags=[],
                document_path="/test/task.md",
                document_name="task.md",
            )
            mock_create_response.return_value = mock_response

            filters = GetTasksFilterParams(limit=2, offset=0)
            result = get_tasks(mock_session, filters)

        assert result.total_count == 5
        assert len(result.results) == 2
        assert result.has_more is True
        assert result.next_offset == 2

    def test_multiple_filters_combined(self):
        """Test that multiple filters are combined with AND logic."""
        today = date.today()

        task = MagicMock(spec=Task)
        task.id = uuid.uuid4()
        task.status = TaskStatus.NOT_COMPLETED.value
        task.description = "Match all"
        task.tags = []
        task.priority = "high"
        task.due = today + timedelta(days=3)
        task.scheduled = None
        task.completion = None

        doc = MagicMock(spec=Document)
        doc.id = uuid.uuid4()
        doc.file_path = "/test/doc.md"
        doc.file_name = "doc.md"
        doc.tags = []

        mock_session = self._create_mock_session_with_tasks([task], [doc])

        with patch(
            "obsidian_rag.mcp_server.tools.tasks.create_task_response"
        ) as mock_create_response:
            from obsidian_rag.mcp_server.models import TaskResponse

            mock_response = TaskResponse(
                id=uuid.uuid4(),
                raw_text="- [ ] Match all",
                status="not_completed",
                description="Match all",
                due=today + timedelta(days=3),
                priority="high",
                tags=[],
                document_path="/test/doc.md",
                document_name="doc.md",
            )
            mock_create_response.return_value = mock_response

            filters = GetTasksFilterParams(
                status=["not_completed"],
                due_after=today,
                due_before=today + timedelta(days=10),
                priority=["high"],
            )
            result = get_tasks(mock_session, filters)

        assert result.total_count == 1
        assert result.results[0].description == "Match all"

    def test_include_tags_all_mode_integration(self):
        """Test include_tags with 'all' mode in full get_tasks flow."""
        task = MagicMock(spec=Task)
        task.id = uuid.uuid4()
        task.status = TaskStatus.NOT_COMPLETED.value
        task.description = "Work task"
        task.tags = ["work", "urgent"]
        task.priority = "normal"
        task.due = None
        task.scheduled = None
        task.completion = None

        doc = MagicMock(spec=Document)
        doc.id = uuid.uuid4()
        doc.file_path = "/test/doc.md"
        doc.file_name = "doc.md"
        doc.tags = []

        mock_session = self._create_mock_session_with_tasks([task], [doc])

        with patch(
            "obsidian_rag.mcp_server.tools.tasks.create_task_response"
        ) as mock_create_response:
            from obsidian_rag.mcp_server.models import TaskResponse

            mock_response = TaskResponse(
                id=uuid.uuid4(),
                raw_text="- [ ] Work task",
                status="not_completed",
                description="Work task",
                due=None,
                priority="normal",
                tags=["work", "urgent"],
                document_path="/test/doc.md",
                document_name="doc.md",
            )
            mock_create_response.return_value = mock_response

            filters = GetTasksFilterParams(
                include_tags=["work", "urgent"],
                tag_match_mode="all",
            )
            result = get_tasks(mock_session, filters)

        assert result.total_count == 1

    def test_include_tags_any_mode_integration(self):
        """Test include_tags with 'any' mode in full get_tasks flow."""
        task = MagicMock(spec=Task)
        task.id = uuid.uuid4()
        task.status = TaskStatus.NOT_COMPLETED.value
        task.description = "Personal task"
        task.tags = ["personal"]
        task.priority = "normal"
        task.due = None
        task.scheduled = None
        task.completion = None

        doc = MagicMock(spec=Document)
        doc.id = uuid.uuid4()
        doc.file_path = "/test/doc.md"
        doc.file_name = "doc.md"
        doc.tags = []

        mock_session = self._create_mock_session_with_tasks([task], [doc])

        with patch(
            "obsidian_rag.mcp_server.tools.tasks.create_task_response"
        ) as mock_create_response:
            from obsidian_rag.mcp_server.models import TaskResponse

            mock_response = TaskResponse(
                id=uuid.uuid4(),
                raw_text="- [ ] Personal task",
                status="not_completed",
                description="Personal task",
                due=None,
                priority="normal",
                tags=["personal"],
                document_path="/test/doc.md",
                document_name="doc.md",
            )
            mock_create_response.return_value = mock_response

            filters = GetTasksFilterParams(
                include_tags=["work", "personal"],
                tag_match_mode="any",
            )
            result = get_tasks(mock_session, filters)

        assert result.total_count == 1

    def test_exclude_tags_integration(self):
        """Test exclude_tags in full get_tasks flow."""
        task = MagicMock(spec=Task)
        task.id = uuid.uuid4()
        task.status = TaskStatus.NOT_COMPLETED.value
        task.description = "Active task"
        task.tags = ["work"]
        task.priority = "normal"
        task.due = None
        task.scheduled = None
        task.completion = None

        doc = MagicMock(spec=Document)
        doc.id = uuid.uuid4()
        doc.file_path = "/test/doc.md"
        doc.file_name = "doc.md"
        doc.tags = []

        mock_session = self._create_mock_session_with_tasks([task], [doc])

        with patch(
            "obsidian_rag.mcp_server.tools.tasks.create_task_response"
        ) as mock_create_response:
            from obsidian_rag.mcp_server.models import TaskResponse

            mock_response = TaskResponse(
                id=uuid.uuid4(),
                raw_text="- [ ] Active task",
                status="not_completed",
                description="Active task",
                due=None,
                priority="normal",
                tags=["work"],
                document_path="/test/doc.md",
                document_name="doc.md",
            )
            mock_create_response.return_value = mock_response

            filters = GetTasksFilterParams(exclude_tags=["blocked"])
            result = get_tasks(mock_session, filters)

        assert result.total_count == 1

    def test_conflicting_tags_raises_error(self):
        """Test that conflicting tags raise ValueError."""
        mock_session = MagicMock()

        filters = GetTasksFilterParams(
            include_tags=["work"],
            exclude_tags=["work"],  # Same tag in both lists
        )

        with pytest.raises(ValueError, match="Conflicting tags found"):
            get_tasks(mock_session, filters)

    def test_backward_compatibility_legacy_tags(self):
        """Test that legacy 'tags' parameter still works."""
        task = MagicMock(spec=Task)
        task.id = uuid.uuid4()
        task.status = TaskStatus.NOT_COMPLETED.value
        task.description = "Legacy tagged task"
        task.tags = ["work", "urgent"]
        task.priority = "normal"
        task.due = None
        task.scheduled = None
        task.completion = None

        doc = MagicMock(spec=Document)
        doc.id = uuid.uuid4()
        doc.file_path = "/test/doc.md"
        doc.file_name = "doc.md"
        doc.tags = []

        mock_session = self._create_mock_session_with_tasks([task], [doc])

        with patch(
            "obsidian_rag.mcp_server.tools.tasks.create_task_response"
        ) as mock_create_response:
            from obsidian_rag.mcp_server.models import TaskResponse

            mock_response = TaskResponse(
                id=uuid.uuid4(),
                raw_text="- [ ] Legacy tagged task",
                status="not_completed",
                description="Legacy tagged task",
                due=None,
                priority="normal",
                tags=["work", "urgent"],
                document_path="/test/doc.md",
                document_name="doc.md",
            )
            mock_create_response.return_value = mock_response

            # Use legacy 'tags' parameter
            filters = GetTasksFilterParams(tags=["work", "urgent"])
            result = get_tasks(mock_session, filters)

        assert result.total_count == 1
