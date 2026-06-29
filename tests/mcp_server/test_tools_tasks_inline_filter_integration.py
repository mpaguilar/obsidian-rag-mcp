"""Integration tests for get_tasks() with inline_filters."""

import uuid
from unittest.mock import MagicMock, patch

import pytest

from obsidian_rag.database.models import Document, Task, TaskStatus
from obsidian_rag.mcp_server.models import PropertyFilter
from obsidian_rag.mcp_server.tools.tasks import get_tasks
from obsidian_rag.mcp_server.tools.tasks_params import GetTasksFilterParams


def _create_mock_session_with_tasks(tasks, documents):
    """Create a mock session with tasks and documents."""
    mock_session = MagicMock()
    mock_session.bind.dialect.name = "postgresql"

    mock_query = MagicMock()
    mock_query.join.return_value = mock_query
    mock_query.filter.return_value = mock_query
    mock_query.order_by.return_value = mock_query
    mock_query.limit.return_value = mock_query
    mock_query.offset.return_value = mock_query
    mock_query.count.return_value = len(tasks)

    results = []
    for i, task in enumerate(tasks):
        doc = documents[i] if i < len(documents) else MagicMock(spec=Document)
        results.append((task, doc))

    mock_query.all.return_value = results
    mock_session.query.return_value = mock_query

    return mock_session


def test_get_tasks_with_inline_equals_filter():
    """Test get_tasks with inline_filters equals operator."""
    task = MagicMock(spec=Task)
    task.id = uuid.uuid4()
    task.status = TaskStatus.NOT_COMPLETED.value
    task.description = "Task with vendor"
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

    mock_session = _create_mock_session_with_tasks([task], [doc])

    with patch(
        "obsidian_rag.mcp_server.tools.tasks.create_task_response"
    ) as mock_create_response:
        from obsidian_rag.mcp_server.models import TaskResponse

        mock_response = TaskResponse(
            id=uuid.uuid4(),
            raw_text="- [ ] Task with vendor",
            status="not_completed",
            description="Task with vendor",
            due=None,
            priority="normal",
            tags=[],
            document_path="/test/doc.md",
            document_name="doc.md",
        )
        mock_create_response.return_value = mock_response

        inline_filters = [
            PropertyFilter(path="vendor", operator="equals", value="acme")
        ]
        filters = GetTasksFilterParams(inline_filters=inline_filters)
        result = get_tasks(mock_session, filters)

    assert result.total_count == 1
    mock_session.query.return_value.filter.assert_called()


def test_get_tasks_with_inline_contains_filter():
    """Test get_tasks with inline_filters contains operator."""
    task = MagicMock(spec=Task)
    task.id = uuid.uuid4()
    task.status = TaskStatus.NOT_COMPLETED.value
    task.description = "Task with vendor"
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

    mock_session = _create_mock_session_with_tasks([task], [doc])

    with patch(
        "obsidian_rag.mcp_server.tools.tasks.create_task_response"
    ) as mock_create_response:
        from obsidian_rag.mcp_server.models import TaskResponse

        mock_response = TaskResponse(
            id=uuid.uuid4(),
            raw_text="- [ ] Task with vendor",
            status="not_completed",
            description="Task with vendor",
            due=None,
            priority="normal",
            tags=[],
            document_path="/test/doc.md",
            document_name="doc.md",
        )
        mock_create_response.return_value = mock_response

        inline_filters = [
            PropertyFilter(path="vendor", operator="contains", value="acme")
        ]
        filters = GetTasksFilterParams(inline_filters=inline_filters)
        result = get_tasks(mock_session, filters)

    assert result.total_count == 1


def test_get_tasks_with_inline_exists_filter():
    """Test get_tasks with inline_filters exists operator."""
    task = MagicMock(spec=Task)
    task.id = uuid.uuid4()
    task.status = TaskStatus.NOT_COMPLETED.value
    task.description = "Task with vendor"
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

    mock_session = _create_mock_session_with_tasks([task], [doc])

    with patch(
        "obsidian_rag.mcp_server.tools.tasks.create_task_response"
    ) as mock_create_response:
        from obsidian_rag.mcp_server.models import TaskResponse

        mock_response = TaskResponse(
            id=uuid.uuid4(),
            raw_text="- [ ] Task with vendor",
            status="not_completed",
            description="Task with vendor",
            due=None,
            priority="normal",
            tags=[],
            document_path="/test/doc.md",
            document_name="doc.md",
        )
        mock_create_response.return_value = mock_response

        inline_filters = [PropertyFilter(path="vendor", operator="exists")]
        filters = GetTasksFilterParams(inline_filters=inline_filters)
        result = get_tasks(mock_session, filters)

    assert result.total_count == 1


def test_get_tasks_with_inline_in_filter():
    """Test get_tasks with inline_filters in operator."""
    task = MagicMock(spec=Task)
    task.id = uuid.uuid4()
    task.status = TaskStatus.NOT_COMPLETED.value
    task.description = "Task with vendor"
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

    mock_session = _create_mock_session_with_tasks([task], [doc])

    with patch(
        "obsidian_rag.mcp_server.tools.tasks.create_task_response"
    ) as mock_create_response:
        from obsidian_rag.mcp_server.models import TaskResponse

        mock_response = TaskResponse(
            id=uuid.uuid4(),
            raw_text="- [ ] Task with vendor",
            status="not_completed",
            description="Task with vendor",
            due=None,
            priority="normal",
            tags=[],
            document_path="/test/doc.md",
            document_name="doc.md",
        )
        mock_create_response.return_value = mock_response

        inline_filters = [
            PropertyFilter(path="vendor", operator="in", value=["acme", "globex"])
        ]
        filters = GetTasksFilterParams(inline_filters=inline_filters)
        result = get_tasks(mock_session, filters)

    assert result.total_count == 1


def test_get_tasks_with_inline_starts_with_filter():
    """Test get_tasks with inline_filters starts_with operator."""
    task = MagicMock(spec=Task)
    task.id = uuid.uuid4()
    task.status = TaskStatus.NOT_COMPLETED.value
    task.description = "Task with vendor"
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

    mock_session = _create_mock_session_with_tasks([task], [doc])

    with patch(
        "obsidian_rag.mcp_server.tools.tasks.create_task_response"
    ) as mock_create_response:
        from obsidian_rag.mcp_server.models import TaskResponse

        mock_response = TaskResponse(
            id=uuid.uuid4(),
            raw_text="- [ ] Task with vendor",
            status="not_completed",
            description="Task with vendor",
            due=None,
            priority="normal",
            tags=[],
            document_path="/test/doc.md",
            document_name="doc.md",
        )
        mock_create_response.return_value = mock_response

        inline_filters = [
            PropertyFilter(path="vendor", operator="starts_with", value="acme")
        ]
        filters = GetTasksFilterParams(inline_filters=inline_filters)
        result = get_tasks(mock_session, filters)

    assert result.total_count == 1


def test_get_tasks_with_inline_regex_filter():
    """Test get_tasks with inline_filters regex operator."""
    task = MagicMock(spec=Task)
    task.id = uuid.uuid4()
    task.status = TaskStatus.NOT_COMPLETED.value
    task.description = "Task with vendor"
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

    mock_session = _create_mock_session_with_tasks([task], [doc])

    with patch(
        "obsidian_rag.mcp_server.tools.tasks.create_task_response"
    ) as mock_create_response:
        from obsidian_rag.mcp_server.models import TaskResponse

        mock_response = TaskResponse(
            id=uuid.uuid4(),
            raw_text="- [ ] Task with vendor",
            status="not_completed",
            description="Task with vendor",
            due=None,
            priority="normal",
            tags=[],
            document_path="/test/doc.md",
            document_name="doc.md",
        )
        mock_create_response.return_value = mock_response

        inline_filters = [
            PropertyFilter(path="vendor", operator="regex", value=r"acme.*")
        ]
        filters = GetTasksFilterParams(inline_filters=inline_filters)
        result = get_tasks(mock_session, filters)

    assert result.total_count == 1


def test_get_tasks_with_multiple_inline_filters():
    """Test get_tasks with multiple inline_filters (AND logic)."""
    task = MagicMock(spec=Task)
    task.id = uuid.uuid4()
    task.status = TaskStatus.NOT_COMPLETED.value
    task.description = "Task with vendor"
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

    mock_session = _create_mock_session_with_tasks([task], [doc])

    with patch(
        "obsidian_rag.mcp_server.tools.tasks.create_task_response"
    ) as mock_create_response:
        from obsidian_rag.mcp_server.models import TaskResponse

        mock_response = TaskResponse(
            id=uuid.uuid4(),
            raw_text="- [ ] Task with vendor",
            status="not_completed",
            description="Task with vendor",
            due=None,
            priority="normal",
            tags=[],
            document_path="/test/doc.md",
            document_name="doc.md",
        )
        mock_create_response.return_value = mock_response

        inline_filters = [
            PropertyFilter(path="vendor", operator="equals", value="acme"),
            PropertyFilter(path="region", operator="equals", value="us-east"),
        ]
        filters = GetTasksFilterParams(inline_filters=inline_filters)
        result = get_tasks(mock_session, filters)

    assert result.total_count == 1
    assert mock_session.query.return_value.filter.call_count >= 2


def test_get_tasks_with_inline_filters_none():
    """Test get_tasks with inline_filters=None does not apply filters."""
    task = MagicMock(spec=Task)
    task.id = uuid.uuid4()
    task.status = TaskStatus.NOT_COMPLETED.value
    task.description = "Task without inline filters"
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

    mock_session = _create_mock_session_with_tasks([task], [doc])

    with patch(
        "obsidian_rag.mcp_server.tools.tasks.create_task_response"
    ) as mock_create_response:
        from obsidian_rag.mcp_server.models import TaskResponse

        mock_response = TaskResponse(
            id=uuid.uuid4(),
            raw_text="- [ ] Task without inline filters",
            status="not_completed",
            description="Task without inline filters",
            due=None,
            priority="normal",
            tags=[],
            document_path="/test/doc.md",
            document_name="doc.md",
        )
        mock_create_response.return_value = mock_response

        filters = GetTasksFilterParams(inline_filters=None)
        result = get_tasks(mock_session, filters)

    assert result.total_count == 1


def test_get_tasks_inline_filter_validation_error():
    """Test get_tasks with invalid inline_filters raises ValueError."""
    mock_session = MagicMock()

    mock_filter = MagicMock()
    mock_filter.path = "vendor"
    mock_filter.operator = "invalid_operator"
    mock_filter.value = "acme"
    inline_filters = [mock_filter]
    filters = GetTasksFilterParams(inline_filters=inline_filters)

    with pytest.raises(ValueError, match="Invalid operator"):
        get_tasks(mock_session, filters)


def test_get_tasks_inline_filter_combined_with_status_filter():
    """Test get_tasks combining inline_filters and status filters."""
    task = MagicMock(spec=Task)
    task.id = uuid.uuid4()
    task.status = TaskStatus.NOT_COMPLETED.value
    task.description = "Task with vendor"
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

    mock_session = _create_mock_session_with_tasks([task], [doc])

    with patch(
        "obsidian_rag.mcp_server.tools.tasks.create_task_response"
    ) as mock_create_response:
        from obsidian_rag.mcp_server.models import TaskResponse

        mock_response = TaskResponse(
            id=uuid.uuid4(),
            raw_text="- [ ] Task with vendor",
            status="not_completed",
            description="Task with vendor",
            due=None,
            priority="normal",
            tags=[],
            document_path="/test/doc.md",
            document_name="doc.md",
        )
        mock_create_response.return_value = mock_response

        inline_filters = [
            PropertyFilter(path="vendor", operator="equals", value="acme")
        ]
        filters = GetTasksFilterParams(
            status=["not_completed"],
            inline_filters=inline_filters,
        )
        result = get_tasks(mock_session, filters)

    assert result.total_count == 1
