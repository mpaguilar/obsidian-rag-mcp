"""End-to-end integration tests for get_tasks() MCP tool with inline_filters parameter.

Tests the full chain: server.py -> handlers.py -> tasks_params.py -> tasks.py -> tasks_inline_filters.py.
"""

import uuid
from unittest.mock import MagicMock, patch

from obsidian_rag.database.models import Document, Task, TaskStatus
from obsidian_rag.mcp_server.models import PropertyFilter
from obsidian_rag.mcp_server.server import get_tasks


def _create_mock_db_manager_with_tasks(tasks, documents):
    """Create a mock db_manager that yields a session with tasks and documents."""
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

    mock_db_manager = MagicMock()
    mock_db_manager.get_session.return_value.__enter__ = MagicMock(
        return_value=mock_session
    )
    mock_db_manager.get_session.return_value.__exit__ = MagicMock(return_value=False)

    return mock_db_manager, mock_session


def _make_task_and_doc():
    """Create a mock task and document pair."""
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

    return task, doc


def test_get_tasks_inline_filters_equals():
    """Test get_tasks with inline_filters equals operator through full chain."""
    task, doc = _make_task_and_doc()
    mock_db_manager, mock_session = _create_mock_db_manager_with_tasks([task], [doc])

    with (
        patch("obsidian_rag.mcp_server.server._get_registry") as mock_registry,
        patch(
            "obsidian_rag.mcp_server.tools.tasks.create_task_response"
        ) as mock_create_response,
    ):
        from obsidian_rag.mcp_server.models import TaskResponse

        mock_registry.return_value = MagicMock(db_manager=mock_db_manager)
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
        result = get_tasks(inline_filters=inline_filters)

    assert result["total_count"] == 1
    mock_session.query.return_value.filter.assert_called()


def test_get_tasks_inline_filters_contains():
    """Test get_tasks with inline_filters contains operator through full chain."""
    task, doc = _make_task_and_doc()
    mock_db_manager, mock_session = _create_mock_db_manager_with_tasks([task], [doc])

    with (
        patch("obsidian_rag.mcp_server.server._get_registry") as mock_registry,
        patch(
            "obsidian_rag.mcp_server.tools.tasks.create_task_response"
        ) as mock_create_response,
    ):
        from obsidian_rag.mcp_server.models import TaskResponse

        mock_registry.return_value = MagicMock(db_manager=mock_db_manager)
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
        result = get_tasks(inline_filters=inline_filters)

    assert result["total_count"] == 1


def test_get_tasks_inline_filters_exists():
    """Test get_tasks with inline_filters exists operator through full chain."""
    task, doc = _make_task_and_doc()
    mock_db_manager, mock_session = _create_mock_db_manager_with_tasks([task], [doc])

    with (
        patch("obsidian_rag.mcp_server.server._get_registry") as mock_registry,
        patch(
            "obsidian_rag.mcp_server.tools.tasks.create_task_response"
        ) as mock_create_response,
    ):
        from obsidian_rag.mcp_server.models import TaskResponse

        mock_registry.return_value = MagicMock(db_manager=mock_db_manager)
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
        result = get_tasks(inline_filters=inline_filters)

    assert result["total_count"] == 1


def test_get_tasks_inline_filters_in():
    """Test get_tasks with inline_filters in operator through full chain."""
    task, doc = _make_task_and_doc()
    mock_db_manager, mock_session = _create_mock_db_manager_with_tasks([task], [doc])

    with (
        patch("obsidian_rag.mcp_server.server._get_registry") as mock_registry,
        patch(
            "obsidian_rag.mcp_server.tools.tasks.create_task_response"
        ) as mock_create_response,
    ):
        from obsidian_rag.mcp_server.models import TaskResponse

        mock_registry.return_value = MagicMock(db_manager=mock_db_manager)
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
        result = get_tasks(inline_filters=inline_filters)

    assert result["total_count"] == 1


def test_get_tasks_inline_filters_starts_with():
    """Test get_tasks with inline_filters starts_with operator through full chain."""
    task, doc = _make_task_and_doc()
    mock_db_manager, mock_session = _create_mock_db_manager_with_tasks([task], [doc])

    with (
        patch("obsidian_rag.mcp_server.server._get_registry") as mock_registry,
        patch(
            "obsidian_rag.mcp_server.tools.tasks.create_task_response"
        ) as mock_create_response,
    ):
        from obsidian_rag.mcp_server.models import TaskResponse

        mock_registry.return_value = MagicMock(db_manager=mock_db_manager)
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
        result = get_tasks(inline_filters=inline_filters)

    assert result["total_count"] == 1


def test_get_tasks_inline_filters_regex():
    """Test get_tasks with inline_filters regex operator through full chain."""
    task, doc = _make_task_and_doc()
    mock_db_manager, mock_session = _create_mock_db_manager_with_tasks([task], [doc])

    with (
        patch("obsidian_rag.mcp_server.server._get_registry") as mock_registry,
        patch(
            "obsidian_rag.mcp_server.tools.tasks.create_task_response"
        ) as mock_create_response,
    ):
        from obsidian_rag.mcp_server.models import TaskResponse

        mock_registry.return_value = MagicMock(db_manager=mock_db_manager)
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
        result = get_tasks(inline_filters=inline_filters)

    assert result["total_count"] == 1


def test_get_tasks_inline_filters_json_string_input():
    """Test get_tasks with inline_filters as JSON string through full chain."""
    task, doc = _make_task_and_doc()
    mock_db_manager, mock_session = _create_mock_db_manager_with_tasks([task], [doc])

    with (
        patch("obsidian_rag.mcp_server.server._get_registry") as mock_registry,
        patch(
            "obsidian_rag.mcp_server.tools.tasks.create_task_response"
        ) as mock_create_response,
    ):
        from obsidian_rag.mcp_server.models import TaskResponse

        mock_registry.return_value = MagicMock(db_manager=mock_db_manager)
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

        json_input = '[{"path": "vendor", "operator": "equals", "value": "acme"}]'
        result = get_tasks(inline_filters=json_input)

    assert result["total_count"] == 1


def test_get_tasks_inline_filters_dict_input():
    """Test get_tasks with inline_filters as dict through full chain."""
    task, doc = _make_task_and_doc()
    mock_db_manager, mock_session = _create_mock_db_manager_with_tasks([task], [doc])

    with (
        patch("obsidian_rag.mcp_server.server._get_registry") as mock_registry,
        patch(
            "obsidian_rag.mcp_server.tools.tasks.create_task_response"
        ) as mock_create_response,
    ):
        from obsidian_rag.mcp_server.models import TaskResponse

        mock_registry.return_value = MagicMock(db_manager=mock_db_manager)
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

        dict_input = {"path": "vendor", "operator": "equals", "value": "acme"}
        result = get_tasks(inline_filters=dict_input)

    assert result["total_count"] == 1


def test_get_tasks_inline_filters_combined_with_status():
    """Test get_tasks combining inline_filters and status through full chain."""
    task, doc = _make_task_and_doc()
    mock_db_manager, mock_session = _create_mock_db_manager_with_tasks([task], [doc])

    with (
        patch("obsidian_rag.mcp_server.server._get_registry") as mock_registry,
        patch(
            "obsidian_rag.mcp_server.tools.tasks.create_task_response"
        ) as mock_create_response,
    ):
        from obsidian_rag.mcp_server.models import TaskResponse

        mock_registry.return_value = MagicMock(db_manager=mock_db_manager)
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
        result = get_tasks(
            status=["not_completed"],
            inline_filters=inline_filters,
        )

    assert result["total_count"] == 1
    assert mock_session.query.return_value.filter.call_count >= 1


def test_get_tasks_inline_filters_combined_with_tags():
    """Test get_tasks combining inline_filters and tag_filters through full chain."""
    task, doc = _make_task_and_doc()
    mock_db_manager, mock_session = _create_mock_db_manager_with_tasks([task], [doc])

    with (
        patch("obsidian_rag.mcp_server.server._get_registry") as mock_registry,
        patch(
            "obsidian_rag.mcp_server.tools.tasks.create_task_response"
        ) as mock_create_response,
    ):
        from obsidian_rag.mcp_server.models import TaskResponse

        mock_registry.return_value = MagicMock(db_manager=mock_db_manager)
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
        result = get_tasks(
            tag_filters={"include_tags": ["work"], "match_mode": "all"},
            inline_filters=inline_filters,
        )

    assert result["total_count"] == 1


def test_get_tasks_inline_filters_max_count_validation():
    """Test get_tasks raises ValueError when inline_filters exceed max count."""
    mock_db_manager, _ = _create_mock_db_manager_with_tasks([], [])

    with patch("obsidian_rag.mcp_server.server._get_registry") as mock_registry:
        mock_registry.return_value = MagicMock(db_manager=mock_db_manager)

        inline_filters = [
            PropertyFilter(path=f"field_{i}", operator="equals", value="test")
            for i in range(11)
        ]

        result = get_tasks(inline_filters=inline_filters)
        assert result["success"] is False
        assert "Maximum 10 inline filters allowed" in result["error"]


def test_get_tasks_inline_filters_invalid_operator():
    """Test get_tasks raises ValueError for invalid inline_filters operator."""
    mock_db_manager, _ = _create_mock_db_manager_with_tasks([], [])

    with patch("obsidian_rag.mcp_server.server._get_registry") as mock_registry:
        mock_registry.return_value = MagicMock(db_manager=mock_db_manager)

        mock_filter = MagicMock()
        mock_filter.path = "vendor"
        mock_filter.operator = "invalid_operator"
        mock_filter.value = "acme"
        inline_filters = [mock_filter]

        result = get_tasks(inline_filters=inline_filters)
        assert result["success"] is False
        assert "Invalid operator" in result["error"]


def test_get_tasks_inline_filters_dot_path_rejected():
    """Test get_tasks raises ValueError for dot notation in inline_filters path."""
    mock_db_manager, _ = _create_mock_db_manager_with_tasks([], [])

    with patch("obsidian_rag.mcp_server.server._get_registry") as mock_registry:
        mock_registry.return_value = MagicMock(db_manager=mock_db_manager)

        inline_filters = [
            PropertyFilter(path="vendor.region", operator="equals", value="us-east")
        ]

        result = get_tasks(inline_filters=inline_filters)
        assert result["success"] is False
        assert "flat keys" in result["error"]
