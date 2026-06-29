"""Tests for get_tasks include_content parameter."""

import uuid
from unittest.mock import MagicMock, patch

from obsidian_rag.mcp_server.models import TaskResponse
from obsidian_rag.mcp_server.tools.tasks import get_tasks
from obsidian_rag.mcp_server.tools.tasks_params import GetTasksFilterParams


def _create_mock_session(results: list[tuple[MagicMock, MagicMock]]) -> MagicMock:
    """Create a mock session returning the given (task, document) tuples."""
    mock_session = MagicMock()
    mock_session.bind.dialect.name = "postgresql"

    mock_query = MagicMock()
    mock_query.join.return_value = mock_query
    mock_query.filter.return_value = mock_query
    mock_query.order_by.return_value = mock_query
    mock_query.limit.return_value = mock_query
    mock_query.offset.return_value = mock_query
    mock_query.count.return_value = len(results)
    mock_query.all.return_value = results

    mock_session.query.return_value = mock_query
    return mock_session


def _create_task(
    raw_text: str = "- [ ] Test task",
    inline_fields: dict[str, str] | None = None,
) -> MagicMock:
    """Create a mock Task model with common attributes."""
    task = MagicMock()
    task.id = uuid.uuid4()
    task.status = "not_completed"
    task.raw_text = raw_text
    task.description = "Test task"
    task.tags = []
    task.priority = "normal"
    task.due = None
    task.scheduled = None
    task.completion = None
    task.inline_fields = inline_fields
    return task


def _create_document(frontmatter_json: dict[str, object] | None = None) -> MagicMock:
    """Create a mock Document model with optional frontmatter."""
    doc = MagicMock()
    doc.id = uuid.uuid4()
    doc.file_path = "/test/doc.md"
    doc.file_name = "doc.md"
    doc.tags = []
    doc.frontmatter_json = frontmatter_json
    return doc


def test_get_tasks_include_content_true_default() -> None:
    """Test that include_content=True is passed by default."""
    task = _create_task()
    doc = _create_document()
    mock_session = _create_mock_session([(task, doc)])

    with patch(
        "obsidian_rag.mcp_server.tools.tasks.create_task_response"
    ) as mock_create_response:
        mock_response = TaskResponse(
            id=task.id,
            raw_text=task.raw_text,
            status=task.status,
            description=task.description,
            due=task.due,
            priority=task.priority,
            tags=task.tags,
            document_path=doc.file_path,
            document_name=doc.file_name,
        )
        mock_create_response.return_value = mock_response

        filters = GetTasksFilterParams()
        result = get_tasks(mock_session, filters)

    assert result.total_count == 1
    assert len(result.results) == 1
    mock_create_response.assert_called_once_with(
        task,
        doc,
        include_content=True,
    )


def test_get_tasks_include_content_false_empty_raw_text() -> None:
    """Test that include_content=False returns an empty raw_text."""
    task = _create_task(raw_text="- [ ] Something important")
    doc = _create_document()
    mock_session = _create_mock_session([(task, doc)])

    filters = GetTasksFilterParams(include_content=False)
    result = get_tasks(mock_session, filters)

    assert result.total_count == 1
    assert len(result.results) == 1
    assert result.results[0].raw_text == ""


def test_get_tasks_include_content_false_keeps_description() -> None:
    """Test that description is preserved when include_content=False."""
    task = _create_task(raw_text="- [ ] Something important")
    task.description = "Important task"
    doc = _create_document()
    mock_session = _create_mock_session([(task, doc)])

    filters = GetTasksFilterParams(include_content=False)
    result = get_tasks(mock_session, filters)

    assert result.results[0].description == "Important task"


def test_get_tasks_include_content_false_keeps_properties() -> None:
    """Test that properties are populated when include_content=False."""
    task = _create_task()
    doc = _create_document({"author": "Alice", "tags": ["work"]})
    mock_session = _create_mock_session([(task, doc)])

    filters = GetTasksFilterParams(include_content=False)
    result = get_tasks(mock_session, filters)

    assert result.results[0].properties == {"author": "Alice"}


def test_get_tasks_properties_populated_from_document_frontmatter() -> None:
    """Test that properties come from document frontmatter."""
    task = _create_task()
    frontmatter = {"project": "obsidian-rag", "status": "active", "tags": ["note"]}
    doc = _create_document(frontmatter)
    mock_session = _create_mock_session([(task, doc)])

    filters = GetTasksFilterParams(include_content=True)
    result = get_tasks(mock_session, filters)

    assert result.results[0].properties == {
        "project": "obsidian-rag",
        "status": "active",
    }


def test_get_tasks_properties_none_no_frontmatter() -> None:
    """Test that properties are None when document has no frontmatter."""
    task = _create_task()
    doc = _create_document(frontmatter_json=None)
    mock_session = _create_mock_session([(task, doc)])

    filters = GetTasksFilterParams(include_content=True)
    result = get_tasks(mock_session, filters)

    assert result.results[0].properties is None


def test_get_tasks_include_content_false_keeps_inline_fields() -> None:
    """Test that inline_fields is populated when include_content=False."""
    task = _create_task(
        raw_text="- [ ] Something important",
        inline_fields={"due": "2026-01-01", "priority": "high"},
    )
    doc = _create_document()
    mock_session = _create_mock_session([(task, doc)])

    filters = GetTasksFilterParams(include_content=False)
    result = get_tasks(mock_session, filters)

    assert result.results[0].raw_text == ""
    assert result.results[0].inline_fields == {"due": "2026-01-01", "priority": "high"}


def test_get_tasks_include_content_true_keeps_inline_fields() -> None:
    """Test that inline_fields is populated when include_content=True."""
    task = _create_task(
        raw_text="- [ ] Something important",
        inline_fields={"repeat": "daily", "custom": "value"},
    )
    doc = _create_document()
    mock_session = _create_mock_session([(task, doc)])

    filters = GetTasksFilterParams(include_content=True)
    result = get_tasks(mock_session, filters)

    assert result.results[0].raw_text == "- [ ] Something important"
    assert result.results[0].inline_fields == {"repeat": "daily", "custom": "value"}


def test_get_tasks_inline_fields_none_when_task_has_none() -> None:
    """Test that inline_fields is None when task has no inline_fields."""
    task = _create_task(inline_fields=None)
    doc = _create_document()
    mock_session = _create_mock_session([(task, doc)])

    filters = GetTasksFilterParams(include_content=False)
    result = get_tasks(mock_session, filters)

    assert result.results[0].inline_fields is None


def test_get_tasks_inline_fields_preserved_with_properties() -> None:
    """Test that inline_fields and properties are both preserved when include_content=False."""
    task = _create_task(
        raw_text="- [ ] Something important",
        inline_fields={"scheduled": "2026-01-01", "repeat": "weekly"},
    )
    doc = _create_document({"author": "Alice", "tags": ["work"]})
    mock_session = _create_mock_session([(task, doc)])

    filters = GetTasksFilterParams(include_content=False)
    result = get_tasks(mock_session, filters)

    response = result.results[0]
    assert response.raw_text == ""
    assert response.inline_fields == {"scheduled": "2026-01-01", "repeat": "weekly"}
    assert response.properties == {"author": "Alice"}
