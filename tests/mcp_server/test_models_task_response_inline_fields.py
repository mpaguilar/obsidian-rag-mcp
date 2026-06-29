"""Tests for TaskResponse inline_fields field and create_task_response population."""

import uuid
from types import SimpleNamespace

from obsidian_rag.mcp_server.models import TaskResponse, create_task_response


def _make_task(*, inline_fields: dict[str, str] | None = None) -> SimpleNamespace:
    """Build a minimal mock task for create_task_response."""
    return SimpleNamespace(
        id=uuid.uuid4(),
        raw_text="- [ ] Test task",
        status="not_completed",
        description="Test task",
        due=None,
        priority="normal",
        tags=["work"],
        inline_fields=inline_fields,
    )


def _make_document() -> SimpleNamespace:
    """Build a minimal mock document for create_task_response."""
    return SimpleNamespace(
        file_path="notes/tasks.md",
        file_name="tasks.md",
        frontmatter_json=None,
    )


def test_task_response_has_inline_fields_field() -> None:
    """TaskResponse model includes inline_fields field."""
    response = TaskResponse(
        id=uuid.uuid4(),
        raw_text="- [ ] Test task",
        status="not_completed",
        description="Test task",
        due=None,
        priority="normal",
        tags=["work"],
        document_path="notes/tasks.md",
        document_name="tasks.md",
        inline_fields={"due": "2025-03-15"},
    )

    assert response.inline_fields == {"due": "2025-03-15"}


def test_task_response_inline_fields_default_none() -> None:
    """Default value of inline_fields is None."""
    response = TaskResponse(
        id=uuid.uuid4(),
        raw_text="- [ ] Test task",
        status="not_completed",
        description="Test task",
        due=None,
        priority="normal",
        tags=["work"],
        document_path="notes/tasks.md",
        document_name="tasks.md",
    )

    assert response.inline_fields is None


def test_create_task_response_populates_inline_fields() -> None:
    """inline_fields populated from Task.inline_fields."""
    task = _make_task(inline_fields={"due": "2025-03-15"})
    document = _make_document()

    response = create_task_response(task, document)  # type: ignore[arg-type]

    assert response.inline_fields == {"due": "2025-03-15"}


def test_create_task_response_inline_fields_none_when_task_has_none() -> None:
    """Task with inline_fields=None results in response inline_fields=None."""
    task = _make_task(inline_fields=None)
    document = _make_document()

    response = create_task_response(task, document)  # type: ignore[arg-type]

    assert response.inline_fields is None


def test_create_task_response_inline_fields_with_content_false() -> None:
    """include_content=False still populates inline_fields (REQ-005)."""
    task = _make_task(inline_fields={"priority": "high"})
    document = _make_document()

    response = create_task_response(task, document, include_content=False)  # type: ignore[arg-type]

    assert response.raw_text == ""
    assert response.inline_fields == {"priority": "high"}


def test_create_task_response_inline_fields_with_dict_values() -> None:
    """Dict with multiple keys survives serialization."""
    fields = {"due": "2025-03-15", "priority": "high", "repeat": "daily"}
    task = _make_task(inline_fields=fields)
    document = _make_document()

    response = create_task_response(task, document)  # type: ignore[arg-type]

    assert response.inline_fields == fields
    assert set(response.model_dump()["inline_fields"].keys()) == set(fields.keys())


def test_create_task_response_inline_fields_with_well_known_fields() -> None:
    """Task with well-known and custom inline_fields both appear in response."""
    task = _make_task(inline_fields={"due": "2026-03-20", "vendor": "Amazon"})
    document = _make_document()

    response = create_task_response(task, document)  # type: ignore[arg-type]

    assert response.inline_fields == {"due": "2026-03-20", "vendor": "Amazon"}


def test_create_task_response_inline_fields_empty_dict() -> None:
    """Task with inline_fields={} results in response inline_fields={}."""
    task = _make_task(inline_fields={})
    document = _make_document()

    response = create_task_response(task, document)  # type: ignore[arg-type]

    assert response.inline_fields == {}


def test_create_task_response_inline_fields_serialization() -> None:
    """Response serializes correctly via .model_dump()."""
    task = _make_task(inline_fields={"due": "2026-03-20", "vendor": "Amazon"})
    document = _make_document()

    response = create_task_response(task, document)  # type: ignore[arg-type]
    dumped = response.model_dump()

    assert dumped["inline_fields"] == {"due": "2026-03-20", "vendor": "Amazon"}
    assert "inline_fields" in dumped
