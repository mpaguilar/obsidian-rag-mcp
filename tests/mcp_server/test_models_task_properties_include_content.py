"""Tests for TaskResponse properties and include_content support."""

import uuid
from datetime import date
from unittest.mock import MagicMock

from obsidian_rag.mcp_server.models import create_task_response


def test_task_response_has_properties_field():
    """TaskResponse should expose a properties field."""
    task = MagicMock()
    task.inline_fields = None
    task.id = uuid.uuid4()
    task.raw_text = "- [ ] task text"
    task.status = "not_completed"
    task.description = "task text"
    task.due = None
    task.priority = "normal"
    task.tags = ["work"]
    document = MagicMock()
    document.file_path = "notes/task.md"
    document.file_name = "task.md"
    document.frontmatter_json = None

    response = create_task_response(task, document)

    assert hasattr(response, "properties")
    assert response.properties is None


def test_task_response_properties_with_dict():
    """TaskResponse should carry a dict of properties."""
    task = MagicMock()
    task.inline_fields = None
    task.id = uuid.uuid4()
    task.raw_text = "- [ ] task text"
    task.status = "not_completed"
    task.description = "task text"
    task.due = date(2026, 1, 1)
    task.priority = "high"
    task.tags = ["work"]
    document = MagicMock()
    document.file_path = "notes/task.md"
    document.file_name = "task.md"
    document.frontmatter_json = {"author": "alice", "tags": ["work"]}

    response = create_task_response(task, document)

    assert response.properties == {"author": "alice"}


def test_create_task_response_populates_properties():
    """create_task_response should populate properties from frontmatter_json."""
    task = MagicMock()
    task.inline_fields = None
    task.id = uuid.uuid4()
    task.raw_text = "- [ ] task text"
    task.status = "not_completed"
    task.description = "task text"
    task.due = None
    task.priority = "normal"
    task.tags = []
    document = MagicMock()
    document.file_path = "notes/task.md"
    document.file_name = "task.md"
    document.frontmatter_json = {"project": "obsidian", "year": 2025}

    response = create_task_response(task, document)

    assert response.properties == {"project": "obsidian", "year": 2025}


def test_create_task_response_properties_none_no_frontmatter():
    """properties should be None when frontmatter_json is absent."""
    task = MagicMock()
    task.inline_fields = None
    task.id = uuid.uuid4()
    task.raw_text = "- [ ] task text"
    task.status = "not_completed"
    task.description = "task text"
    task.due = None
    task.priority = "normal"
    task.tags = []
    document = MagicMock()
    document.file_path = "notes/task.md"
    document.file_name = "task.md"
    document.frontmatter_json = None

    response = create_task_response(task, document)

    assert response.properties is None


def test_create_task_response_properties_excludes_tags_key():
    """properties should exclude the tags key from frontmatter_json."""
    task = MagicMock()
    task.inline_fields = None
    task.id = uuid.uuid4()
    task.raw_text = "- [ ] task text"
    task.status = "not_completed"
    task.description = "task text"
    task.due = None
    task.priority = "normal"
    task.tags = []
    document = MagicMock()
    document.file_path = "notes/task.md"
    document.file_name = "task.md"
    document.frontmatter_json = {"tags": ["a", "b"], "title": "doc"}

    response = create_task_response(task, document)

    assert "tags" not in response.properties
    assert response.properties == {"title": "doc"}


def test_create_task_include_content_true_preserves_raw_text():
    """When include_content is True, raw_text should be preserved."""
    task = MagicMock()
    task.inline_fields = None
    task.id = uuid.uuid4()
    task.raw_text = "- [ ] secret task"
    task.status = "not_completed"
    task.description = "secret task"
    task.due = None
    task.priority = "normal"
    task.tags = []
    document = MagicMock()
    document.file_path = "notes/task.md"
    document.file_name = "task.md"
    document.frontmatter_json = None

    response = create_task_response(task, document, include_content=True)

    assert response.raw_text == "- [ ] secret task"


def test_create_task_include_content_false_empty_raw_text():
    """When include_content is False, raw_text should be empty."""
    task = MagicMock()
    task.inline_fields = None
    task.id = uuid.uuid4()
    task.raw_text = "- [ ] secret task"
    task.status = "not_completed"
    task.description = "secret task"
    task.due = None
    task.priority = "normal"
    task.tags = []
    document = MagicMock()
    document.file_path = "notes/task.md"
    document.file_name = "task.md"
    document.frontmatter_json = None

    response = create_task_response(task, document, include_content=False)

    assert response.raw_text == ""


def test_create_task_include_content_false_keeps_description():
    """When include_content is False, description should remain unchanged."""
    task = MagicMock()
    task.inline_fields = None
    task.id = uuid.uuid4()
    task.raw_text = "- [ ] secret task"
    task.status = "not_completed"
    task.description = "secret task"
    task.due = None
    task.priority = "normal"
    task.tags = []
    document = MagicMock()
    document.file_path = "notes/task.md"
    document.file_name = "task.md"
    document.frontmatter_json = None

    response = create_task_response(task, document, include_content=False)

    assert response.description == "secret task"


def test_create_task_include_content_default_is_true():
    """Default value for include_content should be True."""
    task = MagicMock()
    task.inline_fields = None
    task.id = uuid.uuid4()
    task.raw_text = "- [ ] default task"
    task.status = "not_completed"
    task.description = "default task"
    task.due = None
    task.priority = "normal"
    task.tags = []
    document = MagicMock()
    document.file_path = "notes/task.md"
    document.file_name = "task.md"
    document.frontmatter_json = None

    response = create_task_response(task, document)

    assert response.raw_text == "- [ ] default task"
