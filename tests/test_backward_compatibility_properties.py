"""Backward compatibility tests for properties field and include_content parameter.

These tests verify that existing code continues to work after the introduction
of the `properties` field on response models and the `include_content` parameter
on tool functions.
"""

import uuid
from datetime import datetime
from unittest.mock import MagicMock

from obsidian_rag.mcp_server.handlers import GetTasksRequest
from obsidian_rag.mcp_server.models import (
    DocumentResponse,
    TaskResponse,
    create_document_response,
    create_task_response,
)
from obsidian_rag.mcp_server.tools.documents_params import (
    GetDocumentParams,
    ListDocumentsParams,
    PaginationParams,
)
from obsidian_rag.mcp_server.tools.tasks_params import (
    GetTasksFilterParams,
    GetTasksRequest as TasksParamsGetTasksRequest,
)


class _MockVault:
    def __init__(self, name: str = "Test Vault") -> None:
        self.name = name


class _MockDocument:
    def __init__(
        self,
        frontmatter_json: dict[str, object] | None = None,
        content: str = "# Content",
    ) -> None:
        self.id = uuid.uuid4()
        self.file_path = "path/to/doc.md"
        self.file_name = "doc.md"
        self.content = content
        self.frontmatter_json = frontmatter_json
        self.tags = ["tag1", "tag2"]
        self.created_at_fs = datetime.now()
        self.modified_at_fs = datetime.now()
        self.vault = _MockVault()


def test_existing_document_response_tests_pass_with_properties_none() -> None:
    """DocumentResponse without properties should default to None."""
    now = datetime.now()
    response = DocumentResponse(
        id=uuid.uuid4(),
        vault_name="Vault",
        file_path="path/doc.md",
        relative_path="path/doc.md",
        file_name="doc.md",
        content="content",
        kind=None,
        tags=[],
        similarity_score=0.0,
        created_at_fs=now,
        modified_at_fs=now,
        obsidian_uri="obsidian://open?vault=Vault&file=path%2Fdoc.md",
    )

    assert response.properties is None


def test_existing_task_response_tests_pass_with_properties_none() -> None:
    """TaskResponse without properties should default to None."""
    response = TaskResponse(
        id=uuid.uuid4(),
        raw_text="- [ ] task",
        status="not_completed",
        description="task",
        due=None,
        priority="normal",
        tags=[],
        document_path="notes/task.md",
        document_name="task.md",
    )

    assert response.properties is None


def test_create_document_response_backward_compatible_no_include_content() -> None:
    """create_document_response without include_content should include content."""
    doc = _MockDocument(content="original content")

    response = create_document_response(doc, 0.25)  # type: ignore[arg-type]

    assert response.content == "original content"


def test_create_task_response_backward_compatible_no_include_content() -> None:
    """create_task_response without include_content should include raw_text."""
    task = MagicMock()
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

    response = create_task_response(task, document)

    assert response.raw_text == "- [ ] secret task"


def test_properties_none_is_not_in_json_output_when_not_set() -> None:
    """When properties is None, exclude_none=True omits it from JSON."""
    now = datetime.now()
    response = DocumentResponse(
        id=uuid.uuid4(),
        vault_name="Vault",
        file_path="path/doc.md",
        relative_path="path/doc.md",
        file_name="doc.md",
        content="content",
        kind=None,
        tags=[],
        similarity_score=0.0,
        created_at_fs=now,
        modified_at_fs=now,
        obsidian_uri="obsidian://open?vault=Vault&file=path%2Fdoc.md",
    )

    data = response.model_dump(exclude_none=True)
    assert "properties" not in data


def test_include_content_true_is_default_all_tools() -> None:
    """All tool parameter dataclasses default include_content to True."""
    pagination = PaginationParams(limit=20, offset=0)
    assert pagination.include_content is True

    get_doc_params = GetDocumentParams()
    assert get_doc_params.include_content is True

    list_doc_params = ListDocumentsParams()
    assert list_doc_params.include_content is True

    tasks_filter = GetTasksFilterParams()
    assert tasks_filter.include_content is True

    tasks_request = TasksParamsGetTasksRequest()
    assert tasks_request.include_content is True

    handler_input = GetTasksRequest()
    assert handler_input.include_content is True

    handler_request = GetTasksRequest()
    assert handler_request.include_content is True
