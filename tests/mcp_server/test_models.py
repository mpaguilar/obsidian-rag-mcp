"""Unit tests for MCP models module."""

import uuid
from datetime import date, datetime

import pytest

from obsidian_rag.mcp_server.models import (
    DocumentListResponse,
    DocumentResponse,
    HealthResponse,
    TaskListResponse,
    TaskResponse,
    _validate_limit,
    _validate_offset,
    create_document_response,
    create_task_response,
)


class TestTaskResponse:
    """Tests for TaskResponse model."""

    def test_task_response_creation(self):
        """Test creating a TaskResponse with all fields."""
        task_id = uuid.uuid4()
        response = TaskResponse(
            id=task_id,
            raw_text="- [ ] Test task #tag",
            status="not_completed",
            description="Test task",
            due=date(2025, 3, 15),
            priority="high",
            tags=["tag"],
            document_path="/path/to/doc.md",
            document_name="doc.md",
        )

        assert response.id == task_id
        assert response.raw_text == "- [ ] Test task #tag"
        assert response.status == "not_completed"
        assert response.description == "Test task"
        assert response.due == date(2025, 3, 15)
        assert response.priority == "high"
        assert response.tags == ["tag"]
        assert response.document_path == "/path/to/doc.md"
        assert response.document_name == "doc.md"

    def test_task_response_optional_fields(self):
        """Test TaskResponse with optional fields as None."""
        task_id = uuid.uuid4()
        response = TaskResponse(
            id=task_id,
            raw_text="- [ ] Test task",
            status="completed",
            description="Test task",
            due=None,
            priority="normal",
            tags=[],
            document_path="/path/to/doc.md",
            document_name="doc.md",
        )

        assert response.due is None
        assert response.tags == []


class TestTaskListResponse:
    """Tests for TaskListResponse model."""

    def test_task_list_response(self):
        """Test creating a TaskListResponse."""
        task = TaskResponse(
            id=uuid.uuid4(),
            raw_text="- [ ] Test",
            status="not_completed",
            description="Test",
            due=None,
            priority="normal",
            tags=[],
            document_path="/path",
            document_name="test.md",
        )

        response = TaskListResponse(
            results=[task],
            total_count=1,
            has_more=False,
            next_offset=None,
        )

        assert len(response.results) == 1
        assert response.total_count == 1
        assert response.has_more is False
        assert response.next_offset is None

    def test_task_list_response_with_pagination(self):
        """Test TaskListResponse with pagination."""
        tasks = [
            TaskResponse(
                id=uuid.uuid4(),
                raw_text=f"- [ ] Task {i}",
                status="not_completed",
                description=f"Task {i}",
                due=None,
                priority="normal",
                tags=[],
                document_path="/path",
                document_name="test.md",
            )
            for i in range(2)
        ]

        response = TaskListResponse(
            results=tasks,
            total_count=10,
            has_more=True,
            next_offset=2,
        )

        assert len(response.results) == 2
        assert response.total_count == 10
        assert response.has_more is True
        assert response.next_offset == 2


class TestDocumentResponse:
    """Tests for DocumentResponse model."""

    def test_document_response_creation(self):
        """Test creating a DocumentResponse with all fields."""
        doc_id = uuid.uuid4()
        now = datetime.now()
        response = DocumentResponse(
            id=doc_id,
            file_path="/path/to/doc.md",
            file_name="doc.md",
            content="# Test Content",
            kind="article",
            tags=["tag1", "tag2"],
            similarity_score=0.15,
            created_at_fs=now,
            modified_at_fs=now,
        )

        assert response.id == doc_id
        assert response.file_path == "/path/to/doc.md"
        assert response.file_name == "doc.md"
        assert response.content == "# Test Content"
        assert response.kind == "article"
        assert response.tags == ["tag1", "tag2"]
        assert response.similarity_score == 0.15
        assert response.created_at_fs == now
        assert response.modified_at_fs == now

    def test_document_response_optional_fields(self):
        """Test DocumentResponse with optional fields as None."""
        doc_id = uuid.uuid4()
        now = datetime.now()
        response = DocumentResponse(
            id=doc_id,
            file_path="/path/to/doc.md",
            file_name="doc.md",
            content="Content",
            kind=None,
            tags=[],
            similarity_score=0.5,
            created_at_fs=now,
            modified_at_fs=now,
        )

        assert response.kind is None
        assert response.tags == []


class TestDocumentListResponse:
    """Tests for DocumentListResponse model."""

    def test_document_list_response(self):
        """Test creating a DocumentListResponse."""
        now = datetime.now()
        doc = DocumentResponse(
            id=uuid.uuid4(),
            file_path="/path/doc.md",
            file_name="doc.md",
            content="Content",
            kind=None,
            tags=[],
            similarity_score=0.1,
            created_at_fs=now,
            modified_at_fs=now,
        )

        response = DocumentListResponse(
            results=[doc],
            total_count=1,
            has_more=False,
            next_offset=None,
        )

        assert len(response.results) == 1
        assert response.total_count == 1


class TestHealthResponse:
    """Tests for HealthResponse model."""

    def test_health_response(self):
        """Test creating a HealthResponse."""
        response = HealthResponse(
            status="healthy",
            version="0.2.3",
            database="connected",
        )

        assert response.status == "healthy"
        assert response.version == "0.2.3"
        assert response.database == "connected"


class TestValidateLimit:
    """Tests for _validate_limit function."""

    def test_valid_limit(self):
        """Test valid limit values."""
        assert _validate_limit(1) == 1
        assert _validate_limit(50) == 50
        assert _validate_limit(100) == 100

    def test_limit_below_minimum(self):
        """Test limit below minimum is clamped to 1."""
        assert _validate_limit(0) == 1
        assert _validate_limit(-10) == 1

    def test_limit_above_maximum(self):
        """Test limit above maximum is clamped to 100."""
        assert _validate_limit(101) == 100
        assert _validate_limit(1000) == 100


class TestValidateOffset:
    """Tests for _validate_offset function."""

    def test_valid_offset(self):
        """Test valid offset values."""
        assert _validate_offset(0) == 0
        assert _validate_offset(10) == 10
        assert _validate_offset(100) == 100

    def test_negative_offset(self):
        """Test negative offset is clamped to 0."""
        assert _validate_offset(-1) == 0
        assert _validate_offset(-100) == 0


class MockTask:
    """Mock task object for testing."""

    def __init__(self):
        self.id = uuid.uuid4()
        self.raw_text = "- [ ] Test task"
        self.status = "not_completed"
        self.description = "Test task"
        self.due = date(2025, 3, 15)
        self.priority = "high"
        self.tags = ["tag1"]


class MockDocument:
    """Mock document object for testing."""

    def __init__(self):
        self.id = uuid.uuid4()
        self.file_path = "/path/to/doc.md"
        self.file_name = "doc.md"
        self.content = "# Content"
        self.kind = "article"
        self.tags = ["tag1", "tag2"]
        self.created_at_fs = datetime.now()
        self.modified_at_fs = datetime.now()


class TestCreateTaskResponse:
    """Tests for create_task_response function."""

    def test_create_task_response(self):
        """Test creating a TaskResponse from mock models."""
        task = MockTask()
        doc = MockDocument()

        response = create_task_response(task, doc)

        assert response.id == task.id
        assert response.raw_text == task.raw_text
        assert response.status == task.status
        assert response.description == task.description
        assert response.due == task.due
        assert response.priority == task.priority
        assert response.tags == task.tags
        assert response.document_path == doc.file_path
        assert response.document_name == doc.file_name


class TestCreateDocumentResponse:
    """Tests for create_document_response function."""

    def test_create_document_response(self):
        """Test creating a DocumentResponse from mock model."""
        doc = MockDocument()
        similarity = 0.25

        response = create_document_response(doc, similarity)

        assert response.id == doc.id
        assert response.file_path == doc.file_path
        assert response.file_name == doc.file_name
        assert response.content == doc.content
        assert response.kind == doc.kind
        assert response.tags == doc.tags
        assert response.similarity_score == similarity
        assert response.created_at_fs == doc.created_at_fs
        assert response.modified_at_fs == doc.modified_at_fs
