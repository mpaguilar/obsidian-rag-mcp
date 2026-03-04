"""Pydantic models for MCP request/response schemas."""

import logging
import uuid
from datetime import date, datetime
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

log = logging.getLogger(__name__)


class TaskResponse(BaseModel):
    """Response model for a single task.

    Attributes:
        id: Unique task identifier (UUID).
        raw_text: Full verbatim text of the task line.
        status: Task status (not_completed, completed, in_progress, cancelled).
        description: Task description without metadata.
        due: Due date for the task (or None).
        priority: Task priority level (highest, high, normal, low, lowest).
        tags: List of tags extracted from the task.
        document_path: Path to the parent document.
        document_name: Name of the parent document.

    """

    id: uuid.UUID
    raw_text: str
    status: str
    description: str
    due: date | None
    priority: str
    tags: list[str]
    document_path: str
    document_name: str


class TaskListResponse(BaseModel):
    """Response model for task list queries with pagination.

    Attributes:
        results: List of task responses.
        total_count: Total number of matching tasks.
        has_more: Whether more results are available.
        next_offset: Offset for the next page (or None if no more results).

    """

    results: list[TaskResponse]
    total_count: int
    has_more: bool
    next_offset: int | None


class DocumentResponse(BaseModel):
    """Response model for a single document.

    Attributes:
        id: Unique document identifier (UUID).
        file_path: Absolute path to the document file.
        file_name: Name of the document file.
        content: Full text content of the document.
        kind: Document kind from FrontMatter (or None).
        tags: List of tags from FrontMatter.
        similarity_score: Cosine distance score (lower is better).
        created_at_fs: Filesystem creation timestamp.
        modified_at_fs: Filesystem modification timestamp.

    """

    id: uuid.UUID
    file_path: str
    file_name: str
    content: str
    kind: str | None
    tags: list[str]
    similarity_score: float
    created_at_fs: datetime
    modified_at_fs: datetime


class DocumentListResponse(BaseModel):
    """Response model for document list queries with pagination.

    Attributes:
        results: List of document responses.
        total_count: Total number of matching documents.
        has_more: Whether more results are available.
        next_offset: Offset for the next page (or None if no more results).

    """

    results: list[DocumentResponse]
    total_count: int
    has_more: bool
    next_offset: int | None


class TagListResponse(BaseModel):
    """Response model for tag list queries with pagination.

    Attributes:
        tags: List of unique tag strings.
        total_count: Total number of unique tags.
        has_more: Whether more results are available.
        next_offset: Offset for the next page (or None if no more results).

    """

    tags: list[str]
    total_count: int
    has_more: bool
    next_offset: int | None


class HealthResponse(BaseModel):
    """Response model for health check endpoint.

    Attributes:
        status: Health status string ("healthy" or "unhealthy").
        version: Application version string.
        database: Database connectivity status.

    """

    status: str
    version: str
    database: str


def _validate_limit(limit: int) -> int:
    """Validate and clamp limit parameter.

    Args:
        limit: The requested limit value.

    Returns:
        Validated limit (clamped between 1 and 100).

    """
    if limit < 1:
        return 1
    if limit > 100:
        return 100
    return limit


def _validate_offset(offset: int) -> int:
    """Validate offset parameter.

    Args:
        offset: The requested offset value.

    Returns:
        Validated offset (minimum 0).

    """
    if offset < 0:
        return 0
    return offset


if TYPE_CHECKING:
    from obsidian_rag.database.models import Document as DocumentModel
    from obsidian_rag.database.models import Task as TaskModel


def create_task_response(
    task: "TaskModel",
    document: "DocumentModel",
) -> TaskResponse:
    """Create a TaskResponse from database models.

    Args:
        task: Task model instance.
        document: Document model instance.

    Returns:
        TaskResponse populated from the models.

    """
    _msg = "Creating TaskResponse from models"
    log.debug(_msg)

    return TaskResponse(
        id=task.id,
        raw_text=task.raw_text,
        status=task.status,
        description=task.description,
        due=task.due,
        priority=task.priority,
        tags=task.tags or [],
        document_path=document.file_path,
        document_name=document.file_name,
    )


def create_document_response(
    document: "DocumentModel",
    similarity_score: float,
) -> DocumentResponse:
    """Create a DocumentResponse from database model.

    Args:
        document: Document model instance.
        similarity_score: Cosine distance score from vector search.

    Returns:
        DocumentResponse populated from the model.

    """
    _msg = "Creating DocumentResponse from model"
    log.debug(_msg)

    return DocumentResponse(
        id=document.id,
        file_path=document.file_path,
        file_name=document.file_name,
        content=document.content,
        kind=document.kind,
        tags=document.tags or [],
        similarity_score=similarity_score,
        created_at_fs=document.created_at_fs,
        modified_at_fs=document.modified_at_fs,
    )
