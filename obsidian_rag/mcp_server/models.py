"""Pydantic models for MCP request/response schemas."""

import logging
import urllib.parse
import uuid
from datetime import date, datetime
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, Field

log = logging.getLogger(__name__)

# Constants for validation
MAX_PAGINATION_LIMIT = 100


class PropertyFilter(BaseModel):
    """Filter for document frontmatter properties.

    Attributes:
        path: Property path using dot notation (e.g., "author.name").
        operator: Comparison operator (equals, contains, exists, in, starts_with, regex).
        value: Value to compare against (optional for 'exists' operator).

    """

    path: str = Field(
        ...,
        description="Property path using dot notation (e.g., 'author.name')",
    )
    operator: Literal["equals", "contains", "exists", "in", "starts_with", "regex"] = (
        Field(
            ...,
            description="Comparison operator",
        )
    )
    value: str | int | float | bool | None | list[str | int | float] = Field(
        default=None,
        description="Value to compare against (not needed for 'exists' operator, can be list for 'in' operator)",
    )


class TagFilter(BaseModel):
    """Filter for document tags with include/exclude semantics.

    Attributes:
        include_tags: List of tags that documents must have.
        exclude_tags: List of tags that documents must NOT have.
        match_mode: Whether document must have ALL or ANY of the include_tags.

    """

    include_tags: list[str] = Field(
        default_factory=list,
        description="Tags that documents must have",
    )
    exclude_tags: list[str] = Field(
        default_factory=list,
        description="Tags that documents must NOT have",
    )
    match_mode: Literal["all", "any"] = Field(
        default="all",
        description="Whether document must have ALL or ANY of the include_tags",
    )


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


class VaultResponse(BaseModel):
    """Response model for a single vault.

    Attributes:
        id: Unique vault identifier (UUID).
        name: Vault name.
        description: Optional vault description.
        host_path: Path on host system for link construction.
        document_count: Number of documents in the vault.

    """

    id: uuid.UUID
    name: str
    description: str | None
    host_path: str
    document_count: int


class VaultListResponse(BaseModel):
    """Response model for vault list queries with pagination.

    Attributes:
        results: List of vault responses.
        total_count: Total number of vaults.
        has_more: Whether more results are available.
        next_offset: Offset for the next page (or None if no more results).

    """

    results: list[VaultResponse]
    total_count: int
    has_more: bool
    next_offset: int | None


class DocumentResponse(BaseModel):
    """Response model for a single document.

    Attributes:
        id: Unique document identifier (UUID).
        vault_name: Name of the vault containing this document.
        file_path: Relative path from vault root.
        relative_path: Relative path from vault root (same as file_path).
        file_name: Name of the document file.
        content: Full text content of the document.
        kind: Document kind from FrontMatter (or None).
        tags: List of tags from FrontMatter.
        similarity_score: Cosine distance score (lower is better).
        created_at_fs: Filesystem creation timestamp.
        modified_at_fs: Filesystem modification timestamp.
        obsidian_uri: Obsidian URI for opening the document.

    """

    id: uuid.UUID
    vault_name: str
    file_path: str
    relative_path: str
    file_name: str
    content: str
    kind: str | None
    tags: list[str]
    similarity_score: float
    created_at_fs: datetime
    modified_at_fs: datetime
    obsidian_uri: str


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


class SessionMetrics(BaseModel):
    """Session metrics for health check endpoint.

    Attributes:
        total_created: Total number of sessions created.
        total_destroyed: Total number of sessions destroyed.
        active_count: Current number of active sessions.
        total_requests: Total number of requests processed.
        peak_concurrent: Peak number of concurrent sessions.
        connection_rate: Average connections per second.
        active_sessions_by_ip: Sessions per client IP.

    """

    total_created: int = 0
    total_destroyed: int = 0
    active_count: int = 0
    total_requests: int = 0
    peak_concurrent: int = 0
    connection_rate: float = 0.0
    active_sessions_by_ip: dict[str, int] = Field(default_factory=dict)


class HealthResponse(BaseModel):
    """Response model for health check endpoint.

    Attributes:
        status: Health status string ("healthy" or "unhealthy").
        version: Application version string.
        database: Database connectivity status.
        sessions: Session metrics and statistics.

    """

    status: str
    version: str
    database: str
    sessions: SessionMetrics = Field(default_factory=SessionMetrics)


def _validate_limit(limit: int) -> int:
    """Validate and clamp limit parameter.

    Args:
        limit: The requested limit value.

    Returns:
        Validated limit (clamped between 1 and 100).

    """
    _msg = "_validate_limit starting"
    log.debug(_msg)
    if limit < 1:
        _msg = "_validate_limit returning"
        log.debug(_msg)
        return 1
    if limit > MAX_PAGINATION_LIMIT:
        _msg = "_validate_limit returning"
        log.debug(_msg)
        return 100
    _msg = "_validate_limit returning"
    log.debug(_msg)
    return limit


def _validate_offset(offset: int) -> int:
    """Validate offset parameter.

    Args:
        offset: The requested offset value.

    Returns:
        Validated offset (minimum 0).

    """
    _msg = "_validate_offset starting"
    log.debug(_msg)
    if offset < 0:
        _msg = "_validate_offset returning"
        log.debug(_msg)
        return 0
    _msg = "_validate_offset returning"
    log.debug(_msg)
    return offset


if TYPE_CHECKING:
    from obsidian_rag.database.models import Document as DocumentModel
    from obsidian_rag.database.models import Task as TaskModel
    from obsidian_rag.database.models import Vault as VaultModel


def _build_obsidian_uri(vault_name: str, relative_path: str) -> str:
    """Build Obsidian URI for opening a document.

    Args:
        vault_name: Name of the vault.
        relative_path: Relative path to the document.

    Returns:
        Obsidian URI string.

    """
    # URL-encode vault name and file path
    encoded_vault = urllib.parse.quote(vault_name)
    encoded_path = urllib.parse.quote(relative_path)
    return f"obsidian://open?vault={encoded_vault}&file={encoded_path}"


def create_vault_response(
    vault: "VaultModel",
    document_count: int,
) -> VaultResponse:
    """Create a VaultResponse from database model.

    Args:
        vault: Vault model instance.
        document_count: Number of documents in the vault.

    Returns:
        VaultResponse populated from the model.

    """
    _msg = "create_vault_response starting"
    log.debug(_msg)

    result = VaultResponse(
        id=vault.id,
        name=vault.name,
        description=vault.description,
        host_path=vault.host_path,
        document_count=document_count,
    )
    _msg = "create_vault_response returning"
    log.debug(_msg)
    return result


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
    _msg = "create_task_response starting"
    log.debug(_msg)

    result = TaskResponse(
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
    _msg = "create_task_response returning"
    log.debug(_msg)
    return result


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
    _msg = "create_document_response starting"
    log.debug(_msg)

    # Get vault name from the vault relationship
    vault_name = document.vault.name if document.vault else "Unknown"
    relative_path = document.file_path

    # Build Obsidian URI
    obsidian_uri = _build_obsidian_uri(vault_name, relative_path)

    result = DocumentResponse(
        id=document.id,
        vault_name=vault_name,
        file_path=relative_path,
        relative_path=relative_path,
        file_name=document.file_name,
        content=document.content,
        kind=document.kind,
        tags=document.tags or [],
        similarity_score=similarity_score,
        created_at_fs=document.created_at_fs,
        modified_at_fs=document.modified_at_fs,
        obsidian_uri=obsidian_uri,
    )
    _msg = "create_document_response returning"
    log.debug(_msg)
    return result
