"""Pydantic models for MCP request/response schemas."""

import logging
import urllib.parse
import uuid
from datetime import date, datetime
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, Field

log = logging.getLogger(__name__)

# Constants for validation
MAX_PAGINATION_LIMIT = 10000


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
        properties: Parent document's frontmatter key-value pairs (excluding tags), or None.
        inline_fields: Inline field key-value pairs from the task (includes well-known fields like due, priority after re-ingestion).

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
    properties: dict[str, object] | None = None
    inline_fields: dict[str, str] | None = None


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
        container_path: Path inside container/Docker.
        host_path: Path on host system for link construction.
        document_count: Number of documents in the vault.
        created_at: Timestamp when the vault was created.
        ingest_status: Current ingest state ('idle', 'in_progress', or 'failed').
        ingest_started_at: Timestamp the current ingest started (or None).
        ingest_pid: OS PID of the ingest process (or None).
        ingest_force: Whether the running ingest is a force re-ingest.

    """

    id: uuid.UUID
    name: str
    description: str | None
    container_path: str
    host_path: str
    document_count: int
    created_at: datetime
    ingest_status: str = "idle"
    ingest_started_at: datetime | None = None
    ingest_pid: int | None = None
    ingest_force: bool = False


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
        matching_chunk: Text of the best matching chunk when found via chunk search.
        created_at_fs: Filesystem creation timestamp.
        modified_at_fs: Filesystem modification timestamp.
        obsidian_uri: Obsidian URI for opening the document.
        properties: Frontmatter key-value pairs (excluding tags), or None.

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
    matching_chunk: str | None = None
    created_at_fs: datetime
    modified_at_fs: datetime
    obsidian_uri: str
    properties: dict[str, object] | None = None


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


class OutputFileConfig(BaseModel):
    """Configuration for writing tool results to an output file.

    Attributes:
        type: Target type — "local" for filesystem, "s3" for S3-compatible endpoint.
        path: Local filesystem path (required when type="local"). Must be under /tmp/.
        endpoint: S3 endpoint URL (required when type="s3").
        bucket: S3 bucket name (required when type="s3").
        key: S3 object key (required when type="s3").
        access_key_id: S3 access key (required when type="s3"). Never logged.
        secret_access_key: S3 secret key (required when type="s3"). Never logged.
        addressing_style: S3 addressing style. "virtual" (default) uses AWS
            virtual-hosted style where the bucket is part of the hostname
            (bucket.endpoint/key). "path" uses path-style where the bucket is
            part of the URL path (endpoint/bucket/key); REQUIRED for non-AWS
            S3-compatible services such as Garage and MinIO. Clients targeting
            Garage/MinIO MUST set addressing_style="path".
        region: Optional SigV4 signing region override. Highest precedence in the
            region resolution chain: per-call region → app-config default
            (OBSIDIAN_RAG_MCP_OUTPUT_FILE_S3_REGION) → URL-derived region →
            GetBucketLocation probe (non-AWS endpoints only) → us-east-1 fallback.
            Set this for Garage/MinIO/Ceph endpoints whose configured region
            is not derivable from the hostname (e.g. 'garage', 'eu-west-1').
    """

    type: Literal["local", "s3"]
    path: str | None = None
    endpoint: str | None = None
    bucket: str | None = None
    key: str | None = None
    access_key_id: str | None = None
    secret_access_key: str | None = None
    addressing_style: str | None = "virtual"
    region: str | None = Field(
        default=None,
        description=(
            "Optional SigV4 signing region override. Highest precedence in the "
            "region resolution chain: per-call region → app-config default "
            "(OBSIDIAN_RAG_MCP_OUTPUT_FILE_S3_REGION) → URL-derived region → "
            "GetBucketLocation probe (non-AWS endpoints only) → us-east-1 fallback. "
            "Set this for Garage/MinIO/Ceph endpoints whose configured region "
            "is not derivable from the hostname (e.g. 'garage', 'eu-west-1')."
        ),
    )


class OutputFileResult(BaseModel):
    """Compact summary returned to LLM context after writing result to output file.

    Attributes:
        type: "local" or "s3" — same as OutputFileConfig.type.
        path: Local path written (present when type="local").
        bucket: S3 bucket (present when type="s3").
        key: S3 object key (present when type="s3").
        bytes: Number of bytes written.
        item_count: Number of result items written.
    """

    type: Literal["local", "s3"]
    path: str | None = None
    bucket: str | None = None
    key: str | None = None
    bytes: int
    item_count: int


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
        Validated limit (clamped between 1 and 10000).

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
        return MAX_PAGINATION_LIMIT
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
        container_path=vault.container_path,
        host_path=vault.host_path,
        document_count=document_count,
        created_at=vault.created_at,
        ingest_status=getattr(vault, "ingest_status", "idle"),
        ingest_started_at=getattr(vault, "ingest_started_at", None),
        ingest_pid=getattr(vault, "ingest_pid", None),
        ingest_force=getattr(vault, "ingest_force", False),
    )
    _msg = "create_vault_response returning"
    log.debug(_msg)
    return result


def create_task_response(
    task: "TaskModel",
    document: "DocumentModel",
    *,
    include_content: bool = True,
) -> TaskResponse:
    """Create a TaskResponse from database models.

    Args:
        task: Task model instance.
        document: Document model instance.
        include_content: When False, the returned raw_text is an empty string.

    Returns:
        TaskResponse populated from the models.

    """
    _msg = "create_task_response starting"
    log.debug(_msg)

    raw_text = task.raw_text if include_content else ""

    properties: dict[str, object] | None = None
    if document.frontmatter_json:
        properties = {
            key: value
            for key, value in document.frontmatter_json.items()
            if key != "tags"
        }

    result = TaskResponse(
        id=task.id,
        raw_text=raw_text,
        status=task.status,
        description=task.description,
        due=task.due,
        priority=task.priority,
        tags=task.tags or [],
        document_path=document.file_path,
        document_name=document.file_name,
        properties=properties,
        inline_fields=getattr(task, "inline_fields", None),
    )
    _msg = "create_task_response returning"
    log.debug(_msg)
    return result


def create_document_response(
    document: "DocumentModel",
    similarity_score: float,
    matching_chunk: str | None = None,
    *,
    include_content: bool = True,
) -> DocumentResponse:
    """Create a DocumentResponse from database model.

    Args:
        document: Document model instance.
        similarity_score: Cosine distance score from vector search.
        matching_chunk: Text of the best matching chunk (optional).
        include_content: When False, the returned content is an empty string.

    Returns:
        DocumentResponse populated from the model.

    Notes:
        The 'kind' field is now derived from frontmatter_json for backward compatibility.

    """
    _msg = "create_document_response starting"
    log.debug(_msg)

    # Get vault name from the vault relationship
    vault_name = document.vault.name if document.vault else "Unknown"
    relative_path = document.file_path

    # Build Obsidian URI
    obsidian_uri = _build_obsidian_uri(vault_name, relative_path)

    # Derive kind from frontmatter_json for backward compatibility
    kind = None
    if document.frontmatter_json:
        kind = document.frontmatter_json.get("kind")

    # Build properties from frontmatter_json, excluding the tags key
    properties: dict[str, object] | None = None
    if document.frontmatter_json is not None:
        properties = {
            key: value
            for key, value in document.frontmatter_json.items()
            if key != "tags"
        }

    content = document.content if include_content else ""

    result = DocumentResponse(
        id=document.id,
        vault_name=vault_name,
        file_path=relative_path,
        relative_path=relative_path,
        file_name=document.file_name,
        content=content,
        kind=kind,
        tags=document.tags or [],
        similarity_score=similarity_score,
        matching_chunk=matching_chunk,
        created_at_fs=document.created_at_fs,
        modified_at_fs=document.modified_at_fs,
        obsidian_uri=obsidian_uri,
        properties=properties,
    )
    _msg = "create_document_response returning"
    log.debug(_msg)
    return result
