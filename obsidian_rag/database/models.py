"""Database models for obsidian-rag.

This module defines the SQLAlchemy models for documents and tasks,
including the pg_vector extension for vector embeddings.
"""

import logging
import uuid
from datetime import UTC, datetime
from enum import Enum as PyEnum
from typing import TYPE_CHECKING, Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    JSON,
    Date,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    TypeDecorator,
    UniqueConstraint,
    event,
    text,
)
from sqlalchemy.dialects.postgresql import ARRAY as PG_ARRAY
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.types import TypeEngine

if TYPE_CHECKING:
    from sqlalchemy.engine import Connection, Dialect

log = logging.getLogger(__name__)


class ArrayType(TypeDecorator[list[str]]):
    """PostgreSQL array type.

    Uses PostgreSQL ARRAY type for storing string arrays.
    This type is only supported with PostgreSQL databases.

    Attributes:
        impl: The underlying type implementation.

    """

    impl = JSON
    cache_ok = True

    def load_dialect_impl(self, dialect: "Dialect") -> TypeEngine[Any]:
        """Load the PostgreSQL ARRAY implementation.

        Args:
            dialect: The SQLAlchemy dialect in use.

        Returns:
            The PostgreSQL ARRAY type implementation.

        Raises:
            RuntimeError: If the dialect is not PostgreSQL.

        """
        if dialect.name == "postgresql":
            return dialect.type_descriptor(PG_ARRAY(Text))
        _msg = f"ArrayType only supports PostgreSQL, got {dialect.name}"
        raise RuntimeError(_msg)


# Vector dimension - configurable, default 1536 for OpenAI embeddings
VECTOR_DIMENSION = 1536


class Base(DeclarativeBase):
    """Base class for all models."""


class TaskStatus(PyEnum):
    """Enumeration of possible task statuses."""

    NOT_COMPLETED = "not_completed"
    COMPLETED = "completed"
    IN_PROGRESS = "in_progress"
    CANCELLED = "cancelled"


class TaskPriority(PyEnum):
    """Enumeration of possible task priorities."""

    HIGHEST = "highest"
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"
    LOWEST = "lowest"


class Vault(Base):
    """Represents an Obsidian vault.

    Attributes:
        id: Unique identifier (UUID).
        name: Vault name (unique, max 100 chars).
        description: Optional vault description.
        container_path: Path inside container/Docker for file operations.
        host_path: Path on host system for link construction.
        created_at: When the vault record was created.
        documents: Related documents (one-to-many relationship).

    """

    __tablename__ = "vaults"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    name: Mapped[str] = mapped_column(
        String(100),
        unique=True,
        nullable=False,
        index=True,
    )
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    container_path: Mapped[str] = mapped_column(Text, nullable=False)
    host_path: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(UTC),
    )

    # Relationships
    documents: Mapped[list["Document"]] = relationship(
        "Document",
        back_populates="vault",
        cascade="all, delete-orphan",
    )

    def __repr__(self) -> str:
        """Return string representation of the vault."""
        return f"<Vault(id={self.id}, name={self.name})>"


class Document(Base):
    """Represents an Obsidian markdown document.

    Attributes:
        id: Unique identifier (UUID).
        vault_id: Foreign key to the parent vault.
        file_path: Relative path from vault root (unique per vault).
        file_name: Name of the file.
        content: Full text content of the document.
        content_vector: Vector embedding of the content (pg_vector).
        checksum_md5: MD5 checksum for change detection.
        created_at_fs: Filesystem creation timestamp.
        modified_at_fs: Filesystem modification timestamp.
        ingested_at: When the document was last ingested.
        tags: Array of tags from FrontMatter.
        frontmatter_json: All FrontMatter properties as JSON (includes 'kind').
        vault: Parent vault relationship.
        tasks: Related tasks (one-to-many relationship).

    """

    __tablename__ = "documents"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    vault_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("vaults.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    file_path: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        index=True,
    )
    file_name: Mapped[str] = mapped_column(Text, nullable=False, index=True)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    content_vector: Mapped[list[float] | None] = mapped_column(
        Vector(VECTOR_DIMENSION),
        nullable=True,
    )
    checksum_md5: Mapped[str] = mapped_column(String(32), nullable=False)
    created_at_fs: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    modified_at_fs: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    ingested_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        default=lambda: datetime.now(UTC),
    )
    tags: Mapped[list[str] | None] = mapped_column(ArrayType, nullable=True)
    frontmatter_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    # Relationships
    vault: Mapped["Vault"] = relationship("Vault", back_populates="documents")
    tasks: Mapped[list["Task"]] = relationship(
        "Task",
        back_populates="document",
        cascade="all, delete-orphan",
    )
    chunks: Mapped[list["DocumentChunk"]] = relationship(
        "DocumentChunk",
        back_populates="document",
        cascade="all, delete-orphan",
    )

    __table_args__ = (
        UniqueConstraint("vault_id", "file_path", name="uq_document_vault_path"),
    )

    def __repr__(self) -> str:
        """Return string representation of the document."""
        return f"<Document(id={self.id}, file_path={self.file_path})>"


class Task(Base):
    """Represents a task extracted from a document.

    Attributes:
        id: Unique identifier (UUID).
        document_id: Foreign key to the parent document.
        line_number: Line number where the task appears.
        raw_text: Full text of the task line.
        status: Task status (not_completed, completed, in_progress, cancelled).
        description: Task description without metadata.
        tags: Array of tags extracted from the task.
        repeat: Recurrence pattern string.
        scheduled: Scheduled date for the task.
        due: Due date for the task.
        completion: Completion date for the task.
        priority: Task priority level.
        custom_metadata: Additional metadata as JSON.
        document: Parent document relationship.

    """

    __tablename__ = "tasks"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    document_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    line_number: Mapped[int] = mapped_column(Integer, nullable=False)
    raw_text: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default=TaskStatus.NOT_COMPLETED.value,
    )
    description: Mapped[str] = mapped_column(Text, nullable=False)
    tags: Mapped[list[str] | None] = mapped_column(ArrayType, nullable=True)
    repeat: Mapped[str | None] = mapped_column(Text, nullable=True)
    scheduled: Mapped[datetime | None] = mapped_column(Date, nullable=True)
    due: Mapped[datetime | None] = mapped_column(Date, nullable=True)
    completion: Mapped[datetime | None] = mapped_column(Date, nullable=True)
    priority: Mapped[str] = mapped_column(
        String(10),
        nullable=False,
        default=TaskPriority.NORMAL.value,
    )
    custom_metadata: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    # Relationships
    document: Mapped["Document"] = relationship("Document", back_populates="tasks")

    __table_args__ = (
        UniqueConstraint("document_id", "line_number", name="uq_task_document_line"),
    )

    def __repr__(self) -> str:
        """Return string representation of the task."""
        return f"<Task(id={self.id}, status={self.status}, description={self.description[:50]})>"


class DocumentChunk(Base):
    """Represents a chunk of a document for large document handling.

    Documents exceeding the embedding token limit are split into overlapping
    chunks stored in this table. Each chunk has its own vector embedding
    for semantic search.

    Attributes:
        id: Unique identifier (UUID).
        document_id: Foreign key to the parent document.
        chunk_index: Index of this chunk within the document (0-based).
        chunk_text: Text content of this chunk.
        chunk_vector: Vector embedding of the chunk text.
        start_char: Starting character position in the original document.
        end_char: Ending character position in the original document.
        token_count: Number of tokens in this chunk (for statistics).
        chunk_type: Type of chunk ('content' or 'task').
        created_at: When the chunk was created.
        document: Parent document relationship.

    """

    __tablename__ = "document_chunks"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    document_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    chunk_text: Mapped[str] = mapped_column(Text, nullable=False)
    chunk_vector: Mapped[list[float]] = mapped_column(
        Vector(VECTOR_DIMENSION),
        nullable=False,
    )
    start_char: Mapped[int] = mapped_column(Integer, nullable=False)
    end_char: Mapped[int] = mapped_column(Integer, nullable=False)
    token_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    chunk_type: Mapped[str | None] = mapped_column(String(20), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(UTC),
    )

    # Relationships
    document: Mapped["Document"] = relationship(
        "Document",
        back_populates="chunks",
    )

    __table_args__ = (
        UniqueConstraint(
            "document_id",
            "chunk_index",
            name="uq_chunk_document_index",
        ),
        Index(
            "ix_document_chunks_chunk_vector_hnsw",
            "chunk_vector",
            postgresql_using="hnsw",
            postgresql_with={
                "M": 32,
                "ef_construction": 128,
            },
            postgresql_ops={
                "chunk_vector": "vector_cosine_ops",
            },
        ),
    )

    def __repr__(self) -> str:
        """Return string representation of the chunk."""
        return (
            f"<DocumentChunk(id={self.id}, document_id={self.document_id}, "
            f"chunk_index={self.chunk_index}, start_char={self.start_char}, "
            f"end_char={self.end_char})>"
        )


@event.listens_for(Base.metadata, "before_create")
def _create_pgvector_extension(
    _target: Base,
    connection: "Connection",
    **_kwargs: object,
) -> None:
    """Create pgvector extension before creating tables (PostgreSQL only)."""
    # Only create extension on PostgreSQL
    dialect = connection.dialect.name
    if dialect == "postgresql":
        _msg = "Creating pgvector extension"
        log.debug(_msg)
        connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
