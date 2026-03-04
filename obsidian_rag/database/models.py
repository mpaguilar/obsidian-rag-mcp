"""Database models for obsidian-rag.

This module defines the SQLAlchemy models for documents and tasks,
including the pg_vector extension for vector embeddings.
"""

import logging
import uuid
from datetime import datetime
from enum import Enum as PyEnum
from typing import TYPE_CHECKING, Any, List

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    JSON,
    Date,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    TypeDecorator,
    UniqueConstraint,
    event,
    text,
)
from sqlalchemy.dialects.postgresql import UUID, ARRAY as PG_ARRAY
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.types import TypeEngine

if TYPE_CHECKING:
    from sqlalchemy.engine import Connection, Dialect

log = logging.getLogger(__name__)


class ArrayType(TypeDecorator[List[str]]):
    """Platform-independent array type.

    Uses PostgreSQL ARRAY type when available, falls back to JSON for SQLite.
    This allows testing with SQLite while using proper PostgreSQL arrays in production.

    Attributes:
        impl: The underlying type implementation (JSON for fallback).

    """

    impl = JSON
    cache_ok = True

    def load_dialect_impl(self, dialect: "Dialect") -> TypeEngine[Any]:
        """Load the appropriate implementation for the dialect.

        Args:
            dialect: The SQLAlchemy dialect in use.

        Returns:
            The type implementation for the dialect (ARRAY for PostgreSQL, JSON otherwise).

        """
        if dialect.name == "postgresql":
            return dialect.type_descriptor(PG_ARRAY(Text))
        return dialect.type_descriptor(JSON())


# Vector dimension - configurable, default 1536 for OpenAI embeddings
VECTOR_DIMENSION = 1536


class Base(DeclarativeBase):
    """Base class for all models."""

    pass


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


class Document(Base):
    """Represents an Obsidian markdown document.

    Attributes:
        id: Unique identifier (UUID).
        file_path: Absolute path to the file (unique).
        file_name: Name of the file.
        content: Full text content of the document.
        content_vector: Vector embedding of the content (pg_vector).
        checksum_md5: MD5 checksum for change detection.
        created_at_fs: Filesystem creation timestamp.
        modified_at_fs: Filesystem modification timestamp.
        ingested_at: When the document was last ingested.
        kind: Document kind from FrontMatter.
        tags: Array of tags from FrontMatter.
        frontmatter_json: All other FrontMatter properties as JSON.
        tasks: Related tasks (one-to-many relationship).

    """

    __tablename__ = "documents"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    file_path: Mapped[str] = mapped_column(
        Text, unique=True, nullable=False, index=True
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
        DateTime, nullable=False, default=datetime.now
    )
    kind: Mapped[str | None] = mapped_column(Text, nullable=True)
    tags: Mapped[list[str] | None] = mapped_column(ArrayType, nullable=True)
    frontmatter_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    # Relationships
    tasks: Mapped[list["Task"]] = relationship(
        "Task", back_populates="document", cascade="all, delete-orphan"
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
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
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


@event.listens_for(Base.metadata, "before_create")
def _create_pgvector_extension(
    target: Base,  # noqa: ARG001
    connection: "Connection",
    **kwargs,  # noqa: ARG001
) -> None:
    """Create pgvector extension before creating tables (PostgreSQL only)."""
    # Only create extension on PostgreSQL
    dialect = connection.dialect.name
    if dialect == "postgresql":  # pragma: no cover
        _msg = "Creating pgvector extension"
        log.debug(_msg)
        connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
