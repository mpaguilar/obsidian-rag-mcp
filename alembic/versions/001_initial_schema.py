"""Initial migration - create documents and tasks tables.

Revision ID: 001
Revises:
Create Date: 2026-03-02

"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

from obsidian_rag.config import get_settings

# revision identifiers, used by Alembic.
revision = "001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create initial schema with documents and tasks tables."""
    # Create pgvector extension
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    # Create documents table
    op.create_table(
        "documents",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("file_path", sa.Text(), nullable=False),
        sa.Column("file_name", sa.Text(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column(
            "content_vector",
            sa.Text(),  # Created as TEXT, will be altered to VECTOR type below
            nullable=True,
        ),
        sa.Column("checksum_md5", sa.String(length=32), nullable=False),
        sa.Column("created_at_fs", sa.DateTime(), nullable=False),
        sa.Column("modified_at_fs", sa.DateTime(), nullable=False),
        sa.Column("ingested_at", sa.DateTime(), nullable=False),
        sa.Column("kind", sa.Text(), nullable=True),
        sa.Column("tags", postgresql.ARRAY(sa.Text()), nullable=True),
        sa.Column("frontmatter_json", postgresql.JSONB(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("file_path"),
    )
    op.create_index("ix_documents_file_name", "documents", ["file_name"])
    op.create_index("ix_documents_file_path", "documents", ["file_path"])

    # Create tasks table
    op.create_table(
        "tasks",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("document_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("line_number", sa.Integer(), nullable=False),
        sa.Column("raw_text", sa.Text(), nullable=False),
        sa.Column("status", sa.String(length=20), nullable=False),
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column("tags", postgresql.ARRAY(sa.Text()), nullable=True),
        sa.Column("repeat", sa.Text(), nullable=True),
        sa.Column("scheduled", sa.Date(), nullable=True),
        sa.Column("due", sa.Date(), nullable=True),
        sa.Column("completion", sa.Date(), nullable=True),
        sa.Column("priority", sa.String(length=10), nullable=False),
        sa.Column("custom_metadata", postgresql.JSONB(), nullable=True),
        sa.ForeignKeyConstraint(
            ["document_id"],
            ["documents.id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("document_id", "line_number", name="uq_task_document_line"),
    )
    op.create_index("ix_tasks_document_id", "tasks", ["document_id"])

    # Alter content_vector column to VECTOR type with configurable dimension
    settings = get_settings()
    dimension = settings.database.vector_dimension
    op.execute(
        f"ALTER TABLE documents ALTER COLUMN content_vector TYPE vector({dimension}) USING content_vector::vector({dimension})"
    )

    # Create vector index for documents using HNSW
    # HNSW and IVFFLAT both have 2000 dimension limits in pgvector
    op.execute(
        "CREATE INDEX ix_documents_content_vector ON documents USING hnsw (content_vector vector_cosine_ops)"
    )


def downgrade() -> None:
    """Drop tables."""
    op.drop_index("ix_tasks_document_id", table_name="tasks")
    op.drop_table("tasks")
    op.drop_index("ix_documents_file_path", table_name="documents")
    op.drop_index("ix_documents_file_name", table_name="documents")
    op.execute("DROP INDEX IF EXISTS ix_documents_content_vector")
    op.drop_table("documents")
    # Note: We don't drop the vector extension because it requires superuser
    # privileges. The extension can remain installed without issues.
