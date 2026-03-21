"""Add document_chunks table

Revision ID: 005
Revises: 004
Create Date: 2026-03-21

"""

from typing import Sequence

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects.postgresql import UUID

# Vector dimension must match VECTOR_DIMENSION in models.py
VECTOR_DIMENSION = 1536

# revision identifiers
revision: str = "005"
down_revision: str | None = "004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create document_chunks table with HNSW index."""
    # Create table
    op.create_table(
        "document_chunks",
        sa.Column("id", UUID(), nullable=False),
        sa.Column("document_id", UUID(), nullable=False),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("chunk_text", sa.Text(), nullable=False),
        sa.Column("chunk_vector", Vector(VECTOR_DIMENSION), nullable=False),
        sa.Column("start_char", sa.Integer(), nullable=False),
        sa.Column("end_char", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(
            ["document_id"],
            ["documents.id"],
            ondelete="CASCADE",
        ),
        sa.UniqueConstraint("document_id", "chunk_index"),
    )

    # Create HNSW index on chunk_vector
    op.execute("""
        CREATE INDEX idx_document_chunks_vector
        ON document_chunks USING hnsw (chunk_vector vector_cosine_ops)
    """)

    # Create index on document_id for joins
    op.create_index(
        "idx_document_chunks_document_id",
        "document_chunks",
        ["document_id"],
    )


def downgrade() -> None:
    """Drop document_chunks table."""
    op.drop_index("idx_document_chunks_document_id")
    op.execute("DROP INDEX IF EXISTS idx_document_chunks_vector")
    op.drop_table("document_chunks")
