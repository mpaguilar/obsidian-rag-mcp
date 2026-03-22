"""Add token_count and chunk_type to document_chunks.

Revision ID: 006
Revises: 005
Create Date: 2026-03-22

"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers
revision: str = "006"
down_revision: str | None = "005"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add token_count and chunk_type columns to document_chunks table."""
    # Add token_count column (nullable for backward compatibility)
    op.add_column(
        "document_chunks",
        sa.Column("token_count", sa.Integer(), nullable=True),
    )

    # Add chunk_type column (nullable, stores 'content' or 'task')
    op.add_column(
        "document_chunks",
        sa.Column("chunk_type", sa.String(length=20), nullable=True),
    )

    # Create index on chunk_type for filtering queries
    op.create_index(
        "ix_document_chunks_chunk_type",
        "document_chunks",
        ["chunk_type"],
    )


def downgrade() -> None:
    """Remove token_count and chunk_type columns."""
    # Drop index first
    op.drop_index("ix_document_chunks_chunk_type", table_name="document_chunks")

    # Drop columns
    op.drop_column("document_chunks", "chunk_type")
    op.drop_column("document_chunks", "token_count")
