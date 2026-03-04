"""Add vault_root column and indexes for document filtering.

Revision ID: 002
Revises: 001
Create Date: 2026-03-04

"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = "002"
down_revision = "001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Add vault_root column and indexes for tag filtering."""
    # Add vault_root column to documents table
    op.add_column("documents", sa.Column("vault_root", sa.Text(), nullable=True))

    # Create GIN index on tags for fast filtering
    # GIN indexes are optimized for array operations
    op.execute("CREATE INDEX ix_documents_tags ON documents USING gin (tags)")

    # Create B-tree index on vault_root for filtering by root path
    op.create_index("ix_documents_vault_root", "documents", ["vault_root"])


def downgrade() -> None:
    """Remove vault_root column and indexes."""
    # Drop indexes
    op.drop_index("ix_documents_vault_root", table_name="documents")
    op.execute("DROP INDEX IF EXISTS ix_documents_tags")

    # Drop column
    op.drop_column("documents", "vault_root")
