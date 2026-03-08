"""Add vaults table and update documents for vault support.

Revision ID: 003_add_vaults_table
Revises: 002_add_vault_root_and_indexes
Create Date: 2026-03-07 00:00:00.000000

"""

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

# revision identifiers, used by Alembic.
revision = "003_add_vaults_table"
down_revision = "002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create vaults table and update documents table."""
    # Create vaults table
    op.create_table(
        "vaults",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            nullable=False,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column("name", sa.String(100), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("container_path", sa.Text(), nullable=False),
        sa.Column("host_path", sa.Text(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("NOW()"),
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("name"),
    )
    op.create_index("ix_vaults_name", "vaults", ["name"])

    # Add vault_id column to documents (nullable initially for migration)
    op.add_column(
        "documents",
        sa.Column(
            "vault_id",
            postgresql.UUID(as_uuid=True),
            nullable=True,
        ),
    )
    op.create_index("ix_documents_vault_id", "documents", ["vault_id"])
    op.create_foreign_key(
        "fk_documents_vault_id",
        "documents",
        "vaults",
        ["vault_id"],
        ["id"],
        ondelete="CASCADE",
    )

    # Remove vault_root column from documents
    op.drop_column("documents", "vault_root")

    # Update unique constraint on documents to be per-vault
    op.drop_constraint("documents_file_path_key", "documents", type_="unique")
    op.create_unique_constraint(
        "uq_document_vault_path",
        "documents",
        ["vault_id", "file_path"],
    )


def downgrade() -> None:
    """Revert vaults table creation and document updates."""
    # Restore unique constraint on file_path only
    op.drop_constraint("uq_document_vault_path", "documents", type_="unique")
    op.create_unique_constraint(
        "documents_file_path_key",
        "documents",
        ["file_path"],
    )

    # Add back vault_root column
    op.add_column(
        "documents",
        sa.Column("vault_root", sa.Text(), nullable=True),
    )

    # Recreate indexes that were dropped when vault_root column was removed
    # These indexes are needed for migration 002's state to be valid
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_documents_vault_root ON documents (vault_root)"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_documents_tags ON documents USING gin (tags)"
    )

    # Remove foreign key and vault_id column
    op.drop_constraint("fk_documents_vault_id", "documents", type_="foreignkey")
    op.drop_index("ix_documents_vault_id", "documents")
    op.drop_column("documents", "vault_id")

    # Drop vaults table
    op.drop_index("ix_vaults_name", "vaults")
    op.drop_table("vaults")
