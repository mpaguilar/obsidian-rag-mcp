"""Drop kind column from documents table.

Revision ID: 004
Revises: 003
Create Date: 2026-03-09

"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy import inspect

# revision identifiers, used by Alembic.
revision = "004"
down_revision = "003_add_vaults_table"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Drop kind column from documents table."""
    # Check if column exists before dropping (idempotent)
    conn = op.get_bind()
    inspector = inspect(conn)
    columns = [col["name"] for col in inspector.get_columns("documents")]

    if "kind" in columns:
        op.drop_column("documents", "kind")


def downgrade() -> None:
    """Add kind column back to documents table."""
    # Check if column doesn't exist before adding (idempotent)
    conn = op.get_bind()
    inspector = inspect(conn)
    columns = [col["name"] for col in inspector.get_columns("documents")]

    if "kind" not in columns:
        op.add_column(
            "documents",
            sa.Column("kind", sa.Text(), nullable=True),
        )
