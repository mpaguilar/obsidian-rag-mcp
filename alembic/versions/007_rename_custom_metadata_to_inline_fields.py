"""Rename custom_metadata to inline_fields on tasks table."""

from alembic import op
from sqlalchemy import inspect

# revision identifiers, used by Alembic.
revision = "007"
down_revision = "006"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Rename custom_metadata column to inline_fields."""
    inspector = inspect(op.get_bind())
    columns = {col["name"] for col in inspector.get_columns("tasks")}
    if "custom_metadata" in columns:
        op.alter_column("tasks", "custom_metadata", new_column_name="inline_fields")


def downgrade() -> None:
    """Rename inline_fields column back to custom_metadata."""
    inspector = inspect(op.get_bind())
    columns = {col["name"] for col in inspector.get_columns("tasks")}
    if "inline_fields" in columns:
        op.alter_column("tasks", "inline_fields", new_column_name="custom_metadata")
