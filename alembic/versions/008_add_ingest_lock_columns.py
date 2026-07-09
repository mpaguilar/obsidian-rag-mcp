"""Add ingest lock columns to vaults table."""

import sqlalchemy as sa

from alembic import op

revision = "008"
down_revision = "007"
branch_labels = None
depends_on = None

_NEW_COLUMNS = [
    ("ingest_status", sa.String(20), "idle", False),
    ("ingest_started_at", sa.DateTime(timezone=True), None, True),
    ("ingest_pid", sa.Integer, None, True),
    ("ingest_force", sa.Boolean, sa.false(), False),
]


def upgrade() -> None:
    """Add ingest lock columns to vaults."""
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    existing = {c["name"] for c in inspector.get_columns("vaults")}
    for name, type_, server_default, nullable in _NEW_COLUMNS:
        if name not in existing:
            op.add_column(
                "vaults",
                sa.Column(name, type_, nullable=nullable, server_default=server_default),
            )


def downgrade() -> None:
    """Drop ingest lock columns from vaults."""
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    existing = {c["name"] for c in inspector.get_columns("vaults")}
    for name, *_ in _NEW_COLUMNS:
        if name in existing:
            op.drop_column("vaults", name)
