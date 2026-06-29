"""Tests for Alembic migration 007: rename custom_metadata to inline_fields."""

import importlib.util
import logging
from pathlib import Path

import pytest
from alembic.operations import Operations
from alembic.runtime.migration import MigrationContext
from sqlalchemy import (
    JSON,
    Column,
    Integer,
    MetaData,
    Table,
    create_engine,
    inspect,
    text,
)
from sqlalchemy.pool import StaticPool

log = logging.getLogger(__name__)


@pytest.fixture
def sqlite_engine():
    """Create an in-memory SQLite engine with a tasks table."""
    engine = create_engine(
        "sqlite:///:memory:",
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )
    metadata = MetaData()
    Table(
        "tasks",
        metadata,
        Column("id", Integer, primary_key=True),
        Column("custom_metadata", JSON, nullable=True),
    )
    metadata.create_all(engine)
    yield engine
    engine.dispose()


@pytest.fixture
def migration_op(sqlite_engine):
    """Create an alembic Operations instance bound to the SQLite engine."""
    conn = sqlite_engine.connect()
    ctx = MigrationContext.configure(conn)
    op = Operations(ctx)
    yield op
    conn.close()


def _load_migration():
    """Load migration 007 as a module."""
    migration_path = Path(
        "alembic/versions/007_rename_custom_metadata_to_inline_fields.py"
    )
    spec = importlib.util.spec_from_file_location(
        "migration_007",
        str(migration_path),
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestMigration007RenameCustomMetadata:
    """Test migration 007 renames custom_metadata to inline_fields."""

    def test_migration_007_upgrade_renames_column(
        self,
        sqlite_engine,
        migration_op,
    ):
        """Verify inline_fields column exists after upgrade."""
        _msg = "test_migration_007_upgrade_renames_column starting"
        log.debug(_msg)

        migration = _load_migration()
        migration.op = migration_op
        migration.upgrade()

        inspector = inspect(sqlite_engine)
        columns = {col["name"] for col in inspector.get_columns("tasks")}
        assert "inline_fields" in columns
        assert "custom_metadata" not in columns

        _msg = "test_migration_007_upgrade_renames_column returning"
        log.debug(_msg)

    def test_migration_007_downgrade_renames_back(
        self,
        sqlite_engine,
        migration_op,
    ):
        """Verify custom_metadata column exists after downgrade."""
        _msg = "test_migration_007_downgrade_renames_back starting"
        log.debug(_msg)

        migration = _load_migration()
        migration.op = migration_op
        migration.upgrade()
        migration.downgrade()

        inspector = inspect(sqlite_engine)
        columns = {col["name"] for col in inspector.get_columns("tasks")}
        assert "custom_metadata" in columns
        assert "inline_fields" not in columns

        _msg = "test_migration_007_downgrade_renames_back returning"
        log.debug(_msg)

    def test_migration_007_idempotent_upgrade(
        self,
        sqlite_engine,
        migration_op,
    ):
        """Verify running upgrade twice does not fail."""
        _msg = "test_migration_007_idempotent_upgrade starting"
        log.debug(_msg)

        migration = _load_migration()
        migration.op = migration_op
        migration.upgrade()
        migration.upgrade()

        inspector = inspect(sqlite_engine)
        columns = {col["name"] for col in inspector.get_columns("tasks")}
        assert "inline_fields" in columns

        _msg = "test_migration_007_idempotent_upgrade returning"
        log.debug(_msg)

    def test_migration_007_preserves_data(
        self,
        sqlite_engine,
        migration_op,
    ):
        """Verify JSONB data in custom_metadata survives rename to inline_fields."""
        _msg = "test_migration_007_preserves_data starting"
        log.debug(_msg)

        with sqlite_engine.connect() as conn:
            conn.execute(
                text("INSERT INTO tasks (custom_metadata) VALUES (:data)"),
                {"data": '{"key": "value"}'},
            )
            conn.commit()

        migration = _load_migration()
        migration.op = migration_op
        migration.upgrade()

        with sqlite_engine.connect() as conn:
            result = conn.execute(
                text("SELECT inline_fields FROM tasks WHERE id = 1"),
            ).scalar()
            assert result == '{"key": "value"}'

        _msg = "test_migration_007_preserves_data returning"
        log.debug(_msg)

    def test_migration_007_idempotent_downgrade(
        self,
        sqlite_engine,
        migration_op,
    ):
        """Verify running downgrade twice does not fail."""
        _msg = "test_migration_007_idempotent_downgrade starting"
        log.debug(_msg)

        migration = _load_migration()
        migration.op = migration_op
        migration.upgrade()
        migration.downgrade()
        migration.downgrade()

        inspector = inspect(sqlite_engine)
        columns = {col["name"] for col in inspector.get_columns("tasks")}
        assert "custom_metadata" in columns

        _msg = "test_migration_007_idempotent_downgrade returning"
        log.debug(_msg)

    def test_migration_007_downgrade_preserves_data(
        self,
        sqlite_engine,
        migration_op,
    ):
        """Verify JSONB data survives downgrade back to custom_metadata."""
        _msg = "test_migration_007_downgrade_preserves_data starting"
        log.debug(_msg)

        with sqlite_engine.connect() as conn:
            conn.execute(
                text("INSERT INTO tasks (custom_metadata) VALUES (:data)"),
                {"data": '{"key": "value"}'},
            )
            conn.commit()

        migration = _load_migration()
        migration.op = migration_op
        migration.upgrade()
        migration.downgrade()

        with sqlite_engine.connect() as conn:
            result = conn.execute(
                text("SELECT custom_metadata FROM tasks WHERE id = 1"),
            ).scalar()
            assert result == '{"key": "value"}'

        _msg = "test_migration_007_downgrade_preserves_data returning"
        log.debug(_msg)

    def test_migration_007_preserves_null(
        self,
        sqlite_engine,
        migration_op,
    ):
        """Verify NULL values survive column rename."""
        _msg = "test_migration_007_preserves_null starting"
        log.debug(_msg)

        with sqlite_engine.connect() as conn:
            conn.execute(
                text("INSERT INTO tasks (custom_metadata) VALUES (NULL)"),
            )
            conn.commit()

        migration = _load_migration()
        migration.op = migration_op
        migration.upgrade()

        with sqlite_engine.connect() as conn:
            result = conn.execute(
                text("SELECT inline_fields FROM tasks WHERE id = 1"),
            ).scalar()
            assert result is None

        _msg = "test_migration_007_preserves_null returning"
        log.debug(_msg)

    def test_migration_007_round_trip(
        self,
        sqlite_engine,
        migration_op,
    ):
        """Verify upgrade -> downgrade -> upgrade preserves data."""
        _msg = "test_migration_007_round_trip starting"
        log.debug(_msg)

        with sqlite_engine.connect() as conn:
            conn.execute(
                text("INSERT INTO tasks (custom_metadata) VALUES (:data)"),
                {"data": '{"a": 1}'},
            )
            conn.commit()

        migration = _load_migration()
        migration.op = migration_op
        migration.upgrade()
        migration.downgrade()
        migration.upgrade()

        with sqlite_engine.connect() as conn:
            result = conn.execute(
                text("SELECT inline_fields FROM tasks WHERE id = 1"),
            ).scalar()
            assert result == '{"a": 1}'

        inspector = inspect(sqlite_engine)
        columns = {col["name"] for col in inspector.get_columns("tasks")}
        assert "inline_fields" in columns
        assert "custom_metadata" not in columns

        _msg = "test_migration_007_round_trip returning"
        log.debug(_msg)

    def test_migration_007_preserves_multiple_rows(
        self,
        sqlite_engine,
        migration_op,
    ):
        """Verify multiple rows survive column rename."""
        _msg = "test_migration_007_preserves_multiple_rows starting"
        log.debug(_msg)

        with sqlite_engine.connect() as conn:
            conn.execute(
                text("INSERT INTO tasks (custom_metadata) VALUES (:d1)"),
                {"d1": '{"row": 1}'},
            )
            conn.execute(
                text("INSERT INTO tasks (custom_metadata) VALUES (:d2)"),
                {"d2": '{"row": 2}'},
            )
            conn.execute(
                text("INSERT INTO tasks (custom_metadata) VALUES (NULL)"),
            )
            conn.commit()

        migration = _load_migration()
        migration.op = migration_op
        migration.upgrade()

        with sqlite_engine.connect() as conn:
            results = (
                conn.execute(
                    text("SELECT inline_fields FROM tasks ORDER BY id"),
                )
                .scalars()
                .all()
            )
            assert results[0] == '{"row": 1}'
            assert results[1] == '{"row": 2}'
            assert results[2] is None

        _msg = "test_migration_007_preserves_multiple_rows returning"
        log.debug(_msg)
