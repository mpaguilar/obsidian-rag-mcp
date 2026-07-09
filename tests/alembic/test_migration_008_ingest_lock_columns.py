"""Tests for Alembic migration 008: add ingest lock columns to vaults table."""

import importlib.util
import logging
from pathlib import Path

import pytest
from alembic.operations import Operations
from alembic.runtime.migration import MigrationContext
from sqlalchemy import (
    Column,
    DateTime,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    create_engine,
    inspect,
    text,
)
from sqlalchemy.pool import StaticPool

log = logging.getLogger(__name__)

_NEW_COLUMN_NAMES = ["ingest_status", "ingest_started_at", "ingest_pid", "ingest_force"]


@pytest.fixture
def sqlite_engine():
    """Create an in-memory SQLite engine with a vaults table."""
    engine = create_engine(
        "sqlite:///:memory:",
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )
    metadata = MetaData()
    Table(
        "vaults",
        metadata,
        Column("id", Integer, primary_key=True),
        Column("name", String(100)),
        Column("container_path", Text),
        Column("host_path", Text),
        Column("created_at", DateTime),
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
    """Load migration 008 as a module."""
    migration_path = Path("alembic/versions/008_add_ingest_lock_columns.py")
    spec = importlib.util.spec_from_file_location(
        "migration_008",
        str(migration_path),
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_migration_008_upgrade_adds_four_columns(sqlite_engine, migration_op):
    """After upgrade, inspect().get_columns("vaults") contains all 4 names."""
    _msg = "test_migration_008_upgrade_adds_four_columns starting"
    log.debug(_msg)

    migration = _load_migration()
    migration.op = migration_op
    migration.upgrade()

    inspector = inspect(sqlite_engine)
    columns = {col["name"] for col in inspector.get_columns("vaults")}
    for name in _NEW_COLUMN_NAMES:
        assert name in columns

    _msg = "test_migration_008_upgrade_adds_four_columns returning"
    log.debug(_msg)


def test_migration_008_upgrade_idempotent(sqlite_engine, migration_op):
    """Run upgrade twice, columns still present exactly once (no error)."""
    _msg = "test_migration_008_upgrade_idempotent starting"
    log.debug(_msg)

    migration = _load_migration()
    migration.op = migration_op
    migration.upgrade()
    migration.upgrade()

    inspector = inspect(sqlite_engine)
    columns = [col["name"] for col in inspector.get_columns("vaults")]
    for name in _NEW_COLUMN_NAMES:
        assert columns.count(name) == 1

    _msg = "test_migration_008_upgrade_idempotent returning"
    log.debug(_msg)


def test_migration_008_downgrade_removes_columns(sqlite_engine, migration_op):
    """After downgrade, 4 names absent."""
    _msg = "test_migration_008_downgrade_removes_columns starting"
    log.debug(_msg)

    migration = _load_migration()
    migration.op = migration_op
    migration.upgrade()
    migration.downgrade()

    inspector = inspect(sqlite_engine)
    columns = {col["name"] for col in inspector.get_columns("vaults")}
    for name in _NEW_COLUMN_NAMES:
        assert name not in columns

    _msg = "test_migration_008_downgrade_removes_columns returning"
    log.debug(_msg)


def test_migration_008_downgrade_idempotent(sqlite_engine, migration_op):
    """Downgrade twice, no error."""
    _msg = "test_migration_008_downgrade_idempotent starting"
    log.debug(_msg)

    migration = _load_migration()
    migration.op = migration_op
    migration.upgrade()
    migration.downgrade()
    migration.downgrade()

    inspector = inspect(sqlite_engine)
    columns = {col["name"] for col in inspector.get_columns("vaults")}
    for name in _NEW_COLUMN_NAMES:
        assert name not in columns

    _msg = "test_migration_008_downgrade_idempotent returning"
    log.debug(_msg)


def test_migration_008_round_trip(sqlite_engine, migration_op):
    """Upgrade -> downgrade -> upgrade -> columns present."""
    _msg = "test_migration_008_round_trip starting"
    log.debug(_msg)

    migration = _load_migration()
    migration.op = migration_op
    migration.upgrade()
    migration.downgrade()
    migration.upgrade()

    inspector = inspect(sqlite_engine)
    columns = {col["name"] for col in inspector.get_columns("vaults")}
    for name in _NEW_COLUMN_NAMES:
        assert name in columns

    _msg = "test_migration_008_round_trip returning"
    log.debug(_msg)


def test_migration_008_ingest_status_defaults_idle(sqlite_engine, migration_op):
    """Insert a row post-upgrade, read back ingest_status='idle'."""
    _msg = "test_migration_008_ingest_status_defaults_idle starting"
    log.debug(_msg)

    migration = _load_migration()
    migration.op = migration_op
    migration.upgrade()

    with sqlite_engine.connect() as conn:
        conn.execute(
            text("INSERT INTO vaults (name, container_path) VALUES (:name, :path)"),
            {"name": "TestVault", "path": "/data/test"},
        )
        conn.commit()

    with sqlite_engine.connect() as conn:
        result = conn.execute(
            text("SELECT ingest_status FROM vaults WHERE id = 1"),
        ).scalar()
        assert result == "idle"

    _msg = "test_migration_008_ingest_status_defaults_idle returning"
    log.debug(_msg)


def test_migration_008_ingest_force_defaults_false(sqlite_engine, migration_op):
    """Insert row, read back ingest_force is False/0."""
    _msg = "test_migration_008_ingest_force_defaults_false starting"
    log.debug(_msg)

    migration = _load_migration()
    migration.op = migration_op
    migration.upgrade()

    with sqlite_engine.connect() as conn:
        conn.execute(
            text("INSERT INTO vaults (name, container_path) VALUES (:name, :path)"),
            {"name": "TestVault", "path": "/data/test"},
        )
        conn.commit()

    with sqlite_engine.connect() as conn:
        result = conn.execute(
            text("SELECT ingest_force FROM vaults WHERE id = 1"),
        ).scalar()
        assert result is False or result == 0

    _msg = "test_migration_008_ingest_force_defaults_false returning"
    log.debug(_msg)


def test_migration_008_nullable_columns_accept_null(sqlite_engine, migration_op):
    """ingest_started_at and ingest_pid accept NULL."""
    _msg = "test_migration_008_nullable_columns_accept_null starting"
    log.debug(_msg)

    migration = _load_migration()
    migration.op = migration_op
    migration.upgrade()

    with sqlite_engine.connect() as conn:
        conn.execute(
            text("INSERT INTO vaults (name, container_path) VALUES (:name, :path)"),
            {"name": "TestVault", "path": "/data/test"},
        )
        conn.commit()

    with sqlite_engine.connect() as conn:
        started_at = conn.execute(
            text("SELECT ingest_started_at FROM vaults WHERE id = 1"),
        ).scalar()
        pid = conn.execute(
            text("SELECT ingest_pid FROM vaults WHERE id = 1"),
        ).scalar()
        assert started_at is None
        assert pid is None

    _msg = "test_migration_008_nullable_columns_accept_null returning"
    log.debug(_msg)
