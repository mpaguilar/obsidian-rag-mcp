"""Tests for Alembic migration downgrade chain integrity.

These tests verify that the migration chain can be properly upgraded
and downgraded without errors.
"""

import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from alembic import command
from alembic.config import Config

log = logging.getLogger(__name__)


@pytest.fixture
def alembic_config(tmp_path):
    """Create Alembic configuration for testing.

    Args:
        tmp_path: Pytest fixture providing a temporary path.

    Returns:
        Config: Alembic configuration object.
    """
    _msg = "alembic_config fixture starting"
    log.debug(_msg)

    # Create alembic.ini content with PostgreSQL URL
    alembic_ini = tmp_path / "alembic.ini"
    alembic_ini.write_text("""
[alembic]
script_location = alembic
prepend_sys_path = .
version_path_separator = os
sqlalchemy.url = postgresql+psycopg://localhost/test

[post_write_hooks]
""")

    config = Config(str(alembic_ini))

    _msg = "alembic_config fixture returning"
    log.debug(_msg)

    return config


@pytest.fixture
def mock_postgresql_engine():
    """Create mock PostgreSQL engine for testing.

    Returns:
        MagicMock: Mocked SQLAlchemy engine instance.
    """
    _msg = "mock_postgresql_engine fixture starting"
    log.debug(_msg)

    engine = MagicMock()
    engine.url = "postgresql+psycopg://localhost/test"

    _msg = "mock_postgresql_engine fixture returning"
    log.debug(_msg)

    return engine


class TestMigration002Downgrade:
    """Test migration 002 downgrade uses IF EXISTS pattern."""

    def test_migration_002_file_uses_if_exists(self):
        """Verify migration 002 uses IF EXISTS when dropping indexes."""
        _msg = "test_migration_002_file_uses_if_exists starting"
        log.debug(_msg)

        migration_path = Path("alembic/versions/002_add_vault_root_and_indexes.py")
        content = migration_path.read_text()

        # Verify IF EXISTS pattern is used
        assert (
            'op.execute("DROP INDEX IF EXISTS ix_documents_vault_root")' in content
        ), "Migration 002 should use IF EXISTS when dropping ix_documents_vault_root"
        assert 'op.execute("DROP INDEX IF EXISTS ix_documents_tags")' in content, (
            "Migration 002 should use IF EXISTS when dropping ix_documents_tags"
        )

        _msg = "test_migration_002_file_uses_if_exists returning"
        log.debug(_msg)


class TestMigration003Downgrade:
    """Test migration 003 downgrade recreates indexes."""

    def test_migration_003_file_recreates_indexes(self):
        """Verify migration 003 recreates indexes in downgrade."""
        _msg = "test_migration_003_file_recreates_indexes starting"
        log.debug(_msg)

        migration_path = Path("alembic/versions/003_add_vaults_table.py")
        content = migration_path.read_text()

        # Verify indexes are recreated after adding vault_root column
        assert (
            'op.execute(\n        "CREATE INDEX IF NOT EXISTS ix_documents_vault_root ON documents (vault_root)"\n    )'
            in content
        ), "Migration 003 should recreate ix_documents_vault_root index in downgrade"
        assert (
            'op.execute(\n        "CREATE INDEX IF NOT EXISTS ix_documents_tags ON documents USING gin (tags)"\n    )'
            in content
        ), "Migration 003 should recreate ix_documents_tags index in downgrade"

        _msg = "test_migration_003_file_recreates_indexes returning"
        log.debug(_msg)


class TestMigrationChainIntegrity:
    """Test full migration chain can be upgraded and downgraded."""

    @pytest.mark.skip(reason="Requires PostgreSQL with pgvector for full test")
    def test_full_upgrade_downgrade_cycle(self, alembic_config, mock_postgresql_engine):
        """Test upgrade head -> downgrade base -> upgrade head cycle.

        Args:
            alembic_config: Alembic configuration fixture.
            mock_postgresql_engine: Mock PostgreSQL engine fixture.

        Notes:
            This test requires PostgreSQL with pgvector extension.
            Skipped in CI as it requires actual PostgreSQL database.
        """
        _msg = "test_full_upgrade_downgrade_cycle starting"
        log.debug(_msg)

        # Configure alembic to use our mock engine
        alembic_config.attributes["connection"] = mock_postgresql_engine.connect()

        # Upgrade to head
        _msg = "Upgrading to head"
        log.debug(_msg)
        command.upgrade(alembic_config, "head")

        # Verify we're at head
        # Note: This would need actual database inspection

        # Downgrade to base
        _msg = "Downgrading to base"
        log.debug(_msg)
        command.downgrade(alembic_config, "base")

        # Upgrade back to head
        _msg = "Upgrading back to head"
        log.debug(_msg)
        command.upgrade(alembic_config, "head")

        _msg = "test_full_upgrade_downgrade_cycle returning"
        log.debug(_msg)


class TestMigrationSyntax:
    """Test migration files have valid Python syntax."""

    def test_migration_002_syntax(self):
        """Verify migration 002 has valid Python syntax."""
        _msg = "test_migration_002_syntax starting"
        log.debug(_msg)

        migration_path = Path("alembic/versions/002_add_vault_root_and_indexes.py")
        content = migration_path.read_text()

        # Compile to check syntax
        compile(content, str(migration_path), "exec")

        _msg = "test_migration_002_syntax returning"
        log.debug(_msg)

    def test_migration_003_syntax(self):
        """Verify migration 003 has valid Python syntax."""
        _msg = "test_migration_003_syntax starting"
        log.debug(_msg)

        migration_path = Path("alembic/versions/003_add_vaults_table.py")
        content = migration_path.read_text()

        # Compile to check syntax
        compile(content, str(migration_path), "exec")

        _msg = "test_migration_003_syntax returning"
        log.debug(_msg)
