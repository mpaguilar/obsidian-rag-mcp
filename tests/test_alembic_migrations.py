"""Tests for alembic migration integrity.

This module tests that the alembic migration chain is valid
and can be traversed in both upgrade and downgrade directions.
"""

import logging
from pathlib import Path

import pytest
from alembic.config import Config
from alembic.script import ScriptDirectory

log = logging.getLogger(__name__)


class TestMigrationChain:
    """Tests for migration chain integrity."""

    def test_migration_chain_is_valid(self) -> None:
        """Test that all migrations form a valid chain.

        Verifies that:
        - All revisions can be loaded
        - The chain from base to head is valid
        - No orphaned revisions exist
        - down_revision references are resolvable
        """
        _msg = "test_migration_chain_is_valid starting"
        log.debug(_msg)

        # Load alembic configuration
        alembic_ini = Path(__file__).parent.parent / "alembic.ini"
        if not alembic_ini.exists():
            pytest.skip("alembic.ini not found")

        alembic_cfg = Config(str(alembic_ini))
        script = ScriptDirectory.from_config(alembic_cfg)

        # Get the revision map - this validates the chain
        # KeyError would be raised if down_revision references are invalid
        revisions = list(script.walk_revisions())

        # Verify we have the expected migrations
        revision_ids = {rev.revision for rev in revisions}
        expected_ids = {
            "001",
            "002",
            "003_add_vaults_table",
        }

        assert expected_ids.issubset(revision_ids), (
            f"Missing expected revisions. Found: {revision_ids}, "
            f"Expected to contain: {expected_ids}"
        )

        # Verify the chain order: 001 -> 002 -> 003_add_vaults_table
        revision_map = {rev.revision: rev for rev in revisions}

        # 001 should have no down_revision
        rev_001 = revision_map.get("001")
        assert rev_001 is not None, "Migration 001 not found"
        assert rev_001.down_revision is None, (
            "Migration 001 should have no down_revision"
        )

        # 002 should point to 001
        rev_002 = revision_map.get("002")
        assert rev_002 is not None, "Migration 002 not found"
        assert rev_002.down_revision == "001", (
            f"Migration 002 should point to 001, but points to {rev_002.down_revision}"
        )

        # 003 should point to 002
        rev_003 = revision_map.get("003_add_vaults_table")
        assert rev_003 is not None, "Migration 003_add_vaults_table not found"
        assert rev_003.down_revision == "002", (
            f"Migration 003 should point to 002, but points to {rev_003.down_revision}"
        )

        _msg = "test_migration_chain_is_valid returning"
        log.debug(_msg)

    def test_migration_002_exists(self) -> None:
        """Test that migration 002 exists in the revision chain."""
        _msg = "test_migration_002_exists starting"
        log.debug(_msg)

        alembic_ini = Path(__file__).parent.parent / "alembic.ini"
        if not alembic_ini.exists():
            pytest.skip("alembic.ini not found")

        alembic_cfg = Config(str(alembic_ini))
        script = ScriptDirectory.from_config(alembic_cfg)

        # Try to get revision
        try:
            rev_002 = script.get_revision("002")
        except KeyError as err:
            pytest.fail(f"Migration 002 not found: {err}")

        assert rev_002 is not None, "Migration 002 should exist"
        assert rev_002.revision == "002", (
            f"Expected revision ID '002', got '{rev_002.revision}'"
        )

        _msg = "test_migration_002_exists returning"
        log.debug(_msg)


class TestMigrationFiles:
    """Tests for migration file structure and content."""

    def test_migration_002_file_has_correct_revision(self) -> None:
        """Test that the migration 002 file has the correct revision identifier."""
        _msg = "test_migration_002_file_has_correct_revision starting"
        log.debug(_msg)

        migration_file = (
            Path(__file__).parent.parent
            / "alembic"
            / "versions"
            / "002_add_vault_root_and_indexes.py"
        )

        assert migration_file.exists(), f"Migration file not found: {migration_file}"

        content = migration_file.read_text()

        # Verify the revision line has the correct ID
        assert 'revision = "002"' in content, (
            "Migration 002 file should have revision = '002'"
        )

        _msg = "test_migration_002_file_has_correct_revision returning"
        log.debug(_msg)

    def test_migration_003_references_002_correctly(self) -> None:
        """Test that migration 003 correctly references migration 002."""
        _msg = "test_migration_003_references_002_correctly starting"
        log.debug(_msg)

        migration_file = (
            Path(__file__).parent.parent
            / "alembic"
            / "versions"
            / "003_add_vaults_table.py"
        )

        assert migration_file.exists(), f"Migration file not found: {migration_file}"

        content = migration_file.read_text()

        # Verify the down_revision points to 002
        assert 'down_revision = "002"' in content, (
            "Migration 003 should reference '002'"
        )

        _msg = "test_migration_003_references_002_correctly returning"
        log.debug(_msg)
