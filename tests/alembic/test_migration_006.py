"""Tests for migration 006: Add chunk token fields."""

import logging
from pathlib import Path

log = logging.getLogger(__name__)


class TestMigration006Syntax:
    """Test migration file syntax and structure."""

    def test_migration_006_syntax(self):
        """Verify migration 006 has valid Python syntax."""
        _msg = "test_migration_006_syntax starting"
        log.debug(_msg)

        migration_path = Path("alembic/versions/006_add_chunk_token_fields.py")
        content = migration_path.read_text()

        # Compile to check syntax
        compile(content, str(migration_path), "exec")

        _msg = "test_migration_006_syntax returning"
        log.debug(_msg)

    def test_migration_006_has_correct_revision(self):
        """Verify migration 006 has correct revision identifiers."""
        _msg = "test_migration_006_has_correct_revision starting"
        log.debug(_msg)

        migration_path = Path("alembic/versions/006_add_chunk_token_fields.py")
        content = migration_path.read_text()

        # Verify revision is "006"
        assert 'revision: str = "006"' in content

        # Verify down_revision is "005"
        assert 'down_revision: str | None = "005"' in content

        _msg = "test_migration_006_has_correct_revision returning"
        log.debug(_msg)

    def test_migration_006_has_upgrade_function(self):
        """Verify migration 006 has upgrade function."""
        _msg = "test_migration_006_has_upgrade_function starting"
        log.debug(_msg)

        migration_path = Path("alembic/versions/006_add_chunk_token_fields.py")
        content = migration_path.read_text()

        # Verify upgrade function exists
        assert "def upgrade()" in content
        assert "op.add_column" in content
        assert "token_count" in content
        assert "chunk_type" in content

        _msg = "test_migration_006_has_upgrade_function returning"
        log.debug(_msg)

    def test_migration_006_has_downgrade_function(self):
        """Verify migration 006 has downgrade function."""
        _msg = "test_migration_006_has_downgrade_function starting"
        log.debug(_msg)

        migration_path = Path("alembic/versions/006_add_chunk_token_fields.py")
        content = migration_path.read_text()

        # Verify downgrade function exists
        assert "def downgrade()" in content
        assert "op.drop_column" in content
        assert "op.drop_index" in content

        _msg = "test_migration_006_has_downgrade_function returning"
        log.debug(_msg)

    def test_migration_006_creates_chunk_type_index(self):
        """Verify migration 006 creates index on chunk_type."""
        _msg = "test_migration_006_creates_chunk_type_index starting"
        log.debug(_msg)

        migration_path = Path("alembic/versions/006_add_chunk_token_fields.py")
        content = migration_path.read_text()

        # Verify index creation
        assert "op.create_index" in content
        assert "ix_document_chunks_chunk_type" in content

        _msg = "test_migration_006_creates_chunk_type_index returning"
        log.debug(_msg)

    def test_migration_006_token_count_is_integer(self):
        """Verify token_count column is Integer type."""
        _msg = "test_migration_006_token_count_is_integer starting"
        log.debug(_msg)

        migration_path = Path("alembic/versions/006_add_chunk_token_fields.py")
        content = migration_path.read_text()

        # Verify Integer type for token_count
        assert "sa.Integer()" in content

        _msg = "test_migration_006_token_count_is_integer returning"
        log.debug(_msg)

    def test_migration_006_chunk_type_is_string(self):
        """Verify chunk_type column is String type."""
        _msg = "test_migration_006_chunk_type_is_string starting"
        log.debug(_msg)

        migration_path = Path("alembic/versions/006_add_chunk_token_fields.py")
        content = migration_path.read_text()

        # Verify String type for chunk_type with length 20
        assert "sa.String(length=20)" in content

        _msg = "test_migration_006_chunk_type_is_string returning"
        log.debug(_msg)

    def test_migration_006_columns_are_nullable(self):
        """Verify columns are nullable for backward compatibility."""
        _msg = "test_migration_006_columns_are_nullable starting"
        log.debug(_msg)

        migration_path = Path("alembic/versions/006_add_chunk_token_fields.py")
        content = migration_path.read_text()

        # Verify nullable=True for both columns
        assert "nullable=True" in content

        _msg = "test_migration_006_columns_are_nullable returning"
        log.debug(_msg)

    def test_migration_006_imports_sqlalchemy(self):
        """Verify migration imports sqlalchemy."""
        _msg = "test_migration_006_imports_sqlalchemy starting"
        log.debug(_msg)

        migration_path = Path("alembic/versions/006_add_chunk_token_fields.py")
        content = migration_path.read_text()

        # Verify imports
        assert "import sqlalchemy as sa" in content
        assert "from alembic import op" in content

        _msg = "test_migration_006_imports_sqlalchemy returning"
        log.debug(_msg)

    def test_migration_006_has_docstring(self):
        """Verify migration has module docstring."""
        _msg = "test_migration_006_has_docstring starting"
        log.debug(_msg)

        migration_path = Path("alembic/versions/006_add_chunk_token_fields.py")
        content = migration_path.read_text()

        # Verify module docstring exists
        assert '"""Add token_count and chunk_type to document_chunks.' in content
        assert "Revision ID: 006" in content
        assert "Revises: 005" in content

        _msg = "test_migration_006_has_docstring returning"
        log.debug(_msg)

    def test_migration_006_upgrade_has_docstring(self):
        """Verify upgrade function has docstring."""
        _msg = "test_migration_006_upgrade_has_docstring starting"
        log.debug(_msg)

        migration_path = Path("alembic/versions/006_add_chunk_token_fields.py")
        content = migration_path.read_text()

        # Verify upgrade docstring
        assert 'def upgrade() -> None:\n    """Add token_count' in content

        _msg = "test_migration_006_upgrade_has_docstring returning"
        log.debug(_msg)

    def test_migration_006_downgrade_has_docstring(self):
        """Verify downgrade function has docstring."""
        _msg = "test_migration_006_downgrade_has_docstring starting"
        log.debug(_msg)

        migration_path = Path("alembic/versions/006_add_chunk_token_fields.py")
        content = migration_path.read_text()

        # Verify downgrade docstring
        assert 'def downgrade() -> None:\n    """Remove token_count' in content

        _msg = "test_migration_006_downgrade_has_docstring returning"
        log.debug(_msg)

    def test_migration_006_downgrade_drops_index_first(self):
        """Verify downgrade drops index before columns."""
        _msg = "test_migration_006_downgrade_drops_index_first starting"
        log.debug(_msg)

        migration_path = Path("alembic/versions/006_add_chunk_token_fields.py")
        content = migration_path.read_text()

        # Find positions of drop_index and drop_column
        drop_index_pos = content.find("op.drop_index")
        drop_column_pos = content.find("op.drop_column")

        # drop_index should come before drop_column
        assert drop_index_pos < drop_column_pos

        _msg = "test_migration_006_downgrade_drops_index_first returning"
        log.debug(_msg)

    def test_migration_006_uses_collections_abc_sequence(self):
        """Verify migration uses collections.abc.Sequence for annotations."""
        _msg = "test_migration_006_uses_collections_abc_sequence starting"
        log.debug(_msg)

        migration_path = Path("alembic/versions/006_add_chunk_token_fields.py")
        content = migration_path.read_text()

        # Verify collections.abc import
        assert "from collections.abc import Sequence" in content

        _msg = "test_migration_006_uses_collections_abc_sequence returning"
        log.debug(_msg)
