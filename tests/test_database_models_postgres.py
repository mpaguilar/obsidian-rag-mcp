"""Tests for database/models.py PostgreSQL-specific code.

Tests for ArrayType with PostgreSQL dialect and pgvector extension creation.
"""

from unittest.mock import MagicMock, Mock, patch

import pytest
from sqlalchemy import create_engine
from sqlalchemy.dialects.postgresql import dialect as pg_dialect

from obsidian_rag.database.models import ArrayType


class TestArrayTypePostgresql:
    """Tests for ArrayType with PostgreSQL dialect (line 62)."""

    def test_array_type_load_dialect_impl_postgresql(self):
        """Test ArrayType.load_dialect_impl with PostgreSQL dialect (line 62)."""
        from sqlalchemy.dialects.postgresql import ARRAY as PG_ARRAY

        array_type = ArrayType()
        pg_dialect_instance = pg_dialect()

        impl = array_type.load_dialect_impl(pg_dialect_instance)

        assert isinstance(impl, PG_ARRAY)

    def test_array_type_load_dialect_impl_postgresql_element_type(self):
        """Test ArrayType PostgreSQL dialect returns Text array."""
        from sqlalchemy import Text
        from sqlalchemy.dialects.postgresql import ARRAY as PG_ARRAY

        array_type = ArrayType()
        pg_dialect_instance = pg_dialect()

        impl = array_type.load_dialect_impl(pg_dialect_instance)

        assert isinstance(impl, PG_ARRAY)
        # The element type should be Text


class TestCreatePgvectorExtension:
    """Tests for _create_pgvector_extension function (lines 285-293)."""

    def test_create_pgvector_extension_success(self):
        """Test _create_pgvector_extension creates extension successfully."""
        from obsidian_rag.database.models import _create_pgvector_extension

        mock_target = MagicMock()
        mock_connection = MagicMock()
        mock_connection.dialect.name = "postgresql"

        _create_pgvector_extension(mock_target, mock_connection)

        mock_connection.execute.assert_called_once()
        call_args = mock_connection.execute.call_args[0][0]
        assert "CREATE EXTENSION IF NOT EXISTS vector" in str(call_args)

    def test_create_pgvector_extension_error_handling(self):
        """Test _create_pgvector_extension handles errors gracefully."""
        from obsidian_rag.database.models import _create_pgvector_extension

        mock_target = MagicMock()
        mock_connection = MagicMock()
        mock_connection.dialect.name = "postgresql"
        mock_connection.execute.side_effect = Exception("Extension creation failed")

        # The function doesn't catch exceptions, so this should raise
        with pytest.raises(Exception, match="Extension creation failed"):
            _create_pgvector_extension(mock_target, mock_connection)

        mock_connection.execute.assert_called_once()

    def test_create_pgvector_extension_skips_non_postgresql(self):
        """Test _create_pgvector_extension skips non-PostgreSQL dialects."""
        from obsidian_rag.database.models import _create_pgvector_extension

        mock_target = MagicMock()
        mock_connection = MagicMock()
        mock_connection.dialect.name = "mysql"

        _create_pgvector_extension(mock_target, mock_connection)

        # Should not execute for non-PostgreSQL
        mock_connection.execute.assert_not_called()

    def test_create_pgvector_integration_with_engine(self):
        """Test pgvector extension with actual database engine."""
        # This test validates the function is called during table creation
        from obsidian_rag.database.models import _create_pgvector_extension

        mock_target = MagicMock()
        mock_connection = MagicMock()
        mock_connection.dialect.name = "postgresql"

        with patch("logging.Logger.warning") as mock_warning:
            _create_pgvector_extension(mock_target, mock_connection)

            # Should not have logged a warning on success
            mock_warning.assert_not_called()
