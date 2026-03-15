"""Tests for database/engine.py PostgreSQL URL normalization.

Tests for _normalize_postgres_url function that converts postgres:// to postgresql+psycopg://.
"""

import pytest

from obsidian_rag.database.engine import _normalize_postgres_url


class TestNormalizePostgresUrl:
    """Tests for _normalize_postgres_url function (lines 16-44)."""

    def test_normalize_postgres_url_with_postgres_prefix(self):
        """Test normalizing postgres:// URL."""
        url = "postgres://user:pass@localhost:5432/dbname"
        result = _normalize_postgres_url(url)

        assert result == "postgresql+psycopg://user:pass@localhost:5432/dbname"

    def test_normalize_postgres_url_with_postgresql_prefix(self):
        """Test URL that already has postgresql:// prefix."""
        url = "postgresql://user:pass@localhost:5432/dbname"
        result = _normalize_postgres_url(url)

        assert result == "postgresql+psycopg://user:pass@localhost:5432/dbname"

    def test_normalize_postgres_url_already_normalized(self):
        """Test URL that is already postgresql+psycopg://."""
        url = "postgresql+psycopg://user:pass@localhost:5432/dbname"
        result = _normalize_postgres_url(url)

        # Should remain unchanged
        assert result == "postgresql+psycopg://user:pass@localhost:5432/dbname"

    def test_normalize_postgres_url_with_unrelated_prefix(self):
        """Test URL with unrelated prefix."""
        url = "mysql://user:pass@localhost:3306/dbname"
        result = _normalize_postgres_url(url)

        # Should not be modified
        assert result == "mysql://user:pass@localhost:3306/dbname"

    def test_normalize_postgres_url_with_special_chars(self):
        """Test URL with special characters in password."""
        url = "postgres://user:p%40ss@localhost:5432/dbname"
        result = _normalize_postgres_url(url)

        assert result == "postgresql+psycopg://user:p%40ss@localhost:5432/dbname"

    def test_normalize_postgres_url_with_query_params(self):
        """Test URL with query parameters."""
        url = "postgres://user:pass@localhost:5432/dbname?sslmode=require"
        result = _normalize_postgres_url(url)

        assert (
            result
            == "postgresql+psycopg://user:pass@localhost:5432/dbname?sslmode=require"
        )

    def test_normalize_postgres_url_with_at_in_password(self):
        """Test URL with @ in password."""
        url = "postgres://user:pass%40word@localhost:5432/dbname"
        result = _normalize_postgres_url(url)

        assert result == "postgresql+psycopg://user:pass%40word@localhost:5432/dbname"

    def test_normalize_postgres_url_empty_string(self):
        """Test empty URL string."""
        url = ""
        result = _normalize_postgres_url(url)

        assert result == ""

    def test_normalize_postgres_url_postgres_with_plus(self):
        """Test postgres+something:// URL pattern."""
        url = "postgres+psycopg://user:pass@localhost:5432/dbname"
        result = _normalize_postgres_url(url)

        # The regex doesn't handle postgres+psycopg pattern, it remains unchanged
        # because it already has a + in it (driver specified)
        assert result == "postgres+psycopg://user:pass@localhost:5432/dbname"
