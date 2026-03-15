"""Tests for database engine module."""

from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from obsidian_rag.database.engine import DatabaseManager


class TestDatabaseManager:
    """Test cases for DatabaseManager class."""

    def test_init_with_postgresql_url(self):
        """Test DatabaseManager initialization with PostgreSQL URL."""
        with patch("obsidian_rag.database.engine.create_engine") as mock_create_engine:
            mock_engine = MagicMock()
            mock_create_engine.return_value = mock_engine

            db_manager = DatabaseManager("postgresql+psycopg://localhost/test")

            mock_create_engine.assert_called_once_with(
                "postgresql+psycopg://localhost/test"
            )
            assert db_manager.engine is mock_engine

    def test_create_tables(self):
        """Test creating all tables."""
        with patch("obsidian_rag.database.engine.create_engine") as mock_create_engine:
            mock_engine = MagicMock()
            mock_create_engine.return_value = mock_engine

            with patch(
                "obsidian_rag.database.models.Base.metadata.create_all"
            ) as mock_create:
                db_manager = DatabaseManager("postgresql+psycopg://localhost/test")
                db_manager.create_tables()

                mock_create.assert_called_once_with(bind=mock_engine)

    def test_get_session(self):
        """Test getting a database session."""
        with patch("obsidian_rag.database.engine.create_engine") as mock_create_engine:
            mock_engine = MagicMock()
            mock_session_class = MagicMock()
            mock_session = MagicMock()
            mock_session_class.return_value = mock_session

            mock_create_engine.return_value = mock_engine

            with patch(
                "obsidian_rag.database.engine.sessionmaker",
                return_value=mock_session_class,
            ):
                db_manager = DatabaseManager("postgresql+psycopg://localhost/test")

                with db_manager.get_session() as session:
                    assert session is mock_session

                mock_session.commit.assert_called_once()

    def test_get_session_rollback_on_error(self):
        """Test session rollback on error."""
        with patch("obsidian_rag.database.engine.create_engine") as mock_create_engine:
            mock_engine = MagicMock()
            mock_session_class = MagicMock()
            mock_session = MagicMock()
            mock_session_class.return_value = mock_session

            mock_create_engine.return_value = mock_engine

            with patch(
                "obsidian_rag.database.engine.sessionmaker",
                return_value=mock_session_class,
            ):
                db_manager = DatabaseManager("postgresql+psycopg://localhost/test")

                with pytest.raises(ValueError):
                    with db_manager.get_session() as session:
                        assert session is mock_session
                        raise ValueError("Test error")

                mock_session.rollback.assert_called_once()
                mock_session.close.assert_called_once()

    def test_drop_tables(self):
        """Test dropping all tables."""
        with patch("obsidian_rag.database.engine.create_engine") as mock_create_engine:
            mock_engine = MagicMock()
            mock_create_engine.return_value = mock_engine

            with patch(
                "obsidian_rag.database.models.Base.metadata.drop_all"
            ) as mock_drop:
                db_manager = DatabaseManager("postgresql+psycopg://localhost/test")
                db_manager.drop_tables()

                mock_drop.assert_called_once_with(bind=mock_engine)

    def test_close(self):
        """Test closing database engine."""
        with patch("obsidian_rag.database.engine.create_engine") as mock_create_engine:
            mock_engine = MagicMock()
            mock_create_engine.return_value = mock_engine

            db_manager = DatabaseManager("postgresql+psycopg://localhost/test")
            db_manager.close()

            mock_engine.dispose.assert_called_once()


class TestNormalizePostgresUrl:
    """Test cases for _normalize_postgres_url function."""

    def test_normalize_postgres_url_adds_driver(self):
        """Test that postgres URL gets psycopg driver added."""
        from obsidian_rag.database.engine import _normalize_postgres_url

        result = _normalize_postgres_url("postgresql://localhost/db")
        assert result == "postgresql+psycopg://localhost/db"

    def test_normalize_postgres_url_preserves_existing_driver(self):
        """Test that URL with existing driver is preserved."""
        from obsidian_rag.database.engine import _normalize_postgres_url

        # URL already has a driver specified
        result = _normalize_postgres_url("postgresql+psycopg://localhost/db")
        assert result == "postgresql+psycopg://localhost/db"

    def test_normalize_postgres_url_short_form(self):
        """Test that postgres:// short form is normalized."""
        from obsidian_rag.database.engine import _normalize_postgres_url

        result = _normalize_postgres_url("postgres://localhost/db")
        assert result == "postgresql+psycopg://localhost/db"

    def test_normalize_postgres_url_logs_debug(self, caplog):
        """Test that URL normalization logs debug messages."""
        from obsidian_rag.database.engine import _normalize_postgres_url

        with caplog.at_level("DEBUG", logger="obsidian_rag.database.engine"):
            _normalize_postgres_url("postgresql://localhost/db")

        assert "Normalizing database URL" in caplog.text

    def test_normalize_postgres_url_with_existing_driver_logs_debug(self, caplog):
        """Test that URL with existing driver logs skip message."""
        from obsidian_rag.database.engine import _normalize_postgres_url

        with caplog.at_level("DEBUG", logger="obsidian_rag.database.engine"):
            _normalize_postgres_url("postgresql+psycopg://localhost/db")

        assert "URL already has driver specified" in caplog.text
