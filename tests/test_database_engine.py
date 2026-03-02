"""Tests for database engine module."""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from obsidian_rag.database.engine import DatabaseManager
from obsidian_rag.database.models import Base, Document


class TestDatabaseManager:
    """Test cases for DatabaseManager class."""

    def test_init_with_url(self):
        """Test initializing DatabaseManager with database URL."""
        db_url = "sqlite:///:memory:"
        manager = DatabaseManager(db_url)

        assert manager.engine is not None
        assert manager.SessionLocal is not None

    def test_create_tables(self):
        """Test creating database tables."""
        db_url = "sqlite:///:memory:"
        manager = DatabaseManager(db_url)

        # Should not raise
        manager.create_tables()

        # Verify tables exist by querying
        with manager.get_session() as session:
            result = session.query(Document).all()
            assert result == []

    def test_drop_tables(self):
        """Test dropping database tables."""
        db_url = "sqlite:///:memory:"
        manager = DatabaseManager(db_url)
        manager.create_tables()

        # Should not raise
        manager.drop_tables()

    def test_get_session_context_manager(self):
        """Test using get_session as context manager."""
        db_url = "sqlite:///:memory:"
        manager = DatabaseManager(db_url)
        manager.create_tables()

        with manager.get_session() as session:
            assert session is not None
            # Add a document
            doc = Document(
                file_path="/test/file.md",
                file_name="file.md",
                content="Test",
                checksum_md5="abc123",
                created_at_fs=__import__("datetime").datetime.now(),
                modified_at_fs=__import__("datetime").datetime.now(),
            )
            session.add(doc)

        # Verify document was committed
        with manager.get_session() as session:
            result = session.query(Document).first()
            assert result is not None
            assert result.file_path == "/test/file.md"

    def test_get_session_rollback_on_error(self):
        """Test that session rolls back on error."""
        db_url = "sqlite:///:memory:"
        manager = DatabaseManager(db_url)
        manager.create_tables()

        # Create a document first
        with manager.get_session() as session:
            doc = Document(
                file_path="/test/file.md",
                file_name="file.md",
                content="Test",
                checksum_md5="abc123",
                created_at_fs=__import__("datetime").datetime.now(),
                modified_at_fs=__import__("datetime").datetime.now(),
            )
            session.add(doc)

        # Try to add duplicate (should fail due to unique constraint)
        try:
            with manager.get_session() as session:
                doc2 = Document(
                    file_path="/test/file.md",  # Same path
                    file_name="file.md",
                    content="Test 2",
                    checksum_md5="def456",
                    created_at_fs=__import__("datetime").datetime.now(),
                    modified_at_fs=__import__("datetime").datetime.now(),
                )
                session.add(doc2)
                # Force the error by flushing
                session.flush()
        except Exception:
            pass  # Expected to fail

        # Verify original document still exists
        with manager.get_session() as session:
            count = session.query(Document).count()
            assert count == 1

    def test_close(self):
        """Test closing database engine."""
        db_url = "sqlite:///:memory:"
        manager = DatabaseManager(db_url)

        # Should not raise
        manager.close()
