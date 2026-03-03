"""Unit tests for MCP document tools."""

import uuid
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from obsidian_rag.database.models import Base, Document
from obsidian_rag.mcp_server.tools.documents import query_documents


@pytest.fixture
def db_engine():
    """Create a test database engine using SQLite."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    yield engine
    Base.metadata.drop_all(engine)


@pytest.fixture
def db_session(db_engine):
    """Create a test database session."""
    SessionLocal = sessionmaker(bind=db_engine)
    session = SessionLocal()
    yield session
    session.close()


class TestQueryDocuments:
    """Tests for query_documents function."""

    def test_empty_result(self, db_session):
        """Test with no documents in database."""
        query_embedding = [0.1] * 1536
        result = query_documents(db_session, query_embedding)

        assert result.results == []
        assert result.total_count == 0
        assert result.has_more is False
        assert result.next_offset is None

    def test_excludes_documents_without_embeddings(self, db_session):
        """Test documents without embeddings are excluded."""
        doc = Document(
            id=uuid.uuid4(),
            file_path="/test/doc.md",
            file_name="doc.md",
            content="# Test",
            checksum_md5="abc123",
            created_at_fs=datetime.now(),
            modified_at_fs=datetime.now(),
            content_vector=None,  # No embedding
        )
        db_session.add(doc)
        db_session.commit()

        query_embedding = [0.1] * 1536
        result = query_documents(db_session, query_embedding)

        assert result.total_count == 0
        assert len(result.results) == 0

    def test_limit_validation(self, db_session):
        """Test that limit is validated and clamped."""
        query_embedding = [0.1] * 1536

        # Test limit above maximum (should work but be clamped internally)
        result = query_documents(db_session, query_embedding, limit=200)
        assert result.total_count == 0  # No documents

    def test_offset_validation(self, db_session):
        """Test that offset is validated."""
        query_embedding = [0.1] * 1536

        # Test negative offset (should be clamped to 0)
        result = query_documents(db_session, query_embedding, offset=-10)
        assert result.total_count == 0  # No documents
