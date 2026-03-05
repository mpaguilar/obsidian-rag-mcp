"""Tests for documents_sqlite module.

Tests for SQLite-specific document query implementations.
"""

from datetime import datetime
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from obsidian_rag.database.models import Base, Document
from obsidian_rag.mcp_server.models import PropertyFilter, TagFilter
from obsidian_rag.mcp_server.tools.documents_params import (
    PaginationParams,
    PropertyFilterParams,
    PropertyQueryParams,
    TagFilterParams,
)
from obsidian_rag.mcp_server.tools.documents_sqlite import (
    get_documents_by_property_sqlite,
    query_documents_sqlite,
)


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


@pytest.fixture
def sample_documents(db_session):
    """Create sample documents for testing."""
    docs = [
        Document(
            id=uuid4(),
            file_path="/data/vault1/work.md",
            file_name="work.md",
            content="# Work Document",
            checksum_md5="abc123",
            created_at_fs=datetime.now(),
            modified_at_fs=datetime.now(),
            tags=["work", "urgent"],
            frontmatter_json={"status": "draft"},
            vault_root="/data/vault1",
        ),
        Document(
            id=uuid4(),
            file_path="/data/vault2/personal.md",
            file_name="personal.md",
            content="# Personal Document",
            checksum_md5="def456",
            created_at_fs=datetime.now(),
            modified_at_fs=datetime.now(),
            tags=["personal", "ideas"],
            frontmatter_json={"status": "published"},
            vault_root="/data/vault2",
        ),
    ]
    db_session.add_all(docs)
    db_session.commit()
    return docs


class TestQueryDocumentsSqlite:
    """Tests for query_documents_sqlite function."""

    def test_query_documents_sqlite_basic(self, db_session, sample_documents):
        """Test query_documents_sqlite with basic query."""
        from obsidian_rag.mcp_server.tools.documents_params import (
            DocumentQueryParams,
            QueryFilterParams,
        )

        filter_params = QueryFilterParams(
            property_filters=PropertyFilterParams(
                include_filters=None, exclude_filters=None
            ),
            tag_params=TagFilterParams(tag_filter=None),
        )
        pagination = PaginationParams(limit=20, offset=0)
        params = DocumentQueryParams(
            session=db_session,
            query_embedding=[0.1] * 1536,
            filter_params=filter_params,
            pagination=pagination,
        )

        result = query_documents_sqlite(params)

        assert result.total_count == 2
        assert len(result.results) == 2

    def test_query_documents_sqlite_with_tag_filter(self, db_session, sample_documents):
        """Test query_documents_sqlite with tag filter."""
        from obsidian_rag.mcp_server.tools.documents_params import (
            DocumentQueryParams,
            QueryFilterParams,
        )

        tag_filter = TagFilter(include_tags=["work"])
        filter_params = QueryFilterParams(
            property_filters=PropertyFilterParams(
                include_filters=None, exclude_filters=None
            ),
            tag_params=TagFilterParams(tag_filter=tag_filter),
        )
        pagination = PaginationParams(limit=20, offset=0)
        params = DocumentQueryParams(
            session=db_session,
            query_embedding=[0.1] * 1536,
            filter_params=filter_params,
            pagination=pagination,
        )

        result = query_documents_sqlite(params)

        assert result.total_count == 1
        assert result.results[0].file_name == "work.md"

    def test_query_documents_sqlite_empty_database(self, db_session):
        """Test query_documents_sqlite with empty database."""
        from obsidian_rag.mcp_server.tools.documents_params import (
            DocumentQueryParams,
            QueryFilterParams,
        )

        filter_params = QueryFilterParams(
            property_filters=PropertyFilterParams(
                include_filters=None, exclude_filters=None
            ),
            tag_params=TagFilterParams(tag_filter=None),
        )
        pagination = PaginationParams(limit=20, offset=0)
        params = DocumentQueryParams(
            session=db_session,
            query_embedding=[0.1] * 1536,
            filter_params=filter_params,
            pagination=pagination,
        )

        result = query_documents_sqlite(params)

        assert result.total_count == 0
        assert result.results == []


class TestGetDocumentsByPropertySqlite:
    """Tests for get_documents_by_property_sqlite function (TASK-095)."""

    def test_get_documents_by_property_sqlite_basic(self, db_session, sample_documents):
        """Test get_documents_by_property_sqlite with basic query."""
        property_filters = PropertyFilterParams(
            include_filters=None, exclude_filters=None
        )
        tag_params = TagFilterParams(tag_filter=None)
        pagination = PaginationParams(limit=20, offset=0)
        params = PropertyQueryParams(
            session=db_session,
            property_filters=property_filters,
            tag_params=tag_params,
            vault_root=None,
            pagination=pagination,
        )

        results, total_count = get_documents_by_property_sqlite(params)

        assert total_count == 2
        assert len(results) == 2

    def test_get_documents_by_property_sqlite_with_vault_root(
        self, db_session, sample_documents
    ):
        """Test get_documents_by_property_sqlite with vault_root filter (TASK-095)."""
        property_filters = PropertyFilterParams(
            include_filters=None, exclude_filters=None
        )
        tag_params = TagFilterParams(tag_filter=None)
        pagination = PaginationParams(limit=20, offset=0)
        params = PropertyQueryParams(
            session=db_session,
            property_filters=property_filters,
            tag_params=tag_params,
            vault_root="/data/vault1",
            pagination=pagination,
        )

        results, total_count = get_documents_by_property_sqlite(params)

        # Should only return documents from vault1
        assert total_count == 1
        assert len(results) == 1
        assert results[0].file_name == "work.md"

    def test_get_documents_by_property_sqlite_with_property_filter(
        self, db_session, sample_documents
    ):
        """Test get_documents_by_property_sqlite with property filter."""
        property_filters = PropertyFilterParams(
            include_filters=[
                PropertyFilter(path="status", operator="equals", value="draft")
            ],
            exclude_filters=None,
        )
        tag_params = TagFilterParams(tag_filter=None)
        pagination = PaginationParams(limit=20, offset=0)
        params = PropertyQueryParams(
            session=db_session,
            property_filters=property_filters,
            tag_params=tag_params,
            vault_root=None,
            pagination=pagination,
        )

        results, total_count = get_documents_by_property_sqlite(params)

        assert total_count == 1
        assert len(results) == 1
        assert results[0].file_name == "work.md"

    def test_get_documents_by_property_sqlite_with_tag_filter(
        self, db_session, sample_documents
    ):
        """Test get_documents_by_property_sqlite with tag filter."""
        property_filters = PropertyFilterParams(
            include_filters=None, exclude_filters=None
        )
        tag_filter = TagFilter(include_tags=["personal"])
        tag_params = TagFilterParams(tag_filter=tag_filter)
        pagination = PaginationParams(limit=20, offset=0)
        params = PropertyQueryParams(
            session=db_session,
            property_filters=property_filters,
            tag_params=tag_params,
            vault_root=None,
            pagination=pagination,
        )

        results, total_count = get_documents_by_property_sqlite(params)

        assert total_count == 1
        assert len(results) == 1
        assert results[0].file_name == "personal.md"

    def test_get_documents_by_property_sqlite_empty_database(self, db_session):
        """Test get_documents_by_property_sqlite with empty database."""
        property_filters = PropertyFilterParams(
            include_filters=None, exclude_filters=None
        )
        tag_params = TagFilterParams(tag_filter=None)
        pagination = PaginationParams(limit=20, offset=0)
        params = PropertyQueryParams(
            session=db_session,
            property_filters=property_filters,
            tag_params=tag_params,
            vault_root=None,
            pagination=pagination,
        )

        results, total_count = get_documents_by_property_sqlite(params)

        assert total_count == 0
        assert results == []
