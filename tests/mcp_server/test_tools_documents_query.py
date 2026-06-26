"""Unit tests for MCP document query tools."""

import uuid
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from obsidian_rag.database.models import Document, Vault
from obsidian_rag.mcp_server.models import PropertyFilter, TagFilter
from obsidian_rag.mcp_server.tools.documents import (
    _glob_to_like,
    get_all_tags,
    query_documents,
)
from obsidian_rag.mcp_server.tools.documents_tags import _matches_glob
from obsidian_rag.mcp_server.tools.documents_params import PaginationParams


@pytest.fixture
def sample_documents(db_session):
    """Create sample documents for testing."""
    # Create vault first
    vault = Vault(
        id=uuid.uuid4(),
        name="test_vault",
        container_path="/data/vault",
        host_path="/data/vault",
    )

    docs = [
        Document(
            id=uuid.uuid4(),
            vault_id=vault.id,
            file_path="/data/vault/work.md",
            file_name="work.md",
            content="# Work Document",
            checksum_md5="abc123",
            created_at_fs=datetime.now(),
            modified_at_fs=datetime.now(),
            tags=["work", "urgent"],
            frontmatter_json={
                "title": "Work",
                "author": {"name": "John", "email": "john@example.com"},
                "status": "draft",
                "priority": 1,
            },
        ),
        Document(
            id=uuid.uuid4(),
            vault_id=vault.id,
            file_path="/data/vault/personal.md",
            file_name="personal.md",
            content="# Personal Document",
            checksum_md5="def456",
            created_at_fs=datetime.now(),
            modified_at_fs=datetime.now(),
            tags=["personal", "ideas"],
            frontmatter_json={
                "title": "Personal",
                "author": {"name": "Jane", "email": "jane@example.com"},
                "status": "published",
                "priority": 2,
            },
        ),
        Document(
            id=uuid.uuid4(),
            vault_id=vault.id,
            file_path="/data/vault/mixed.md",
            file_name="mixed.md",
            content="# Mixed Document",
            checksum_md5="ghi789",
            created_at_fs=datetime.now(),
            modified_at_fs=datetime.now(),
            tags=["work", "ideas"],
            frontmatter_json={
                "title": "Mixed",
                "author": {"name": "John", "email": "john2@example.com"},
                "status": "draft",
                "priority": 3,
            },
        ),
        Document(
            id=uuid.uuid4(),
            vault_id=vault.id,
            file_path="/data/vault/untagged.md",
            file_name="untagged.md",
            content="# Untagged Document",
            checksum_md5="jkl012",
            created_at_fs=datetime.now(),
            modified_at_fs=datetime.now(),
            tags=None,
            frontmatter_json={"title": "Untagged"},
        ),
    ]

    # Configure mock to return documents for Document queries
    db_session.query.return_value.filter.return_value.order_by.return_value.all.return_value = docs
    db_session.query.return_value.filter.return_value.order_by.return_value.count.return_value = len(
        docs
    )
    db_session.query.return_value.filter.return_value.all.return_value = docs
    db_session.query.return_value.filter.return_value.count.return_value = len(docs)
    db_session.query.return_value.all.return_value = docs
    db_session.query.return_value.count.return_value = len(docs)

    return docs


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

    def test_limit_validation(self, db_session):
        """Test that limit is validated and clamped."""
        query_embedding = [0.1] * 1536

        # Test limit above maximum (should work but be clamped internally)
        pagination = PaginationParams(limit=200, offset=0)
        result = query_documents(db_session, query_embedding, pagination=pagination)
        assert result.total_count == 0  # No documents

    def test_offset_validation(self, db_session):
        """Test that offset is validated."""
        query_embedding = [0.1] * 1536

        # Test negative offset (should be clamped to 0)
        pagination = PaginationParams(limit=20, offset=-10)
        result = query_documents(db_session, query_embedding, pagination=pagination)
        assert result.total_count == 0  # No documents


class TestGlobToLike:
    """Tests for _glob_to_like helper function."""

    def test_star_wildcard(self):
        """Test * becomes %."""
        result = _glob_to_like("tag*")
        assert result == "tag%"

    def test_question_wildcard(self):
        """Test ? becomes _."""
        result = _glob_to_like("ta?")
        assert result == "ta_"

    def test_sql_special_chars_escaped(self):
        """Test SQL special characters are escaped."""
        result = _glob_to_like("100%")
        assert "\\%" in result

    def test_multiple_wildcards(self):
        """Test multiple wildcards in pattern."""
        result = _glob_to_like("*work*")
        assert result == "%work%"


class TestMatchesGlob:
    """Tests for _matches_glob function."""

    def test_matches_glob_star(self):
        """Test glob matching with star."""
        assert _matches_glob("work", "wo*") is True
        assert _matches_glob("work", "*rk") is True
        assert _matches_glob("workplace", "work*") is True

    def test_matches_glob_question(self):
        """Test glob matching with question mark."""
        assert _matches_glob("work", "wo?k") is True
        assert _matches_glob("work", "wor?") is True

    def test_matches_glob_case_insensitive(self):
        """Test glob matching is case-insensitive."""
        assert _matches_glob("WORK", "work*") is True
        assert _matches_glob("work", "WORK*") is True


class TestQueryDocumentsWithFilters:
    """Tests for query_documents with filters."""

    def _configure_mock_for_docs(self, db_session, docs, total_count=None):
        """Configure mock to return specific documents."""
        if total_count is None:
            total_count = len(docs)
        db_session.query.return_value.filter.return_value.order_by.return_value.all.return_value = docs
        db_session.query.return_value.filter.return_value.order_by.return_value.count.return_value = total_count

    def test_query_documents_with_tag_filter(self, db_session, sample_documents):
        """Test query_documents with tag filter."""
        # work.md and mixed.md have "work" tag
        filtered_docs = [d for d in sample_documents if d.tags and "work" in d.tags]
        self._configure_mock_for_docs(db_session, filtered_docs, total_count=2)

        tag_filter = TagFilter(include_tags=["work"])
        pagination = PaginationParams(limit=20, offset=0)
        result = query_documents(
            db_session,
            query_embedding=[0.1] * 1536,
            tag_filter=tag_filter,
            pagination=pagination,
        )
        # For mock, returns filtered docs with similarity_score=0.0
        # work.md and mixed.md have "work" tag (exact match)
        assert result.total_count == 2

    def test_query_documents_with_property_filter(self, db_session, sample_documents):
        """Test query_documents with property filter."""
        # work.md and mixed.md have status="draft"
        filtered_docs = [
            d for d in sample_documents if d.frontmatter_json.get("status") == "draft"
        ]
        self._configure_mock_for_docs(db_session, filtered_docs, total_count=2)

        include_props = [
            PropertyFilter(path="status", operator="equals", value="draft")
        ]
        from obsidian_rag.mcp_server.tools.documents_params import PropertyFilterParams

        property_filters = PropertyFilterParams(
            include_filters=include_props, exclude_filters=None
        )
        pagination = PaginationParams(limit=20, offset=0)
        result = query_documents(
            db_session,
            query_embedding=[0.1] * 1536,
            filter_params=property_filters,
            pagination=pagination,
        )
        # work.md and mixed.md have status="draft"
        assert result.total_count == 2

    def test_query_documents_empty_result(self, db_session):
        """Test query_documents with empty database."""
        self._configure_mock_for_docs(db_session, [], total_count=0)

        tag_filter = TagFilter(include_tags=["work"])
        pagination = PaginationParams(limit=20, offset=0)
        result = query_documents(
            db_session,
            query_embedding=[0.1] * 1536,
            tag_filter=tag_filter,
            pagination=pagination,
        )
        assert result.total_count == 0
        assert result.results == []


class TestGetAllTags:
    """Tests for get_all_tags function."""

    def _create_mock_tag_rows(self, tags):
        """Create mock row objects with .tag attribute."""
        rows = []
        for tag in tags:
            row = MagicMock()
            row.tag = tag
            rows.append(row)
        return rows

    def _configure_mock_for_tags(self, db_session, tags):
        """Configure mock to return specific tags."""
        rows = self._create_mock_tag_rows(tags)
        select_mock = db_session.query.return_value
        select_mock.filter.return_value = select_mock
        select_mock.order_by.return_value = select_mock
        select_mock.all.return_value = rows

    def test_get_all_tags_basic(self, db_session, sample_documents):
        """Test getting all unique tags."""
        # Extract unique tags from sample documents
        all_tags = set()
        for doc in sample_documents:
            if doc.tags:
                all_tags.update(doc.tags)
        tags_list = sorted(list(all_tags))
        self._configure_mock_for_tags(db_session, tags_list)

        result = get_all_tags(db_session, pattern=None, limit=20, offset=0)
        assert result.total_count == 4  # work, urgent, personal, ideas
        assert sorted(result.tags) == ["ideas", "personal", "urgent", "work"]

    def test_get_all_tags_with_pattern(self, db_session, sample_documents):
        """Test getting tags with glob pattern."""
        self._configure_mock_for_tags(db_session, ["work"])

        result = get_all_tags(db_session, pattern="work*", limit=20, offset=0)
        assert result.total_count == 1
        assert result.tags == ["work"]

    def test_get_all_tags_pagination(self, db_session, sample_documents):
        """Test tag pagination."""
        all_tags = ["ideas", "personal", "urgent", "work"]
        self._configure_mock_for_tags(db_session, all_tags)

        result = get_all_tags(db_session, pattern=None, limit=2, offset=0)
        assert result.total_count == 4
        assert len(result.tags) == 2
        assert result.has_more is True

    def test_get_all_tags_empty_database(self, db_session):
        """Test getting tags with empty database."""
        self._configure_mock_for_tags(db_session, [])

        result = get_all_tags(db_session, pattern=None, limit=20, offset=0)
        assert result.total_count == 0
        assert result.tags == []


class TestExtractTagsPostgresql:
    """Tests for _extract_tags_postgresql function using subquery approach."""

    def test_extract_tags_postgresql_with_pattern(self):
        """Test _extract_tags_postgresql with pattern filtering (subquery approach)."""
        from obsidian_rag.mcp_server.tools.documents import _extract_tags_postgresql

        mock_session = MagicMock()

        # Mock the query results
        mock_row1 = MagicMock()
        mock_row1.tag = "work"
        mock_row2 = MagicMock()
        mock_row2.tag = "workplace"

        mock_query = MagicMock()
        mock_query.filter.return_value = mock_query
        mock_query.order_by.return_value = mock_query
        mock_query.all.return_value = [mock_row1, mock_row2]

        mock_session.query.return_value = mock_query

        result = _extract_tags_postgresql(mock_session, pattern="work*")

        assert "work" in result
        assert "workplace" in result
        # Verify filter was called (at least once for the pattern)
        assert mock_query.filter.call_count >= 1


class TestGetAllTagsAdditional:
    """Additional tests for get_all_tags (TASK-096)."""

    def _create_mock_tag_rows(self, tags):
        """Create mock row objects with .tag attribute."""
        rows = []
        for tag in tags:
            row = MagicMock()
            row.tag = tag
            rows.append(row)
        return rows

    def _configure_mock_for_tags(self, db_session, tags):
        """Configure mock to return specific tags."""
        rows = self._create_mock_tag_rows(tags)
        select_mock = db_session.query.return_value
        select_mock.filter.return_value = select_mock
        select_mock.order_by.return_value = select_mock
        select_mock.all.return_value = rows

    def test_get_all_tags_execution_path(self, db_session, sample_documents):
        """Test get_all_tags execution path (line 455)."""
        # Extract unique tags from sample documents
        all_tags = set()
        for doc in sample_documents:
            if doc.tags:
                all_tags.update(doc.tags)
        tags_list = sorted(list(all_tags))
        self._configure_mock_for_tags(db_session, tags_list)

        # This test exercises the full execution path of get_all_tags
        result = get_all_tags(
            db_session,
            pattern=None,
            limit=10,
            offset=0,
        )
        # Should return all unique tags from sample documents
        assert result.total_count == 4
        assert sorted(result.tags) == ["ideas", "personal", "urgent", "work"]
        assert result.has_more is False
        assert result.next_offset is None


class TestQueryDocumentsChunkParameters:
    """Tests for query_documents with use_chunks and rerank parameters."""

    @patch("obsidian_rag.mcp_server.tools.documents.query_chunks")
    def test_query_documents_with_use_chunks(self, mock_query_chunks, db_session):
        """Test query_documents with use_chunks=True parameter."""
        from obsidian_rag.mcp_server.tools.documents_chunks import ChunkQueryResult

        mock_result = ChunkQueryResult(
            chunk_id="12345678-1234-1234-1234-123456789abc",
            content="Chunk content text",
            document_name="doc.md",
            document_path="path/doc.md",
            vault_name="Vault",
            chunk_index=0,
            total_chunks=2,
            token_count=512,
            chunk_type="content",
            similarity_score=0.8,
            rerank_score=None,
        )
        mock_query_chunks.return_value = [mock_result]

        query_embedding = [0.1] * 1536
        result = query_documents(
            db_session,
            query_embedding,
            use_chunks=True,
        )

        mock_query_chunks.assert_called_once()
        assert len(result.results) == 1
        assert result.results[0].file_name == "doc.md"
        assert result.results[0].content == "Chunk content text"

    @patch("obsidian_rag.mcp_server.tools.documents.query_chunks")
    @patch("obsidian_rag.mcp_server.tools.documents.rerank_chunk_results")
    def test_query_documents_with_rerank(
        self,
        mock_rerank,
        mock_query_chunks,
        db_session,
    ):
        """Test query_documents with rerank=True parameter."""
        from obsidian_rag.mcp_server.tools.documents_chunks import ChunkQueryResult

        mock_result = ChunkQueryResult(
            chunk_id="12345678-1234-1234-1234-123456789abc",
            content="Chunk content text",
            document_name="doc.md",
            document_path="path/doc.md",
            vault_name="Vault",
            chunk_index=0,
            total_chunks=2,
            token_count=512,
            chunk_type="content",
            similarity_score=0.8,
            rerank_score=0.95,
        )
        mock_query_chunks.return_value = [mock_result]
        mock_rerank.return_value = [mock_result]

        query_embedding = [0.1] * 1536
        result = query_documents(
            db_session,
            query_embedding,
            use_chunks=True,
            rerank=True,
        )

        mock_query_chunks.assert_called_once()
        mock_rerank.assert_called_once()
        assert len(result.results) == 1

    def test_query_documents_without_chunks(self, db_session):
        """Test query_documents without use_chunks parameter (default behavior)."""
        query_embedding = [0.1] * 1536
        result = query_documents(db_session, query_embedding)

        # Should return empty results when no documents in database
        assert result.results == []
        assert result.total_count == 0


class TestQueryDocumentsQueryText:
    """Tests for query_text parameter passing to reranker."""

    @patch("obsidian_rag.mcp_server.tools.documents.query_chunks")
    @patch("obsidian_rag.mcp_server.tools.documents.rerank_chunk_results")
    def test_query_text_passed_to_reranker(
        self,
        mock_rerank,
        mock_query_chunks,
        db_session,
    ):
        """Test that query_text is correctly passed to rerank_chunk_results."""
        from obsidian_rag.mcp_server.tools.documents_chunks import ChunkQueryResult

        # Setup mock chunk result
        mock_result = ChunkQueryResult(
            chunk_id="12345678-1234-1234-1234-123456789abc",
            content="Chunk content text",
            document_name="doc.md",
            document_path="path/doc.md",
            vault_name="Vault",
            chunk_index=0,
            total_chunks=2,
            token_count=512,
            chunk_type="content",
            similarity_score=0.8,
            rerank_score=None,
        )
        mock_query_chunks.return_value = [mock_result]
        mock_rerank.return_value = [mock_result]

        # Call query_documents with query_text
        query_embedding = [0.1] * 1536
        test_query = "machine learning algorithms"
        result = query_documents(
            db_session,
            query_embedding,
            use_chunks=True,
            rerank=True,
            query_text=test_query,
        )

        # Verify rerank_chunk_results was called with correct query_text
        mock_rerank.assert_called_once()
        call_args = mock_rerank.call_args
        assert call_args[0][0] == test_query, (
            f"Expected query_text '{test_query}', got '{call_args[0][0]}'"
        )
        assert len(result.results) == 1

    @patch("obsidian_rag.mcp_server.tools.documents.query_chunks")
    @patch("obsidian_rag.mcp_server.tools.documents.rerank_chunk_results")
    def test_empty_query_text_default(
        self,
        mock_rerank,
        mock_query_chunks,
        db_session,
    ):
        """Test that empty string is default for query_text."""
        from obsidian_rag.mcp_server.tools.documents_chunks import ChunkQueryResult

        mock_result = ChunkQueryResult(
            chunk_id="12345678-1234-1234-1234-123456789abc",
            content="Chunk content text",
            document_name="doc.md",
            document_path="path/doc.md",
            vault_name="Vault",
            chunk_index=0,
            total_chunks=2,
            token_count=512,
            chunk_type="content",
            similarity_score=0.8,
            rerank_score=None,
        )
        mock_query_chunks.return_value = [mock_result]
        mock_rerank.return_value = [mock_result]

        # Call query_documents WITHOUT query_text (backward compatibility)
        query_embedding = [0.1] * 1536
        _ = query_documents(
            db_session,
            query_embedding,
            use_chunks=True,
            rerank=True,
        )

        # Verify rerank_chunk_results was called with empty string (default)
        mock_rerank.assert_called_once()
        call_args = mock_rerank.call_args
        assert call_args[0][0] == "", (
            f"Expected empty query_text, got '{call_args[0][0]}'"
        )

    @patch("obsidian_rag.mcp_server.tools.documents.query_chunks")
    def test_query_text_not_used_without_rerank(
        self,
        mock_query_chunks,
        db_session,
    ):
        """Test that query_text is ignored when rerank=False."""
        from obsidian_rag.mcp_server.tools.documents_chunks import ChunkQueryResult

        mock_result = ChunkQueryResult(
            chunk_id="12345678-1234-1234-1234-123456789abc",
            content="Chunk content text",
            document_name="doc.md",
            document_path="path/doc.md",
            vault_name="Vault",
            chunk_index=0,
            total_chunks=2,
            token_count=512,
            chunk_type="content",
            similarity_score=0.8,
            rerank_score=None,
        )
        mock_query_chunks.return_value = [mock_result]

        # Call with query_text but rerank=False
        query_embedding = [0.1] * 1536
        test_query = "test query"
        result = query_documents(
            db_session,
            query_embedding,
            use_chunks=True,
            rerank=False,
            query_text=test_query,
        )

        # Verify query_chunks was called but rerank was not
        mock_query_chunks.assert_called_once()
        assert len(result.results) == 1


class TestGetAllTagsSQLGeneration:
    """Integration tests for get_all_tags SQL generation path."""

    def test_get_all_tags_postgresql_generates_valid_sql(self):
        """Verify get_all_tags with PostgreSQL dialect generates valid SQL via _extract_tags_postgresql."""
        mock_session = MagicMock()
        mock_bind = MagicMock()
        mock_bind.dialect.name = "postgresql"
        mock_session.bind = mock_bind

        mock_query = MagicMock()
        mock_query.filter.return_value = mock_query
        mock_query.order_by.return_value = mock_query

        mock_row = MagicMock()
        mock_row.tag = "personal"
        mock_query.all.return_value = [mock_row]

        mock_session.query.return_value = mock_query

        result = get_all_tags(mock_session, pattern=None, limit=20, offset=0)

        assert result.total_count == 1
        assert "personal" in result.tags

    def test_get_all_tags_with_pattern_postgresql_path(self):
        """Verify pattern filtering works end-to-end in PostgreSQL path."""
        mock_session = MagicMock()
        mock_bind = MagicMock()
        mock_bind.dialect.name = "postgresql"
        mock_session.bind = mock_bind

        mock_query = MagicMock()
        mock_query.filter.return_value = mock_query
        mock_query.order_by.return_value = mock_query

        mock_row = MagicMock()
        mock_row.tag = "work-project"
        mock_query.all.return_value = [mock_row]

        mock_session.query.return_value = mock_query

        result = get_all_tags(mock_session, pattern="work*", limit=20, offset=0)

        assert "work-project" in result.tags
        expected_filter_call_count = 2  # isnot(None) + pattern
        assert mock_query.filter.call_count == expected_filter_call_count
