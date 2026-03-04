"""Tests for MCP document tools PostgreSQL path."""

import uuid
from unittest.mock import MagicMock

import pytest


class TestQueryDocumentsPostgres:
    """Tests for query_documents with PostgreSQL dialect."""

    def test_query_documents_postgresql(self):
        """Test query_documents with PostgreSQL dialect."""
        from obsidian_rag.mcp_server.tools.documents import query_documents

        # Create a mock session with PostgreSQL dialect
        mock_session = MagicMock()
        mock_bind = MagicMock()
        mock_bind.dialect.name = "postgresql"
        mock_session.bind = mock_bind

        # Mock the query results
        mock_doc = MagicMock()
        mock_doc.id = uuid.uuid4()
        mock_doc.file_path = "/test/path.md"
        mock_doc.file_name = "path.md"
        mock_doc.content = "Test content"
        mock_doc.content_vector = [0.1] * 1536
        mock_doc.kind = "note"
        mock_doc.tags = ["test"]

        mock_distance = 0.5

        # Setup query chain mocks
        mock_query = MagicMock()
        mock_query.filter.return_value = mock_query
        mock_query.order_by.return_value = mock_query
        mock_query.count.return_value = 1
        mock_query.offset.return_value.limit.return_value.all.return_value = [
            (mock_doc, mock_distance)
        ]

        mock_session.query.return_value = mock_query

        query_embedding = [0.1] * 1536
        result = query_documents(mock_session, query_embedding, limit=20, offset=0)

        assert result.total_count == 1
        assert len(result.results) == 1
        assert result.results[0].file_name == "path.md"
        assert result.has_more is False
        assert result.next_offset is None

    def test_query_documents_postgresql_empty_results(self):
        """Test query_documents with PostgreSQL returning empty results."""
        from obsidian_rag.mcp_server.tools.documents import query_documents

        # Create a mock session with PostgreSQL dialect
        mock_session = MagicMock()
        mock_bind = MagicMock()
        mock_bind.dialect.name = "postgresql"
        mock_session.bind = mock_bind

        # Setup query chain mocks for empty results
        mock_query = MagicMock()
        mock_query.filter.return_value = mock_query
        mock_query.order_by.return_value = mock_query
        mock_query.count.return_value = 0
        mock_query.offset.return_value.limit.return_value.all.return_value = []

        mock_session.query.return_value = mock_query

        query_embedding = [0.1] * 1536
        result = query_documents(mock_session, query_embedding, limit=20, offset=0)

        assert result.total_count == 0
        assert len(result.results) == 0
        assert result.has_more is False
        assert result.next_offset is None
