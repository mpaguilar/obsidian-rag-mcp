"""Tests for MCP tool definitions module."""

from unittest.mock import MagicMock, patch

import pytest

from obsidian_rag.mcp_server.handlers import QueryFilterParams
from obsidian_rag.mcp_server.tool_definitions import query_documents_tool


class TestQueryDocumentsTool:
    """Tests for query_documents_tool function."""

    @patch("obsidian_rag.mcp_server.tools.documents.query_documents")
    def test_query_text_passed_to_impl(
        self,
        mock_query_documents,
    ):
        """Test that query text is passed to query_documents."""
        from obsidian_rag.mcp_server.tools.documents_params import (
            PaginationParams,
        )

        # Setup mocks
        mock_db_manager = MagicMock()
        mock_embedding_provider = MagicMock()
        mock_embedding_provider.generate_embedding.return_value = [0.1] * 1536
        mock_session = MagicMock()
        mock_db_manager.get_session.return_value.__enter__ = MagicMock(
            return_value=mock_session
        )
        mock_db_manager.get_session.return_value.__exit__ = MagicMock(return_value=None)

        from obsidian_rag.mcp_server.models import DocumentListResponse

        mock_query_documents.return_value = DocumentListResponse(
            results=[],
            total_count=0,
            has_more=False,
            next_offset=None,
        )

        test_query = "semantic search query"
        query_documents_tool(
            db_manager=mock_db_manager,
            embedding_provider=mock_embedding_provider,
            query=test_query,
            filters=QueryFilterParams(
                include_properties=None,
                exclude_properties=None,
                include_tags=None,
                exclude_tags=None,
            ),
            pagination=PaginationParams(limit=20, offset=0),
            use_chunks=True,
            rerank=True,
        )

        # Verify query_documents was called with query_text
        mock_query_documents.assert_called_once()
        call_kwargs = mock_query_documents.call_args.kwargs
        assert call_kwargs.get("query_text") == test_query, (
            f"Expected query_text '{test_query}', got '{call_kwargs.get('query_text')}'"
        )

    def test_raises_error_without_embedding_provider(self):
        """Test that error is raised when embedding provider is None."""
        mock_db_manager = MagicMock()

        with pytest.raises(RuntimeError) as exc_info:
            query_documents_tool(
                db_manager=mock_db_manager,
                embedding_provider=None,
                query="test query",
            )

        assert "Embedding provider not configured" in str(exc_info.value)
