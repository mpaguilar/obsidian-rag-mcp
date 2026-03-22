"""Tests for chunk search option in query_documents."""

from unittest.mock import Mock, patch

import pytest

from obsidian_rag.mcp_server.tools.documents import query_documents
from obsidian_rag.mcp_server.tools.documents_chunks import ChunkQueryResult


class TestQueryDocumentsChunkOption:
    """Test cases for chunk-level search option."""

    @patch("obsidian_rag.mcp_server.tools.documents.query_documents_postgresql")
    def test_document_level_search_default(self, mock_postgres_query):
        """Test that document-level search is default."""
        mock_session = Mock()
        mock_postgres_query.return_value = Mock(results=[], total_count=0)

        result = query_documents(
            mock_session,
            [0.1] * 1536,
            use_chunks=False,  # Explicit
        )

        # Should call document-level query
        mock_postgres_query.assert_called_once()

    @patch("obsidian_rag.mcp_server.tools.documents.query_chunks")
    def test_chunk_level_search(self, mock_query_chunks):
        """Test chunk-level search when enabled."""
        mock_session = Mock()
        mock_query_chunks.return_value = []

        result = query_documents(
            mock_session,
            [0.1] * 1536,
            use_chunks=True,
        )

        # Should call chunk-level query
        mock_query_chunks.assert_called_once()
        assert result.total_count == 0

    @patch("obsidian_rag.mcp_server.tools.documents.rerank_chunk_results")
    @patch("obsidian_rag.mcp_server.tools.documents.query_chunks")
    def test_chunk_search_with_rerank(self, mock_query_chunks, mock_rerank):
        """Test chunk search with re-ranking enabled."""
        mock_session = Mock()

        # Create mock chunk results
        chunk_results = [
            ChunkQueryResult(
                chunk_id="11111111-1111-1111-1111-111111111111",
                content="Test content",
                document_name="doc.md",
                document_path="path/doc.md",
                vault_name="Vault",
                chunk_index=0,
                total_chunks=1,
                token_count=512,
                chunk_type="content",
                similarity_score=0.8,
                rerank_score=None,
            ),
        ]
        mock_query_chunks.return_value = chunk_results

        # Mock rerank to return results with rerank_score
        reranked_results = [
            ChunkQueryResult(
                chunk_id="12345678-1234-1234-1234-123456789abc",
                content="Test content",
                document_name="doc.md",
                document_path="path/doc.md",
                vault_name="Vault",
                chunk_index=0,
                total_chunks=1,
                token_count=512,
                chunk_type="content",
                similarity_score=0.8,
                rerank_score=0.95,
            ),
        ]
        mock_rerank.return_value = reranked_results

        result = query_documents(
            mock_session,
            [0.1] * 1536,
            use_chunks=True,
            rerank=True,
            rerank_model="ms-marco-MiniLM-L-12-v2",
        )

        # Should call both chunk query and rerank
        mock_query_chunks.assert_called_once()
        mock_rerank.assert_called_once()
        assert result.total_count == 1

    @patch("obsidian_rag.mcp_server.tools.documents.query_chunks")
    def test_chunk_search_empty_results(self, mock_query_chunks):
        """Test chunk search with no results."""
        mock_session = Mock()
        mock_query_chunks.return_value = []

        result = query_documents(
            mock_session,
            [0.1] * 1536,
            use_chunks=True,
        )

        assert result.total_count == 0
        assert result.results == []
        assert result.has_more is False
        assert result.next_offset is None

    @patch("obsidian_rag.mcp_server.tools.documents.query_chunks")
    def test_chunk_search_with_pagination(self, mock_query_chunks):
        """Test chunk search respects pagination limit."""
        mock_session = Mock()

        # Create multiple chunk results
        chunk_results = [
            ChunkQueryResult(
                chunk_id=f"{i:08d}-0000-0000-0000-00000000000{i}",
                content=f"Content {i}",
                document_name="doc.md",
                document_path="path/doc.md",
                vault_name="Vault",
                chunk_index=i,
                total_chunks=5,
                token_count=512,
                chunk_type="content",
                similarity_score=0.9 - (i * 0.1),
                rerank_score=None,
            )
            for i in range(5)
        ]
        mock_query_chunks.return_value = chunk_results

        from obsidian_rag.mcp_server.tools.documents_params import PaginationParams

        result = query_documents(
            mock_session,
            [0.1] * 1536,
            use_chunks=True,
            pagination=PaginationParams(limit=3, offset=0),
        )

        # Should respect the limit parameter
        mock_query_chunks.assert_called_once_with(
            mock_session,
            [0.1] * 1536,
            limit=3,
        )

    @patch("obsidian_rag.mcp_server.tools.documents.rerank_chunk_results")
    @patch("obsidian_rag.mcp_server.tools.documents.query_chunks")
    def test_chunk_search_rerank_disabled(self, mock_query_chunks, mock_rerank):
        """Test that reranking is not applied when disabled."""
        mock_session = Mock()

        chunk_results = [
            ChunkQueryResult(
                chunk_id="11111111-1111-1111-1111-111111111111",
                content="Test content",
                document_name="doc.md",
                document_path="path/doc.md",
                vault_name="Vault",
                chunk_index=0,
                total_chunks=1,
                token_count=512,
                chunk_type="content",
                similarity_score=0.8,
                rerank_score=None,
            ),
        ]
        mock_query_chunks.return_value = chunk_results

        result = query_documents(
            mock_session,
            [0.1] * 1536,
            use_chunks=True,
            rerank=False,  # Explicitly disabled
        )

        # Should not call rerank
        mock_query_chunks.assert_called_once()
        mock_rerank.assert_not_called()

    @patch("obsidian_rag.mcp_server.tools.documents.rerank_chunk_results")
    @patch("obsidian_rag.mcp_server.tools.documents.query_chunks")
    def test_chunk_search_rerank_no_results(self, mock_query_chunks, mock_rerank):
        """Test that reranking is skipped when no chunk results."""
        mock_session = Mock()
        mock_query_chunks.return_value = []

        result = query_documents(
            mock_session,
            [0.1] * 1536,
            use_chunks=True,
            rerank=True,
        )

        # Should not call rerank when no results
        mock_query_chunks.assert_called_once()
        mock_rerank.assert_not_called()
        assert result.total_count == 0
