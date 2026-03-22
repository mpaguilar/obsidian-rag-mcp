"""Integration tests for chunk-level search."""

from unittest.mock import Mock, patch

from obsidian_rag.mcp_server.tools.documents_chunks import (
    ChunkQueryResult,
    query_chunks,
    rerank_chunk_results,
)

# Constants for test values
EXPECTED_SIMILARITY_SCORE = 0.85
EXPECTED_RERANK_SCORE = 0.9
VECTOR_DIMENSION = 1536
TEST_DISTANCE = 0.15


class TestChunkSearchIntegration:
    """Integration tests for chunk search."""

    def test_query_chunks_returns_results(self):
        """Test that query_chunks returns properly formatted results."""
        # This would be tested with a real database session
        # For now, we verify the function signature and mocking
        mock_session = Mock()

        # Mock the query chain
        mock_chunk = Mock()
        mock_chunk.id = "chunk-1"
        mock_chunk.chunk_text = "Test content"
        mock_chunk.chunk_index = 0
        mock_chunk.token_count = 512
        mock_chunk.chunk_type = "content"
        mock_chunk.document.file_name = "test.md"
        mock_chunk.document.file_path = "test.md"
        mock_chunk.document.vault.name = "Test Vault"
        mock_chunk.document_id = "doc-1"

        # Setup mock chain for the main query
        mock_query = Mock()
        mock_query.join.return_value = mock_query
        mock_query.filter.return_value = mock_query
        mock_query.order_by.return_value = mock_query
        mock_query.limit.return_value = mock_query
        mock_query.all.return_value = [
            (mock_chunk, TEST_DISTANCE),  # distance
        ]

        # Setup mock for count query
        mock_count_query = Mock()
        mock_count_query.filter.return_value = mock_count_query
        mock_count_query.scalar.return_value = 5

        # Configure session.query to return different mocks based on call
        def mock_query_side_effect(*args):
            if (
                len(args) == 1
                and hasattr(args[0], "__name__")
                and args[0].__name__ == "count"
            ):
                return mock_count_query
            return mock_query

        mock_session.query.side_effect = mock_query_side_effect

        results = query_chunks(mock_session, [0.1] * VECTOR_DIMENSION, limit=10)

        assert len(results) == 1
        assert results[0].chunk_id == "chunk-1"
        assert results[0].similarity_score == EXPECTED_SIMILARITY_SCORE

    def test_rerank_integration(self):
        """Test re-ranking integration."""
        chunks = [
            ChunkQueryResult(
                chunk_id="c1",
                content="Content 1",
                document_name="doc.md",
                document_path="path/doc.md",
                vault_name="Vault",
                chunk_index=0,
                total_chunks=2,
                token_count=512,
                chunk_type="content",
                similarity_score=0.8,
                rerank_score=None,
            ),
        ]

        with patch(
            "obsidian_rag.mcp_server.tools.documents_chunks.create_reranker"
        ) as mock_create:
            mock_reranker = Mock()
            mock_reranker.rerank.return_value = [
                {"id": "c1", "text": "Content 1", "score": EXPECTED_RERANK_SCORE},
            ]
            mock_create.return_value = mock_reranker

            result = rerank_chunk_results("query", chunks, "model", 128, 10)

            assert len(result) == 1
            assert result[0].rerank_score == EXPECTED_RERANK_SCORE
