"""Tests for chunk-level document search."""

from unittest.mock import Mock, patch

from obsidian_rag.mcp_server.tools.documents_chunks import (
    ChunkQueryResult,
    query_chunks,
    rerank_chunk_results,
)


class TestChunkQueryResult:
    """Test cases for ChunkQueryResult dataclass."""

    def test_chunk_query_result_creation(self):
        """Test ChunkQueryResult creation."""
        result = ChunkQueryResult(
            chunk_id="chunk-1",
            content="Test content",
            document_name="doc.md",
            document_path="path/doc.md",
            vault_name="Test Vault",
            chunk_index=0,
            total_chunks=5,
            token_count=512,
            chunk_type="content",
            similarity_score=0.89,
            rerank_score=None,
        )
        assert result.chunk_id == "chunk-1"
        assert result.similarity_score == 0.89
        assert result.rerank_score is None


class TestQueryChunks:
    """Test cases for query_chunks function."""

    def test_query_chunks_postgresql(self):
        """Test chunk query with PostgreSQL."""
        mock_session = Mock()

        # Mock query results
        mock_chunk = Mock()
        mock_chunk.id = "chunk-1"
        mock_chunk.chunk_text = "Test content"
        mock_chunk.document.file_name = "doc.md"
        mock_chunk.document.file_path = "path/doc.md"
        mock_chunk.document.vault.name = "Test Vault"
        mock_chunk.chunk_index = 0
        mock_chunk.token_count = 512
        mock_chunk.chunk_type = "content"
        mock_chunk.document_id = "doc-1"

        # Setup mock chain for the main query
        mock_query = Mock()
        mock_query.join.return_value = mock_query
        mock_query.filter.return_value = mock_query
        mock_query.order_by.return_value = mock_query
        mock_query.limit.return_value = mock_query
        mock_query.all.return_value = [
            (mock_chunk, 0.15),  # distance = 0.15, similarity = 0.85
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

        result = query_chunks(
            mock_session,
            [0.1] * 1536,
            limit=10,
        )

        assert len(result) == 1
        assert result[0].chunk_id == "chunk-1"
        assert result[0].similarity_score == 0.85  # 1.0 - 0.15
        assert result[0].content == "Test content"
        assert result[0].document_name == "doc.md"
        assert result[0].vault_name == "Test Vault"

    def test_query_chunks_with_vault_filter(self):
        """Test chunk query with vault name filter."""
        mock_session = Mock()

        # Mock query results
        mock_chunk = Mock()
        mock_chunk.id = "chunk-1"
        mock_chunk.chunk_text = "Test content"
        mock_chunk.document.file_name = "doc.md"
        mock_chunk.document.file_path = "path/doc.md"
        mock_chunk.document.vault.name = "Test Vault"
        mock_chunk.chunk_index = 0
        mock_chunk.token_count = 512
        mock_chunk.chunk_type = "content"
        mock_chunk.document_id = "doc-1"

        # Setup mock chain for main query
        mock_query = Mock()
        mock_query.join.return_value = mock_query
        mock_filtered_query = Mock()
        mock_query.filter.return_value = mock_filtered_query
        mock_filtered_query.order_by.return_value = mock_filtered_query
        mock_filtered_query.limit.return_value = mock_filtered_query
        mock_filtered_query.all.return_value = [
            (mock_chunk, 0.10),
        ]

        # Setup mock for count query (separate chain)
        mock_count_query = Mock()
        mock_count_filtered = Mock()
        mock_count_query.filter.return_value = mock_count_filtered
        mock_count_filtered.scalar.return_value = 3

        def mock_query_side_effect(*args):
            # Main query has 2 args: DocumentChunk, distance label
            # Count query has 1 arg: func.count
            if len(args) == 2:
                return mock_query
            return mock_count_query

        mock_session.query.side_effect = mock_query_side_effect

        result = query_chunks(
            mock_session,
            [0.1] * 1536,
            vault_name="Test Vault",
            limit=10,
        )

        assert len(result) == 1
        assert result[0].chunk_id == "chunk-1"
        # Verify filter was called on main query (vault filter applied)
        assert mock_query.filter.call_count >= 1

    def test_query_chunks_empty_results(self):
        """Test chunk query with no results."""
        mock_session = Mock()

        # Setup mock chain for empty results
        mock_query = Mock()
        mock_query.join.return_value = mock_query
        mock_query.filter.return_value = mock_query
        mock_query.order_by.return_value = mock_query
        mock_query.limit.return_value = mock_query
        mock_query.all.return_value = []

        mock_session.query.return_value = mock_query

        result = query_chunks(
            mock_session,
            [0.1] * 1536,
            limit=10,
        )

        assert len(result) == 0


class TestRerankChunkResults:
    """Test cases for rerank_chunk_results function."""

    @patch("obsidian_rag.mcp_server.tools.documents_chunks.create_reranker")
    @patch("obsidian_rag.mcp_server.tools.documents_chunks.rerank_chunks")
    def test_rerank_results(self, mock_rerank, mock_create_reranker):
        """Test re-ranking chunk results."""
        mock_reranker = Mock()
        mock_create_reranker.return_value = mock_reranker

        mock_rerank.return_value = [
            Mock(
                chunk_id="chunk-2",
                content="Content 2",
                score=0.95,
                original_rank=2,
                new_rank=1,
            ),
        ]

        chunks = [
            ChunkQueryResult(
                chunk_id="chunk-1",
                content="Content 1",
                document_name="doc.md",
                document_path="path/doc.md",
                vault_name="Vault",
                chunk_index=0,
                total_chunks=2,
                token_count=512,
                chunk_type="content",
                similarity_score=0.80,
                rerank_score=None,
            ),
            ChunkQueryResult(
                chunk_id="chunk-2",
                content="Content 2",
                document_name="doc.md",
                document_path="path/doc.md",
                vault_name="Vault",
                chunk_index=1,
                total_chunks=2,
                token_count=512,
                chunk_type="content",
                similarity_score=0.75,
                rerank_score=None,
            ),
        ]

        result = rerank_chunk_results("query", chunks, "ms-marco", 128, 10)

        assert len(result) == 1
        assert result[0].rerank_score == 0.95
        assert result[0].chunk_id == "chunk-2"
        assert result[0].similarity_score == 0.75  # Original score preserved

    @patch("obsidian_rag.mcp_server.tools.documents_chunks.create_reranker")
    def test_rerank_empty_chunks(self, mock_create_reranker):
        """Test re-ranking with empty chunks list."""
        result = rerank_chunk_results("query", [], "ms-marco", 128, 10)

        assert len(result) == 0
        mock_create_reranker.assert_not_called()

    @patch("obsidian_rag.mcp_server.tools.documents_chunks.create_reranker")
    def test_rerank_unavailable(self, mock_create_reranker):
        """Test re-ranking when reranker is unavailable."""
        mock_create_reranker.return_value = None

        chunks = [
            ChunkQueryResult(
                chunk_id="chunk-1",
                content="Content 1",
                document_name="doc.md",
                document_path="path/doc.md",
                vault_name="Vault",
                chunk_index=0,
                total_chunks=1,
                token_count=512,
                chunk_type="content",
                similarity_score=0.80,
                rerank_score=None,
            ),
        ]

        result = rerank_chunk_results("query", chunks, "ms-marco", 128, 10)

        # Should return original chunks when reranker unavailable
        assert len(result) == 1
        assert result[0].chunk_id == "chunk-1"
        assert result[0].rerank_score is None

    @patch("obsidian_rag.mcp_server.tools.documents_chunks.create_reranker")
    @patch("obsidian_rag.mcp_server.tools.documents_chunks.rerank_chunks")
    def test_rerank_multiple_results(self, mock_rerank, mock_create_reranker):
        """Test re-ranking with multiple results."""
        mock_reranker = Mock()
        mock_create_reranker.return_value = mock_reranker

        mock_rerank.return_value = [
            Mock(
                chunk_id="chunk-2",
                content="Content 2",
                score=0.95,
                original_rank=2,
                new_rank=1,
            ),
            Mock(
                chunk_id="chunk-1",
                content="Content 1",
                score=0.85,
                original_rank=1,
                new_rank=2,
            ),
        ]

        chunks = [
            ChunkQueryResult(
                chunk_id="chunk-1",
                content="Content 1",
                document_name="doc.md",
                document_path="path/doc.md",
                vault_name="Vault",
                chunk_index=0,
                total_chunks=2,
                token_count=512,
                chunk_type="content",
                similarity_score=0.80,
                rerank_score=None,
            ),
            ChunkQueryResult(
                chunk_id="chunk-2",
                content="Content 2",
                document_name="doc.md",
                document_path="path/doc.md",
                vault_name="Vault",
                chunk_index=1,
                total_chunks=2,
                token_count=512,
                chunk_type="content",
                similarity_score=0.75,
                rerank_score=None,
            ),
        ]

        result = rerank_chunk_results("query", chunks, "ms-marco", 128, 10)

        assert len(result) == 2
        # Results should be in reranked order
        assert result[0].chunk_id == "chunk-2"
        assert result[0].rerank_score == 0.95
        assert result[1].chunk_id == "chunk-1"
        assert result[1].rerank_score == 0.85
