"""Integration tests for flashrank re-ranking."""

import builtins
import sys
import types
from unittest.mock import Mock, patch

from obsidian_rag.reranking import rerank_chunks


class TestFlashrankIntegration:
    """Integration tests for flashrank."""

    def test_create_reranker_integration(self):
        """Test creating a real reranker."""
        mock_reranker = Mock()
        mock_reranker_class = Mock(return_value=mock_reranker)

        # Create a proper mock module with Reranker class using types.ModuleType
        mock_flashrank = types.ModuleType("flashrank")
        mock_flashrank.Reranker = mock_reranker_class  # type: ignore[attr-defined]

        # Custom import function that returns our mock for flashrank
        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "flashrank":
                return mock_flashrank
            return original_import(name, *args, **kwargs)

        # Clear the module from cache
        if "obsidian_rag.reranking" in sys.modules:
            del sys.modules["obsidian_rag.reranking"]

        with patch.object(builtins, "__import__", mock_import):
            from obsidian_rag.reranking import create_reranker

            result = create_reranker("ms-marco-MiniLM-L-12-v2")

        mock_reranker_class.assert_called_once()
        assert result is not None

    def test_rerank_chunks_integration(self):
        """Test re-ranking chunks."""
        expected_top_k = 2
        expected_score = 0.95

        mock_reranker = Mock()
        mock_reranker.rerank.return_value = [
            {"id": "chunk-2", "text": "Content 2", "score": expected_score},
            {"id": "chunk-1", "text": "Content 1", "score": 0.85},
        ]

        chunks = [
            {"chunk_id": "chunk-1", "content": "Content 1"},
            {"chunk_id": "chunk-2", "content": "Content 2"},
        ]

        result = rerank_chunks(
            "test query", chunks, mock_reranker, top_k=expected_top_k
        )

        assert len(result) == expected_top_k
        assert result[0].chunk_id == "chunk-2"
        assert result[0].score == expected_score

    def test_reranker_unavailable_graceful(self):
        """Test graceful handling when reranker is unavailable."""
        # Clear the module from cache to force re-import
        if "obsidian_rag.reranking" in sys.modules:
            del sys.modules["obsidian_rag.reranking"]

        # Simulate flashrank not being installed
        with patch.dict("sys.modules", {"flashrank": None}):
            from obsidian_rag.reranking import create_reranker

            result = create_reranker("ms-marco-MiniLM-L-12-v2")

        assert result is None
