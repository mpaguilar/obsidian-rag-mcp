"""Tests for reranking module."""

import builtins
import sys
import types
from unittest.mock import Mock, patch

from obsidian_rag.reranking import (
    RerankConfig,
    RerankError,
    RerankResult,
    rerank_chunks,
)


class TestRerankConfig:
    """Test cases for RerankConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = RerankConfig()
        assert config.model == "ms-marco-MiniLM-L-12-v2"
        assert config.max_length == 128
        assert config.top_k == 10
        assert config.enabled is True

    def test_custom_config(self):
        """Test custom configuration values."""
        config = RerankConfig(
            model="rank-T5-flan",
            max_length=256,
            top_k=5,
            enabled=False,
        )
        assert config.model == "rank-T5-flan"
        assert config.max_length == 256
        assert config.top_k == 5
        assert config.enabled is False


class TestRerankResult:
    """Test cases for RerankResult dataclass."""

    def test_rerank_result_creation(self):
        """Test RerankResult creation."""
        result = RerankResult(
            chunk_id="uuid-123",
            content="Test content",
            score=0.95,
            original_rank=5,
            new_rank=1,
        )
        assert result.chunk_id == "uuid-123"
        assert result.content == "Test content"
        assert result.score == 0.95
        assert result.original_rank == 5
        assert result.new_rank == 1


class TestCreateReranker:
    """Test cases for create_reranker function."""

    def _reload_module_with_flashrank_mock(self, mock_flashrank_module):
        """Helper to reload reranking module with mocked flashrank."""
        # Clear the module from cache
        if "obsidian_rag.reranking" in sys.modules:
            del sys.modules["obsidian_rag.reranking"]

        # Set up the mock in sys.modules
        with patch.dict("sys.modules", {"flashrank": mock_flashrank_module}):
            from obsidian_rag.reranking import create_reranker

            return create_reranker

    def test_create_reranker_success(self):
        """Test successful reranker creation."""
        mock_reranker = Mock()
        mock_reranker_class = Mock(return_value=mock_reranker)

        # Create a proper mock module with Ranker class using types.ModuleType
        mock_flashrank = types.ModuleType("flashrank")
        mock_flashrank.Ranker = mock_reranker_class  # type: ignore[attr-defined]
        mock_flashrank.RerankRequest = Mock  # type: ignore[attr-defined]

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

        mock_reranker_class.assert_called_once_with(
            model_name="ms-marco-MiniLM-L-12-v2",
            max_length=128,
        )
        assert result == mock_reranker

    def test_create_reranker_import_error(self):
        """Test handling of import error."""
        # Clear the module from cache to force re-import
        if "obsidian_rag.reranking" in sys.modules:
            del sys.modules["obsidian_rag.reranking"]

        # Simulate flashrank not being installed
        with patch.dict("sys.modules", {"flashrank": None}):
            from obsidian_rag.reranking import create_reranker

            result = create_reranker("ms-marco-MiniLM-L-12-v2")

        assert result is None

    def test_create_reranker_oserror(self):
        """Test handling of OSError during reranker creation."""
        mock_reranker_class = Mock(side_effect=OSError("Model file not found"))

        # Create a proper mock module with Ranker class using types.ModuleType
        mock_flashrank = types.ModuleType("flashrank")
        mock_flashrank.Ranker = mock_reranker_class  # type: ignore[attr-defined]
        mock_flashrank.RerankRequest = Mock  # type: ignore[attr-defined]

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

        assert result is None


class TestRerankChunks:
    """Test cases for rerank_chunks function."""

    def test_rerank_chunks_success(self):
        """Test successful chunk re-ranking with RerankRequest."""
        import types

        # Create a mock flashrank module with RerankRequest
        mock_flashrank = types.ModuleType("flashrank")
        mock_flashrank.Ranker = Mock()  # type: ignore[attr-defined]

        # Create a simple RerankRequest class for testing
        class RerankRequest:
            def __init__(self, query, passages):
                self.query = query
                self.passages = passages

        mock_flashrank.RerankRequest = RerankRequest  # type: ignore[attr-defined]

        # Custom import function
        import builtins

        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "flashrank":
                return mock_flashrank
            return original_import(name, *args, **kwargs)

        # Clear module cache and reload with mock
        if "obsidian_rag.reranking" in sys.modules:
            del sys.modules["obsidian_rag.reranking"]

        with patch.object(builtins, "__import__", mock_import):
            from obsidian_rag.reranking import rerank_chunks

            mock_reranker = Mock()
            mock_reranker.rerank.return_value = [
                {"id": "chunk-2", "text": "Content 2", "score": 0.95},
                {"id": "chunk-1", "text": "Content 1", "score": 0.85},
            ]

            chunks = [
                {"chunk_id": "chunk-1", "content": "Content 1"},
                {"chunk_id": "chunk-2", "content": "Content 2"},
            ]

            result = rerank_chunks("test query", chunks, mock_reranker, top_k=2)

            # Verify RerankRequest was passed to rerank
            call_args = mock_reranker.rerank.call_args
            assert call_args is not None
            rerank_request = call_args[0][0]  # First positional argument
            assert isinstance(rerank_request, RerankRequest)
            assert rerank_request.query == "test query"
            assert len(rerank_request.passages) == 2

            assert len(result) == 2
            assert result[0].chunk_id == "chunk-2"
            assert result[0].score == 0.95
            assert result[0].new_rank == 1
            assert result[1].chunk_id == "chunk-1"
            assert result[1].score == 0.85
            assert result[1].new_rank == 2

    def test_rerank_chunks_empty(self):
        """Test re-ranking with empty chunks."""
        mock_reranker = Mock()

        result = rerank_chunks("query", [], mock_reranker, top_k=10)

        assert result == []
        mock_reranker.rerank.assert_not_called()

    def test_rerank_chunks_error(self):
        """Test handling of reranking error with RerankRequest."""
        import types

        # Create a mock flashrank module with RerankRequest
        mock_flashrank = types.ModuleType("flashrank")
        mock_flashrank.Ranker = Mock()  # type: ignore[attr-defined]

        class RerankRequest:
            def __init__(self, query, passages):
                self.query = query
                self.passages = passages

        mock_flashrank.RerankRequest = RerankRequest  # type: ignore[attr-defined]

        import builtins

        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "flashrank":
                return mock_flashrank
            return original_import(name, *args, **kwargs)

        if "obsidian_rag.reranking" in sys.modules:
            del sys.modules["obsidian_rag.reranking"]

        with patch.object(builtins, "__import__", mock_import):
            from obsidian_rag.reranking import rerank_chunks

            mock_reranker = Mock()
            mock_reranker.rerank.side_effect = RuntimeError("Inference failed")

            chunks = [
                {"chunk_id": "chunk-1", "content": "Content 1"},
            ]

            result = rerank_chunks("query", chunks, mock_reranker, top_k=10)

            # Should return empty list on error
            assert result == []

    def test_rerank_chunks_none_reranker(self):
        """Test re-ranking with None reranker."""
        chunks = [
            {"chunk_id": "chunk-1", "content": "Content 1"},
        ]

        result = rerank_chunks("query", chunks, None, top_k=10)

        # Should return empty list when reranker is None
        assert result == []

    def test_rerank_chunks_no_rerankrequest(self):
        """Test re-ranking when RerankRequest is not available."""
        import types

        # Create a mock flashrank module without RerankRequest
        mock_flashrank = types.ModuleType("flashrank")
        mock_flashrank.Ranker = Mock()  # type: ignore[attr-defined]
        # No RerankRequest attribute

        import builtins

        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "flashrank":
                return mock_flashrank
            return original_import(name, *args, **kwargs)

        if "obsidian_rag.reranking" in sys.modules:
            del sys.modules["obsidian_rag.reranking"]

        with patch.object(builtins, "__import__", mock_import):
            from obsidian_rag.reranking import rerank_chunks

            mock_reranker = Mock()
            chunks = [
                {"chunk_id": "chunk-1", "content": "Content 1"},
            ]

            result = rerank_chunks("query", chunks, mock_reranker, top_k=10)

            # Should return empty list when RerankRequest is not available
            assert result == []


class TestRerankError:
    """Test cases for RerankError exception."""

    def test_rerank_error_is_exception(self):
        """Test that RerankError is an Exception subclass."""
        err = RerankError("Test error")
        assert isinstance(err, Exception)
        assert str(err) == "Test error"
