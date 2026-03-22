"""Tests for tokenizer module."""

import sys
from unittest.mock import Mock, patch

# Clear any mocked tokenizer module from sys.modules to ensure we test the real module
if "obsidian_rag.tokenizer" in sys.modules:
    del sys.modules["obsidian_rag.tokenizer"]

from obsidian_rag.tokenizer import (
    TokenizerConfig,
    TokenizerError,
    count_tokens,
    get_tokenizer,
    initialize_tokenizer,
    _tokenizer_cache,
    clear_tokenizer_cache,
)


class TestTokenizerConfig:
    """Test cases for TokenizerConfig dataclass."""

    def test_default_config_values(self):
        """Test default configuration values."""
        config = TokenizerConfig()
        assert config.chunk_size == 512
        assert config.chunk_overlap == 50
        assert config.model_name == "gpt2"

    def test_custom_config_values(self):
        """Test custom configuration values."""
        config = TokenizerConfig(
            chunk_size=256, chunk_overlap=25, model_name="cl100k_base"
        )
        assert config.chunk_size == 256
        assert config.chunk_overlap == 25
        assert config.model_name == "cl100k_base"


class TestInitializeTokenizer:
    """Test cases for initialize_tokenizer function."""

    @patch("tokenizers.Tokenizer")
    def test_initialize_success(self, mock_tokenizer_class):
        """Test successful tokenizer initialization."""
        mock_tokenizer = Mock()
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        result = initialize_tokenizer("gpt2")

        mock_tokenizer_class.from_pretrained.assert_called_once_with("gpt2")
        assert result == mock_tokenizer

    @patch("tokenizers.Tokenizer")
    def test_initialize_fallback_on_error(self, mock_tokenizer_class):
        """Test fallback to character heuristic on tokenizer error."""
        mock_tokenizer_class.from_pretrained.side_effect = OSError("Download failed")

        result = initialize_tokenizer("gpt2")
        assert result is None


class TestCountTokens:
    """Test cases for count_tokens function."""

    def test_count_tokens_with_tokenizer(self):
        """Test token counting with initialized tokenizer."""
        mock_tokenizer = Mock()
        mock_tokenizer.encode.return_value = Mock(ids=list(range(512)))

        result = count_tokens("Test text", mock_tokenizer)
        assert result == 512

    def test_count_tokens_fallback(self):
        """Test character heuristic fallback when tokenizer is None."""
        # ~4 chars per token, so 100 chars = ~25 tokens
        result = count_tokens("a" * 100, None)
        assert result == 25

    def test_count_tokens_exception_fallback(self):
        """Test character heuristic when tokenizer.encode raises exception."""
        mock_tokenizer = Mock()
        mock_tokenizer.encode.side_effect = RuntimeError("Tokenization error")

        result = count_tokens("a" * 100, mock_tokenizer)
        # Should fall back to character heuristic: 100 // 4 = 25
        assert result == 25


class TestGetTokenizer:
    """Test cases for get_tokenizer function."""

    def setup_method(self):
        """Clear cache before each test."""
        clear_tokenizer_cache()

    @patch.object(sys.modules["obsidian_rag.tokenizer"], "initialize_tokenizer")
    def test_get_tokenizer_creates_new(self, mock_init):
        """Test that get_tokenizer creates tokenizer if not cached."""
        mock_tokenizer = Mock()
        mock_init.return_value = mock_tokenizer

        result = get_tokenizer("gpt2")

        mock_init.assert_called_once_with("gpt2")
        assert result == mock_tokenizer

    @patch.object(sys.modules["obsidian_rag.tokenizer"], "initialize_tokenizer")
    def test_get_tokenizer_returns_cached(self, mock_init):
        """Test that get_tokenizer returns cached tokenizer."""
        mock_tokenizer = Mock()
        mock_init.return_value = mock_tokenizer

        # First call creates and caches
        result1 = get_tokenizer("gpt2")
        assert result1 == mock_tokenizer
        assert mock_init.call_count == 1

        # Second call should return cached
        result2 = get_tokenizer("gpt2")
        assert result2 == mock_tokenizer
        # Should not call initialize again
        assert mock_init.call_count == 1


class TestTokenizerError:
    """Test cases for TokenizerError exception."""

    def test_tokenizer_error_is_exception(self):
        """Test that TokenizerError is an Exception subclass."""
        err = TokenizerError("Test error")
        assert isinstance(err, Exception)
        assert str(err) == "Test error"


class TestClearTokenizerCache:
    """Test cases for clear_tokenizer_cache function."""

    def setup_method(self):
        """Clear cache before each test."""
        clear_tokenizer_cache()

    def test_clear_cache(self):
        """Test that clear_tokenizer_cache clears the cache."""
        # Add something to cache
        _tokenizer_cache["test"] = Mock()
        assert len(_tokenizer_cache) == 1

        # Clear it
        clear_tokenizer_cache()
        assert len(_tokenizer_cache) == 0
