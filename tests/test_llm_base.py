"""Tests for LLM base module."""

from unittest.mock import Mock, patch

import pytest

from obsidian_rag.llm.base import (
    ChatError,
    ChatProvider,
    EmbeddingError,
    EmbeddingProvider,
    ProviderError,
    ProviderFactory,
)


class TestProviderError:
    """Test cases for ProviderError exception."""

    def test_is_exception(self):
        """Test that ProviderError is an exception."""
        with pytest.raises(ProviderError):
            raise ProviderError("test error")


class TestEmbeddingError:
    """Test cases for EmbeddingError exception."""

    def test_is_provider_error(self):
        """Test that EmbeddingError is a ProviderError."""
        with pytest.raises(ProviderError):
            raise EmbeddingError("embedding failed")


class TestChatError:
    """Test cases for ChatError exception."""

    def test_is_provider_error(self):
        """Test that ChatError is a ProviderError."""
        with pytest.raises(ProviderError):
            raise ChatError("chat failed")


class TestEmbeddingProvider:
    """Test cases for EmbeddingProvider abstract class."""

    def test_cannot_instantiate_directly(self):
        """Test that EmbeddingProvider cannot be instantiated directly."""
        with pytest.raises(TypeError):
            EmbeddingProvider()

    def test_subclass_must_implement_methods(self):
        """Test that subclass must implement abstract methods."""

        class IncompleteProvider(EmbeddingProvider):
            pass

        with pytest.raises(TypeError):
            IncompleteProvider()

    def test_can_create_complete_subclass(self):
        """Test creating a complete subclass."""

        class CompleteProvider(EmbeddingProvider):
            def generate_embedding(self, text):
                return [0.1, 0.2, 0.3]

            def get_dimension(self):
                return 3

        provider = CompleteProvider()
        assert provider.generate_embedding("test") == [0.1, 0.2, 0.3]
        assert provider.get_dimension() == 3


class TestChatProvider:
    """Test cases for ChatProvider abstract class."""

    def test_cannot_instantiate_directly(self):
        """Test that ChatProvider cannot be instantiated directly."""
        with pytest.raises(TypeError):
            ChatProvider()

    def test_subclass_must_implement_chat(self):
        """Test that subclass must implement chat method."""

        class IncompleteProvider(ChatProvider):
            pass

        with pytest.raises(TypeError):
            IncompleteProvider()

    def test_can_create_complete_subclass(self):
        """Test creating a complete subclass."""

        class CompleteProvider(ChatProvider):
            def chat(self, messages, **kwargs):
                return "Response"

        provider = CompleteProvider()
        assert provider.chat([{"role": "user", "content": "Hello"}]) == "Response"


class TestProviderFactory:
    """Test cases for ProviderFactory class."""

    def test_create_openai_embedding_provider_no_api_key(self):
        """Test creating OpenAI embedding provider without API key."""
        # When API key is not provided, ValueError should be raised
        with pytest.raises(ValueError, match="API key is required"):
            ProviderFactory.create_embedding_provider("openai")

    def test_create_huggingface_embedding_provider(self):
        """Test creating HuggingFace embedding provider."""
        from obsidian_rag.llm.providers import HuggingFaceEmbeddingProvider

        with patch("obsidian_rag.llm.providers.log"):
            with patch(
                "langchain_huggingface.HuggingFaceEmbeddings",
            ) as mock_hf:
                mock_hf.return_value = Mock()
                provider = ProviderFactory.create_embedding_provider("huggingface")

        assert isinstance(provider, HuggingFaceEmbeddingProvider)

    def test_create_unknown_embedding_provider(self):
        """Test creating unknown embedding provider."""
        with pytest.raises(ValueError, match="Unknown embedding provider"):
            ProviderFactory.create_embedding_provider("unknown")

    def test_create_unknown_chat_provider(self):
        """Test creating unknown chat provider."""
        with pytest.raises(ValueError, match="Unknown chat provider"):
            ProviderFactory.create_chat_provider("unknown")

    def test_create_openai_chat_provider_no_api_key(self):
        """Test creating OpenAI chat provider without API key."""
        # When API key is not provided, ValueError should be raised
        with pytest.raises(ValueError, match="API key is required"):
            ProviderFactory.create_chat_provider("openai")

    def test_create_openrouter_embedding_provider(self):
        """Test creating OpenRouter embedding provider."""
        from obsidian_rag.llm.providers import OpenRouterEmbeddingProvider

        with patch("obsidian_rag.llm.providers.log"):
            with patch("litellm.embedding"):
                provider = ProviderFactory.create_embedding_provider(
                    "openrouter",
                    api_key="test-key",
                )

        assert isinstance(provider, OpenRouterEmbeddingProvider)
        assert provider.model == "qwen/qwen3-embedding-8b"
        assert provider.get_dimension() == 4096

    def test_create_openrouter_embedding_provider_no_api_key(self):
        """Test creating OpenRouter embedding provider without API key."""
        # When API key is not provided, ValueError should be raised
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="API key is required"):
                ProviderFactory.create_embedding_provider("openrouter")

    def test_create_openrouter_chat_provider(self):
        """Test creating OpenRouter chat provider."""
        from obsidian_rag.llm.providers import OpenRouterChatProvider

        with patch("obsidian_rag.llm.providers.log"):
            with patch("litellm.completion"):
                provider = ProviderFactory.create_chat_provider(
                    "openrouter",
                    api_key="test-key",
                )

        assert isinstance(provider, OpenRouterChatProvider)
        assert provider.model == "anthropic/claude-3-opus"

    def test_create_openrouter_chat_provider_no_api_key(self):
        """Test creating OpenRouter chat provider without API key."""
        # When API key is not provided, ValueError should be raised
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="API key is required"):
                ProviderFactory.create_chat_provider("openrouter")
