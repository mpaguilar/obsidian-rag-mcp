"""Tests for LLM provider implementations."""

from unittest.mock import Mock, patch

import pytest

from obsidian_rag.llm.base import ChatError, EmbeddingError


class TestOpenAIEmbeddingProvider:
    """Test cases for OpenAIEmbeddingProvider."""

    def test_init_with_api_key(self):
        """Test initialization with explicit API key."""
        from obsidian_rag.llm.providers import OpenAIEmbeddingProvider

        with patch("obsidian_rag.llm.providers.log"):
            provider = OpenAIEmbeddingProvider(api_key="test-key")

        assert provider.api_key == "test-key"
        assert provider.model == "text-embedding-3-small"
        assert provider._dimension == 1536

    def test_init_without_api_key_uses_env_var(self):
        """Test initialization uses OPENAI_API_KEY env var when api_key not provided."""
        from obsidian_rag.llm.providers import OpenAIEmbeddingProvider

        with patch("obsidian_rag.llm.providers.log"):
            with patch.dict("os.environ", {"OPENAI_API_KEY": "env-key"}):
                provider = OpenAIEmbeddingProvider()

        assert provider.api_key == "env-key"

    def test_init_without_api_key_raises_error(self):
        """Test initialization raises ValueError when no API key available."""
        from obsidian_rag.llm.providers import OpenAIEmbeddingProvider

        with patch("obsidian_rag.llm.providers.log"):
            with patch.dict("os.environ", {}, clear=True):
                with pytest.raises(ValueError, match="API key is required"):
                    OpenAIEmbeddingProvider()

    def test_init_with_custom_model(self):
        """Test initialization with custom model name."""
        from obsidian_rag.llm.providers import OpenAIEmbeddingProvider

        with patch("obsidian_rag.llm.providers.log"):
            provider = OpenAIEmbeddingProvider(
                api_key="test-key",
                model="text-embedding-3-large",
            )

        assert provider.model == "text-embedding-3-large"
        assert provider._dimension == 3072

    def test_init_with_base_url(self):
        """Test initialization with custom base URL."""
        from obsidian_rag.llm.providers import OpenAIEmbeddingProvider

        with patch("obsidian_rag.llm.providers.log"):
            provider = OpenAIEmbeddingProvider(
                api_key="test-key",
                base_url="https://custom.api.com",
            )

        assert provider.base_url == "https://custom.api.com"

    def test_get_dimension(self):
        """Test get_dimension returns correct dimension."""
        from obsidian_rag.llm.providers import OpenAIEmbeddingProvider

        with patch("obsidian_rag.llm.providers.log"):
            provider = OpenAIEmbeddingProvider(api_key="test-key")

        assert provider.get_dimension() == 1536

    def test_generate_embedding_success(self):
        """Test successful embedding generation."""
        from obsidian_rag.llm.providers import OpenAIEmbeddingProvider

        mock_response = {
            "data": [{"embedding": [0.1, 0.2, 0.3]}],
        }

        with patch("obsidian_rag.llm.providers.log"):
            with patch("litellm.embedding", return_value=mock_response):
                provider = OpenAIEmbeddingProvider(api_key="test-key")
                result = provider.generate_embedding("test text")

        assert result == [0.1, 0.2, 0.3]

    def test_generate_embedding_error(self):
        """Test embedding generation error handling."""
        from obsidian_rag.llm.providers import OpenAIEmbeddingProvider

        with patch("obsidian_rag.llm.providers.log"):
            with patch("litellm.embedding", side_effect=Exception("API Error")):
                provider = OpenAIEmbeddingProvider(api_key="test-key")

                with pytest.raises(EmbeddingError, match="API Error"):
                    provider.generate_embedding("test text")

    def test_generate_embedding_with_base_url(self):
        """Test embedding generation with custom base URL."""
        from obsidian_rag.llm.providers import OpenAIEmbeddingProvider

        mock_response = {
            "data": [{"embedding": [0.1, 0.2, 0.3]}],
        }

        with patch("obsidian_rag.llm.providers.log"):
            with patch("litellm.embedding", return_value=mock_response) as mock_embed:
                provider = OpenAIEmbeddingProvider(
                    api_key="test-key",
                    base_url="https://custom.api.com",
                )
                provider.generate_embedding("test text")

        mock_embed.assert_called_once()
        call_kwargs = mock_embed.call_args.kwargs
        assert call_kwargs["api_base"] == "https://custom.api.com"


class TestOpenAIChatProvider:
    """Test cases for OpenAIChatProvider."""

    def test_init_with_api_key(self):
        """Test initialization with explicit API key."""
        from obsidian_rag.llm.providers import OpenAIChatProvider

        with patch("obsidian_rag.llm.providers.log"):
            provider = OpenAIChatProvider(api_key="test-key")

        assert provider.api_key == "test-key"
        assert provider.model == "gpt-4"
        assert provider.temperature == 0.7
        assert provider.max_tokens is None

    def test_init_with_custom_params(self):
        """Test initialization with custom parameters."""
        from obsidian_rag.llm.providers import OpenAIChatProvider

        with patch("obsidian_rag.llm.providers.log"):
            provider = OpenAIChatProvider(
                api_key="test-key",
                model="gpt-3.5-turbo",
                temperature=0.5,
                max_tokens=100,
            )

        assert provider.model == "gpt-3.5-turbo"
        assert provider.temperature == 0.5
        assert provider.max_tokens == 100

    def test_init_without_api_key_uses_env_var(self):
        """Test initialization uses OPENAI_API_KEY env var when api_key not provided."""
        from obsidian_rag.llm.providers import OpenAIChatProvider

        with patch("obsidian_rag.llm.providers.log"):
            with patch.dict("os.environ", {"OPENAI_API_KEY": "env-key"}):
                provider = OpenAIChatProvider()

        assert provider.api_key == "env-key"

    def test_init_without_api_key_raises_error(self):
        """Test initialization raises ValueError when no API key available."""
        from obsidian_rag.llm.providers import OpenAIChatProvider

        with patch("obsidian_rag.llm.providers.log"):
            with patch.dict("os.environ", {}, clear=True):
                with pytest.raises(ValueError, match="API key is required"):
                    OpenAIChatProvider()

    def test_chat_success(self):
        """Test successful chat completion."""
        from obsidian_rag.llm.providers import OpenAIChatProvider

        mock_response = {
            "choices": [{"message": {"content": "Test response"}}],
        }

        with patch("obsidian_rag.llm.providers.log"):
            with patch("litellm.completion", return_value=mock_response):
                provider = OpenAIChatProvider(api_key="test-key")
                messages = [{"role": "user", "content": "Hello"}]
                result = provider.chat(messages)

        assert result == "Test response"

    def test_chat_with_empty_response(self):
        """Test chat with empty response content."""
        from obsidian_rag.llm.providers import OpenAIChatProvider

        mock_response = {
            "choices": [{"message": {"content": None}}],
        }

        with patch("obsidian_rag.llm.providers.log"):
            with patch("litellm.completion", return_value=mock_response):
                provider = OpenAIChatProvider(api_key="test-key")
                messages = [{"role": "user", "content": "Hello"}]
                result = provider.chat(messages)

        assert result == ""

    def test_chat_error(self):
        """Test chat error handling."""
        from obsidian_rag.llm.providers import OpenAIChatProvider

        with patch("obsidian_rag.llm.providers.log"):
            with patch("litellm.completion", side_effect=Exception("API Error")):
                provider = OpenAIChatProvider(api_key="test-key")
                messages = [{"role": "user", "content": "Hello"}]

                with pytest.raises(ChatError, match="API Error"):
                    provider.chat(messages)

    def test_chat_with_custom_temperature(self):
        """Test chat with custom temperature parameter."""
        from obsidian_rag.llm.providers import OpenAIChatProvider

        mock_response = {
            "choices": [{"message": {"content": "Test"}}],
        }

        with patch("obsidian_rag.llm.providers.log"):
            with patch("litellm.completion", return_value=mock_response) as mock_chat:
                provider = OpenAIChatProvider(
                    api_key="test-key",
                    temperature=0.9,
                )
                messages = [{"role": "user", "content": "Hello"}]
                provider.chat(messages, temperature=0.5)

        mock_chat.assert_called_once()
        call_kwargs = mock_chat.call_args.kwargs
        assert call_kwargs["temperature"] == 0.5

    def test_chat_with_max_tokens(self):
        """Test chat with max_tokens parameter."""
        from obsidian_rag.llm.providers import OpenAIChatProvider

        mock_response = {
            "choices": [{"message": {"content": "Test"}}],
        }

        with patch("obsidian_rag.llm.providers.log"):
            with patch("litellm.completion", return_value=mock_response) as mock_chat:
                provider = OpenAIChatProvider(
                    api_key="test-key",
                    max_tokens=100,
                )
                messages = [{"role": "user", "content": "Hello"}]
                provider.chat(messages)

        mock_chat.assert_called_once()
        call_kwargs = mock_chat.call_args.kwargs
        assert call_kwargs["max_tokens"] == 100

    def test_chat_with_base_url(self):
        """Test chat with custom base URL."""
        from obsidian_rag.llm.providers import OpenAIChatProvider

        mock_response = {
            "choices": [{"message": {"content": "Test"}}],
        }

        with patch("obsidian_rag.llm.providers.log"):
            with patch("litellm.completion", return_value=mock_response) as mock_chat:
                provider = OpenAIChatProvider(
                    api_key="test-key",
                    base_url="https://custom.api.com",
                )
                messages = [{"role": "user", "content": "Hello"}]
                provider.chat(messages)

        mock_chat.assert_called_once()
        call_kwargs = mock_chat.call_args.kwargs
        assert call_kwargs["api_base"] == "https://custom.api.com"


class TestHuggingFaceEmbeddingProvider:
    """Test cases for HuggingFaceEmbeddingProvider."""

    def test_init_with_default_model(self):
        """Test initialization with default model."""
        from obsidian_rag.llm.providers import HuggingFaceEmbeddingProvider

        mock_embeddings = Mock()

        with patch("obsidian_rag.llm.providers.log"):
            with patch(
                "langchain_huggingface.HuggingFaceEmbeddings",
                return_value=mock_embeddings,
            ):
                provider = HuggingFaceEmbeddingProvider()

        assert provider.model_name == "all-MiniLM-L6-v2"
        assert provider._dimension == 384

    def test_init_with_custom_model(self):
        """Test initialization with custom model name."""
        from obsidian_rag.llm.providers import HuggingFaceEmbeddingProvider

        mock_embeddings = Mock()

        with patch("obsidian_rag.llm.providers.log"):
            with patch(
                "langchain_huggingface.HuggingFaceEmbeddings",
                return_value=mock_embeddings,
            ):
                provider = HuggingFaceEmbeddingProvider(model="all-mpnet-base-v2")

        assert provider.model_name == "all-mpnet-base-v2"
        assert provider._dimension == 768

    def test_init_with_device(self):
        """Test initialization with custom device."""
        from obsidian_rag.llm.providers import HuggingFaceEmbeddingProvider

        mock_embeddings = Mock()

        with patch("obsidian_rag.llm.providers.log"):
            with patch(
                "langchain_huggingface.HuggingFaceEmbeddings",
                return_value=mock_embeddings,
            ) as mock_hf:
                HuggingFaceEmbeddingProvider(device="cuda")

        mock_hf.assert_called_once()
        call_kwargs = mock_hf.call_args.kwargs
        assert call_kwargs["model_kwargs"]["device"] == "cuda"

    def test_init_import_error(self):
        """Test initialization raises ImportError when langchain-huggingface not installed."""
        from obsidian_rag.llm.providers import HuggingFaceEmbeddingProvider

        with patch("obsidian_rag.llm.providers.log"):
            # Mock the import statement to raise ImportError
            import builtins

            original_import = builtins.__import__

            def mock_import(name, *args, **kwargs):
                if name == "langchain_huggingface":
                    raise ImportError("No module named 'langchain_huggingface'")
                return original_import(name, *args, **kwargs)

            with patch.object(builtins, "__import__", mock_import):
                with pytest.raises(
                    ImportError, match="langchain-huggingface package is required"
                ):
                    HuggingFaceEmbeddingProvider()

    def test_get_dimension(self):
        """Test get_dimension returns correct dimension."""
        from obsidian_rag.llm.providers import HuggingFaceEmbeddingProvider

        mock_embeddings = Mock()

        with patch("obsidian_rag.llm.providers.log"):
            with patch(
                "langchain_huggingface.HuggingFaceEmbeddings",
                return_value=mock_embeddings,
            ):
                provider = HuggingFaceEmbeddingProvider()

        assert provider.get_dimension() == 384

    def test_generate_embedding_success(self):
        """Test successful embedding generation."""
        from obsidian_rag.llm.providers import HuggingFaceEmbeddingProvider

        mock_embeddings = Mock()
        mock_embeddings.embed_query.return_value = [0.1, 0.2, 0.3]

        with patch("obsidian_rag.llm.providers.log"):
            with patch(
                "langchain_huggingface.HuggingFaceEmbeddings",
                return_value=mock_embeddings,
            ):
                provider = HuggingFaceEmbeddingProvider()
                result = provider.generate_embedding("test text")

        assert result == [0.1, 0.2, 0.3]
        mock_embeddings.embed_query.assert_called_once_with("test text")

    def test_generate_embedding_error(self):
        """Test embedding generation error handling."""
        from obsidian_rag.llm.providers import HuggingFaceEmbeddingProvider

        mock_embeddings = Mock()
        mock_embeddings.embed_query.side_effect = Exception("Model Error")

        with patch("obsidian_rag.llm.providers.log"):
            with patch(
                "langchain_huggingface.HuggingFaceEmbeddings",
                return_value=mock_embeddings,
            ):
                provider = HuggingFaceEmbeddingProvider()

                with pytest.raises(EmbeddingError, match="Model Error"):
                    provider.generate_embedding("test text")

    def test_dimension_mapping_all_mini_lm_l12(self):
        """Test dimension mapping for all-MiniLM-L12-v2."""
        from obsidian_rag.llm.providers import HuggingFaceEmbeddingProvider

        mock_embeddings = Mock()

        with patch("obsidian_rag.llm.providers.log"):
            with patch(
                "langchain_huggingface.HuggingFaceEmbeddings",
                return_value=mock_embeddings,
            ):
                provider = HuggingFaceEmbeddingProvider(model="all-MiniLM-L12-v2")

        assert provider._dimension == 384

    def test_dimension_mapping_paraphrase_multilingual(self):
        """Test dimension mapping for paraphrase-multilingual model."""
        from obsidian_rag.llm.providers import HuggingFaceEmbeddingProvider

        mock_embeddings = Mock()

        with patch("obsidian_rag.llm.providers.log"):
            with patch(
                "langchain_huggingface.HuggingFaceEmbeddings",
                return_value=mock_embeddings,
            ):
                provider = HuggingFaceEmbeddingProvider(
                    model="paraphrase-multilingual-MiniLM-L12-v2",
                )

        assert provider._dimension == 384

    def test_dimension_mapping_unknown_model(self):
        """Test dimension mapping for unknown model uses default."""
        from obsidian_rag.llm.providers import HuggingFaceEmbeddingProvider

        mock_embeddings = Mock()

        with patch("obsidian_rag.llm.providers.log"):
            with patch(
                "langchain_huggingface.HuggingFaceEmbeddings",
                return_value=mock_embeddings,
            ):
                provider = HuggingFaceEmbeddingProvider(model="unknown-model")

        assert provider._dimension == 384  # Default dimension
