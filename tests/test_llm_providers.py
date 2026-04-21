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

    def test_init_with_unresolved_env_var_api_key_raises_error(self):
        """Test initialization raises ValueError when api_key is an unresolved ${...} pattern."""
        from obsidian_rag.llm.providers import OpenAIEmbeddingProvider

        with patch("obsidian_rag.llm.providers.log"):
            with pytest.raises(ValueError, match="API key is required"):
                OpenAIEmbeddingProvider(api_key="${OPENAI_API_KEY}")

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

    def test_init_with_unresolved_env_var_api_key_raises_error(self):
        """Test initialization raises ValueError when api_key is an unresolved ${...} pattern."""
        from obsidian_rag.llm.providers import OpenAIChatProvider

        with patch("obsidian_rag.llm.providers.log"):
            with pytest.raises(ValueError, match="API key is required"):
                OpenAIChatProvider(api_key="${OPENAI_API_KEY}")

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
        mock_hf_class = Mock(return_value=mock_embeddings)

        with patch("obsidian_rag.llm.providers.log"):
            with patch(
                "obsidian_rag.llm.providers.HuggingFaceEmbeddings",
                mock_hf_class,
            ):
                provider = HuggingFaceEmbeddingProvider()

        assert provider.model_name == "all-MiniLM-L6-v2"
        assert provider._dimension == 384

    def test_init_with_custom_model(self):
        """Test initialization with custom model name."""
        from obsidian_rag.llm.providers import HuggingFaceEmbeddingProvider

        mock_embeddings = Mock()
        mock_hf_class = Mock(return_value=mock_embeddings)

        with patch("obsidian_rag.llm.providers.log"):
            with patch(
                "obsidian_rag.llm.providers.HuggingFaceEmbeddings",
                mock_hf_class,
            ):
                provider = HuggingFaceEmbeddingProvider(model="all-mpnet-base-v2")

        assert provider.model_name == "all-mpnet-base-v2"
        assert provider._dimension == 768

    def test_init_with_device(self):
        """Test initialization with custom device."""
        from obsidian_rag.llm.providers import HuggingFaceEmbeddingProvider

        mock_embeddings = Mock()
        mock_hf_class = Mock(return_value=mock_embeddings)

        with patch("obsidian_rag.llm.providers.log"):
            with patch(
                "obsidian_rag.llm.providers.HuggingFaceEmbeddings",
                mock_hf_class,
            ):
                HuggingFaceEmbeddingProvider(device="cuda")

        mock_hf_class.assert_called_once()
        call_kwargs = mock_hf_class.call_args.kwargs
        assert call_kwargs["model_kwargs"]["device"] == "cuda"

    def test_init_import_error(self):
        """Test initialization raises ImportError when langchain-huggingface not installed."""
        from obsidian_rag.llm.providers import HuggingFaceEmbeddingProvider

        with patch("obsidian_rag.llm.providers.log"):
            # Patch the HuggingFaceEmbeddings module variable to None
            with patch("obsidian_rag.llm.providers.HuggingFaceEmbeddings", None):
                with pytest.raises(
                    ImportError, match="langchain-huggingface package is required"
                ):
                    HuggingFaceEmbeddingProvider()

    def test_get_dimension(self):
        """Test get_dimension returns correct dimension."""
        from obsidian_rag.llm.providers import HuggingFaceEmbeddingProvider

        mock_embeddings = Mock()
        mock_hf_class = Mock(return_value=mock_embeddings)

        with patch("obsidian_rag.llm.providers.log"):
            with patch(
                "obsidian_rag.llm.providers.HuggingFaceEmbeddings",
                mock_hf_class,
            ):
                provider = HuggingFaceEmbeddingProvider()

        assert provider.get_dimension() == 384

    def test_generate_embedding_success(self):
        """Test successful embedding generation."""
        from obsidian_rag.llm.providers import HuggingFaceEmbeddingProvider

        mock_embeddings = Mock()
        mock_embeddings.embed_query.return_value = [0.1, 0.2, 0.3]
        mock_hf_class = Mock(return_value=mock_embeddings)

        with patch("obsidian_rag.llm.providers.log"):
            with patch(
                "obsidian_rag.llm.providers.HuggingFaceEmbeddings",
                mock_hf_class,
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
        mock_hf_class = Mock(return_value=mock_embeddings)

        with patch("obsidian_rag.llm.providers.log"):
            with patch(
                "obsidian_rag.llm.providers.HuggingFaceEmbeddings",
                mock_hf_class,
            ):
                provider = HuggingFaceEmbeddingProvider()

                with pytest.raises(EmbeddingError, match="Model Error"):
                    provider.generate_embedding("test text")

    def test_dimension_mapping_all_mini_lm_l12(self):
        """Test dimension mapping for all-MiniLM-L12-v2."""
        from obsidian_rag.llm.providers import HuggingFaceEmbeddingProvider

        mock_embeddings = Mock()
        mock_hf_class = Mock(return_value=mock_embeddings)

        with patch("obsidian_rag.llm.providers.log"):
            with patch(
                "obsidian_rag.llm.providers.HuggingFaceEmbeddings",
                mock_hf_class,
            ):
                provider = HuggingFaceEmbeddingProvider(model="all-MiniLM-L12-v2")

        assert provider._dimension == 384

    def test_dimension_mapping_paraphrase_multilingual(self):
        """Test dimension mapping for paraphrase-multilingual model."""
        from obsidian_rag.llm.providers import HuggingFaceEmbeddingProvider

        mock_embeddings = Mock()
        mock_hf_class = Mock(return_value=mock_embeddings)

        with patch("obsidian_rag.llm.providers.log"):
            with patch(
                "obsidian_rag.llm.providers.HuggingFaceEmbeddings",
                mock_hf_class,
            ):
                provider = HuggingFaceEmbeddingProvider(
                    model="paraphrase-multilingual-MiniLM-L12-v2",
                )

        assert provider._dimension == 384

    def test_dimension_mapping_unknown_model(self):
        """Test dimension mapping for unknown model uses default."""
        from obsidian_rag.llm.providers import HuggingFaceEmbeddingProvider

        mock_embeddings = Mock()
        mock_hf_class = Mock(return_value=mock_embeddings)

        with patch("obsidian_rag.llm.providers.log"):
            with patch(
                "obsidian_rag.llm.providers.HuggingFaceEmbeddings",
                mock_hf_class,
            ):
                provider = HuggingFaceEmbeddingProvider(model="unknown-model")

        assert provider._dimension == 384  # Default dimension


class TestOpenRouterEmbeddingProvider:
    """Test cases for OpenRouterEmbeddingProvider."""

    def test_init_with_api_key(self):
        """Test initialization with explicit API key."""
        from obsidian_rag.llm.providers import OpenRouterEmbeddingProvider

        with patch("obsidian_rag.llm.providers.log"):
            provider = OpenRouterEmbeddingProvider(api_key="test-key")

        assert provider.api_key == "test-key"
        assert provider.model == "qwen/qwen3-embedding-8b"
        assert provider._dimension == 4096
        assert provider.base_url == "https://openrouter.ai/api/v1"

    def test_init_without_api_key_uses_env_var(self):
        """Test initialization uses OPENROUTER_API_KEY env var when api_key not provided."""
        from obsidian_rag.llm.providers import OpenRouterEmbeddingProvider

        with patch("obsidian_rag.llm.providers.log"):
            with patch.dict("os.environ", {"OPENROUTER_API_KEY": "env-key"}):
                provider = OpenRouterEmbeddingProvider()

        assert provider.api_key == "env-key"

    def test_init_without_api_key_raises_error(self):
        """Test initialization raises ValueError when no API key available."""
        from obsidian_rag.llm.providers import OpenRouterEmbeddingProvider

        with patch("obsidian_rag.llm.providers.log"):
            with patch.dict("os.environ", {}, clear=True):
                with pytest.raises(ValueError, match="API key is required"):
                    OpenRouterEmbeddingProvider()

    def test_init_with_unresolved_env_var_api_key_raises_error(self):
        """Test initialization raises ValueError when api_key is an unresolved ${...} pattern."""
        from obsidian_rag.llm.providers import OpenRouterEmbeddingProvider

        with patch("obsidian_rag.llm.providers.log"):
            with pytest.raises(ValueError, match="API key is required"):
                OpenRouterEmbeddingProvider(api_key="${OPENROUTER_API_KEY}")

    def test_init_with_custom_model(self):
        """Test initialization with custom model name."""
        from obsidian_rag.llm.providers import OpenRouterEmbeddingProvider

        with patch("obsidian_rag.llm.providers.log"):
            provider = OpenRouterEmbeddingProvider(
                api_key="test-key",
                model="openai/text-embedding-3-small",
            )

        assert provider.model == "openai/text-embedding-3-small"
        assert provider._dimension == 4096  # Falls back to default

    def test_init_with_custom_base_url(self):
        """Test initialization with custom base URL."""
        from obsidian_rag.llm.providers import OpenRouterEmbeddingProvider

        with patch("obsidian_rag.llm.providers.log"):
            provider = OpenRouterEmbeddingProvider(
                api_key="test-key",
                base_url="https://custom.openrouter.api.com",
            )

        assert provider.base_url == "https://custom.openrouter.api.com"

    def test_get_dimension(self):
        """Test get_dimension returns correct dimension."""
        from obsidian_rag.llm.providers import OpenRouterEmbeddingProvider

        with patch("obsidian_rag.llm.providers.log"):
            provider = OpenRouterEmbeddingProvider(api_key="test-key")

        assert provider.get_dimension() == 4096

    def test_generate_embedding_success(self):
        """Test successful embedding generation."""
        from obsidian_rag.llm.providers import OpenRouterEmbeddingProvider

        mock_response = {
            "data": [{"embedding": [0.1, 0.2, 0.3]}],
        }

        with patch("obsidian_rag.llm.providers.log"):
            with patch("litellm.embedding", return_value=mock_response):
                provider = OpenRouterEmbeddingProvider(api_key="test-key")
                result = provider.generate_embedding("test text")

        assert result == [0.1, 0.2, 0.3]

    def test_generate_embedding_error(self):
        """Test embedding generation error handling."""
        from obsidian_rag.llm.providers import OpenRouterEmbeddingProvider

        with patch("obsidian_rag.llm.providers.log"):
            with patch("litellm.embedding", side_effect=Exception("API Error")):
                provider = OpenRouterEmbeddingProvider(api_key="test-key")

                with pytest.raises(EmbeddingError, match="API Error"):
                    provider.generate_embedding("test text")

    def test_generate_embedding_adds_openrouter_prefix_for_default_model(self):
        """Test that default model gets 'openrouter/' prefix for litellm 1.83+ native routing.

        This test verifies that the model name receives the 'openrouter/'
        prefix for litellm 1.83+ native OpenRouter routing support.
        """
        from obsidian_rag.llm.providers import OpenRouterEmbeddingProvider

        mock_response = {
            "data": [{"embedding": [0.1, 0.2, 0.3]}],
        }

        with patch("obsidian_rag.llm.providers.log"):
            with patch("litellm.embedding", return_value=mock_response) as mock_embed:
                provider = OpenRouterEmbeddingProvider(
                    api_key="test-key",
                    model="qwen/qwen3-embedding-8b",
                )
                provider.generate_embedding("test text")

        mock_embed.assert_called_once()
        call_kwargs = mock_embed.call_args.kwargs
        # Model should have openrouter/ prefix for litellm 1.83+ native support
        assert call_kwargs["model"] == "openrouter/qwen/qwen3-embedding-8b"

    def test_generate_embedding_with_openai_model_via_openrouter(self):
        """Test using OpenAI model through OpenRouter with correct routing.

        Verifies that the model name gets 'openrouter/' prefix added
        for litellm native OpenRouter routing.
        """
        from obsidian_rag.llm.providers import OpenRouterEmbeddingProvider

        mock_response = {
            "data": [{"embedding": [0.1, 0.2, 0.3]}],
        }

        with patch("obsidian_rag.llm.providers.log"):
            with patch("litellm.embedding", return_value=mock_response) as mock_embed:
                provider = OpenRouterEmbeddingProvider(
                    api_key="test-key",
                    model="openai/text-embedding-3-small",
                )
                provider.generate_embedding("test text")

        mock_embed.assert_called_once()
        call_kwargs = mock_embed.call_args.kwargs
        assert call_kwargs["model"] == "openrouter/openai/text-embedding-3-small"
        assert call_kwargs["api_key"] == "test-key"

    def test_generate_embedding_with_custom_base_url(self):
        """Test that custom base_url is passed to litellm.embedding() call."""
        from obsidian_rag.llm.providers import OpenRouterEmbeddingProvider

        custom_url = "https://custom.openrouter.api.com"
        mock_response = {
            "data": [{"embedding": [0.1, 0.2, 0.3]}],
        }

        with patch("obsidian_rag.llm.providers.log"):
            with patch("litellm.embedding", return_value=mock_response) as mock_embed:
                provider = OpenRouterEmbeddingProvider(
                    api_key="test-key",
                    model="qwen/qwen3-embedding-8b",
                    base_url=custom_url,
                )
                assert provider.base_url == custom_url
                provider.generate_embedding("test text")

        mock_embed.assert_called_once()
        call_kwargs = mock_embed.call_args.kwargs
        assert call_kwargs["api_base"] == custom_url

    def test_dimension_mapping_qwen3_embedding_8b(self):
        """Test dimension mapping for qwen/qwen3-embedding-8b."""
        from obsidian_rag.llm.providers import OpenRouterEmbeddingProvider

        with patch("obsidian_rag.llm.providers.log"):
            provider = OpenRouterEmbeddingProvider(
                api_key="test-key",
                model="qwen/qwen3-embedding-8b",
            )

        assert provider._dimension == 4096

    def test_init_reads_openrouter_api_key_from_env(self):
        """Test that initialization reads OPENROUTER_API_KEY from env if no key provided."""
        from obsidian_rag.llm import providers as providers_module
        from obsidian_rag.llm.providers import OpenRouterEmbeddingProvider

        with patch("obsidian_rag.llm.providers.log"):
            with patch.dict(
                providers_module.os.environ,
                {"OPENROUTER_API_KEY": "env-api-key"},
                clear=True,
            ):
                provider = OpenRouterEmbeddingProvider()

                assert provider.api_key == "env-api-key"

    def test_init_prefers_explicit_api_key_over_env(self):
        """Test that explicit api_key takes precedence over OPENROUTER_API_KEY env var."""
        from obsidian_rag.llm import providers as providers_module
        from obsidian_rag.llm.providers import OpenRouterEmbeddingProvider

        with patch("obsidian_rag.llm.providers.log"):
            with patch.dict(
                providers_module.os.environ,
                {"OPENROUTER_API_KEY": "env-api-key"},
            ):
                provider = OpenRouterEmbeddingProvider(api_key="explicit-key")

                assert provider.api_key == "explicit-key"

    def test_init_import_error(self):
        """Test initialization raises ImportError when litellm not installed."""
        from obsidian_rag.llm import providers as providers_module
        from obsidian_rag.llm.providers import OpenRouterEmbeddingProvider

        with patch("obsidian_rag.llm.providers.log"):
            # Patch the litellm module variable to None to simulate it not being installed
            with patch.object(providers_module, "litellm", None):
                with pytest.raises(ImportError, match="litellm package is required"):
                    OpenRouterEmbeddingProvider(api_key="test-key")


class TestAddOpenrouterPrefix:
    """Test cases for _add_openrouter_prefix helper function."""

    def test_adds_prefix_to_bare_model(self):
        """Test that prefix is added to model without it."""
        from obsidian_rag.llm.providers import _add_openrouter_prefix

        assert (
            _add_openrouter_prefix("anthropic/claude-3-opus")
            == "openrouter/anthropic/claude-3-opus"
        )

    def test_preserves_existing_prefix(self):
        """Test that already-prefixed model is returned unchanged."""
        from obsidian_rag.llm.providers import _add_openrouter_prefix

        assert (
            _add_openrouter_prefix("openrouter/anthropic/claude-3-opus")
            == "openrouter/anthropic/claude-3-opus"
        )

    def test_adds_prefix_to_simple_model(self):
        """Test that prefix is added to simple model name."""
        from obsidian_rag.llm.providers import _add_openrouter_prefix

        assert (
            _add_openrouter_prefix("qwen/qwen3-embedding-8b")
            == "openrouter/qwen/qwen3-embedding-8b"
        )


class TestOpenRouterEmbeddingProviderPrefixStripping:
    """Test cases for openrouter/ prefix handling in generate_embedding."""

    def test_generate_embedding_adds_openrouter_prefix_for_openai_models(self):
        """Test that 'openai/' models get 'openrouter/' prefix added."""
        from obsidian_rag.llm.providers import OpenRouterEmbeddingProvider

        mock_response = {
            "data": [{"embedding": [0.1, 0.2, 0.3]}],
        }

        with patch("obsidian_rag.llm.providers.log"):
            with patch("litellm.embedding", return_value=mock_response) as mock_embed:
                provider = OpenRouterEmbeddingProvider(
                    api_key="test-key",
                    model="openai/text-embedding-3-small",
                )
                provider.generate_embedding("test text")

        mock_embed.assert_called_once()
        call_kwargs = mock_embed.call_args.kwargs
        # Model name should have 'openrouter/' prefix added
        assert call_kwargs["model"] == "openrouter/openai/text-embedding-3-small"

    def test_generate_embedding_adds_openrouter_prefix_for_non_openai_models(self):
        """Test that non-OpenAI model names get 'openrouter/' prefix added."""
        from obsidian_rag.llm.providers import OpenRouterEmbeddingProvider

        mock_response = {
            "data": [{"embedding": [0.1, 0.2, 0.3]}],
        }

        with patch("obsidian_rag.llm.providers.log"):
            with patch("litellm.embedding", return_value=mock_response) as mock_embed:
                provider = OpenRouterEmbeddingProvider(
                    api_key="test-key",
                    model="qwen/qwen3-embedding-8b",
                )
                provider.generate_embedding("test text")

        mock_embed.assert_called_once()
        call_kwargs = mock_embed.call_args.kwargs
        # Non-OpenAI model names should get openrouter/ prefix
        assert call_kwargs["model"] == "openrouter/qwen/qwen3-embedding-8b"

    def test_generate_embedding_adds_openrouter_prefix_for_openai_ada(self):
        """Test prefix handling for OpenAI ada model."""
        from obsidian_rag.llm.providers import OpenRouterEmbeddingProvider

        mock_response = {
            "data": [{"embedding": [0.1, 0.2, 0.3]}],
        }

        with patch("obsidian_rag.llm.providers.log"):
            with patch("litellm.embedding", return_value=mock_response) as mock_embed:
                provider = OpenRouterEmbeddingProvider(
                    api_key="test-key",
                    model="openai/text-embedding-ada-002",
                )
                provider.generate_embedding("test text")

        mock_embed.assert_called_once()
        call_kwargs = mock_embed.call_args.kwargs
        assert call_kwargs["model"] == "openrouter/openai/text-embedding-ada-002"

    def test_generate_embedding_handles_model_without_prefix(self):
        """Test that model names without prefix get openrouter/ prefix."""
        from obsidian_rag.llm.providers import OpenRouterEmbeddingProvider

        mock_response = {
            "data": [{"embedding": [0.1, 0.2, 0.3]}],
        }

        with patch("obsidian_rag.llm.providers.log"):
            with patch("litellm.embedding", return_value=mock_response) as mock_embed:
                provider = OpenRouterEmbeddingProvider(
                    api_key="test-key",
                    model="text-embedding-3-small",  # No prefix
                )
                provider.generate_embedding("test text")

        mock_embed.assert_called_once()
        call_kwargs = mock_embed.call_args.kwargs
        # Model without prefix should get openrouter/ prefix
        assert call_kwargs["model"] == "openrouter/text-embedding-3-small"

    def test_generate_embedding_preserves_existing_openrouter_prefix(self):
        """Test that model names already with openrouter/ prefix are unchanged."""
        from obsidian_rag.llm.providers import OpenRouterEmbeddingProvider

        mock_response = {
            "data": [{"embedding": [0.1, 0.2, 0.3]}],
        }

        with patch("obsidian_rag.llm.providers.log"):
            with patch("litellm.embedding", return_value=mock_response) as mock_embed:
                provider = OpenRouterEmbeddingProvider(
                    api_key="test-key",
                    model="openrouter/openai/text-embedding-3-small",  # Already has prefix
                )
                provider.generate_embedding("test text")

        mock_embed.assert_called_once()
        call_kwargs = mock_embed.call_args.kwargs
        # Model already with openrouter/ prefix should be unchanged
        assert call_kwargs["model"] == "openrouter/openai/text-embedding-3-small"


class TestOpenRouterEmbeddingProviderDebugLogging:
    """Test cases for debug logging in generate_embedding."""

    def test_generate_embedding_logs_model_name(self):
        """Test that debug logging includes model name."""
        from obsidian_rag.llm.providers import OpenRouterEmbeddingProvider

        mock_response = {
            "data": [{"embedding": [0.1, 0.2, 0.3]}],
        }

        with patch("obsidian_rag.llm.providers.log") as mock_log:
            with patch("litellm.embedding", return_value=mock_response):
                provider = OpenRouterEmbeddingProvider(
                    api_key="test-key",
                    model="openai/text-embedding-3-small",
                )
                provider.generate_embedding("test text")

        # Verify debug logging was called
        debug_calls = mock_log.debug.call_args_list
        assert len(debug_calls) >= 1

    def test_generate_embedding_logs_non_openai_model(self):
        """Test that logging works for non-OpenAI models."""
        from obsidian_rag.llm.providers import OpenRouterEmbeddingProvider

        mock_response = {
            "data": [{"embedding": [0.1, 0.2, 0.3]}],
        }

        with patch("obsidian_rag.llm.providers.log") as mock_log:
            with patch("litellm.embedding", return_value=mock_response):
                provider = OpenRouterEmbeddingProvider(
                    api_key="test-key",
                    model="qwen/qwen3-embedding-8b",
                )
                provider.generate_embedding("test text")

        # Verify debug logging was called
        debug_calls = mock_log.debug.call_args_list
        assert len(debug_calls) >= 1


class TestOpenRouterEmbeddingProviderEdgeCases:
    """Test edge cases for model name handling."""

    def test_generate_embedding_handles_empty_model_name(self):
        """Test behavior with empty model name - uses default model."""
        from obsidian_rag.llm.providers import OpenRouterEmbeddingProvider

        mock_response = {
            "data": [{"embedding": [0.1, 0.2, 0.3]}],
        }

        with patch("obsidian_rag.llm.providers.log"):
            with patch("litellm.embedding", return_value=mock_response) as mock_embed:
                provider = OpenRouterEmbeddingProvider(
                    api_key="test-key",
                    model="",  # Empty model name - should use default
                )
                provider.generate_embedding("test text")

        mock_embed.assert_called_once()
        call_kwargs = mock_embed.call_args.kwargs
        # Empty model name uses default model with openrouter/ prefix
        assert call_kwargs["model"] == "openrouter/qwen/qwen3-embedding-8b"

    def test_generate_embedding_idempotent_multiple_calls(self):
        """Test that multiple calls with same model work correctly."""
        from obsidian_rag.llm.providers import OpenRouterEmbeddingProvider

        mock_response = {
            "data": [{"embedding": [0.1, 0.2, 0.3]}],
        }

        with patch("obsidian_rag.llm.providers.log"):
            with patch("litellm.embedding", return_value=mock_response) as mock_embed:
                provider = OpenRouterEmbeddingProvider(
                    api_key="test-key",
                    model="openai/text-embedding-3-small",
                )
                # Call multiple times
                provider.generate_embedding("test 1")
                provider.generate_embedding("test 2")
                provider.generate_embedding("test 3")

        # All calls should use the openrouter/ prefixed model name
        assert mock_embed.call_count == 3
        for call in mock_embed.call_args_list:
            assert call.kwargs["model"] == "openrouter/openai/text-embedding-3-small"


class TestOpenRouterEmbeddingProviderIntegration:
    """Integration-style tests for OpenRouter embedding provider."""

    def test_full_workflow_with_openai_model(self):
        """Test complete workflow from init to embedding generation with OpenAI model."""
        from obsidian_rag.llm import providers as providers_module
        from obsidian_rag.llm.providers import OpenRouterEmbeddingProvider

        mock_response = {
            "data": [{"embedding": [0.1] * 1536}],
        }

        with patch("obsidian_rag.llm.providers.log") as mock_log:
            with patch.dict(
                providers_module.os.environ,
                {"OPENROUTER_API_KEY": "test-key"},
                clear=True,
            ):
                with patch(
                    "litellm.embedding", return_value=mock_response
                ) as mock_embed:
                    # Initialize provider with OpenAI model
                    provider = OpenRouterEmbeddingProvider(
                        api_key="test-key",
                        model="openai/text-embedding-3-small",
                    )

                    # Verify api_key is stored
                    assert provider.api_key == "test-key"

                    # Generate embedding
                    result = provider.generate_embedding("test query")

                    # Verify embedding returned
                    assert len(result) == 1536

                    # Verify litellm was called with openrouter/ prefixed model name
                    mock_embed.assert_called_once()
                    call_kwargs = mock_embed.call_args.kwargs
                    assert (
                        call_kwargs["model"]
                        == "openrouter/openai/text-embedding-3-small"
                    )

                    # Verify debug logging occurred
                    debug_messages = [
                        str(call.args[0])
                        for call in mock_log.debug.call_args_list
                        if call.args
                    ]
                    assert any("Generating embedding" in msg for msg in debug_messages)
                    assert any(
                        "generate_embedding returning" in msg for msg in debug_messages
                    )

    def test_full_workflow_with_qwen_model(self):
        """Test complete workflow with non-OpenAI model (qwen)."""
        from obsidian_rag.llm import providers as providers_module
        from obsidian_rag.llm.providers import OpenRouterEmbeddingProvider

        mock_response = {
            "data": [{"embedding": [0.1] * 4096}],
        }

        with patch("obsidian_rag.llm.providers.log"):
            with patch.dict(
                providers_module.os.environ,
                {"OPENROUTER_API_KEY": "test-key"},
                clear=True,
            ):
                with patch(
                    "litellm.embedding", return_value=mock_response
                ) as mock_embed:
                    # Initialize provider with Qwen model
                    provider = OpenRouterEmbeddingProvider(
                        api_key="test-key",
                        model="qwen/qwen3-embedding-8b",
                    )

                    # Verify dimension is correct for qwen model
                    assert provider.get_dimension() == 4096

                    # Generate embedding
                    result = provider.generate_embedding("test query")

                    # Verify embedding returned
                    assert len(result) == 4096

                    # Verify litellm was called with openrouter/ prefixed model name
                    mock_embed.assert_called_once()
                    call_kwargs = mock_embed.call_args.kwargs
                    assert call_kwargs["model"] == "openrouter/qwen/qwen3-embedding-8b"

    def test_error_handling_preserves_context(self):
        """Test that errors include proper context for debugging."""
        from obsidian_rag.llm import providers as providers_module
        from obsidian_rag.llm.base import EmbeddingError
        from obsidian_rag.llm.providers import OpenRouterEmbeddingProvider

        with patch("obsidian_rag.llm.providers.log"):
            with patch.dict(
                providers_module.os.environ,
                {"OPENROUTER_API_KEY": "test-key"},
                clear=True,
            ):
                with patch(
                    "litellm.embedding",
                    side_effect=Exception("Authentication failed"),
                ):
                    provider = OpenRouterEmbeddingProvider(
                        api_key="test-key",
                        model="openai/text-embedding-3-small",
                    )

                    with pytest.raises(EmbeddingError) as exc_info:
                        provider.generate_embedding("test query")

                    # Verify error message includes context
                    assert "OpenRouter embedding generation failed" in str(
                        exc_info.value
                    )
                    assert "Authentication failed" in str(exc_info.value)


class TestOpenRouterChatProvider:
    """Test cases for OpenRouterChatProvider."""

    def test_init_with_api_key(self):
        """Test initialization with explicit API key."""
        from obsidian_rag.llm.providers import OpenRouterChatProvider

        with patch("obsidian_rag.llm.providers.log"):
            provider = OpenRouterChatProvider(api_key="test-key")

        assert provider.api_key == "test-key"
        assert provider.model == "anthropic/claude-3-opus"
        assert provider.temperature == 0.7
        assert provider.max_tokens is None
        assert provider.base_url == "https://openrouter.ai/api/v1"

    def test_init_with_custom_params(self):
        """Test initialization with custom parameters."""
        from obsidian_rag.llm.providers import OpenRouterChatProvider

        with patch("obsidian_rag.llm.providers.log"):
            provider = OpenRouterChatProvider(
                api_key="test-key",
                model="openai/gpt-4",
                temperature=0.5,
                max_tokens=100,
            )

        assert provider.model == "openai/gpt-4"
        assert provider.temperature == 0.5
        assert provider.max_tokens == 100

    def test_init_without_api_key_uses_env_var(self):
        """Test initialization uses OPENROUTER_API_KEY env var when api_key not provided."""
        from obsidian_rag.llm.providers import OpenRouterChatProvider

        with patch("obsidian_rag.llm.providers.log"):
            with patch.dict("os.environ", {"OPENROUTER_API_KEY": "env-key"}):
                provider = OpenRouterChatProvider()

        assert provider.api_key == "env-key"

    def test_init_without_api_key_raises_error(self):
        """Test initialization raises ValueError when no API key available."""
        from obsidian_rag.llm.providers import OpenRouterChatProvider

        with patch("obsidian_rag.llm.providers.log"):
            with patch.dict("os.environ", {}, clear=True):
                with pytest.raises(ValueError, match="API key is required"):
                    OpenRouterChatProvider()

    def test_init_with_unresolved_env_var_api_key_raises_error(self):
        """Test initialization raises ValueError when api_key is an unresolved ${...} pattern."""
        from obsidian_rag.llm.providers import OpenRouterChatProvider

        with patch("obsidian_rag.llm.providers.log"):
            with pytest.raises(ValueError, match="API key is required"):
                OpenRouterChatProvider(api_key="${OPENROUTER_API_KEY}")

    def test_chat_success(self):
        """Test successful chat completion."""
        from obsidian_rag.llm.providers import OpenRouterChatProvider

        mock_response = {
            "choices": [{"message": {"content": "Test response"}}],
        }

        with patch("obsidian_rag.llm.providers.log"):
            with patch("litellm.completion", return_value=mock_response):
                provider = OpenRouterChatProvider(api_key="test-key")
                messages = [{"role": "user", "content": "Hello"}]
                result = provider.chat(messages)

        assert result == "Test response"

    def test_chat_with_empty_response(self):
        """Test chat with empty response content."""
        from obsidian_rag.llm.providers import OpenRouterChatProvider

        mock_response = {
            "choices": [{"message": {"content": None}}],
        }

        with patch("obsidian_rag.llm.providers.log"):
            with patch("litellm.completion", return_value=mock_response):
                provider = OpenRouterChatProvider(api_key="test-key")
                messages = [{"role": "user", "content": "Hello"}]
                result = provider.chat(messages)

        assert result == ""

    def test_chat_error(self):
        """Test chat error handling."""
        from obsidian_rag.llm.providers import OpenRouterChatProvider

        with patch("obsidian_rag.llm.providers.log"):
            with patch("litellm.completion", side_effect=Exception("API Error")):
                provider = OpenRouterChatProvider(api_key="test-key")
                messages = [{"role": "user", "content": "Hello"}]

                with pytest.raises(ChatError, match="API Error"):
                    provider.chat(messages)

    def test_chat_uses_openrouter_prefix(self):
        """Test that model name is prefixed with openrouter/ for litellm."""
        from obsidian_rag.llm.providers import OpenRouterChatProvider

        mock_response = {
            "choices": [{"message": {"content": "Test"}}],
        }

        with patch("obsidian_rag.llm.providers.log"):
            with patch("litellm.completion", return_value=mock_response) as mock_chat:
                provider = OpenRouterChatProvider(
                    api_key="test-key",
                    model="anthropic/claude-3-opus",
                )
                messages = [{"role": "user", "content": "Hello"}]
                provider.chat(messages)

        mock_chat.assert_called_once()
        call_kwargs = mock_chat.call_args.kwargs
        assert call_kwargs["model"] == "openrouter/anthropic/claude-3-opus"
        assert call_kwargs["api_base"] == "https://openrouter.ai/api/v1"

    def test_chat_preserves_existing_openrouter_prefix(self):
        """Test that model name already with openrouter/ prefix is not double-prefixed."""
        from obsidian_rag.llm.providers import OpenRouterChatProvider

        mock_response = {
            "choices": [{"message": {"content": "Test"}}],
        }

        with patch("obsidian_rag.llm.providers.log"):
            with patch("litellm.completion", return_value=mock_response) as mock_chat:
                provider = OpenRouterChatProvider(
                    api_key="test-key",
                    model="openrouter/anthropic/claude-3-opus",
                )
                messages = [{"role": "user", "content": "Hello"}]
                provider.chat(messages)

        mock_chat.assert_called_once()
        call_kwargs = mock_chat.call_args.kwargs
        assert call_kwargs["model"] == "openrouter/anthropic/claude-3-opus"

    def test_chat_with_custom_temperature(self):
        """Test chat with custom temperature parameter."""
        from obsidian_rag.llm.providers import OpenRouterChatProvider

        mock_response = {
            "choices": [{"message": {"content": "Test"}}],
        }

        with patch("obsidian_rag.llm.providers.log"):
            with patch("litellm.completion", return_value=mock_response) as mock_chat:
                provider = OpenRouterChatProvider(
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
        from obsidian_rag.llm.providers import OpenRouterChatProvider

        mock_response = {
            "choices": [{"message": {"content": "Test"}}],
        }

        with patch("obsidian_rag.llm.providers.log"):
            with patch("litellm.completion", return_value=mock_response) as mock_chat:
                provider = OpenRouterChatProvider(
                    api_key="test-key",
                    max_tokens=100,
                )
                messages = [{"role": "user", "content": "Hello"}]
                provider.chat(messages)

        mock_chat.assert_called_once()
        call_kwargs = mock_chat.call_args.kwargs
        assert call_kwargs["max_tokens"] == 100

    def test_chat_with_custom_base_url(self):
        """Test chat with custom base URL."""
        from obsidian_rag.llm.providers import OpenRouterChatProvider

        mock_response = {
            "choices": [{"message": {"content": "Test"}}],
        }

        with patch("obsidian_rag.llm.providers.log"):
            with patch("litellm.completion", return_value=mock_response) as mock_chat:
                provider = OpenRouterChatProvider(
                    api_key="test-key",
                    base_url="https://custom.openrouter.api.com",
                )
                messages = [{"role": "user", "content": "Hello"}]
                provider.chat(messages)

        mock_chat.assert_called_once()
        call_kwargs = mock_chat.call_args.kwargs
        assert call_kwargs["api_base"] == "https://custom.openrouter.api.com"

    def test_init_import_error(self):
        """Test initialization raises ImportError when litellm not installed."""
        from obsidian_rag.llm import providers as providers_module
        from obsidian_rag.llm.providers import OpenRouterChatProvider

        with patch("obsidian_rag.llm.providers.log"):
            # Patch the litellm module variable to None to simulate it not being installed
            with patch.object(providers_module, "litellm", None):
                with pytest.raises(ImportError, match="litellm package is required"):
                    OpenRouterChatProvider(api_key="test-key")

    def test_chat_with_empty_base_url_does_not_set_api_base(self):
        """Test chat with explicitly empty base_url does not set api_base."""
        from obsidian_rag.llm.providers import OpenRouterChatProvider

        mock_response = {
            "choices": [{"message": {"content": "Test"}}],
        }

        with patch("obsidian_rag.llm.providers.log"):
            with patch("litellm.completion", return_value=mock_response) as mock_chat:
                provider = OpenRouterChatProvider(
                    api_key="test-key",
                    base_url="",
                )
                # Manually set base_url to empty to test defensive branch
                provider.base_url = ""
                messages = [{"role": "user", "content": "Hello"}]
                provider.chat(messages)

        mock_chat.assert_called_once()
        call_kwargs = mock_chat.call_args.kwargs
        assert "api_base" not in call_kwargs


class TestIndividualFactoryFunctions:
    """Test cases for individual provider factory functions."""

    def test_create_openai_embedding_provider_success(self):
        """Test creating OpenAI embedding provider with factory function."""
        from obsidian_rag.llm.providers import (
            OpenAIEmbeddingProvider,
            create_openai_embedding_provider,
        )

        with patch("obsidian_rag.llm.providers.log"):
            provider = create_openai_embedding_provider(api_key="test-key")

        assert isinstance(provider, OpenAIEmbeddingProvider)
        assert provider.api_key == "test-key"

    def test_create_openai_embedding_provider_with_model(self):
        """Test creating OpenAI embedding provider with custom model."""
        from obsidian_rag.llm.providers import create_openai_embedding_provider

        with patch("obsidian_rag.llm.providers.log"):
            provider = create_openai_embedding_provider(
                api_key="test-key",
                model="text-embedding-3-large",
            )

        assert provider.model == "text-embedding-3-large"
        assert provider._dimension == 3072

    def test_create_huggingface_embedding_provider_success(self):
        """Test creating HuggingFace embedding provider with factory function."""
        from obsidian_rag.llm.providers import (
            HuggingFaceEmbeddingProvider,
            create_huggingface_embedding_provider,
        )

        mock_embeddings = Mock()
        mock_hf_class = Mock(return_value=mock_embeddings)

        with patch("obsidian_rag.llm.providers.log"):
            with patch(
                "obsidian_rag.llm.providers.HuggingFaceEmbeddings",
                mock_hf_class,
            ):
                provider = create_huggingface_embedding_provider()

        assert isinstance(provider, HuggingFaceEmbeddingProvider)
        assert provider.model_name == "all-MiniLM-L6-v2"

    def test_create_huggingface_embedding_provider_with_device(self):
        """Test creating HuggingFace embedding provider with device parameter."""
        from obsidian_rag.llm.providers import create_huggingface_embedding_provider

        mock_embeddings = Mock()
        mock_hf_class = Mock(return_value=mock_embeddings)

        with patch("obsidian_rag.llm.providers.log"):
            with patch(
                "obsidian_rag.llm.providers.HuggingFaceEmbeddings",
                mock_hf_class,
            ):
                create_huggingface_embedding_provider(device="cuda")

        mock_hf_class.assert_called_once()
        call_kwargs = mock_hf_class.call_args.kwargs
        assert call_kwargs["model_kwargs"]["device"] == "cuda"

    def test_create_openrouter_embedding_provider_success(self):
        """Test creating OpenRouter embedding provider with factory function."""
        from obsidian_rag.llm.providers import (
            OpenRouterEmbeddingProvider,
            create_openrouter_embedding_provider,
        )

        with patch("obsidian_rag.llm.providers.log"):
            provider = create_openrouter_embedding_provider(api_key="test-key")

        assert isinstance(provider, OpenRouterEmbeddingProvider)
        assert provider.api_key == "test-key"
        assert provider.model == "qwen/qwen3-embedding-8b"

    def test_create_openai_chat_provider_success(self):
        """Test creating OpenAI chat provider with factory function."""
        from obsidian_rag.llm.providers import (
            OpenAIChatProvider,
            create_openai_chat_provider,
        )

        with patch("obsidian_rag.llm.providers.log"):
            provider = create_openai_chat_provider(api_key="test-key")

        assert isinstance(provider, OpenAIChatProvider)
        assert provider.api_key == "test-key"
        assert provider.model == "gpt-4"

    def test_create_openai_chat_provider_with_params(self):
        """Test creating OpenAI chat provider with custom parameters."""
        from obsidian_rag.llm.providers import create_openai_chat_provider

        with patch("obsidian_rag.llm.providers.log"):
            provider = create_openai_chat_provider(
                api_key="test-key",
                model="gpt-3.5-turbo",
                temperature=0.5,
                max_tokens=100,
            )

        assert provider.model == "gpt-3.5-turbo"
        assert provider.temperature == 0.5
        assert provider.max_tokens == 100

    def test_create_openrouter_chat_provider_success(self):
        """Test creating OpenRouter chat provider with factory function."""
        from obsidian_rag.llm.providers import (
            OpenRouterChatProvider,
            create_openrouter_chat_provider,
        )

        with patch("obsidian_rag.llm.providers.log"):
            provider = create_openrouter_chat_provider(api_key="test-key")

        assert isinstance(provider, OpenRouterChatProvider)
        assert provider.api_key == "test-key"
        assert provider.model == "anthropic/claude-3-opus"

    def test_create_openrouter_chat_provider_with_model(self):
        """Test creating OpenRouter chat provider with custom model."""
        from obsidian_rag.llm.providers import create_openrouter_chat_provider

        with patch("obsidian_rag.llm.providers.log"):
            provider = create_openrouter_chat_provider(
                api_key="test-key",
                model="openai/gpt-4",
            )

        assert provider.model == "openai/gpt-4"


class TestOptionalDependencyErrors:
    """Test defensive branches when optional dependencies are not installed."""

    def test_openai_embedding_provider_without_litellm(self):
        """Test OpenAIEmbeddingProvider raises ImportError when litellm is None (lines 57-58)."""
        with patch("obsidian_rag.llm.providers.litellm", None):
            with patch("obsidian_rag.llm.providers.log"):
                with pytest.raises(ImportError, match="litellm package is required"):
                    from obsidian_rag.llm.providers import OpenAIEmbeddingProvider

                    OpenAIEmbeddingProvider(api_key="test-key")

    def test_openai_chat_provider_without_litellm(self):
        """Test OpenAIChatProvider raises ImportError when litellm is None (lines 173-174)."""
        with patch("obsidian_rag.llm.providers.litellm", None):
            with patch("obsidian_rag.llm.providers.log"):
                with pytest.raises(ImportError, match="litellm package is required"):
                    from obsidian_rag.llm.providers import OpenAIChatProvider

                    OpenAIChatProvider(api_key="test-key")
