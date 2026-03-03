"""Concrete LLM provider implementations."""

import logging
from typing import TYPE_CHECKING

from obsidian_rag.llm.base import (
    ChatError,
    ChatProvider,
    EmbeddingError,
    EmbeddingProvider,
)

if TYPE_CHECKING:
    pass

log = logging.getLogger(__name__)


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """OpenAI embedding provider implementation."""

    DEFAULT_MODEL = "text-embedding-3-small"
    DEFAULT_DIMENSION = 1536

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
    ) -> None:
        """Initialize OpenAI embedding provider.

        Args:
            api_key: OpenAI API key. If None, uses OPENAI_API_KEY env var.
            model: Model name. Defaults to text-embedding-3-small.
            base_url: Optional custom base URL for API.

        Raises:
            ImportError: If litellm package is not installed.
            ValueError: If API key is not provided.

        """
        _msg = "Initializing OpenAI embedding provider"
        log.debug(_msg)

        try:
            import litellm
        except ImportError as e:
            _msg = "litellm package is required for OpenAIEmbeddingProvider"
            log.error(_msg)
            raise ImportError(_msg) from e

        self.model = model or self.DEFAULT_MODEL
        self._dimension = self._get_dimension_for_model(self.model)

        # Get API key
        if api_key is None:
            import os

            api_key = os.environ.get("OPENAI_API_KEY")

        if not api_key:
            _msg = "OpenAI API key is required"
            log.error(_msg)
            raise ValueError(_msg)

        # Store configuration for litellm
        self.api_key = api_key
        self.base_url = base_url
        self.litellm = litellm

        _msg = f"OpenAI embedding provider initialized with model: {self.model}"
        log.debug(_msg)

    def _get_dimension_for_model(self, model: str) -> int:
        """Get embedding dimension for a specific model."""
        dimensions = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
        }
        return dimensions.get(model, self.DEFAULT_DIMENSION)

    def generate_embedding(self, text: str) -> list[float]:
        """Generate embedding using litellm.

        Args:
            text: Text to embed.

        Returns:
            Embedding vector as list of floats.

        Raises:
            EmbeddingError: If the API request fails.

        Notes:
            Uses litellm.embedding() for provider-agnostic LLM connectivity.

        """
        _msg = f"Generating embedding with OpenAI model: {self.model}"
        log.debug(_msg)

        try:
            # Use litellm for embedding generation
            # Format model as provider/model for litellm
            model_name = f"openai/{self.model}"
            response = self.litellm.embedding(
                model=model_name,
                input=[text],
                api_key=self.api_key,
                api_base=self.base_url,
            )
            return response["data"][0]["embedding"]
        except Exception as e:
            _msg = f"OpenAI embedding generation failed: {e}"
            log.exception(_msg)
            raise EmbeddingError(_msg) from e

    def get_dimension(self) -> int:
        """Get embedding dimension."""
        return self._dimension


class OpenAIChatProvider(ChatProvider):
    """OpenAI chat provider implementation."""

    DEFAULT_MODEL = "gpt-4"

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> None:
        """Initialize OpenAI chat provider.

        Args:
            api_key: OpenAI API key. If None, uses OPENAI_API_KEY env var.
            model: Model name. Defaults to gpt-4.
            base_url: Optional custom base URL for API.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.

        Raises:
            ImportError: If litellm package is not installed.
            ValueError: If API key is not provided.

        """
        _msg = "Initializing OpenAI chat provider"
        log.debug(_msg)

        try:
            import litellm
        except ImportError as e:
            _msg = "litellm package is required for OpenAIChatProvider"
            log.error(_msg)
            raise ImportError(_msg) from e

        self.model = model or self.DEFAULT_MODEL
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Get API key
        if api_key is None:
            import os

            api_key = os.environ.get("OPENAI_API_KEY")

        if not api_key:
            _msg = "OpenAI API key is required"
            log.error(_msg)
            raise ValueError(_msg)

        # Store configuration for litellm
        self.api_key = api_key
        self.base_url = base_url
        self.litellm = litellm

        _msg = f"OpenAI chat provider initialized with model: {self.model}"
        log.debug(_msg)

    def chat(self, messages: list[dict[str, str]], **kwargs) -> str:
        """Generate chat response using litellm.

        Args:
            messages: List of message dicts with 'role' and 'content'.
            **kwargs: Additional parameters to pass to the API.

        Returns:
            Response text from the model.

        Raises:
            ChatError: If the API request fails.

        Notes:
            Uses litellm.completion() for provider-agnostic LLM connectivity.

        """
        _msg = f"Sending chat request to OpenAI model: {self.model}"
        log.debug(_msg)

        request_params = {
            "model": f"openai/{self.model}",
            "messages": messages,
            "temperature": kwargs.get("temperature", self.temperature),
            "api_key": self.api_key,
        }

        if self.base_url:
            request_params["api_base"] = self.base_url

        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        if max_tokens is not None:
            request_params["max_tokens"] = max_tokens

        try:
            response = self.litellm.completion(**request_params)
            return response["choices"][0]["message"]["content"] or ""
        except Exception as e:
            _msg = f"OpenAI chat request failed: {e}"
            log.exception(_msg)
            raise ChatError(_msg) from e


class OpenRouterEmbeddingProvider(EmbeddingProvider):
    """OpenRouter embedding provider implementation.

    Uses LiteLLM to connect to OpenRouter API for embeddings.
    Supports models like qwen/qwen3-embedding-8b with 4096 dimensions.

    Attributes:
        DEFAULT_MODEL: Default embedding model for OpenRouter.
        DEFAULT_DIMENSION: Default embedding dimension (4096 for qwen3-embedding-8b).
        DEFAULT_BASE_URL: OpenRouter API base URL.

    """

    DEFAULT_MODEL = "qwen/qwen3-embedding-8b"
    DEFAULT_DIMENSION = 4096
    DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
    ) -> None:
        """Initialize OpenRouter embedding provider.

        Args:
            api_key: OpenRouter API key. If None, uses OPENROUTER_API_KEY env var.
            model: Model name. Defaults to qwen/qwen3-embedding-8b.
            base_url: Optional custom base URL for API.
                Defaults to https://openrouter.ai/api/v1.

        Raises:
            ImportError: If litellm package is not installed.
            ValueError: If API key is not provided.

        """
        _msg = "Initializing OpenRouter embedding provider"
        log.debug(_msg)

        try:
            import litellm
        except ImportError as e:
            _msg = "litellm package is required for OpenRouterEmbeddingProvider"
            log.error(_msg)
            raise ImportError(_msg) from e

        self.model = model or self.DEFAULT_MODEL
        self._dimension = self._get_dimension_for_model(self.model)

        # Get API key
        if api_key is None:
            import os

            api_key = os.environ.get("OPENROUTER_API_KEY")

        if not api_key:
            _msg = "OpenRouter API key is required"
            log.error(_msg)
            raise ValueError(_msg)

        # Store configuration for litellm
        self.api_key = api_key
        self.base_url = base_url or self.DEFAULT_BASE_URL
        self.litellm = litellm

        _msg = f"OpenRouter embedding provider initialized with model: {self.model}"
        log.debug(_msg)

    def _get_dimension_for_model(self, model: str) -> int:
        """Get embedding dimension for a specific model.

        Args:
            model: The model name.

        Returns:
            The embedding dimension for the model.

        """
        dimensions = {
            "qwen/qwen3-embedding-8b": 4096,
        }
        return dimensions.get(model, self.DEFAULT_DIMENSION)

    def generate_embedding(self, text: str) -> list[float]:
        """Generate embedding using litellm.

        Args:
            text: Text to embed.

        Returns:
            Embedding vector as list of floats.

        Raises:
            EmbeddingError: If the API request fails.

        Notes:
            Uses litellm.embedding() for provider-agnostic LLM connectivity.
            Model format for OpenRouter is "openrouter/{provider}/{model}".

        """
        _msg = f"Generating embedding with OpenRouter model: {self.model}"
        log.debug(_msg)

        try:
            # Use litellm for embedding generation
            # Format model as openrouter/provider/model for litellm
            model_name = f"openrouter/{self.model}"
            response = self.litellm.embedding(
                model=model_name,
                input=[text],
                api_key=self.api_key,
                api_base=self.base_url,
            )
            return response["data"][0]["embedding"]
        except Exception as e:
            _msg = f"OpenRouter embedding generation failed: {e}"
            log.exception(_msg)
            raise EmbeddingError(_msg) from e

    def get_dimension(self) -> int:
        """Get embedding dimension.

        Returns:
            The embedding dimension for the configured model.

        """
        return self._dimension


class OpenRouterChatProvider(ChatProvider):
    """OpenRouter chat provider implementation.

    Uses LiteLLM to connect to OpenRouter API for chat completions.
    Supports various chat models from multiple providers.

    Attributes:
        DEFAULT_MODEL: Default chat model for OpenRouter.
        DEFAULT_BASE_URL: OpenRouter API base URL.

    """

    DEFAULT_MODEL = "anthropic/claude-3-opus"
    DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> None:
        """Initialize OpenRouter chat provider.

        Args:
            api_key: OpenRouter API key. If None, uses OPENROUTER_API_KEY env var.
            model: Model name. Defaults to anthropic/claude-3-opus.
            base_url: Optional custom base URL for API.
                Defaults to https://openrouter.ai/api/v1.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.

        Raises:
            ImportError: If litellm package is not installed.
            ValueError: If API key is not provided.

        """
        _msg = "Initializing OpenRouter chat provider"
        log.debug(_msg)

        try:
            import litellm
        except ImportError as e:
            _msg = "litellm package is required for OpenRouterChatProvider"
            log.error(_msg)
            raise ImportError(_msg) from e

        self.model = model or self.DEFAULT_MODEL
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Get API key
        if api_key is None:
            import os

            api_key = os.environ.get("OPENROUTER_API_KEY")

        if not api_key:
            _msg = "OpenRouter API key is required"
            log.error(_msg)
            raise ValueError(_msg)

        # Store configuration for litellm
        self.api_key = api_key
        self.base_url = base_url or self.DEFAULT_BASE_URL
        self.litellm = litellm

        _msg = f"OpenRouter chat provider initialized with model: {self.model}"
        log.debug(_msg)

    def chat(self, messages: list[dict[str, str]], **kwargs) -> str:
        """Generate chat response using litellm.

        Args:
            messages: List of message dicts with 'role' and 'content'.
            **kwargs: Additional parameters to pass to the API.

        Returns:
            Response text from the model.

        Raises:
            ChatError: If the API request fails.

        Notes:
            Uses litellm.completion() for provider-agnostic LLM connectivity.
            Model format for OpenRouter is "openrouter/{provider}/{model}".

        """
        _msg = f"Sending chat request to OpenRouter model: {self.model}"
        log.debug(_msg)

        request_params = {
            "model": f"openrouter/{self.model}",
            "messages": messages,
            "temperature": kwargs.get("temperature", self.temperature),
            "api_key": self.api_key,
        }

        if self.base_url:
            request_params["api_base"] = self.base_url

        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        if max_tokens is not None:
            request_params["max_tokens"] = max_tokens

        try:
            response = self.litellm.completion(**request_params)
            return response["choices"][0]["message"]["content"] or ""
        except Exception as e:
            _msg = f"OpenRouter chat request failed: {e}"
            log.exception(_msg)
            raise ChatError(_msg) from e


class HuggingFaceEmbeddingProvider(EmbeddingProvider):
    """HuggingFace embedding provider for local embeddings using langchain."""

    DEFAULT_MODEL = "all-MiniLM-L6-v2"
    DEFAULT_DIMENSION = 384

    def __init__(
        self,
        model: str | None = None,
        device: str | None = None,
    ) -> None:
        """Initialize HuggingFace embedding provider.

        Args:
            model: Model name. Defaults to all-MiniLM-L6-v2.
            device: Device to use ('cpu', 'cuda', etc.). Auto-detected if None.

        Raises:
            ImportError: If langchain package is not installed.

        """
        _msg = "Initializing HuggingFace embedding provider"
        log.debug(_msg)

        try:
            from langchain_huggingface import HuggingFaceEmbeddings
        except ImportError as e:
            _msg = "langchain-huggingface package is required for HuggingFaceEmbeddingProvider"
            log.error(_msg)
            raise ImportError(_msg) from e

        self.model_name = model or self.DEFAULT_MODEL

        # Initialize model using langchain
        model_kwargs = {}
        if device:
            model_kwargs["device"] = device

        self.embedding_model = HuggingFaceEmbeddings(
            model_name=self.model_name,
            model_kwargs=model_kwargs,
        )
        self._dimension = self._get_dimension_from_model()

        _msg = (
            f"HuggingFace embedding provider initialized with model: {self.model_name}"
        )
        log.debug(_msg)

    def _get_dimension_from_model(self) -> int:
        """Get embedding dimension from model configuration.

        Returns:
            The embedding dimension.

        """
        # Map of known model dimensions
        dimensions = {
            "all-MiniLM-L6-v2": 384,
            "all-MiniLM-L12-v2": 384,
            "all-mpnet-base-v2": 768,
            "paraphrase-multilingual-MiniLM-L12-v2": 384,
        }
        return dimensions.get(self.model_name, self.DEFAULT_DIMENSION)

    def generate_embedding(self, text: str) -> list[float]:
        """Generate embedding using HuggingFace via langchain.

        Args:
            text: Text to embed.

        Returns:
            Embedding vector as list of floats.

        Raises:
            EmbeddingError: If embedding generation fails.

        Notes:
            Uses langchain.embeddings.HuggingFaceEmbeddings for local embeddings.

        """
        _msg = f"Generating embedding with model: {self.model_name}"
        log.debug(_msg)

        try:
            embedding = self.embedding_model.embed_query(text)
            return list(embedding)
        except Exception as e:
            _msg = f"HuggingFace embedding generation failed: {e}"
            log.exception(_msg)
            raise EmbeddingError(_msg) from e

    def get_dimension(self) -> int:
        """Get embedding dimension."""
        return self._dimension
