"""Concrete LLM provider implementations."""

import logging
import os
from typing import TYPE_CHECKING, TypedDict, cast

if TYPE_CHECKING:
    import litellm
    from langchain_huggingface import HuggingFaceEmbeddings
else:
    try:
        import litellm
    except ImportError:
        litellm = None  # type: ignore[misc,assignment]

    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError:
        HuggingFaceEmbeddings = None  # type: ignore[misc,assignment]

from obsidian_rag.llm.base import (
    ChatError,
    ChatProvider,
    EmbeddingError,
    EmbeddingProvider,
)

log = logging.getLogger(__name__)


class OpenAIProviderConfig(TypedDict, total=False):
    """Configuration for OpenAI providers.

    Attributes:
        api_key: API key for authentication.
        model: Model name to use.
        base_url: Optional custom base URL for API.
        temperature: Sampling temperature for chat.
        max_tokens: Maximum tokens to generate.

    """

    api_key: str | None
    model: str | None
    base_url: str | None
    temperature: float
    max_tokens: int | None


class CompletionRequestParams(TypedDict, total=False):
    """Parameters for litellm completion request.

    Attributes:
        model: Model identifier.
        messages: List of message dictionaries.
        temperature: Sampling temperature.
        api_key: API key for authentication.
        api_base: Custom base URL for API.
        max_tokens: Maximum tokens to generate.

    """

    model: str
    messages: list[dict[str, str]]
    temperature: float
    api_key: str | None
    api_base: str
    max_tokens: int


class HuggingFaceProviderConfig(TypedDict, total=False):
    """Configuration for HuggingFace providers.

    Attributes:
        model: Model name to use.
        device: Device to use ('cpu', 'cuda', etc.).

    """

    model: str | None
    device: str | None


class OpenRouterProviderConfig(TypedDict, total=False):
    """Configuration for OpenRouter providers.

    Attributes:
        api_key: API key for authentication.
        model: Model name to use.
        base_url: Optional custom base URL for API.
        temperature: Sampling temperature for chat.
        max_tokens: Maximum tokens to generate.

    """

    api_key: str | None
    model: str | None
    base_url: str | None
    temperature: float
    max_tokens: int | None


# Union type for embedding provider configs
EmbeddingProviderConfig = (
    OpenAIProviderConfig | HuggingFaceProviderConfig | OpenRouterProviderConfig
)

# Union type for chat provider configs
ChatProviderConfig = OpenAIProviderConfig | OpenRouterProviderConfig


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

        if litellm is None:
            _msg = "litellm package is required for OpenAIEmbeddingProvider"
            log.error(_msg)
            raise ImportError(_msg)

        self.model = model or self.DEFAULT_MODEL
        self._dimension = self._get_dimension_for_model(self.model)

        # Get API key
        if api_key is None:
            api_key = os.environ.get("OPENAI_API_KEY")

        if not api_key or api_key.startswith("${"):
            _msg = (
                "OpenAI API key is required. Set OPENAI_API_KEY "
                "environment variable or configure api_key in config file."
            )
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
        _msg = "_get_dimension_for_model starting"
        log.debug(_msg)
        dimensions = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
        }
        result = dimensions.get(model, self.DEFAULT_DIMENSION)
        _msg = "_get_dimension_for_model returning"
        log.debug(_msg)
        return result

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
        except Exception as e:
            _msg = f"OpenAI embedding generation failed: {e}"
            log.exception(_msg)
            raise EmbeddingError(_msg) from e
        else:
            result = response["data"][0]["embedding"]
            _msg = "generate_embedding returning"
            log.debug(_msg)
            return result

    def get_dimension(self) -> int:
        """Get embedding dimension."""
        _msg = "get_dimension starting"
        log.debug(_msg)
        _msg = "get_dimension returning"
        log.debug(_msg)
        return self._dimension


# Individual factory functions for type-safe provider creation
# These replace the overloaded create_embedding_provider method


def create_openai_embedding_provider(
    api_key: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
) -> "OpenAIEmbeddingProvider":
    """Create an OpenAI embedding provider.

    Args:
        api_key: OpenAI API key. If None, uses OPENAI_API_KEY env var.
        model: Model name. Defaults to text-embedding-3-small.
        base_url: Optional custom base URL for API.

    Returns:
        OpenAIEmbeddingProvider instance.

    Raises:
        ImportError: If litellm package is not installed.
        ValueError: If API key is not provided.

    """
    _msg = "Creating OpenAI embedding provider"
    log.debug(_msg)
    return OpenAIEmbeddingProvider(
        api_key=api_key,
        model=model,
        base_url=base_url,
    )


def create_huggingface_embedding_provider(
    model: str | None = None,
    device: str | None = None,
) -> "HuggingFaceEmbeddingProvider":
    """Create a HuggingFace embedding provider.

    Args:
        model: Model name. Defaults to all-MiniLM-L6-v2.
        device: Device to use ('cpu', 'cuda', etc.). Auto-detected if None.

    Returns:
        HuggingFaceEmbeddingProvider instance.

    Raises:
        ImportError: If langchain-huggingface package is not installed.

    """
    _msg = "Creating HuggingFace embedding provider"
    log.debug(_msg)
    return HuggingFaceEmbeddingProvider(
        model=model,
        device=device,
    )


def create_openrouter_embedding_provider(
    api_key: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
) -> "OpenRouterEmbeddingProvider":
    """Create an OpenRouter embedding provider.

    Args:
        api_key: OpenRouter API key. If None, uses OPENROUTER_API_KEY env var.
        model: Model name. Defaults to qwen/qwen3-embedding-8b.
        base_url: Optional custom base URL for API.

    Returns:
        OpenRouterEmbeddingProvider instance.

    Raises:
        ImportError: If litellm package is not installed.
        ValueError: If API key is not provided.

    """
    _msg = "Creating OpenRouter embedding provider"
    log.debug(_msg)
    return OpenRouterEmbeddingProvider(
        api_key=api_key,
        model=model,
        base_url=base_url,
    )


def create_openai_chat_provider(
    api_key: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
    temperature: float = 0.7,
    max_tokens: int | None = None,
) -> "OpenAIChatProvider":
    """Create an OpenAI chat provider.

    Args:
        api_key: OpenAI API key. If None, uses OPENAI_API_KEY env var.
        model: Model name. Defaults to gpt-4.
        base_url: Optional custom base URL for API.
        temperature: Sampling temperature.
        max_tokens: Maximum tokens to generate.

    Returns:
        OpenAIChatProvider instance.

    Raises:
        ImportError: If litellm package is not installed.
        ValueError: If API key is not provided.

    """
    _msg = "Creating OpenAI chat provider"
    log.debug(_msg)
    return OpenAIChatProvider(
        api_key=api_key,
        model=model,
        base_url=base_url,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def create_openrouter_chat_provider(
    api_key: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
    temperature: float = 0.7,
    max_tokens: int | None = None,
) -> "OpenRouterChatProvider":
    """Create an OpenRouter chat provider.

    Args:
        api_key: OpenRouter API key. If None, uses OPENROUTER_API_KEY env var.
        model: Model name. Defaults to anthropic/claude-3-opus.
        base_url: Optional custom base URL for API.
        temperature: Sampling temperature.
        max_tokens: Maximum tokens to generate.

    Returns:
        OpenRouterChatProvider instance.

    Raises:
        ImportError: If litellm package is not installed.
        ValueError: If API key is not provided.

    """
    _msg = "Creating OpenRouter chat provider"
    log.debug(_msg)
    return OpenRouterChatProvider(
        api_key=api_key,
        model=model,
        base_url=base_url,
        temperature=temperature,
        max_tokens=max_tokens,
    )


class ProviderFactory:
    """Factory for creating provider instances.

    Creates appropriate provider instances based on configuration.

    Note:
        This factory maintains backward compatibility. For type-safe
        creation with proper IDE support, use the dedicated factory
        functions: create_openai_embedding_provider,
        create_huggingface_embedding_provider, etc.

    """

    @staticmethod
    def create_embedding_provider(
        provider_name: str,
        config: dict[str, object] | None = None,
    ) -> EmbeddingProvider:
        """Create an embedding provider instance.

        Args:
            provider_name: Name of the provider ('openai', 'huggingface', 'openrouter').
            config: Provider-specific configuration dictionary.

        Returns:
            EmbeddingProvider instance.

        Raises:
            ValueError: If the provider name is unknown.

        """
        _msg = f"Creating embedding provider: {provider_name}"
        log.debug(_msg)

        cfg = config or {}

        if provider_name == "openai":
            result: EmbeddingProvider = create_openai_embedding_provider(**cfg)  # type: ignore[arg-type]
            _msg = "create_embedding_provider returning"
            log.debug(_msg)
            return result
        if provider_name == "huggingface":
            result = create_huggingface_embedding_provider(**cfg)  # type: ignore[arg-type]
            _msg = "create_embedding_provider returning"
            log.debug(_msg)
            return result
        if provider_name == "openrouter":
            result = create_openrouter_embedding_provider(**cfg)  # type: ignore[arg-type]
            _msg = "create_embedding_provider returning"
            log.debug(_msg)
            return result
        _msg = f"Unknown embedding provider: {provider_name}"
        log.error(_msg)
        raise ValueError(_msg)

    @staticmethod
    def create_chat_provider(
        provider_name: str,
        config: dict[str, object] | None = None,
    ) -> ChatProvider:
        """Create a chat provider instance.

        Args:
            provider_name: Name of the provider ('openai', 'openrouter').
            config: Provider-specific configuration dictionary.

        Returns:
            ChatProvider instance.

        Raises:
            ValueError: If the provider name is unknown.

        """
        _msg = f"Creating chat provider: {provider_name}"
        log.debug(_msg)

        cfg = config or {}

        if provider_name == "openai":
            result: ChatProvider = create_openai_chat_provider(**cfg)  # type: ignore[arg-type]
            _msg = "create_chat_provider returning"
            log.debug(_msg)
            return result
        if provider_name == "openrouter":
            result = create_openrouter_chat_provider(**cfg)  # type: ignore[arg-type]
            _msg = "create_chat_provider returning"
            log.debug(_msg)
            return result
        _msg = f"Unknown chat provider: {provider_name}"
        log.error(_msg)
        raise ValueError(_msg)


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

        if litellm is None:
            _msg = "litellm package is required for OpenAIChatProvider"
            log.error(_msg)
            raise ImportError(_msg)

        self.model = model or self.DEFAULT_MODEL
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Get API key
        if api_key is None:
            api_key = os.environ.get("OPENAI_API_KEY")

        if not api_key or api_key.startswith("${"):
            _msg = (
                "OpenAI API key is required. Set OPENAI_API_KEY "
                "environment variable or configure api_key in config file."
            )
            log.error(_msg)
            raise ValueError(_msg)

        # Store configuration for litellm
        self.api_key = api_key
        self.base_url = base_url
        self.litellm = litellm

        _msg = f"OpenAI chat provider initialized with model: {self.model}"
        log.debug(_msg)

    def chat(self, messages: list[dict[str, str]], **kwargs: object) -> str:
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

        temperature = kwargs.get("temperature", self.temperature)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)

        request_params: "CompletionRequestParams" = {
            "model": f"openai/{self.model}",
            "messages": messages,
            "temperature": cast("float", temperature)
            if temperature is not None
            else self.temperature,
            "api_key": self.api_key,
        }

        if self.base_url:
            request_params["api_base"] = self.base_url

        if max_tokens is not None:
            request_params["max_tokens"] = cast("int", max_tokens)

        try:
            response = self.litellm.completion(**request_params)
        except Exception as e:
            _msg = f"OpenAI chat request failed: {e}"
            log.exception(_msg)
            raise ChatError(_msg) from e
        else:
            result = response["choices"][0]["message"]["content"] or ""
            _msg = "chat returning"
            log.debug(_msg)
            return result


def _add_openrouter_prefix(model: str) -> str:
    """Add 'openrouter/' prefix to model name for litellm 1.83+ native routing.

    If the model name already starts with 'openrouter/', it is returned
    unchanged to avoid double-prefixing.

    Args:
        model: Model name (e.g., 'anthropic/claude-3-opus' or
            'openrouter/anthropic/claude-3-opus').

    Returns:
        Model name with 'openrouter/' prefix added if not already present.

    """
    if model.startswith("openrouter/"):
        return model
    return f"openrouter/{model}"


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

        if litellm is None:
            _msg = "litellm package is required for OpenRouterEmbeddingProvider"
            log.error(_msg)
            raise ImportError(_msg)

        self.model = model or self.DEFAULT_MODEL
        self._dimension = self._get_dimension_for_model(self.model)

        # Get API key
        if api_key is None:
            api_key = os.environ.get("OPENROUTER_API_KEY")

        if not api_key or api_key.startswith("${"):
            _msg = (
                "OpenRouter API key is required. Set OPENROUTER_API_KEY "
                "environment variable or configure api_key in config file."
            )
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
        _msg = "_get_dimension_for_model starting"
        log.debug(_msg)
        dimensions = {
            "qwen/qwen3-embedding-8b": 4096,
        }
        result = dimensions.get(model, self.DEFAULT_DIMENSION)
        _msg = "_get_dimension_for_model returning"
        log.debug(_msg)
        return result

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
            Prepends 'openrouter/' prefix to model names for litellm 1.83+
            native OpenRouter routing support.

        """
        _msg = f"Generating embedding with OpenRouter model: {self.model}"
        log.debug(_msg)

        try:
            # litellm 1.83+ natively supports OpenRouter routing via the
            # 'openrouter/' prefix. This replaces the previous workaround that
            # stripped 'openai/' and set OPENAI_API_BASE/OPENAI_BASE_URL env vars.
            model_name = _add_openrouter_prefix(self.model)

            _msg = f"Calling litellm.embedding with model={model_name}"
            log.debug(_msg)

            response = self.litellm.embedding(
                model=model_name,
                input=[text],
                api_key=self.api_key,
                api_base=self.base_url,
                # OpenRouter requires explicit encoding_format, unlike OpenAI
                encoding_format="float",
            )
        except Exception as e:
            _msg = f"OpenRouter embedding generation failed: {e}"
            log.exception(_msg)
            raise EmbeddingError(_msg) from e
        else:
            result = response["data"][0]["embedding"]
            _msg = "generate_embedding returning"
            log.debug(_msg)
            return result

    def get_dimension(self) -> int:
        """Get embedding dimension.

        Returns:
            The embedding dimension for the configured model.

        """
        _msg = "get_dimension starting"
        log.debug(_msg)
        _msg = "get_dimension returning"
        log.debug(_msg)
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

        if litellm is None:
            _msg = "litellm package is required for OpenRouterChatProvider"
            log.error(_msg)
            raise ImportError(_msg)

        self.model = model or self.DEFAULT_MODEL
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Get API key
        if api_key is None:
            api_key = os.environ.get("OPENROUTER_API_KEY")

        if not api_key or api_key.startswith("${"):
            _msg = (
                "OpenRouter API key is required. Set OPENROUTER_API_KEY "
                "environment variable or configure api_key in config file."
            )
            log.error(_msg)
            raise ValueError(_msg)

        # Store configuration for litellm
        self.api_key = api_key
        self.base_url = base_url or self.DEFAULT_BASE_URL
        self.litellm = litellm

        _msg = f"OpenRouter chat provider initialized with model: {self.model}"
        log.debug(_msg)

    def chat(self, messages: list[dict[str, str]], **kwargs: object) -> str:
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

        # litellm 1.83+ natively supports OpenRouter routing via the
        # 'openrouter/' prefix. Avoid double-prefixing if already present.
        model_name = _add_openrouter_prefix(self.model)

        temperature = kwargs.get("temperature", self.temperature)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)

        request_params: CompletionRequestParams = {
            "model": model_name,
            "messages": messages,
            "temperature": cast("float", temperature)
            if temperature is not None
            else self.temperature,
            "api_key": self.api_key,
        }

        if self.base_url:
            request_params["api_base"] = self.base_url

        if max_tokens is not None:
            request_params["max_tokens"] = cast("int", max_tokens)

        try:
            response = self.litellm.completion(**request_params)
        except Exception as e:
            _msg = f"OpenRouter chat request failed: {e}"
            log.exception(_msg)
            raise ChatError(_msg) from e
        else:
            result = response["choices"][0]["message"]["content"] or ""
            _msg = "chat returning"
            log.debug(_msg)
            return result


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

        if HuggingFaceEmbeddings is None:
            _msg = "langchain-huggingface package is required for HuggingFaceEmbeddingProvider"
            log.error(_msg)
            raise ImportError(_msg)

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
        _msg = "_get_dimension_from_model starting"
        log.debug(_msg)
        # Map of known model dimensions
        dimensions = {
            "all-MiniLM-L6-v2": 384,
            "all-MiniLM-L12-v2": 384,
            "all-mpnet-base-v2": 768,
            "paraphrase-multilingual-MiniLM-L12-v2": 384,
        }
        result = dimensions.get(self.model_name, self.DEFAULT_DIMENSION)
        _msg = "_get_dimension_from_model returning"
        log.debug(_msg)
        return result

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
        except Exception as e:
            _msg = f"HuggingFace embedding generation failed: {e}"
            log.exception(_msg)
            raise EmbeddingError(_msg) from e
        else:
            result = list(embedding)
            _msg = "generate_embedding returning"
            log.debug(_msg)
            return result

    def get_dimension(self) -> int:
        """Get embedding dimension."""
        _msg = "get_dimension starting"
        log.debug(_msg)
        _msg = "get_dimension returning"
        log.debug(_msg)
        return self._dimension
