"""Type stubs for litellm library.

litellm provides provider-agnostic LLM connectivity.
"""

from typing import Any, TypedDict

class EmbeddingData(TypedDict):
    """Single embedding result."""

    embedding: list[float]
    index: int
    object: str

class EmbeddingResponse(TypedDict):
    """Response from embedding API call."""

    data: list[EmbeddingData]
    model: str
    object: str
    usage: dict[str, int]

class CompletionChoice(TypedDict):
    """Single completion choice."""

    finish_reason: str
    index: int
    message: dict[str, str]

class CompletionResponse(TypedDict):
    """Response from completion API call."""

    choices: list[CompletionChoice]
    created: int
    id: str
    model: str
    object: str
    usage: dict[str, int]

def embedding(
    model: str,
    input: str | list[str],
    api_key: str | None = None,
    api_base: str | None = None,
    encoding_format: str = "float",
    **kwargs: Any,
) -> EmbeddingResponse:
    """Generate embeddings using specified model.

    Args:
        model: Model identifier (e.g., "text-embedding-3-small").
        input: Text or list of texts to embed.
        api_key: API key for authentication.
        api_base: Custom base URL for API endpoint.
        encoding_format: Format for embeddings ("float" or "base64").
        **kwargs: Additional provider-specific parameters.

    Returns:
        EmbeddingResponse with embeddings and usage info.

    """
    ...

def completion(
    model: str,
    messages: list[dict[str, str]],
    temperature: float | None = None,
    max_tokens: int | None = None,
    api_key: str | None = None,
    api_base: str | None = None,
    **kwargs: Any,
) -> CompletionResponse:
    """Generate completion using specified model.

    Args:
        model: Model identifier (e.g., "gpt-4").
        messages: List of message dicts with 'role' and 'content'.
        temperature: Sampling temperature (0.0 to 2.0).
        max_tokens: Maximum tokens to generate.
        api_key: API key for authentication.
        api_base: Custom base URL for API endpoint.
        **kwargs: Additional provider-specific parameters.

    Returns:
        CompletionResponse with generated text and usage info.

    """
    ...
