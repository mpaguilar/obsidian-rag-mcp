"""Type stubs for langchain-huggingface library.

Provides local HuggingFace embedding models via langchain interface.
"""

from typing import Any

class HuggingFaceEmbeddings:
    """Local HuggingFace embedding model wrapper.

    Loads and runs embedding models locally without requiring API calls.
    Suitable for privacy-sensitive deployments or offline use.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        cache_folder: str | None = None,
        model_kwargs: dict[str, Any] | None = None,
        encode_kwargs: dict[str, Any] | None = None,
        multi_process: bool = False,
    ) -> None:
        """Initialize HuggingFace embeddings.

        Args:
            model_name: HuggingFace model identifier.
                Default: "sentence-transformers/all-MiniLM-L6-v2"
            cache_folder: Directory to cache downloaded models.
            model_kwargs: Additional arguments for model loading.
                Common: {"device": "cuda"} for GPU acceleration.
            encode_kwargs: Additional arguments for encoding.
            multi_process: Enable multi-process encoding.
        """
        ...

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query text.

        Args:
            text: Text to embed.

        Returns:
            Embedding vector as list of floats.
        """
        ...

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple documents.

        Args:
            texts: List of texts to embed.

        Returns:
            List of embedding vectors.
        """
        ...

    @property
    def client(self) -> Any:
        """Access underlying sentence-transformers client."""
        ...
