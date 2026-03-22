"""Type stubs for tiktoken library.

tiktoken provides fast tokenization for OpenAI models using Rust.
"""

class Encoding:
    """Tokenizer encoding for a specific model.

    Provides encode/decode operations for converting between
    text and token IDs.
    """

    def encode(
        self,
        text: str,
        allowed_special: set[str] | str = set(),  # noqa: B006
        disallowed_special: set[str] | str = "all",
    ) -> list[int]:
        """Encode text to token IDs.

        Args:
            text: Text to encode.
            allowed_special: Special tokens to allow.
                Use "all" to allow all, or set of token names.
            disallowed_special: Special tokens to disallow.
                Use "all" to disallow all, or set of token names.

        Returns:
            List of token IDs.
        """
        ...

    def decode(self, tokens: list[int]) -> str:
        """Decode token IDs to text.

        Args:
            tokens: List of token IDs.

        Returns:
            Decoded text string.
        """
        ...

    @property
    def n_vocab(self) -> int:
        """Vocabulary size."""
        ...

    @property
    def name(self) -> str:
        """Encoding name."""
        ...

def get_encoding(encoding_name: str) -> Encoding:
    """Get encoding by name.

    Args:
        encoding_name: Encoding identifier (e.g., "cl100k_base" for GPT-4).

    Returns:
        Encoding instance for the specified encoding.

    Raises:
        ValueError: If encoding name is unknown.
    """
    ...

def encoding_for_model(model_name: str) -> Encoding:
    """Get encoding appropriate for a model.

    Args:
        model_name: Model identifier (e.g., "gpt-4", "text-embedding-3-small").

    Returns:
        Encoding instance suitable for the model.

    Raises:
        KeyError: If model is not known.
    """
    ...
