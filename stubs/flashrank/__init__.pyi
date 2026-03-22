"""Type stubs for flashrank library.

flashrank provides cross-encoder re-ranking capabilities for semantic search.
"""

from typing import Any

class RerankRequest:
    """Request object for re-ranking.

    Attributes:
        query: The search query string.
        passages: List of passage dictionaries with 'id' and 'text' keys.
    """

    def __init__(self, query: str, passages: list[dict[str, Any]]) -> None:
        """Initialize RerankRequest.

        Args:
            query: The search query string.
            passages: List of passage dictionaries. Each dict should have:
                - 'id': Unique identifier (str)
                - 'text': Passage content (str)
                - Optional 'meta': Metadata dict
        """
        ...

class Ranker:
    """Cross-encoder ranker for re-ranking passages.

    This class loads a pre-trained cross-encoder model and provides
    re-ranking capabilities for semantic search results.
    """

    def __init__(
        self,
        model_name: str = "ms-marco-MiniLM-L-12-v2",
        max_length: int = 128,
    ) -> None:
        """Initialize the Ranker with a pre-trained model.

        Args:
            model_name: Name of the pre-trained model to load.
                Default: "ms-marco-MiniLM-L-12-v2"
            max_length: Maximum token length for input sequences.
                Default: 128
        """
        ...

    def rerank(self, request: RerankRequest) -> list[dict[str, Any]]:
        """Re-rank passages based on query relevance.

        Args:
            request: RerankRequest object containing query and passages.

        Returns:
            List of result dictionaries, each containing:
                - 'id': Passage identifier (matches input id)
                - 'text': Passage text
                - 'score': Relevance score (float, higher is better)
                - 'meta': Metadata dict (if provided in input)

            Results are sorted by score in descending order.
        """
        ...
