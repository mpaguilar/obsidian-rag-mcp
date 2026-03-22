"""Type stubs for pgvector SQLAlchemy integration.

pgvector provides vector similarity search for PostgreSQL.
"""

from typing import Any

from sqlalchemy import TypeDecorator
from sqlalchemy.types import TypeEngine

class Vector(TypeDecorator[list[float]]):
    """SQLAlchemy type for PostgreSQL vector columns.

    This TypeDecorator enables storage and retrieval of high-dimensional
    vectors in PostgreSQL using the pgvector extension.

    Example:
        class Document(Base):
            content_vector: Mapped[list[float]] = mapped_column(Vector(1536))
    """

    impl: TypeEngine[Any]
    cache_ok: bool = True

    def __init__(self, dim: int) -> None:
        """Initialize Vector type with specified dimension.

        Args:
            dim: Vector dimension (e.g., 1536 for OpenAI embeddings).
                Must match the output dimension of your embedding model.
        """
        ...

    def cosine_distance(self, other: list[float]) -> Any:
        """Compute cosine distance between vectors.

        Args:
            other: Vector to compare against.

        Returns:
            SQL expression for cosine distance (1 - cosine_similarity).
            Lower values indicate higher similarity.
        """
        ...
