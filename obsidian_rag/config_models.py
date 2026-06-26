"""Pydantic models for configuration management.

This module contains the configuration model classes that define the
structure and validation for various Obsidian RAG configuration sections.
"""

from pathlib import Path

from pydantic import BaseModel, Field, ValidationInfo, field_validator, model_validator

# Constants for validation
PGVECTOR_MAX_DIMENSION = 2000
PORT_MAX = 65535
CHUNK_SIZE_MIN = 64
CHUNK_SIZE_MAX = 2048


class EndpointConfig(BaseModel):
    """Configuration for an LLM endpoint."""

    provider: str = "openai"
    model: str = ""
    api_key: str | None = None
    base_url: str | None = None
    temperature: float = 0.7
    max_tokens: int | None = None


class DatabaseConfig(BaseModel):
    """Configuration for database connection.

    Attributes:
        url: PostgreSQL connection URL.
        vector_dimension: Dimension for vector embeddings (must be positive).
            Common values: 1536 (OpenAI), 768 (HuggingFace), 1024, 384.
        pool_size: Number of persistent connections in the pool (default: 10).
            Increase for high-concurrency deployments with many Gunicorn workers.
        max_overflow: Maximum overflow connections beyond pool_size (default: 20).
        pool_timeout: Seconds to wait for a connection from pool (default: 30).
        pool_recycle: Seconds after which to recycle connections (default: 3600).

    """

    url: str = "postgresql+psycopg://localhost/obsidian_rag"
    vector_dimension: int = 1536
    pool_size: int = 10
    max_overflow: int = 20
    pool_timeout: int = 30
    pool_recycle: int = 3600

    @field_validator("vector_dimension")
    @classmethod
    def validate_vector_dimension(cls, v: int) -> int:
        """Validate vector dimension is positive and within pgvector limits.

        Args:
            v: The vector dimension value to validate.

        Returns:
            The validated value, or default (1536) if invalid.

        Raises:
            ValidationError: If vector_dimension exceeds 2000 (pgvector index limit).

        Notes:
            pgvector HNSW and IVFFLAT indexes both have a 2000 dimension limit.
            Compatible models: text-embedding-3-small (1536), all-MiniLM-L6-v2 (384),
            all-mpnet-base-v2 (768), text-embedding-ada-002 (1536).

        """
        if v <= 0:
            return 1536
        if v > PGVECTOR_MAX_DIMENSION:
            _msg = (
                "vector_dimension must be <= 2000 for pgvector index compatibility. "
                "Compatible models: "
                "text-embedding-3-small (1536), "
                "text-embedding-ada-002 (1536), "
                "all-MiniLM-L6-v2 (384), "
                "all-MiniLM-L12-v2 (384), "
                "all-mpnet-base-v2 (768), "
                "paraphrase-multilingual-MiniLM-L12-v2 (384). "
                f"Got: {v}"
            )
            raise ValueError(_msg)
        return v

    @field_validator("pool_size", "max_overflow", "pool_timeout", "pool_recycle")
    @classmethod
    def validate_positive_int(cls, v: int) -> int:
        """Validate pool configuration values are positive.

        Args:
            v: The integer value to validate.

        Returns:
            The validated value, or raises ValueError if invalid.

        """
        if v <= 0:
            _msg = f"Pool configuration value must be positive, got: {v}"
            raise ValueError(_msg)
        return v


class IngestionConfig(BaseModel):
    """Configuration for document ingestion."""

    batch_size: int = 100
    max_file_size_mb: int = 10
    progress_interval: int = 10
    max_chunk_chars: int = 24000
    chunk_overlap_chars: int = 800

    @field_validator("batch_size")
    @classmethod
    def validate_batch_size(cls, v: int) -> int:
        """Validate batch size is positive."""
        if v <= 0:
            return 100
        return v

    @field_validator("max_chunk_chars")
    @classmethod
    def validate_max_chunk_chars(cls, v: int) -> int:
        """Validate max chunk characters is positive.

        Args:
            v: The max chunk characters value to validate.

        Returns:
            The validated value, or default (24000) if invalid.

        """
        if v <= 0:
            return 24000
        return v

    @field_validator("chunk_overlap_chars")
    @classmethod
    def validate_chunk_overlap_chars(cls, v: int) -> int:
        """Validate chunk overlap characters is non-negative.

        Args:
            v: The chunk overlap characters value to validate.

        Returns:
            The validated value, or default (800) if invalid.

        """
        if v < 0:
            return 800
        return v


class ChunkingConfig(BaseModel):
    """Configuration for token-based document chunking.

    Attributes:
        chunk_size: Target number of tokens per chunk (default: 512, range: 64-2048).
        chunk_overlap: Number of tokens to overlap between chunks (default: 50).
        tokenizer_cache_dir: Directory for caching tokenizer models.
        tokenizer_model: Name of the tokenizer model to use.
        flashrank_enabled: Whether to enable flashrank re-ranking.
        flashrank_model: Name of the flashrank model to use.
        flashrank_top_k: Number of top results to return after re-ranking.

    """

    chunk_size: int = 512
    chunk_overlap: int = 50
    tokenizer_cache_dir: str = "~/.cache/obsidian-rag/tokenizers"
    tokenizer_model: str = "gpt2"
    flashrank_enabled: bool = True
    flashrank_model: str = "ms-marco-MiniLM-L-12-v2"
    flashrank_top_k: int = 10

    @field_validator("chunk_size")
    @classmethod
    def validate_chunk_size(cls, v: int) -> int:
        """Validate chunk size is within acceptable range.

        Args:
            v: The chunk size value to validate.

        Returns:
            Validated chunk size (clamped to 64-2048 range).

        """
        if v < CHUNK_SIZE_MIN:
            return CHUNK_SIZE_MIN
        if v > CHUNK_SIZE_MAX:
            return CHUNK_SIZE_MAX
        return v

    @field_validator("chunk_overlap")
    @classmethod
    def validate_chunk_overlap(cls, v: int, info: ValidationInfo) -> int:
        """Validate chunk overlap is less than chunk size.

        Args:
            v: The chunk overlap value to validate.
            info: Field validation info.

        Returns:
            Validated chunk overlap (ensured less than chunk_size).

        """
        # Get chunk_size from other fields if available
        chunk_size = 512
        if hasattr(info, "data") and "chunk_size" in info.data:  # pragma: no branch
            chunk_size = info.data["chunk_size"]

        if v >= chunk_size:
            return chunk_size - 1
        return v

    @field_validator("tokenizer_cache_dir")
    @classmethod
    def validate_cache_dir(cls, v: str) -> str:
        """Validate and expand tokenizer cache directory.

        Args:
            v: The cache directory path.

        Returns:
            Expanded absolute path.

        """
        expanded = Path(v).expanduser()
        return str(expanded)


class LoggingConfig(BaseModel):
    """Configuration for logging."""

    level: str = "INFO"
    format: str = "text"


class VaultConfig(BaseModel):
    """Configuration for an Obsidian vault.

    Attributes:
        container_path: Path inside container/Docker for file operations.
        host_path: Path on host system for link construction (defaults to container_path).
        description: Optional description of the vault.

    """

    container_path: str
    host_path: str | None = None
    description: str | None = None

    @model_validator(mode="after")
    def set_default_host_path(self) -> "VaultConfig":
        """Set host_path to container_path if not provided.

        Returns:
            The validated VaultConfig instance.

        """
        if self.host_path is None:
            self.host_path = self.container_path
        return self


class MCPConfig(BaseModel):
    """Configuration for MCP server.

    Attributes:
        host: Bind address for the MCP server.
        port: HTTP port for the MCP server.
        token: Bearer token for authentication (required).
        cors_origins: List of allowed CORS origins.
        enable_health_check: Enable health check endpoint.
        stateless_http: Enable stateless mode for horizontal scaling.
        max_concurrent_sessions: Maximum number of concurrent sessions.
        session_timeout_seconds: Session timeout for inactive sessions.
        rate_limit_per_second: Maximum connections per second per IP.
        rate_limit_window: Rate limit window in seconds.
        enable_request_logging: Enable HTTP request/response logging.

    """

    host: str = "0.0.0.0"
    port: int = 8000
    token: str | None = None
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])
    enable_health_check: bool = True
    stateless_http: bool = False
    max_concurrent_sessions: int = 100
    session_timeout_seconds: int = 300
    rate_limit_per_second: float = 10.0
    rate_limit_window: int = 60
    enable_request_logging: bool = True

    @field_validator("host")
    @classmethod
    def validate_host(cls, v: str) -> str:
        """Validate and clean host address.

        Strips surrounding quotes that may be included from environment
        variables and validates the host is not empty.

        Args:
            v: The host string to validate.

        Returns:
            Cleaned host string.

        Raises:
            ValueError: If host is empty after stripping.

        """
        # Strip quotes that may be present in environment variable values
        cleaned = v.strip().strip('"').strip("'")
        if not cleaned:
            _msg = "Host cannot be empty"
            raise ValueError(_msg)
        return cleaned

    @field_validator("port")
    @classmethod
    def validate_port(cls, v: int) -> int:
        """Validate port is in valid range."""
        if v < 1 or v > PORT_MAX:
            _msg = f"Port must be between 1 and 65535, got {v}"
            raise ValueError(_msg)
        return v

    @field_validator("max_concurrent_sessions")
    @classmethod
    def validate_max_concurrent(cls, v: int) -> int:
        """Validate max concurrent sessions is positive."""
        if v < 1:
            return 100
        return v

    @field_validator("session_timeout_seconds")
    @classmethod
    def validate_session_timeout(cls, v: int) -> int:
        """Validate session timeout is positive."""
        if v < 1:
            return 300
        return v

    @field_validator("rate_limit_per_second")
    @classmethod
    def validate_rate_limit(cls, v: float) -> float:
        """Validate rate limit is positive."""
        if v <= 0:
            return 10.0
        return v
