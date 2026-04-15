"""Configuration management for obsidian-rag.

Supports layered configuration with precedence:
CLI flags > Environment variables > Config files > Defaults
"""

import logging
import os
import re
from pathlib import Path
from typing import Any, TypedDict, TypeVar, Unpack

import yaml
from pydantic import BaseModel, Field, ValidationInfo, field_validator, model_validator
from pydantic.fields import FieldInfo
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

log = logging.getLogger(__name__)

# TypeVar for homomorphic _interpolate_env_vars function
T = TypeVar("T", str, int, float, None, dict, list)

# Constants for validation
PGVECTOR_MAX_DIMENSION = 2000
PORT_MAX = 65535
CHUNK_SIZE_MIN = 64
CHUNK_SIZE_MAX = 2048

# Vault name validation pattern: alphanumeric, spaces, hyphens, underscores
# Must start with alphanumeric character
VAULT_NAME_PATTERN = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9 _-]*$")
MAX_VAULTS = 100
MAX_VAULT_NAME_LENGTH = 100

DEFAULT_CONFIG = {
    "endpoints": {
        "embedding": {
            "provider": "openai",
            "model": "text-embedding-3-small",
            "api_key": None,
            "base_url": None,
        },
        "analysis": {
            "provider": "openai",
            "model": "gpt-4",
            "api_key": None,
            "base_url": "https://api.openai.com/v1",
            "temperature": 0.7,
            "max_tokens": 2000,
        },
        "chat": {
            "provider": "openai",
            "model": "gpt-4",
            "api_key": None,
            "base_url": "https://api.openai.com/v1",
            "temperature": 0.8,
        },
    },
    "database": {
        "url": "postgresql+psycopg://localhost/obsidian_rag",
        "vector_dimension": 1536,
        "pool_size": 10,
        "max_overflow": 20,
        "pool_timeout": 30,
        "pool_recycle": 3600,
    },
    "ingestion": {
        "batch_size": 100,
        "max_file_size_mb": 10,
        "progress_interval": 10,
        "max_chunk_chars": 24000,
        "chunk_overlap_chars": 800,
    },
    "logging": {
        "level": "INFO",
        "format": "text",
    },
    "mcp": {
        "host": "0.0.0.0",
        "port": 8000,
        "token": None,
        "cors_origins": ["*"],
        "enable_health_check": True,
        "stateless_http": False,
    },
    "vaults": {
        "Obsidian Vault": {
            "container_path": "/data",
            "host_path": "/data",
            "description": "Default vault",
        },
    },
}


def _get_config_file_path() -> Path | None:
    """Find the config file path.

    Searches in order:
    1. $PWD/.obsidian-rag.yaml
    2. $XDG_CONFIG_HOME/obsidian-rag/config.yaml (or ~/.config/obsidian-rag/config.yaml)

    Returns:
        Path to config file if found, None otherwise.

    """
    _msg = "Searching for config file"
    log.debug(_msg)

    # Check current working directory first
    cwd_config = Path.cwd() / ".obsidian-rag.yaml"
    if cwd_config.exists():
        _msg = f"Found config file: {cwd_config}"
        log.debug(_msg)
        return cwd_config

    # Check XDG config directory
    xdg_config_home = os.environ.get("XDG_CONFIG_HOME")
    if xdg_config_home:
        xdg_config = Path(xdg_config_home) / "obsidian-rag" / "config.yaml"
    else:
        xdg_config = Path.home() / ".config" / "obsidian-rag" / "config.yaml"

    if xdg_config.exists():
        _msg = f"Found config file: {xdg_config}"
        log.debug(_msg)
        return xdg_config

    _msg = "No config file found"
    log.debug(_msg)
    return None


def _replace_env_var(match: re.Match) -> str:
    """Replace a single environment variable match.

    Args:
        match: Regex match object containing the variable expression.

    Returns:
        The replaced value.

    """
    _msg = "_replace_env_var starting"
    log.debug(_msg)
    var_expr = match.group(1)
    if ":-" in var_expr:
        var_name, default = var_expr.split(":-", 1)
        result = os.environ.get(var_name, default)
        _msg = "_replace_env_var returning"
        log.debug(_msg)
        return str(result)
    result = os.environ.get(var_expr, match.group(0))
    _msg = "_replace_env_var returning"
    log.debug(_msg)
    return str(result)


def _interpolate_env_vars(
    value: T,
) -> T:
    """Interpolate environment variables in configuration values.

    Supports ${VAR} and ${VAR:-default} syntax.

    This is a homomorphic function that preserves the input type:
    - str input -> str output (with env var interpolation)
    - dict input -> dict output (recursively interpolate values)
    - list input -> list output (recursively interpolate items)
    - int/float/None input -> same output (unchanged)

    Args:
        value: The value to interpolate (string, dict, list, or primitive).

    Returns:
        The interpolated value with the same type as input.

    """
    _msg = "_interpolate_env_vars starting"
    log.debug(_msg)
    if isinstance(value, str):
        pattern = r"\$\{([^}]+)\}"
        str_result: str = re.sub(pattern, _replace_env_var, value)
        _msg = "_interpolate_env_vars returning"
        log.debug(_msg)
        return str_result  # type: ignore[return-value]
    if isinstance(value, dict):
        dict_result: dict[str, Any] = {
            k: _interpolate_env_vars(v) for k, v in value.items()
        }
        _msg = "_interpolate_env_vars returning"
        log.debug(_msg)
        return dict_result  # type: ignore[return-value]
    if isinstance(value, list):
        list_result: list[Any] = [_interpolate_env_vars(item) for item in value]
        _msg = "_interpolate_env_vars returning"
        log.debug(_msg)
        return list_result  # type: ignore[return-value]
    _msg = "_interpolate_env_vars returning"
    log.debug(_msg)
    return value


def _load_yaml_config() -> dict:
    """Load configuration from YAML file.

    Returns:
        Dictionary with configuration values.

    """
    _msg = "Loading YAML configuration"
    log.debug(_msg)

    config_path = _get_config_file_path()
    if not config_path:
        return {}

    try:
        with open(config_path, encoding="utf-8") as f:
            config = yaml.safe_load(f)
    except (OSError, yaml.YAMLError) as e:
        _msg = f"Error loading config file: {e}"
        log.warning(_msg)
        config = None

    if config is None:
        return {}
    return _interpolate_env_vars(config)


def _deep_merge(base: dict, override: dict) -> dict:
    """Deep merge two configuration dictionaries.

    Args:
        base: The base configuration.
        override: The configuration to merge on top.

    Returns:
        The merged configuration.

    """
    _msg = "_deep_merge starting"
    log.debug(_msg)
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    _msg = "_deep_merge returning"
    log.debug(_msg)
    return result


# Known model dimensions for cross-validation
MODEL_DIMENSIONS: dict[str, dict[str, int]] = {
    "openai": {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    },
    "openrouter": {
        "qwen/qwen3-embedding-8b": 4096,
    },
    "huggingface": {
        "all-MiniLM-L6-v2": 384,
        "all-MiniLM-L12-v2": 384,
        "all-mpnet-base-v2": 768,
        "sentence-transformers/all-MiniLM-L6-v2": 384,
        "sentence-transformers/all-MiniLM-L12-v2": 384,
        "sentence-transformers/all-mpnet-base-v2": 768,
        "paraphrase-multilingual-MiniLM-L12-v2": 384,
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2": 384,
    },
}


class YamlConfigSettingsSource(PydanticBaseSettingsSource):
    """Settings source that loads from YAML config file.

    This source has lower precedence than environment variables but
    higher precedence than default values.

    """

    def get_field_value(
        self,
        _field: FieldInfo | None,
        _field_name: str,
    ) -> tuple[Any, str, bool]:
        """Get field value from YAML config.

        Note: This method is required by the base class but we implement
        __call__ instead for the full dictionary approach.

        Args:
            _field: The field being requested (unused, required by base class).
            _field_name: The name of the field (unused, required by base class).

        Returns:
            A tuple of (None, "", False) indicating no value.

        """
        # We don't use per-field loading; full dictionary is loaded below
        return None, "", False

    def __call__(self) -> dict[str, Any]:
        """Load configuration from YAML file.

        Returns:
            Dictionary with configuration values from YAML file.

        """
        yaml_config = _load_yaml_config()
        # Merge with defaults to ensure all keys exist
        return _deep_merge(DEFAULT_CONFIG.copy(), yaml_config)


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


class SettingsKwargs(TypedDict, total=False):
    """TypedDict for Settings constructor kwargs.

    This TypedDict defines all valid keyword arguments for the Settings
    class constructor, enabling type-safe configuration overrides.

    Uses dict types because users pass dicts that Pydantic converts
    to model instances internally.

    Attributes:
        endpoints: Dictionary of endpoint configurations.
        database: Database connection configuration.
        ingestion: Document ingestion settings.
        logging: Logging configuration.
        mcp: MCP server configuration.
        vaults: Dictionary of vault configurations.

    """

    endpoints: dict[str, dict[str, Any]]
    database: dict[str, Any]
    ingestion: dict[str, Any]
    chunking: dict[str, Any]
    logging: dict[str, Any]
    mcp: dict[str, Any]
    vaults: dict[str, VaultConfig]


class GetSettingsKwargs(TypedDict, total=False):
    """TypedDict for get_settings function kwargs.

    Includes verbose parameter which is passed through to Settings.
    Uses dict types because users pass dicts that Pydantic converts
    to model instances internally.

    Attributes:
        verbose: If True, set logging level to DEBUG.
        endpoints: Dictionary of endpoint configurations.
        database: Database connection configuration.
        ingestion: Document ingestion settings.
        logging: Logging configuration.
        mcp: MCP server configuration.
        vaults: Dictionary of vault configurations.

    """

    verbose: bool
    endpoints: dict[str, dict[str, Any]]
    database: dict[str, Any]
    ingestion: dict[str, Any]
    chunking: dict[str, Any]
    logging: dict[str, Any]
    mcp: dict[str, Any]
    vaults: dict[str, VaultConfig]


class Settings(BaseSettings):
    """Application settings with layered configuration.

    Configuration sources (highest to lowest precedence):
    1. CLI flags (passed directly to Settings constructor)
    2. Environment variables (OBSIDIAN_RAG_*)
    3. Config files (.obsidian-rag.yaml, ~/.config/obsidian-rag/config.yaml)
    4. Default values

    """

    model_config = SettingsConfigDict(
        env_prefix="OBSIDIAN_RAG_",
        env_nested_delimiter="_",
        extra="allow",
    )

    endpoints: dict[str, EndpointConfig] = Field(default_factory=dict)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    ingestion: IngestionConfig = Field(default_factory=IngestionConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    mcp: MCPConfig = Field(default_factory=MCPConfig)
    vaults: dict[str, VaultConfig] = Field(default_factory=dict)

    def __init__(
        self,
        *,
        verbose: bool = False,
        **kwargs: Unpack[SettingsKwargs],
    ) -> None:
        """Initialize settings with merged configuration.

        Args:
            verbose: If True, set logging level to DEBUG.
            **kwargs: Additional configuration overrides (CLI flags).

        """
        _msg = "Initializing application settings"
        log.debug(_msg)

        # Apply verbose flag as a CLI override
        if verbose:
            merged = _deep_merge(dict(kwargs), {"logging": {"level": "DEBUG"}})
            kwargs = merged  # type: ignore[assignment]

        super().__init__(**kwargs)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Customize settings sources to include YAML config.

        Precedence (highest to lowest):
        1. init_settings (CLI flags passed to constructor)
        2. env_settings (environment variables)
        3. file_secret_settings (secret files)
        4. YamlConfigSettingsSource (YAML config file)
        5. dotenv_settings (.env files)
        6. Defaults

        """
        return (
            init_settings,
            env_settings,
            file_secret_settings,
            dotenv_settings,
            YamlConfigSettingsSource(settings_cls),
        )

    def get_endpoint_config(self, endpoint_name: str) -> EndpointConfig | None:
        """Get configuration for a specific endpoint.

        Args:
            endpoint_name: Name of the endpoint (e.g., 'embedding', 'chat').

        Returns:
            Endpoint configuration or None if not found.

        """
        _msg = f"Getting endpoint config for: {endpoint_name}"
        log.debug(_msg)
        return self.endpoints.get(endpoint_name)

    def _get_expected_dimension(self) -> int | None:
        """Get expected embedding dimension from configuration.

        Returns:
            Expected dimension if provider+model is known, None otherwise.

        """
        embedding_config = self.endpoints.get("embedding")
        if not embedding_config:
            return None

        provider = embedding_config.provider
        model = embedding_config.model

        if not model or provider not in MODEL_DIMENSIONS:
            return None

        return MODEL_DIMENSIONS[provider].get(model)

    def _validate_dimension_limit(
        self,
        dimension: int,
        model: str,
        provider: str,
    ) -> None:
        """Validate that dimension does not exceed pgvector limit.

        Args:
            dimension: The embedding dimension to validate.
            model: The model name.
            provider: The provider name.

        Raises:
            ValueError: If dimension exceeds 2000.

        """
        if dimension > PGVECTOR_MAX_DIMENSION:
            _msg = (
                f"Embedding model '{model}' from '{provider}' produces "
                f"{dimension}-dimensional embeddings, which exceeds "
                "the pgvector index limit of 2000. "
                "Please use a compatible model with <= 2000 dimensions, or "
                "reduce the model's output dimensions if supported."
            )
            raise ValueError(_msg)

    def _validate_dimension_match(
        self,
        expected: int,
        configured: int,
        model: str,
    ) -> None:
        """Validate that expected dimension matches configured dimension.

        Args:
            expected: The expected dimension from the model.
            configured: The configured vector_dimension.
            model: The model name.

        Raises:
            ValueError: If dimensions do not match.

        """
        if expected != configured:
            _msg = (
                f"Embedding dimension mismatch: model '{model}' produces "
                f"{expected}-dimensional embeddings, but "
                f"database.vector_dimension is set to {configured}. "
                f"Please set database.vector_dimension to {expected} "
                "in your configuration."
            )
            raise ValueError(_msg)

    @model_validator(mode="after")
    def validate_embedding_dimension_compatibility(self) -> "Settings":
        """Validate that embedding provider dimension matches vector_dimension.

        This validator checks if the configured embedding provider's output
        dimension matches the database vector_dimension setting. A mismatch
        would cause runtime errors during document ingestion.

        Returns:
            The validated Settings instance.

        Raises:
            ValueError: If embedding provider dimension does not match
                vector_dimension setting.

        Notes:
            Only validates for known provider+model combinations.
            Skips validation if embedding endpoint is not configured.

        """
        expected_dimension = self._get_expected_dimension()
        if expected_dimension is None:
            return self

        embedding_config = self.endpoints.get("embedding")
        provider = embedding_config.provider if embedding_config else ""
        model = embedding_config.model if embedding_config else ""

        self._validate_dimension_limit(expected_dimension, model, provider)
        self._validate_dimension_match(
            expected_dimension,
            self.database.vector_dimension,
            model,
        )

        return self

    def _validate_vault_name(self, vault_name: str) -> None:
        """Validate a single vault name.

        Args:
            vault_name: Name of the vault to validate.

        Raises:
            ValueError: If vault name is invalid.

        """
        if not VAULT_NAME_PATTERN.match(vault_name):
            _msg = (
                f"Invalid vault name '{vault_name}'. "
                "Vault names must start with an alphanumeric character "
                "and contain only letters, numbers, spaces, hyphens, and underscores."
            )
            raise ValueError(_msg)

        if len(vault_name) > MAX_VAULT_NAME_LENGTH:
            _msg = (
                f"Vault name '{vault_name}' exceeds {MAX_VAULT_NAME_LENGTH} characters"
            )
            raise ValueError(_msg)

    @model_validator(mode="after")
    def validate_vaults(self) -> "Settings":
        """Validate vault configuration.

        Checks that:
        - Vault names match the allowed pattern
        - Number of vaults does not exceed maximum
        - If no vaults configured, default is created

        Returns:
            The validated Settings instance.

        Raises:
            ValueError: If vault configuration is invalid.

        """
        # If no vaults configured, create default
        if not self.vaults:  # pragma: no cover (coverage issue with pydantic validator)
            self.vaults = {
                "Obsidian Vault": VaultConfig(
                    container_path="/data",
                    host_path="/data",
                    description="Default vault",
                ),
            }
            return self

        # Check maximum number of vaults
        if len(self.vaults) > MAX_VAULTS:
            _msg = f"Maximum {MAX_VAULTS} vaults allowed, got {len(self.vaults)}"
            raise ValueError(_msg)

        # Validate each vault name
        for vault_name in self.vaults:
            self._validate_vault_name(vault_name)

        return self

    def get_vault(self, vault_name: str) -> VaultConfig | None:
        """Get vault configuration by name.

        Args:
            vault_name: Name of the vault.

        Returns:
            VaultConfig if found, None otherwise.

        """
        _msg = f"Getting vault config for: {vault_name}"
        log.debug(_msg)
        return self.vaults.get(vault_name)

    def get_vault_names(self) -> list[str]:
        """Get list of configured vault names.

        Returns:
            List of vault names.

        """
        return list(self.vaults.keys())


def get_settings(**kwargs: Unpack[GetSettingsKwargs]) -> Settings:
    """Get application settings instance.

    Args:
        **kwargs: CLI flag overrides.

    Returns:
        Settings instance.

    """
    _msg = "Creating settings instance"
    log.debug(_msg)
    return Settings(**kwargs)
