"""Configuration management for obsidian-rag.

Supports layered configuration with precedence:
CLI flags > Environment variables > Config files > Defaults
"""

import logging
import os
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, Union, overload

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic.fields import FieldInfo
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

log = logging.getLogger(__name__)

DEFAULT_CONFIG = {
    "endpoints": {
        "embedding": {
            "provider": "openai",
            "model": "text-embedding-3-small",
            "api_key": None,
            "base_url": "https://api.openai.com/v1",
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
    },
    "ingestion": {
        "batch_size": 100,
        "max_file_size_mb": 10,
        "progress_interval": 10,
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
    var_expr = match.group(1)
    if ":-" in var_expr:
        var_name, default = var_expr.split(":-", 1)
        result = os.environ.get(var_name, default)
        return str(result)
    else:
        result = os.environ.get(var_expr, match.group(0))
        return str(result)


@overload
def _interpolate_env_vars(value: str) -> str: ...


@overload
def _interpolate_env_vars(value: dict[str, Any]) -> dict[str, Any]: ...


@overload
def _interpolate_env_vars(value: list[Any]) -> list[Any]: ...


def _interpolate_env_vars(
    value: Union[str, dict[str, Any], list[Any]],
) -> Union[str, dict[str, Any], list[Any]]:
    """Interpolate environment variables in configuration values.

    Supports ${VAR} and ${VAR:-default} syntax.

    Args:
        value: The value to interpolate (string, dict, or list).

    Returns:
        The interpolated value.

    """
    if isinstance(value, str):
        pattern = r"\$\{([^}]+)\}"
        return re.sub(pattern, _replace_env_var, value)
    elif isinstance(value, dict):
        return {k: _interpolate_env_vars(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [_interpolate_env_vars(item) for item in value]
    else:
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
            if config is None:
                return {}
            return _interpolate_env_vars(config)
    except Exception as e:
        _msg = f"Error loading config file: {e}"
        log.warning(_msg)
        return {}


def _deep_merge(base: dict, override: dict) -> dict:
    """Deep merge two configuration dictionaries.

    Args:
        base: The base configuration.
        override: The configuration to merge on top.

    Returns:
        The merged configuration.

    """
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
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
        self, field: FieldInfo, field_name: str
    ) -> tuple[Any, str, bool]:  # noqa: ARG002
        """Get field value from YAML config.

        Note: This method is required by the base class but we implement
        __call__ instead for the full dictionary approach.

        Args:
            field: The field being requested (unused).
            field_name: The name of the field (unused).

        Returns:
            A tuple of (None, "", False) indicating no value.

        """
        # We don't use per-field loading; full dictionary is loaded below
        _ = field  # Explicitly mark as used for ruff
        _ = field_name  # Explicitly mark as used for ruff
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

    """

    url: str = "postgresql+psycopg://localhost/obsidian_rag"
    vector_dimension: int = 1536

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
        if v > 2000:
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


class IngestionConfig(BaseModel):
    """Configuration for document ingestion."""

    batch_size: int = 100
    max_file_size_mb: int = 10
    progress_interval: int = 10

    @field_validator("batch_size")
    @classmethod
    def validate_batch_size(cls, v: int) -> int:
        """Validate batch size is positive."""
        if v <= 0:
            return 100
        return v


class LoggingConfig(BaseModel):
    """Configuration for logging."""

    level: str = "INFO"
    format: str = "text"


class MCPConfig(BaseModel):
    """Configuration for MCP server.

    Attributes:
        host: Bind address for the MCP server.
        port: HTTP port for the MCP server.
        token: Bearer token for authentication (required).
        cors_origins: List of allowed CORS origins.
        enable_health_check: Enable health check endpoint.
        stateless_http: Enable stateless mode for horizontal scaling.

    """

    host: str = "0.0.0.0"
    port: int = 8000
    token: str | None = None
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])
    enable_health_check: bool = True
    stateless_http: bool = False

    @field_validator("port")
    @classmethod
    def validate_port(cls, v: int) -> int:
        """Validate port is in valid range."""
        if v < 1 or v > 65535:
            _msg = f"Port must be between 1 and 65535, got {v}"
            raise ValueError(_msg)
        return v


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
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    mcp: MCPConfig = Field(default_factory=MCPConfig)

    def __init__(self, verbose: bool = False, **kwargs) -> None:  # type: ignore[no-untyped-def]
        """Initialize settings with merged configuration.

        Args:
            verbose: If True, set logging level to DEBUG.
            **kwargs: Additional configuration overrides (CLI flags).

        """
        _msg = "Initializing application settings"
        log.debug(_msg)

        # Apply verbose flag as a CLI override
        if verbose:
            kwargs = _deep_merge(kwargs, {"logging": {"level": "DEBUG"}})

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
        self, dimension: int, model: str, provider: str
    ) -> None:
        """Validate that dimension does not exceed pgvector limit.

        Args:
            dimension: The embedding dimension to validate.
            model: The model name.
            provider: The provider name.

        Raises:
            ValueError: If dimension exceeds 2000.

        """
        if dimension > 2000:
            _msg = (
                f"Embedding model '{model}' from '{provider}' produces "
                f"{dimension}-dimensional embeddings, which exceeds "
                "the pgvector index limit of 2000. "
                "Please use a compatible model with <= 2000 dimensions, or "
                "reduce the model's output dimensions if supported."
            )
            raise ValueError(_msg)

    def _validate_dimension_match(
        self, expected: int, configured: int, model: str
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
            expected_dimension, self.database.vector_dimension, model
        )

        return self


def get_settings(**kwargs) -> Settings:  # type: ignore[no-untyped-def]
    """Get application settings instance.

    Args:
        **kwargs: CLI flag overrides.

    Returns:
        Settings instance.

    """
    _msg = "Creating settings instance"
    log.debug(_msg)
    return Settings(**kwargs)
