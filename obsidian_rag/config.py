"""Configuration management for obsidian-rag.

Supports layered configuration with precedence:
CLI flags > Environment variables > Config files > Defaults
"""

import logging
import os
from pathlib import Path
from typing import Any, TypedDict, Unpack

import yaml
from pydantic import Field, model_validator
from pydantic.fields import FieldInfo
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

log = logging.getLogger(__name__)

from obsidian_rag.config_env import T as T
from obsidian_rag.config_env import _interpolate_env_vars
from obsidian_rag.config_env import _replace_env_var as _replace_env_var
from obsidian_rag.config_models import (
    ChunkingConfig,
    DatabaseConfig,
    EndpointConfig,
    IngestionConfig,
    LoggingConfig,
    MCPConfig,
    VaultConfig,
)
from obsidian_rag.config_validators import (
    MAX_VAULTS,
    convert_endpoint_value,
    get_expected_dimension,
    merge_endpoints_into_data,
    parse_env_var_key,
    try_parse_numeric,
    validate_dimension_limit,
    validate_dimension_match,
    validate_vault_name,
)

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
    "vaults": {},
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

    @staticmethod
    def _try_parse_numeric(field_name: str, value: str) -> float | int | str:
        """Try to parse value as numeric type based on field name."""
        return try_parse_numeric(field_name, value)

    @staticmethod
    def _convert_endpoint_value(field_name: str, value: str) -> object:
        """Convert string env var value to appropriate type."""
        return convert_endpoint_value(field_name, value)

    @staticmethod
    def _parse_env_var_key(key: str) -> tuple[str, str] | None:
        """Parse environment variable key into endpoint and field names."""
        return parse_env_var_key(key)

    @staticmethod
    def _merge_endpoints_into_data(
        endpoints: dict[str, dict[str, object]],
        data: dict[str, object],
    ) -> dict[str, object]:
        """Merge parsed endpoints into the data dictionary."""
        return merge_endpoints_into_data(endpoints, data)

    @model_validator(mode="before")
    @classmethod
    def _parse_endpoint_env_vars(cls, data: dict[str, object]) -> dict[str, object]:
        """Parse endpoint environment variables into endpoints dict.

        Environment variables like OBSIDIAN_RAG_ENDPOINTS_EMBEDDING_PROVIDER
        are parsed into the endpoints dictionary structure since Pydantic
        settings' env_nested_delimiter only works with nested models,
        not with dict fields.

        Args:
            data: The raw data dictionary being validated.

        Returns:
            The data dictionary with parsed endpoints merged in.

        """
        _msg = "Parsing endpoint environment variables"
        log.debug(_msg)

        endpoints: dict[str, dict[str, object]] = {}

        for key, value in os.environ.items():
            parsed = cls._parse_env_var_key(key)
            if parsed is None:
                continue

            endpoint_name, field_name = parsed

            if endpoint_name not in endpoints:
                endpoints[endpoint_name] = {}

            endpoints[endpoint_name][field_name] = cls._convert_endpoint_value(
                field_name,
                value,
            )

        data = cls._merge_endpoints_into_data(endpoints, data)

        _msg = "Endpoint environment variables parsed"
        log.debug(_msg)
        return data

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
        return get_expected_dimension(self.endpoints)

    @staticmethod
    def _validate_dimension_limit(
        dimension: int,
        model: str,
        provider: str,
    ) -> None:
        """Validate that dimension does not exceed pgvector limit."""
        validate_dimension_limit(dimension, model, provider)

    @staticmethod
    def _validate_dimension_match(
        expected: int,
        configured: int,
        model: str,
    ) -> None:
        """Validate that expected dimension matches configured dimension."""
        validate_dimension_match(expected, configured, model)

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

    @staticmethod
    def _validate_vault_name(vault_name: str) -> None:
        """Validate a single vault name."""
        validate_vault_name(vault_name)

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
        if not self.vaults:
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
