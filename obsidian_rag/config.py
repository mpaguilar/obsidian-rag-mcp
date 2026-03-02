"""Configuration management for obsidian-rag.

Supports layered configuration with precedence:
CLI flags > Environment variables > Config files > Defaults
"""

import logging
import os
import re
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

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
        "url": "postgresql://localhost/obsidian_rag",
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


def _interpolate_env_vars(value: Any) -> Any:
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


def _merge_configs(base: dict, override: dict) -> dict:
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
            result[key] = _merge_configs(result[key], value)
        else:
            result[key] = value
    return result


class EndpointConfig(BaseSettings):
    """Configuration for an LLM endpoint."""

    model_config = SettingsConfigDict(extra="allow")

    provider: str = "openai"
    model: str = ""
    api_key: str | None = None
    base_url: str | None = None
    temperature: float = 0.7
    max_tokens: int | None = None


class DatabaseConfig(BaseSettings):
    """Configuration for database connection."""

    url: str = "postgresql://localhost/obsidian_rag"


class IngestionConfig(BaseSettings):
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


class LoggingConfig(BaseSettings):
    """Configuration for logging."""

    level: str = "INFO"
    format: str = "text"


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

    def __init__(self, verbose: bool = False, **kwargs) -> None:  # type: ignore[no-untyped-def]
        """Initialize settings with merged configuration.

        Args:
            verbose: If True, set logging level to DEBUG.
            **kwargs: Additional configuration overrides.

        """
        _msg = "Initializing application settings"
        log.debug(_msg)

        # Start with defaults
        config = DEFAULT_CONFIG.copy()

        # Merge with YAML config file
        yaml_config = _load_yaml_config()
        config = _merge_configs(config, yaml_config)

        # Merge with kwargs (CLI flags)
        config = _merge_configs(config, kwargs)

        # Apply verbose flag
        if verbose:
            if "logging" not in config:
                config["logging"] = {}
            config["logging"]["level"] = "DEBUG"

        super().__init__(**config)

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
