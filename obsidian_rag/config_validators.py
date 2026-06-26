"""Standalone validation helpers for configuration.

This module contains pure validation and parsing functions that are
called by pydantic validators in the Settings class and config models.
"""

import logging
import re
from typing import Any

log = logging.getLogger(__name__)

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

# Vault name validation pattern: alphanumeric, spaces, hyphens, underscores
# Must start with alphanumeric character
VAULT_NAME_PATTERN = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9 _-]*$")
MAX_VAULTS = 100
MAX_VAULT_NAME_LENGTH = 100


def get_expected_dimension(endpoints: dict[str, Any]) -> int | None:
    """Get expected embedding dimension from configuration.

    Args:
        endpoints: Dictionary of endpoint configurations.

    Returns:
        Expected dimension if provider+model is known, None otherwise.

    """
    embedding_config = endpoints.get("embedding")
    if not embedding_config:
        return None

    provider = embedding_config.provider
    model = embedding_config.model

    if not model or provider not in MODEL_DIMENSIONS:
        return None

    return MODEL_DIMENSIONS[provider].get(model)


def validate_dimension_limit(
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
    from obsidian_rag.config_models import PGVECTOR_MAX_DIMENSION

    if dimension > PGVECTOR_MAX_DIMENSION:
        _msg = (
            f"Embedding model '{model}' from '{provider}' produces "
            f"{dimension}-dimensional embeddings, which exceeds "
            "the pgvector index limit of 2000. "
            "Please use a compatible model with <= 2000 dimensions, or "
            "reduce the model's output dimensions if supported."
        )
        raise ValueError(_msg)


def validate_dimension_match(
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


def validate_vault_name(vault_name: str) -> None:
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
        _msg = f"Vault name '{vault_name}' exceeds {MAX_VAULT_NAME_LENGTH} characters"
        raise ValueError(_msg)


def try_parse_numeric(field_name: str, value: str) -> float | int | str:
    """Try to parse value as numeric type based on field name.

    Args:
        field_name: The field name to determine numeric type.
        value: The string value to parse.

    Returns:
        Parsed float, int, or original string if parsing fails.

    """
    if field_name == "temperature":
        try:
            return float(value)
        except ValueError:
            return value
    if field_name == "max_tokens":
        try:
            return int(value)
        except ValueError:
            return value
    return value


def convert_endpoint_value(field_name: str, value: str) -> object:
    """Convert string env var value to appropriate type.

    Args:
        field_name: The field name (e.g., 'temperature', 'max_tokens').
        value: The string value from environment variable.

    Returns:
        The converted value (float, int, bool, None, or str).

    """
    # Try numeric conversion first
    result = try_parse_numeric(field_name, value)
    if result is not value:
        return result

    # Handle boolean and empty values
    if value.lower() in ("true", "false"):
        return value.lower() == "true"
    if value == "":
        return None
    return value


def parse_env_var_key(key: str) -> tuple[str, str] | None:
    """Parse environment variable key into endpoint and field names.

    Args:
        key: The environment variable key.

    Returns:
        Tuple of (endpoint_name, field_name) or None if invalid.

    """
    prefix = "OBSIDIAN_RAG_ENDPOINTS_"
    expected_parts_count = 2

    if not key.startswith(prefix):
        return None

    remaining = key[len(prefix) :]
    parts = remaining.split("_", 1)

    if len(parts) != expected_parts_count:
        return None

    endpoint_name, field_name = parts
    return endpoint_name.lower(), field_name.lower()


def merge_endpoint_config(
    existing_config: dict[str, object],
    env_config: dict[str, object],
) -> None:
    """Merge env var config into existing endpoint config in-place.

    Only sets values that are None or missing in existing_config.

    Args:
        existing_config: The existing endpoint configuration dict.
        env_config: The configuration from environment variables.

    """
    for field_name, value in env_config.items():
        existing_value = existing_config.get(field_name)
        if existing_value is None:
            existing_config[field_name] = value


def apply_endpoint_merge(
    existing_endpoints: dict[str, object],
    endpoint_name: str,
    endpoint_config: dict[str, object],
) -> None:
    """Apply merge for a single endpoint configuration.

    Args:
        existing_endpoints: The existing endpoints dictionary.
        endpoint_name: The name of the endpoint to merge.
        endpoint_config: The configuration from environment variables.

    """
    if endpoint_name not in existing_endpoints:
        existing_endpoints[endpoint_name] = endpoint_config
        return

    existing = existing_endpoints[endpoint_name]
    if isinstance(existing, dict):
        merge_endpoint_config(existing, endpoint_config)


def merge_endpoints_into_data(
    endpoints: dict[str, dict[str, object]],
    data: dict[str, object],
) -> dict[str, object]:
    """Merge parsed endpoints into the data dictionary.

    Fills in missing values and replaces None values with env var values.
    Explicit non-None kwargs take precedence over env vars.

    Args:
        endpoints: The parsed endpoints dictionary (from env vars).
        data: The raw data dictionary being validated.

    Returns:
        The data dictionary with endpoints merged in.

    """
    if not endpoints:
        return data

    existing_endpoints = data.get("endpoints", {})
    if not isinstance(existing_endpoints, dict):
        data["endpoints"] = endpoints
        return data

    for endpoint_name, endpoint_config in endpoints.items():
        apply_endpoint_merge(
            existing_endpoints,
            endpoint_name,
            endpoint_config,
        )

    data["endpoints"] = existing_endpoints
    return data
