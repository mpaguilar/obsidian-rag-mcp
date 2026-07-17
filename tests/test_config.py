"""Tests for MCPConfig output_file_s3_region field via Settings."""

from pathlib import Path
from unittest.mock import patch

import pytest

from obsidian_rag.config import (
    MCPConfig,
    Settings,
)


def test_mcp_config_output_file_s3_region_default_none() -> None:
    """Default output_file_s3_region is None."""
    config = MCPConfig()
    assert config.output_file_s3_region is None


@patch("obsidian_rag.config._get_config_file_path")
def test_mcp_config_output_file_s3_region_env_override(
    mock_get_config: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OBSIDIAN_RAG_MCP_OUTPUT_FILE_S3_REGION env var sets the field."""
    mock_get_config.return_value = None
    monkeypatch.setenv("OBSIDIAN_RAG_MCP_OUTPUT_FILE_S3_REGION", "garage")

    settings = Settings()
    assert settings.mcp.output_file_s3_region == "garage"


@patch("obsidian_rag.config._get_config_file_path")
def test_mcp_config_output_file_s3_region_env_override_multiple_vars(
    mock_get_config: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Multiple OBSIDIAN_RAG_MCP_* env vars exercise the for-loop continue branch."""
    mock_get_config.return_value = None
    monkeypatch.setenv("OBSIDIAN_RAG_MCP_OUTPUT_FILE_S3_REGION", "garage")
    monkeypatch.setenv("OBSIDIAN_RAG_MCP_TOKEN", "test-token")

    settings = Settings()
    assert settings.mcp.output_file_s3_region == "garage"
    assert settings.mcp.token == "test-token"


@patch("obsidian_rag.config._get_config_file_path")
def test_mcp_config_output_file_s3_region_env_override_with_explicit_kwargs(
    mock_get_config: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Explicit kwargs override env vars when field is already set in data."""
    mock_get_config.return_value = None
    monkeypatch.setenv("OBSIDIAN_RAG_MCP_OUTPUT_FILE_S3_REGION", "garage")

    settings = Settings(mcp={"output_file_s3_region": "eu-west-1"})
    assert settings.mcp.output_file_s3_region == "eu-west-1"


@patch("obsidian_rag.config._get_config_file_path")
def test_mcp_config_output_file_s3_region_env_override_model_instance(
    mock_get_config: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Env var sets mcp data when existing mcp is a model instance without the field."""
    mock_get_config.return_value = None
    monkeypatch.setenv("OBSIDIAN_RAG_MCP_OUTPUT_FILE_S3_REGION", "garage")

    # Passing an MCPConfig instance triggers the model_dump() branch in the validator.
    # Since the instance does not have output_file_s3_region set, the env var fills it in.
    settings = Settings(mcp=MCPConfig())
    assert settings.mcp.output_file_s3_region == "garage"


@patch("obsidian_rag.config._get_config_file_path")
def test_mcp_config_output_file_s3_region_env_override_non_dict_non_model(
    mock_get_config: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Env var replaces mcp when existing value is neither dict nor model."""
    mock_get_config.return_value = None
    monkeypatch.setenv("OBSIDIAN_RAG_MCP_OUTPUT_FILE_S3_REGION", "garage")

    # Passing a string triggers the else branch where existing_mcp is not a dict.
    settings = Settings(mcp="not-a-dict")  # type: ignore[arg-type]
    assert settings.mcp.output_file_s3_region == "garage"


def test_mcp_config_output_file_s3_region_config_file_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """YAML config with mcp.output_file_s3_region sets the field."""
    monkeypatch.chdir(tmp_path)
    config_file = tmp_path / ".obsidian-rag.yaml"
    config_file.write_text("mcp:\n  output_file_s3_region: eu-west-1")

    settings = Settings()
    assert settings.mcp.output_file_s3_region == "eu-west-1"
