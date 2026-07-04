"""Tests for _parse_output_file helpers in server.py."""

import json

import pytest

from obsidian_rag.mcp_server.models import OutputFileConfig
from obsidian_rag.mcp_server.server import (
    _parse_output_file,
    _parse_output_file_str_or_dict,
)


def test_parse_output_file_none_returns_none() -> None:
    """Test that None input returns None."""
    result = _parse_output_file(None)
    assert result is None


def test_parse_output_file_model_returns_model() -> None:
    """Test that OutputFileConfig input is returned unchanged."""
    config = OutputFileConfig(type="local", path="/tmp/test.json")
    result = _parse_output_file(config)
    assert result is config


def test_parse_output_file_dict_returns_config() -> None:
    """Test that dict input is parsed into OutputFileConfig."""
    data = {"type": "local", "path": "/tmp/test.json"}
    result = _parse_output_file(data)
    assert isinstance(result, OutputFileConfig)
    assert result.type == "local"
    assert result.path == "/tmp/test.json"


def test_parse_output_file_json_string_returns_config() -> None:
    """Test that JSON string input is parsed into OutputFileConfig."""
    json_str = '{"type": "local", "path": "/tmp/test.json"}'
    result = _parse_output_file(json_str)
    assert isinstance(result, OutputFileConfig)
    assert result.type == "local"
    assert result.path == "/tmp/test.json"


def test_parse_output_file_empty_string_returns_none() -> None:
    """Test that empty string input returns None."""
    result = _parse_output_file("")
    assert result is None


def test_parse_output_file_whitespace_string_returns_none() -> None:
    """Test that whitespace-only string returns None."""
    result = _parse_output_file("   ")
    assert result is None


def test_parse_output_file_invalid_json_returns_none() -> None:
    """Test that invalid JSON string raises JSONDecodeError."""
    with pytest.raises(json.JSONDecodeError):
        _parse_output_file("{invalid json}")


def test_parse_output_file_str_or_dict_dict_input() -> None:
    """Test that dict input to helper returns OutputFileConfig."""
    data = {"type": "s3", "bucket": "my-bucket", "key": "results.json"}
    result = _parse_output_file_str_or_dict(data)
    assert isinstance(result, OutputFileConfig)
    assert result.type == "s3"
    assert result.bucket == "my-bucket"
    assert result.key == "results.json"


def test_parse_output_file_str_or_dict_json_input() -> None:
    """Test that JSON string input to helper returns OutputFileConfig."""
    json_str = '{"type": "s3", "bucket": "my-bucket", "key": "results.json"}'
    result = _parse_output_file_str_or_dict(json_str)
    assert isinstance(result, OutputFileConfig)
    assert result.type == "s3"
    assert result.bucket == "my-bucket"
    assert result.key == "results.json"


def test_parse_output_file_str_or_dict_non_dict_returns_none() -> None:
    """Test that parsed value which is not a dict returns None."""
    result = _parse_output_file_str_or_dict("123")
    assert result is None


def test_parse_output_file_str_or_dict_empty_string_returns_none() -> None:
    """Test that empty string input to helper returns None."""
    result = _parse_output_file_str_or_dict("")
    assert result is None
