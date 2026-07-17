"""Tests for OutputFileConfig and OutputFileResult Pydantic models."""

from pydantic import ValidationError
import pytest

from obsidian_rag.mcp_server.models import OutputFileConfig, OutputFileResult


def test_output_file_config_local_type() -> None:
    """Test OutputFileConfig with type='local' and path."""
    config = OutputFileConfig(type="local", path="/tmp/results.json")
    assert config.type == "local"
    assert config.path == "/tmp/results.json"
    assert config.endpoint is None
    assert config.bucket is None
    assert config.key is None
    assert config.access_key_id is None
    assert config.secret_access_key is None


def test_output_file_config_s3_type() -> None:
    """Test OutputFileConfig with type='s3' and all S3 fields."""
    config = OutputFileConfig(
        type="s3",
        endpoint="https://s3.example.com",
        bucket="my-bucket",
        key="results.json",
        access_key_id="AKIAIOSFODNN7EXAMPLE",
        secret_access_key="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
    )
    assert config.type == "s3"
    assert config.endpoint == "https://s3.example.com"
    assert config.bucket == "my-bucket"
    assert config.key == "results.json"
    assert config.access_key_id == "AKIAIOSFODNN7EXAMPLE"
    assert config.secret_access_key == "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
    assert config.path is None


def test_output_file_config_defaults() -> None:
    """Test OutputFileConfig optional fields default to None."""
    config = OutputFileConfig(type="local")
    assert config.path is None
    assert config.endpoint is None
    assert config.bucket is None
    assert config.key is None
    assert config.access_key_id is None
    assert config.secret_access_key is None


def test_output_file_result_local() -> None:
    """Test OutputFileResult with type='local', path, bytes, item_count."""
    result = OutputFileResult(
        type="local",
        path="/tmp/results.json",
        bytes=1024,
        item_count=5,
    )
    assert result.type == "local"
    assert result.path == "/tmp/results.json"
    assert result.bytes == 1024
    assert result.item_count == 5
    assert result.bucket is None
    assert result.key is None


def test_output_file_result_s3() -> None:
    """Test OutputFileResult with type='s3', bucket, key, bytes, item_count."""
    result = OutputFileResult(
        type="s3",
        bucket="my-bucket",
        key="results.json",
        bytes=2048,
        item_count=10,
    )
    assert result.type == "s3"
    assert result.bucket == "my-bucket"
    assert result.key == "results.json"
    assert result.bytes == 2048
    assert result.item_count == 10
    assert result.path is None


def test_output_file_config_serialization() -> None:
    """Test OutputFileConfig round-trip model_dump + model_validate."""
    original = OutputFileConfig(
        type="s3",
        endpoint="https://s3.example.com",
        bucket="my-bucket",
        key="results.json",
        access_key_id="AKIAIOSFODNN7EXAMPLE",
        secret_access_key="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
    )
    dumped = original.model_dump()
    restored = OutputFileConfig.model_validate(dumped)
    assert restored.type == original.type
    assert restored.endpoint == original.endpoint
    assert restored.bucket == original.bucket
    assert restored.key == original.key
    assert restored.access_key_id == original.access_key_id
    assert restored.secret_access_key == original.secret_access_key
    assert restored.path is None


def test_output_file_result_serialization() -> None:
    """Test OutputFileResult round-trip model_dump + model_validate."""
    original = OutputFileResult(
        type="local",
        path="/tmp/results.json",
        bytes=1024,
        item_count=5,
    )
    dumped = original.model_dump()
    restored = OutputFileResult.model_validate(dumped)
    assert restored.type == original.type
    assert restored.path == original.path
    assert restored.bytes == original.bytes
    assert restored.item_count == original.item_count
    assert restored.bucket is None
    assert restored.key is None


def test_output_file_config_addressing_style_default() -> None:
    """Default addressing_style is 'virtual' for both local and s3 types."""
    config_s3 = OutputFileConfig(
        type="s3",
        endpoint="https://s3.example.com",
        bucket="my-bucket",
        key="results.json",
        access_key_id="AKIAIOSFODNN7EXAMPLE",
        secret_access_key="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
    )
    assert config_s3.addressing_style == "virtual"
    config_local = OutputFileConfig(type="local", path="/tmp/results.json")
    assert config_local.addressing_style == "virtual"


def test_output_file_config_addressing_style_path() -> None:
    """Explicit addressing_style='path' round-trips."""
    config = OutputFileConfig(
        type="s3",
        endpoint="http://garage:3900",
        bucket="my-bucket",
        key="results.json",
        access_key_id="AKIAIOSFODNN7EXAMPLE",
        secret_access_key="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
        addressing_style="path",
    )
    assert config.addressing_style == "path"


def test_output_file_config_addressing_style_none_allowed() -> None:
    """Explicit None is permitted; dispatcher coerces to 'virtual'."""
    config = OutputFileConfig(
        type="s3",
        endpoint="https://s3.example.com",
        bucket="my-bucket",
        key="results.json",
        access_key_id="AKIAIOSFODNN7EXAMPLE",
        secret_access_key="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
        addressing_style=None,
    )
    assert config.addressing_style is None


def test_output_file_config_serialization_addressing_style() -> None:
    """Round-trip a config with addressing_style='path' through model_dump + model_validate."""
    original = OutputFileConfig(
        type="s3",
        endpoint="http://garage:3900",
        bucket="my-bucket",
        key="results.json",
        access_key_id="AKIAIOSFODNN7EXAMPLE",
        secret_access_key="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
        addressing_style="path",
    )
    dumped = original.model_dump()
    restored = OutputFileConfig.model_validate(dumped)
    assert restored.addressing_style == "path"


def test_output_file_config_region_default_none() -> None:
    """Default region is None for both local and s3 types."""
    config_s3 = OutputFileConfig(
        type="s3",
        endpoint="https://s3.example.com",
        bucket="my-bucket",
        key="results.json",
        access_key_id="AKIAIOSFODNN7EXAMPLE",
        secret_access_key="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
    )
    assert config_s3.region is None
    config_local = OutputFileConfig(type="local", path="/tmp/results.json")
    assert config_local.region is None


def test_output_file_config_region_explicit_set() -> None:
    """Explicit region='garage' round-trips through construction."""
    config = OutputFileConfig(
        type="s3",
        endpoint="http://garage:3900",
        bucket="my-bucket",
        key="results.json",
        access_key_id="AKIAIOSFODNN7EXAMPLE",
        secret_access_key="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
        region="garage",
    )
    assert config.region == "garage"


def test_output_file_config_region_serialization() -> None:
    """Round-trip a config with region='eu-west-1' through model_dump + model_validate."""
    original = OutputFileConfig(
        type="s3",
        endpoint="https://s3.example.com",
        bucket="my-bucket",
        key="results.json",
        access_key_id="AKIAIOSFODNN7EXAMPLE",
        secret_access_key="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
        region="eu-west-1",
    )
    dumped = original.model_dump()
    restored = OutputFileConfig.model_validate(dumped)
    assert restored.region == "eu-west-1"


def test_output_file_config_region_none_allowed() -> None:
    """Explicit region=None is permitted."""
    config = OutputFileConfig(
        type="s3",
        endpoint="https://s3.example.com",
        bucket="my-bucket",
        key="results.json",
        access_key_id="AKIAIOSFODNN7EXAMPLE",
        secret_access_key="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
        region=None,
    )
    assert config.region is None


def test_output_file_config_invalid_type() -> None:
    """Test OutputFileConfig with invalid type raises ValidationError."""
    with pytest.raises(ValidationError):
        OutputFileConfig(type="ftp")
