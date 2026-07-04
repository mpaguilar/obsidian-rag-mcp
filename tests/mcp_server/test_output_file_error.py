"""Tests for output file error conditions."""

import errno
import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from botocore.exceptions import ClientError
from pydantic import ValidationError

from obsidian_rag.mcp_server.models import OutputFileConfig, OutputFileResult
from obsidian_rag.mcp_server.output_file import (
    _validate_local_path,
    _validate_s3_config,
    write_output_file,
)


def test_s3_unreachable_returns_error_dict() -> None:
    """Mocked boto3 raises ConnectionError; returns error dict."""
    mock_client = MagicMock()
    mock_client.put_object.side_effect = ConnectionError("Endpoint unreachable")
    with patch("boto3.client", return_value=mock_client):
        config = OutputFileConfig(
            type="s3",
            endpoint="http://bad.example.com",
            bucket="mybucket",
            key="results.json",
            access_key_id="AKIAIOSFODNN7EXAMPLE",
            secret_access_key="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
        )
        result = write_output_file({"documents": []}, config)
        assert result["success"] is False
        assert "Endpoint unreachable" in result["error"]


def test_s3_auth_failure_returns_error_dict() -> None:
    """Mocked boto3 raises ClientError with 403 status; returns error dict."""
    error_response = {"Error": {"Code": "403", "Message": "Forbidden"}}
    mock_client = MagicMock()
    mock_client.put_object.side_effect = ClientError(error_response, "PutObject")
    with patch("boto3.client", return_value=mock_client):
        config = OutputFileConfig(
            type="s3",
            endpoint="http://s3.example.com",
            bucket="mybucket",
            key="results.json",
            access_key_id="bad-key",
            secret_access_key="bad-secret",
        )
        result = write_output_file({"documents": []}, config)
        assert result["success"] is False
        assert "Forbidden" in result["error"]


def test_s3_bucket_not_found_returns_error_dict() -> None:
    """Mocked boto3 raises NoSuchBucket; returns error dict."""
    error_response = {
        "Error": {"Code": "NoSuchBucket", "Message": "The bucket does not exist"},
    }
    mock_client = MagicMock()
    mock_client.put_object.side_effect = ClientError(error_response, "PutObject")
    with patch("boto3.client", return_value=mock_client):
        config = OutputFileConfig(
            type="s3",
            endpoint="http://s3.example.com",
            bucket="missing",
            key="results.json",
            access_key_id="AKIAIOSFODNN7EXAMPLE",
            secret_access_key="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
        )
        result = write_output_file({"documents": []}, config)
        assert result["success"] is False
        assert "bucket does not exist" in result["error"]


def test_local_disk_full_returns_error_dict() -> None:
    """Mock os.fdopen to raise OSError(ENOSPC); returns error dict."""
    with patch(
        "obsidian_rag.mcp_server.output_file.os.fdopen",
        side_effect=OSError(errno.ENOSPC, "No space left on device"),
    ):
        config = OutputFileConfig(type="local", path="/tmp/out.json")
        result = write_output_file({"documents": []}, config)
        assert result["success"] is False
        assert "No space left on device" in result["error"]


def test_local_permission_denied_returns_error_dict(tmp_path: Path) -> None:
    """Mock os.mkdir to raise PermissionError; returns error dict."""
    target = str(tmp_path / "nested" / "out.json")
    with patch(
        "obsidian_rag.mcp_server.output_file.os.mkdir",
        side_effect=PermissionError("Permission denied"),
    ):
        config = OutputFileConfig(type="local", path=target)
        result = write_output_file({"documents": []}, config)
        assert result["success"] is False
        assert "Permission denied" in result["error"]


def test_local_invalid_path_raises_valueerror() -> None:
    """Path outside /tmp/ raises ValueError."""
    with pytest.raises(ValueError, match="must be under /tmp/"):
        _validate_local_path("/etc/passwd")


def test_local_traversal_attack_raises_valueerror() -> None:
    """/tmp/../etc/passwd raises ValueError."""
    with pytest.raises(ValueError, match="must be under /tmp/"):
        _validate_local_path("/tmp/../etc/passwd")


def test_s3_missing_fields_raises_valueerror() -> None:
    """OutputFileConfig with type='s3' but missing fields raises ValueError."""
    config = OutputFileConfig(type="s3", endpoint="http://s3.example.com")
    with pytest.raises(ValueError, match="missing required fields"):
        _validate_s3_config(config)


def test_invalid_type_value_raises_valueerror() -> None:
    """OutputFileConfig with type='ftp' fails Pydantic validation."""
    with pytest.raises(ValidationError):
        OutputFileConfig(type="ftp")  # type: ignore[arg-type]


def test_json_serialization_failure_returns_error_dict() -> None:
    """Result dict with non-serializable value causes error dict."""
    with patch(
        "obsidian_rag.mcp_server.output_file.json.dumps",
        side_effect=OSError("Serialization failed"),
    ):
        config = OutputFileConfig(type="local", path="/tmp/out.json")
        result = write_output_file({"documents": [object()]}, config)
        assert result["success"] is False
        assert "Serialization failed" in result["error"]


def test_s3_no_credentials_in_log(caplog: pytest.LogCaptureFixture) -> None:
    """Verify access_key_id/secret_access_key never appear in any log output."""
    caplog.set_level(logging.DEBUG, logger="obsidian_rag.mcp_server.output_file")
    mock_client = MagicMock()
    mock_client.put_object.side_effect = ConnectionError("fail")
    with patch("boto3.client", return_value=mock_client):
        config = OutputFileConfig(
            type="s3",
            endpoint="http://s3.example.com",
            bucket="mybucket",
            key="results.json",
            access_key_id="SECRET_KEY_ID",
            secret_access_key="SECRET_ACCESS_KEY",
        )
        write_output_file({"documents": []}, config)
    full_log = caplog.text
    assert "SECRET_KEY_ID" not in full_log
    assert "SECRET_ACCESS_KEY" not in full_log


def test_s3_no_credentials_in_summary() -> None:
    """Summary dict does not contain access_key_id or secret_access_key."""
    result = OutputFileResult(
        type="s3",
        bucket="mybucket",
        key="results.json",
        bytes=100,
        item_count=5,
    )
    dump = result.model_dump()
    assert "access_key_id" not in dump
    assert "secret_access_key" not in dump
