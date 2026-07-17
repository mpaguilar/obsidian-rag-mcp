"""Integration tests for S3 region resolution through output_file dispatcher."""

import logging
from unittest.mock import MagicMock, patch

import pytest

from obsidian_rag.mcp_server.models import OutputFileConfig
from obsidian_rag.mcp_server.output_file import write_output_file


@patch("obsidian_rag.mcp_server.output_file.boto3.client")
def test_backward_compat_aws_no_region_no_app_default(mock_client: MagicMock) -> None:
    """AWS endpoint with no region and no app default -> URL-derived, no warning, upload succeeds."""
    config = OutputFileConfig(
        type="s3",
        endpoint="https://s3.eu-west-1.amazonaws.com",
        bucket="b",
        key="k",
        access_key_id="aki",
        secret_access_key="sak",
    )
    result = write_output_file({"data": "x"}, config, app_default_region=None)
    assert mock_client.call_args.kwargs["region_name"] == "eu-west-1"
    assert "output_file" in result


@patch("obsidian_rag.mcp_server.output_file.boto3.client")
def test_backward_compat_aws_bare_amazonaws_no_region(mock_client: MagicMock) -> None:
    """Bare s3.amazonaws.com -> us-east-1, no warning."""
    config = OutputFileConfig(
        type="s3",
        endpoint="https://s3.amazonaws.com",
        bucket="b",
        key="k",
        access_key_id="aki",
        secret_access_key="sak",
    )
    result = write_output_file({"data": "x"}, config, app_default_region=None)
    assert mock_client.call_args.kwargs["region_name"] == "us-east-1"
    assert "output_file" in result


@patch("obsidian_rag.mcp_server.output_file.boto3.client")
def test_dispatcher_threads_app_default_region_end_to_end(
    mock_client: MagicMock,
) -> None:
    """app_default_region="garage" threads through dispatcher."""
    config = OutputFileConfig(
        type="s3",
        endpoint="http://garage:3900",
        bucket="b",
        key="k",
        access_key_id="aki",
        secret_access_key="sak",
    )
    result = write_output_file({"data": "x"}, config, app_default_region="garage")
    assert mock_client.call_args.kwargs["region_name"] == "garage"
    assert "output_file" in result


@patch("obsidian_rag.mcp_server.output_file.boto3.client")
def test_dispatcher_per_call_region_overrides_app_default(
    mock_client: MagicMock,
) -> None:
    """config.region overrides app_default_region."""
    config = OutputFileConfig(
        type="s3",
        endpoint="http://garage:3900",
        bucket="b",
        key="k",
        access_key_id="aki",
        secret_access_key="sak",
        region="eu-west-1",
    )
    result = write_output_file({"data": "x"}, config, app_default_region="garage")
    assert mock_client.call_args.kwargs["region_name"] == "eu-west-1"
    assert "output_file" in result


@patch("obsidian_rag.mcp_server.output_file.boto3.client")
def test_dispatcher_garage_no_region_emits_warning_and_attempts_upload(
    mock_client: MagicMock,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Garage endpoint, no region, no app default -> WARNING emitted, boto3 called with us-east-1."""
    config = OutputFileConfig(
        type="s3",
        endpoint="http://garage:3900",
        bucket="b",
        key="k",
        access_key_id="aki",
        secret_access_key="sak",
    )
    with caplog.at_level(logging.WARNING, logger="obsidian_rag.mcp_server.output_file"):
        result = write_output_file({"data": "x"}, config, app_default_region=None)
    assert any("us-east-1" in record.message for record in caplog.records)
    assert mock_client.call_args.kwargs["region_name"] == "us-east-1"
    assert "output_file" in result
