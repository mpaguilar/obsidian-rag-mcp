from unittest.mock import patch

from botocore.exceptions import ClientError

from obsidian_rag.mcp_server.output_file import (
    _derive_region_from_endpoint,
    _probe_bucket_region,
    _resolve_s3_region,
    _write_s3,
)
from obsidian_rag.mcp_server.models import OutputFileConfig


# Probe tests
@patch("obsidian_rag.mcp_server.output_file.boto3.client")
def test_probe_success_returns_constraint(mock_client):
    """Successful probe returns the LocationConstraint region."""
    mock_client.return_value.get_bucket_location.return_value = {
        "LocationConstraint": "garage"
    }
    result = _probe_bucket_region(
        endpoint="http://garage:3900",
        access_key_id="key",
        secret_access_key="secret",
        addressing_style="path",
        bucket="my-bucket",
    )
    assert result == "garage"
    # Assert probe signed with us-east-1
    assert mock_client.call_args.kwargs["region_name"] == "us-east-1"


@patch("obsidian_rag.mcp_server.output_file.boto3.client")
def test_probe_client_error_falls_through(mock_client):
    """ClientError → probe returns None."""
    mock_client.return_value.get_bucket_location.side_effect = ClientError(
        {}, "GetBucketLocation"
    )
    result = _probe_bucket_region(
        endpoint="http://garage:3900",
        access_key_id="key",
        secret_access_key="secret",
        addressing_style="path",
        bucket="b",
    )
    assert result is None


@patch("obsidian_rag.mcp_server.output_file.boto3.client")
def test_probe_empty_constraint_falls_through(mock_client):
    """Empty LocationConstraint → None (AWS us-east-1 case)."""
    mock_client.return_value.get_bucket_location.return_value = {
        "LocationConstraint": ""
    }
    result = _probe_bucket_region(
        endpoint="http://garage:3900",
        access_key_id="key",
        secret_access_key="secret",
        addressing_style="path",
        bucket="b",
    )
    assert result is None


@patch("obsidian_rag.mcp_server.output_file.boto3.client")
def test_probe_none_constraint_falls_through(mock_client):
    """None LocationConstraint → None."""
    mock_client.return_value.get_bucket_location.return_value = {
        "LocationConstraint": None
    }
    result = _probe_bucket_region(
        endpoint="http://garage:3900",
        access_key_id="key",
        secret_access_key="secret",
        addressing_style="path",
        bucket="b",
    )
    assert result is None


@patch("obsidian_rag.mcp_server.output_file.boto3.client")
def test_probe_network_error_falls_through(mock_client):
    """Network error → None."""
    mock_client.side_effect = ConnectionError("boom")
    result = _probe_bucket_region(
        endpoint="http://garage:3900",
        access_key_id="key",
        secret_access_key="secret",
        addressing_style="path",
        bucket="b",
    )
    assert result is None


@patch("obsidian_rag.mcp_server.output_file.boto3.client")
def test_probe_not_called_when_url_derivation_succeeds(mock_client):
    """AWS endpoint → probe never invoked."""
    config = OutputFileConfig(
        type="s3",
        endpoint="https://s3.us-west-2.amazonaws.com",
        bucket="mybucket",
        key="results.json",
        access_key_id="AKIAIOSFODNN7EXAMPLE",
        secret_access_key="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
    )
    result = _resolve_s3_region(config, app_default_region=None)
    assert result == "us-west-2"
    mock_client.assert_not_called()


@patch("obsidian_rag.mcp_server.output_file.boto3.client")
def test_probe_uses_us_east_1_signing(mock_client):
    """Probe's boto3.client call has region_name='us-east-1'."""
    mock_client.return_value.get_bucket_location.return_value = {
        "LocationConstraint": "eu-west-1"
    }
    result = _probe_bucket_region(
        endpoint="http://garage:3900",
        access_key_id="key",
        secret_access_key="secret",
        addressing_style="path",
        bucket="b",
    )
    assert result == "eu-west-1"
    assert mock_client.call_args.kwargs["region_name"] == "us-east-1"


@patch("obsidian_rag.mcp_server.output_file.boto3.client")
def test_probe_uses_normal_tls_no_verify_false(mock_client):
    """Config passed to probe does NOT contain verify=False."""
    _probe_bucket_region(
        endpoint="http://garage:3900",
        access_key_id="key",
        secret_access_key="secret",
        addressing_style="path",
        bucket="b",
    )
    call_kwargs = mock_client.call_args.kwargs
    assert "verify" not in call_kwargs


# _write_s3 threading tests
@patch("obsidian_rag.mcp_server.output_file.boto3.client")
def test_write_s3_passes_region_name_to_boto3_client(mock_client):
    """_write_s3 threads region_name into boto3.client."""
    mock_client.return_value.put_object.return_value = {}
    _write_s3(
        result_json='{"documents": []}',
        endpoint="http://s3.example.com",
        bucket="mybucket",
        key="results.json",
        access_key_id="AKIAIOSFODNN7EXAMPLE",
        secret_access_key="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
        region_name="garage",
    )
    assert mock_client.call_args.kwargs["region_name"] == "garage"


@patch("obsidian_rag.mcp_server.output_file.boto3.client")
def test_write_s3_default_region_name_us_east_1(mock_client):
    """Default region_name is us-east-1."""
    mock_client.return_value.put_object.return_value = {}
    _write_s3(
        result_json='{"documents": []}',
        endpoint="http://s3.example.com",
        bucket="mybucket",
        key="results.json",
        access_key_id="AKIAIOSFODNN7EXAMPLE",
        secret_access_key="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
    )
    assert mock_client.call_args.kwargs["region_name"] == "us-east-1"


# Region resolution coverage tests
@patch("obsidian_rag.mcp_server.output_file.boto3.client")
def test_resolve_s3_region_app_default_region(mock_client):
    """app_default_region is returned when config.region is absent."""
    config = OutputFileConfig(
        type="s3",
        endpoint="http://garage:3900",
        bucket="mybucket",
        key="results.json",
        access_key_id="AKIAIOSFODNN7EXAMPLE",
        secret_access_key="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
    )
    result = _resolve_s3_region(config, app_default_region="ap-southeast-1")
    assert result == "ap-southeast-1"
    mock_client.assert_not_called()


@patch("obsidian_rag.mcp_server.output_file.boto3.client")
def test_resolve_s3_region_probe_path(mock_client):
    """When URL derivation fails, probe result is used."""
    mock_client.return_value.get_bucket_location.return_value = {
        "LocationConstraint": "garage"
    }
    config = OutputFileConfig(
        type="s3",
        endpoint="http://garage:3900",
        bucket="mybucket",
        key="results.json",
        access_key_id="AKIAIOSFODNN7EXAMPLE",
        secret_access_key="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
        addressing_style="path",
    )
    result = _resolve_s3_region(config, app_default_region=None)
    assert result == "garage"
    mock_client.assert_called_once()


@patch("obsidian_rag.mcp_server.output_file.boto3.client")
def test_resolve_s3_region_fallback_to_us_east_1(mock_client):
    """When probe also fails, fallback to us-east-1 with warning."""
    mock_client.return_value.get_bucket_location.side_effect = ClientError(
        {}, "GetBucketLocation"
    )
    config = OutputFileConfig(
        type="s3",
        endpoint="http://garage:3900",
        bucket="mybucket",
        key="results.json",
        access_key_id="AKIAIOSFODNN7EXAMPLE",
        secret_access_key="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
        addressing_style="path",
    )
    result = _resolve_s3_region(config, app_default_region=None)
    assert result == "us-east-1"


# Endpoint derivation coverage tests
def test_derive_region_empty_endpoint():
    """Empty endpoint returns None."""
    assert _derive_region_from_endpoint("") is None


def test_derive_region_no_match():
    """Unknown hostname returns None."""
    assert _derive_region_from_endpoint("http://unknown.example.com") is None
