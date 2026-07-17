import json
import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from botocore.exceptions import ClientError

from obsidian_rag.mcp_server.output_file import (
    OutputFileConfig,
    OutputFileResult,
    _count_items,
    _derive_region_from_endpoint,
    _probe_bucket_region,
    _resolve_s3_region,
    _validate_local_path,
    _validate_s3_config,
    _warn_fallback_region,
    _write_local,
    _write_s3,
    build_output_file_summary,
    write_output_file,
)


def test_output_file_config_local_valid_path():
    """Validate path under /tmp/."""
    result = _validate_local_path("/tmp/test.json")
    assert result == "/tmp/test.json"


def test_output_file_config_local_invalid_path():
    """Raises ValueError for paths outside /tmp/."""
    with pytest.raises(ValueError, match="must be under /tmp/"):
        _validate_local_path("/etc/passwd")


def test_output_file_config_local_relative_path():
    """Resolves relative paths; rejects if not under /tmp/."""
    with pytest.raises(ValueError, match="must be under /tmp/"):
        _validate_local_path("test.json")


def test_output_file_config_local_traversal_attack():
    """Rejects /tmp/../etc/passwd."""
    with pytest.raises(ValueError, match="must be under /tmp/"):
        _validate_local_path("/tmp/../etc/passwd")


def test_validate_s3_config_all_fields():
    """Passes when all fields present."""
    config = OutputFileConfig(
        type="s3",
        endpoint="http://s3.example.com",
        bucket="mybucket",
        key="results.json",
        access_key_id="AKIAIOSFODNN7EXAMPLE",
        secret_access_key="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
    )
    _validate_s3_config(config)  # should not raise


def test_validate_s3_config_missing_fields():
    """Raises ValueError listing missing fields."""
    config = OutputFileConfig(type="s3", endpoint="http://s3.example.com")
    with pytest.raises(ValueError, match="missing required fields"):
        _validate_s3_config(config)


def test_write_local_atomic_write(tmp_path):
    """Writes to temp then renames; file content matches."""
    target = str(tmp_path / "output.json")
    data = '{"success": true}'
    result = _write_local(data, target)
    assert result.type == "local"
    assert result.path == target
    assert result.bytes == len(data.encode("utf-8"))
    assert Path(target).read_text(encoding="utf-8") == data


def test_write_local_auto_creates_parent_dir(tmp_path):
    """mkdir -p for nested dirs."""
    target = str(tmp_path / "a" / "b" / "output.json")
    data = '{"success": true}'
    _write_local(data, target)
    assert Path(target).exists()


def test_write_local_overwrites_existing(tmp_path):
    """Second write overwrites first."""
    target = str(tmp_path / "output.json")
    _write_local('{"old": true}', target)
    _write_local('{"new": true}', target)
    assert Path(target).read_text(encoding="utf-8") == '{"new": true}'


def test_write_local_empty_result(tmp_path):
    """bytes=0, valid write."""
    target = str(tmp_path / "output.json")
    result = _write_local("", target)
    assert result.bytes == 0
    assert Path(target).read_text(encoding="utf-8") == ""


def test_write_local_cleanup_on_failure(tmp_path):
    """Cleans up temp file when os.replace fails."""
    target = str(tmp_path / "output.json")
    with patch(
        "obsidian_rag.mcp_server.output_file.os.replace",
        side_effect=OSError("replace failed"),
    ):
        with pytest.raises(OSError, match="replace failed"):
            _write_local('{"data": true}', target)
    assert not Path(target).exists()


def test_write_local_cleanup_temp_gone(tmp_path):
    """Handles case where temp file is already gone during cleanup."""
    target = str(tmp_path / "output.json")
    with (
        patch(
            "obsidian_rag.mcp_server.output_file.os.replace",
            side_effect=OSError("replace failed"),
        ),
        patch(
            "obsidian_rag.mcp_server.output_file.os.path.exists",
            return_value=False,
        ),
    ):
        with pytest.raises(OSError, match="replace failed"):
            _write_local('{"data": true}', target)
    assert not Path(target).exists()


def test_write_local_cleanup_unlink_fails(tmp_path):
    """Handles OSError during temp file cleanup."""
    target = str(tmp_path / "output.json")
    with (
        patch(
            "obsidian_rag.mcp_server.output_file.os.replace",
            side_effect=OSError("replace failed"),
        ),
        patch(
            "obsidian_rag.mcp_server.output_file.os.unlink",
            side_effect=OSError("unlink failed"),
        ),
    ):
        with pytest.raises(OSError, match="replace failed"):
            _write_local('{"data": true}', target)
    assert not Path(target).exists()


def test_write_s3_success() -> None:
    """Mocked boto3, verifies PutObject called with correct params and Config constructed with timeouts and default addressing_style."""
    mock_client = MagicMock()
    with (
        patch("boto3.client", return_value=mock_client) as mock_boto3,
        patch("obsidian_rag.mcp_server.output_file.Config") as mock_config,
    ):
        result = _write_s3(
            '{"documents": []}',
            "http://s3.example.com",
            "mybucket",
            "results.json",
            "AKIAIOSFODNN7EXAMPLE",
            "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
        )
        mock_boto3.assert_called_once_with(
            "s3",
            endpoint_url="http://s3.example.com",
            aws_access_key_id="AKIAIOSFODNN7EXAMPLE",
            aws_secret_access_key="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
            region_name="us-east-1",
            config=mock_config.return_value,
        )
        mock_config.assert_called_once_with(
            connect_timeout=10,
            read_timeout=30,
            s3={"addressing_style": "virtual"},
        )
        mock_client.put_object.assert_called_once_with(
            Bucket="mybucket",
            Key="results.json",
            Body=b'{"documents": []}',
            ContentType="application/json",
        )
        assert result.type == "s3"
        assert result.bucket == "mybucket"
        assert result.key == "results.json"


def test_write_s3_addressing_style_path() -> None:
    """addressing_style='path' threads into boto3 Config s3 dict."""
    mock_client = MagicMock()
    with (
        patch("boto3.client", return_value=mock_client),
        patch("obsidian_rag.mcp_server.output_file.Config") as mock_config,
    ):
        _write_s3(
            '{"documents": []}',
            "http://garage:3900",
            "mybucket",
            "results.json",
            "AKIAIOSFODNN7EXAMPLE",
            "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
            addressing_style="path",
        )
        mock_config.assert_called_once_with(
            connect_timeout=10,
            read_timeout=30,
            s3={"addressing_style": "path"},
        )


def test_write_s3_timeouts() -> None:
    """boto3 Config uses connect_timeout=10 and read_timeout=30."""
    mock_client = MagicMock()
    with (
        patch("boto3.client", return_value=mock_client),
        patch("obsidian_rag.mcp_server.output_file.Config") as mock_config,
    ):
        _write_s3(
            '{"documents": []}',
            "http://s3.example.com",
            "mybucket",
            "results.json",
            "AKIAIOSFODNN7EXAMPLE",
            "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
        )
        mock_config.assert_called_once_with(
            connect_timeout=10,
            read_timeout=30,
            s3={"addressing_style": "virtual"},
        )


def test_write_s3_network_error():
    """Mocked boto3 raises endpoint connection error; returns error dict."""
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


def test_write_s3_auth_failure():
    """Mocked boto3 raises ClientError 403; returns error dict."""
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


def test_write_s3_bucket_not_found():
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


def test_write_s3_no_credential_logging():
    """Verifies access_key_id/secret_access_key never in log output."""

    class CaptureHandler(logging.Handler):
        def __init__(self):
            super().__init__()
            self.messages = []

        def emit(self, record):
            self.messages.append(record.getMessage())

    handler = CaptureHandler()
    handler.setLevel(logging.DEBUG)
    logger = logging.getLogger("obsidian_rag.mcp_server.output_file")
    logger.addHandler(handler)
    original_level = logger.level
    logger.setLevel(logging.DEBUG)

    try:
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

        full_log = "\n".join(handler.messages)
        assert "SECRET_KEY_ID" not in full_log
        assert "SECRET_ACCESS_KEY" not in full_log
    finally:
        logger.removeHandler(handler)
        logger.setLevel(original_level)


def test_count_items_documents():
    """Returns len(documents) array."""
    data = json.dumps({"documents": [{"a": 1}, {"b": 2}]})
    assert _count_items(data) == 2


def test_count_items_tasks():
    """Returns len(tasks) array."""
    data = json.dumps({"tasks": [{"a": 1}, {"b": 2}, {"c": 3}]})
    assert _count_items(data) == 3


def test_count_items_tags():
    """Returns len(tags) array."""
    data = json.dumps({"tags": ["a", "b", "c", "d"]})
    assert _count_items(data) == 4


def test_count_items_single_document():
    """Returns 1 for single-doc response."""
    data = json.dumps({"success": True, "result": {"id": "123"}})
    assert _count_items(data) == 1


def test_build_output_file_summary_local():
    """Summary has output_file key with local result."""
    result = OutputFileResult(
        type="local",
        path="/tmp/out.json",
        bytes=100,
        item_count=5,
    )
    summary = build_output_file_summary({}, result)
    assert "output_file" in summary
    assert summary["output_file"]["type"] == "local"
    assert summary["output_file"]["path"] == "/tmp/out.json"


def test_build_output_file_summary_s3():
    """Summary has output_file key with s3 result."""
    result = OutputFileResult(type="s3", bucket="b", key="k", bytes=50, item_count=2)
    summary = build_output_file_summary({}, result)
    assert "output_file" in summary
    assert summary["output_file"]["type"] == "s3"
    assert summary["output_file"]["bucket"] == "b"


def test_write_output_file_dispatcher_local(tmp_path):
    """Dispatches to _write_local."""
    config = OutputFileConfig(type="local", path=str(tmp_path / "out.json"))
    result = write_output_file({"documents": [{"id": "1"}]}, config)
    assert "output_file" in result
    assert result["output_file"]["type"] == "local"


def test_write_output_file_dispatcher_s3():
    """Dispatches to _write_s3."""
    mock_client = MagicMock()
    with patch("boto3.client", return_value=mock_client):
        config = OutputFileConfig(
            type="s3",
            endpoint="http://s3.example.com",
            bucket="mybucket",
            key="results.json",
            access_key_id="AKIAIOSFODNN7EXAMPLE",
            secret_access_key="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
        )
        result = write_output_file({"documents": [{"id": "1"}]}, config)
        assert "output_file" in result
        assert result["output_file"]["type"] == "s3"


def test_write_output_file_dispatcher_threads_addressing_style() -> None:
    """Dispatcher passes addressing_style='path' through to _write_s3."""
    mock_client = MagicMock()
    with (
        patch("boto3.client", return_value=mock_client),
        patch(
            "obsidian_rag.mcp_server.output_file._write_s3",
            return_value=OutputFileResult(
                type="s3", bucket="b", key="k", bytes=50, item_count=2
            ),
        ) as mock_write_s3,
    ):
        config = OutputFileConfig(
            type="s3",
            endpoint="http://garage:3900",
            bucket="mybucket",
            key="results.json",
            access_key_id="AKIAIOSFODNN7EXAMPLE",
            secret_access_key="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
            addressing_style="path",
        )
        result = write_output_file({"documents": [{"id": "1"}]}, config)
        assert "output_file" in result
        mock_write_s3.assert_called_once()
        assert mock_write_s3.call_args.kwargs["addressing_style"] == "path"


def test_write_output_file_dispatcher_addressing_style_none_defaults_virtual() -> None:
    """Dispatcher coerces addressing_style=None to 'virtual'."""
    mock_client = MagicMock()
    with (
        patch("boto3.client", return_value=mock_client),
        patch(
            "obsidian_rag.mcp_server.output_file._write_s3",
            return_value=OutputFileResult(
                type="s3", bucket="b", key="k", bytes=50, item_count=2
            ),
        ) as mock_write_s3,
    ):
        config = OutputFileConfig(
            type="s3",
            endpoint="http://s3.example.com",
            bucket="mybucket",
            key="results.json",
            access_key_id="AKIAIOSFODNN7EXAMPLE",
            secret_access_key="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
            addressing_style=None,
        )
        result = write_output_file({"documents": [{"id": "1"}]}, config)
        assert "output_file" in result
        mock_write_s3.assert_called_once()
        assert mock_write_s3.call_args.kwargs["addressing_style"] == "virtual"


def test_write_output_file_dispatcher_validation_error_raises():
    """ValueError re-raised."""
    config = OutputFileConfig(type="local", path="/etc/passwd")
    with pytest.raises(ValueError, match="must be under /tmp/"):
        write_output_file({"documents": []}, config)


def test_write_output_file_dispatcher_disk_full_returns_error(tmp_path):
    """OSError caught, error dict returned."""
    config = OutputFileConfig(type="local", path=str(tmp_path / "out.json"))
    with patch(
        "obsidian_rag.mcp_server.output_file._write_local",
        side_effect=OSError("No space left on device"),
    ):
        result = write_output_file({"documents": []}, config)
        assert result["success"] is False
        assert "No space left on device" in result["error"]


def test_write_output_file_dispatcher_permission_denied_returns_error(tmp_path):
    """PermissionError caught."""
    config = OutputFileConfig(type="local", path=str(tmp_path / "out.json"))
    with patch(
        "obsidian_rag.mcp_server.output_file._write_local",
        side_effect=PermissionError("Permission denied"),
    ):
        result = write_output_file({"documents": []}, config)
        assert result["success"] is False
        assert "Permission denied" in result["error"]


def test_resolve_region_per_call_wins():
    """config.region set → returned regardless of other layers."""
    config = OutputFileConfig(
        type="s3",
        region="garage",
        endpoint="http://garage:3900",
        bucket="mybucket",
        key="results.json",
        access_key_id="AKIAIOSFODNN7EXAMPLE",
        secret_access_key="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
    )
    assert _resolve_s3_region(config, "eu-west-1") == "garage"


def test_resolve_region_app_default_when_per_call_none():
    """config.region=None, app_default set → app_default returned."""
    config = OutputFileConfig(
        type="s3",
        region=None,
        endpoint="http://s3.example.com",
        bucket="mybucket",
        key="results.json",
        access_key_id="AKIAIOSFODNN7EXAMPLE",
        secret_access_key="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
    )
    assert _resolve_s3_region(config, "eu-west-1") == "eu-west-1"


def test_resolve_region_url_derived_when_both_none():
    """AWS endpoint, no region → URL-derived."""
    config = OutputFileConfig(
        type="s3",
        region=None,
        endpoint="https://s3.eu-west-1.amazonaws.com",
        bucket="mybucket",
        key="results.json",
        access_key_id="AKIAIOSFODNN7EXAMPLE",
        secret_access_key="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
    )
    assert _resolve_s3_region(config, None) == "eu-west-1"


def test_resolve_region_fallback_with_warning(caplog):
    """Garage endpoint, no region → us-east-1 + WARNING called."""
    config = OutputFileConfig(
        type="s3",
        region=None,
        endpoint="http://garage:3900",
        bucket="mybucket",
        key="results.json",
        access_key_id="AKIAIOSFODNN7EXAMPLE",
        secret_access_key="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
    )
    with (
        caplog.at_level(logging.WARNING, logger="obsidian_rag.mcp_server.output_file"),
        patch(
            "obsidian_rag.mcp_server.output_file._derive_region_from_endpoint",
            return_value=None,
        ),
        patch(
            "obsidian_rag.mcp_server.output_file._probe_bucket_region",
            return_value=None,
        ),
    ):
        result = _resolve_s3_region(config, None)
    assert result == "us-east-1"
    assert any(
        "garage:3900" in r.message and "us-east-1" in r.message for r in caplog.records
    )


def test_resolve_region_explicit_overrides_app_default():
    """Both set → per-call wins."""
    config = OutputFileConfig(
        type="s3",
        region="garage",
        endpoint="http://garage:3900",
        bucket="mybucket",
        key="results.json",
        access_key_id="AKIAIOSFODNN7EXAMPLE",
        secret_access_key="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
    )
    assert _resolve_s3_region(config, "eu-west-1") == "garage"


def test_derive_region_aws_hostname_variants():
    """AWS hostname patterns derive the correct region."""
    assert (
        _derive_region_from_endpoint("https://s3.eu-west-1.amazonaws.com")
        == "eu-west-1"
    )
    assert (
        _derive_region_from_endpoint("https://s3-us-west-2.amazonaws.com")
        == "us-west-2"
    )
    assert (
        _derive_region_from_endpoint("https://s3-website.ap-southeast-2.amazonaws.com")
        == "ap-southeast-2"
    )


def test_derive_region_do_spaces_wasabi_b2():
    """Hosted S3 patterns derive region."""
    assert _derive_region_from_endpoint("https://nyc3.digitaloceanspaces.com") == "nyc3"
    assert (
        _derive_region_from_endpoint("https://s3.us-east-1.wasabisys.com")
        == "us-east-1"
    )
    assert (
        _derive_region_from_endpoint("https://s3.us-west-1.backblazeb2.com")
        == "us-west-1"
    )


def test_derive_region_r2_returns_auto():
    """Cloudflare R2 endpoints resolve to 'auto'."""
    assert (
        _derive_region_from_endpoint("https://abc123.r2.cloudflarestorage.com")
        == "auto"
    )


def test_derive_region_garage_minio_ip_returns_none():
    """Unrecognized endpoints return None."""
    assert _derive_region_from_endpoint("http://garage:3900") is None
    assert _derive_region_from_endpoint("http://192.168.1.5:9000") is None


def test_derive_region_bare_s3_returns_us_east_1():
    """Bare s3.amazonaws.com returns us-east-1 without warning."""
    assert _derive_region_from_endpoint("https://s3.amazonaws.com") == "us-east-1"


def test_warn_fallback_called_for_non_aws(caplog):
    """Fallback to us-east-1 for non-AWS endpoint emits WARNING."""
    with caplog.at_level(logging.WARNING, logger="obsidian_rag.mcp_server.output_file"):
        _warn_fallback_region("http://garage:3900")
    assert any(
        "garage:3900" in r.message and "us-east-1" in r.message for r in caplog.records
    )
    assert any("OutputFileConfig.region" in r.message for r in caplog.records)


def test_warn_fallback_not_called_for_aws(caplog):
    """AWS hostname → no warning (derived before fallback)."""
    config = OutputFileConfig(
        type="s3",
        region=None,
        endpoint="https://s3.eu-west-1.amazonaws.com",
        bucket="mybucket",
        key="results.json",
        access_key_id="AKIAIOSFODNN7EXAMPLE",
        secret_access_key="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
    )
    with caplog.at_level(logging.WARNING, logger="obsidian_rag.mcp_server.output_file"):
        result = _resolve_s3_region(config, None)
    assert result == "eu-west-1"
    assert not any("us-east-1" in r.message for r in caplog.records)


def test_warn_fallback_not_called_with_explicit_region(caplog):
    """Explicit region → no warning even for Garage."""
    config = OutputFileConfig(
        type="s3",
        region="garage",
        endpoint="http://garage:3900",
        bucket="mybucket",
        key="results.json",
        access_key_id="AKIAIOSFODNN7EXAMPLE",
        secret_access_key="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
    )
    with caplog.at_level(logging.WARNING, logger="obsidian_rag.mcp_server.output_file"):
        result = _resolve_s3_region(config, None)
    assert result == "garage"
    assert not any("us-east-1" in r.message for r in caplog.records)


def test_derive_region_empty_endpoint_returns_none():
    """Empty endpoint string returns None."""
    assert _derive_region_from_endpoint("") is None


def test_resolve_region_probe_bucket_region_success():
    """Probe bucket region returns valid region when derivation fails."""
    config = OutputFileConfig(
        type="s3",
        region=None,
        endpoint="http://garage:3900",
        bucket="mybucket",
        key="results.json",
        access_key_id="AKIAIOSFODNN7EXAMPLE",
        secret_access_key="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
    )
    with (
        patch(
            "obsidian_rag.mcp_server.output_file._derive_region_from_endpoint",
            return_value=None,
        ),
        patch(
            "obsidian_rag.mcp_server.output_file._probe_bucket_region",
            return_value="eu-west-1",
        ),
    ):
        result = _resolve_s3_region(config, None)
    assert result == "eu-west-1"


def test_probe_bucket_region_success():
    """Probing returns region when get_bucket_location succeeds."""
    mock_client = MagicMock()
    mock_client.get_bucket_location.return_value = {"LocationConstraint": "ap-south-1"}
    with patch("boto3.client", return_value=mock_client):
        result = _probe_bucket_region(
            "http://s3.example.com",
            "AKIAIOSFODNN7EXAMPLE",
            "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
            "virtual",
            "mybucket",
        )
    assert result == "ap-south-1"


def test_probe_bucket_region_client_error():
    """Probing returns None on ClientError."""
    mock_client = MagicMock()
    mock_client.get_bucket_location.side_effect = ClientError(
        {"Error": {"Code": "403", "Message": "Forbidden"}},
        "GetBucketLocation",
    )
    with patch("boto3.client", return_value=mock_client):
        result = _probe_bucket_region(
            "http://s3.example.com",
            "AKIAIOSFODNN7EXAMPLE",
            "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
            "virtual",
            "mybucket",
        )
    assert result is None


def test_probe_bucket_region_empty_location():
    """Probing returns None when LocationConstraint is empty/None."""
    mock_client = MagicMock()
    mock_client.get_bucket_location.return_value = {"LocationConstraint": None}
    with patch("boto3.client", return_value=mock_client):
        result = _probe_bucket_region(
            "http://s3.example.com",
            "AKIAIOSFODNN7EXAMPLE",
            "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
            "virtual",
            "mybucket",
        )
    assert result is None


def test_probe_bucket_region_os_error():
    """Probing returns None on OSError."""
    with patch(
        "boto3.client",
        side_effect=OSError("network unreachable"),
    ):
        result = _probe_bucket_region(
            "http://s3.example.com",
            "AKIAIOSFODNN7EXAMPLE",
            "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
            "virtual",
            "mybucket",
        )
    assert result is None
