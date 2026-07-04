import json
import logging
import os
import tempfile
from pathlib import Path

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

from obsidian_rag.mcp_server.models import OutputFileConfig, OutputFileResult

log = logging.getLogger(__name__)


def write_output_file(
    result: dict[str, object],
    config: OutputFileConfig,
) -> dict[str, object]:
    """Dispatch result writing based on config.type.

    Args:
        result: Full tool result dict (JSON-serializable).
        config: OutputFileConfig specifying target type and parameters.

    Returns:
        Compact summary dict {output_file: OutputFileResult.model_dump()}
        on success, or error dict {"success": False, "error": "..."} on failure.
    """
    _msg = "write_output_file starting"
    log.debug(_msg)
    try:
        if config.type == "local":
            _validate_local_path(config.path or "")
            result_json = json.dumps(result)
            output_result = _write_local(result_json, config.path or "")
        else:
            _validate_s3_config(config)
            result_json = json.dumps(result)
            output_result = _write_s3(
                result_json,
                config.endpoint or "",
                config.bucket or "",
                config.key or "",
                config.access_key_id or "",
                config.secret_access_key or "",
                addressing_style=config.addressing_style or "virtual",
            )
        summary = build_output_file_summary(result, output_result)
        _msg = "write_output_file returning"
        log.debug(_msg)
        return summary
    except ValueError:
        _msg = "write_output_file raising ValueError"
        log.debug(_msg)
        raise
    except (OSError, ClientError) as err:
        _msg = f"{config.type} write failed: {err}"
        log.error(_msg)
        return {"success": False, "error": _msg}


def _write_local(result_json: str, path: str) -> OutputFileResult:
    """Write result_json to local filesystem path with atomic write.

    Args:
        result_json: Serialized JSON string to write.
        path: Target filesystem path (must be under /tmp/).

    Returns:
        OutputFileResult with type="local", path, bytes, item_count.

    Raises:
        ValueError: If path is not under /tmp/.
    """
    _msg = "_write_local starting"
    log.debug(_msg)
    target_path = Path(path)
    parent_dir = target_path.parent
    parent_dir.mkdir(parents=True, exist_ok=True)

    fd, temp_path_str = tempfile.mkstemp(dir=str(parent_dir))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as temp_f:
            temp_f.write(result_json)
        os.replace(temp_path_str, path)
    except Exception:
        try:
            if os.path.exists(temp_path_str):
                os.unlink(temp_path_str)
        except OSError:
            pass
        raise

    item_count = _count_items(result_json)
    result = OutputFileResult(
        type="local",
        path=path,
        bytes=len(result_json.encode("utf-8")),
        item_count=item_count,
    )
    _msg = "_write_local returning"
    log.debug(_msg)
    return result


def _write_s3(
    result_json: str,
    endpoint: str,
    bucket: str,
    key: str,
    access_key_id: str,
    secret_access_key: str,
    addressing_style: str = "virtual",
) -> OutputFileResult:
    """Write result_json to S3-compatible endpoint using boto3.

    Args:
        result_json: Serialized JSON string to write.
        endpoint: S3 endpoint URL.
        bucket: S3 bucket name.
        key: S3 object key.
        access_key_id: S3 access key.
        secret_access_key: S3 secret key.
        addressing_style: S3 addressing style — "virtual" (AWS default,
            bucket in hostname) or "path" (Garage/MinIO, bucket in URL path).
            Defaults to "virtual".

    Returns:
        OutputFileResult with type="s3", bucket, key, bytes, item_count.
    """
    _msg = "_write_s3 starting"
    log.debug(_msg)
    s3_config = Config(
        connect_timeout=10,
        read_timeout=30,
        s3={"addressing_style": addressing_style},
    )
    client = boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=access_key_id,
        aws_secret_access_key=secret_access_key,
        config=s3_config,
    )
    body_bytes = result_json.encode("utf-8")
    client.put_object(
        Bucket=bucket,
        Key=key,
        Body=body_bytes,
        ContentType="application/json",
    )
    item_count = _count_items(result_json)
    result = OutputFileResult(
        type="s3",
        bucket=bucket,
        key=key,
        bytes=len(body_bytes),
        item_count=item_count,
    )
    _msg = "_write_s3 returning"
    log.debug(_msg)
    return result


def _validate_local_path(path: str) -> str:
    """Validate that path is under /tmp/ root.

    Args:
        path: Filesystem path to validate.

    Returns:
        Resolved absolute path if valid.

    Raises:
        ValueError: If path is not under /tmp/.
    """
    _msg = "_validate_local_path starting"
    log.debug(_msg)
    resolved = str(Path(path).resolve())
    if not resolved.startswith("/tmp/"):
        _msg = f"Local path must be under /tmp/: {resolved}"
        log.error(_msg)
        raise ValueError(_msg)
    _msg = "_validate_local_path returning"
    log.debug(_msg)
    return resolved


def _validate_s3_config(config: OutputFileConfig) -> None:
    """Validate that all required S3 fields are present.

    Args:
        config: OutputFileConfig with type="s3".

    Raises:
        ValueError: If any required S3 field is missing. Lists missing fields.
    """
    _msg = "_validate_s3_config starting"
    log.debug(_msg)
    required = ["endpoint", "bucket", "key", "access_key_id", "secret_access_key"]
    missing = [field for field in required if getattr(config, field) is None]
    if missing:
        _msg = f"S3 config missing required fields: {missing}"
        log.error(_msg)
        raise ValueError(_msg)
    _msg = "_validate_s3_config returning"
    log.debug(_msg)


def _count_items(result_json: str) -> int:
    """Count items in the result JSON.

    Args:
        result_json: Serialized JSON string.

    Returns:
        Number of items (len of documents/tasks/tags arrays, or 1).
    """
    if not result_json:
        return 0
    parsed = json.loads(result_json)
    if "documents" in parsed:
        return len(parsed["documents"])  # type: ignore[arg-type]
    if "tasks" in parsed:
        return len(parsed["tasks"])  # type: ignore[arg-type]
    if "tags" in parsed:
        return len(parsed["tags"])  # type: ignore[arg-type]
    return 1


def build_output_file_summary(
    _result: dict[str, object],
    output_file_result: OutputFileResult,
) -> dict[str, object]:
    """Build the compact summary response that replaces the full result.

    Args:
        _result: Original full result dict (unused; item_count is computed
            during the write operation).
        output_file_result: OutputFileResult from the write operation.

    Returns:
        Dict with {"output_file": OutputFileResult.model_dump()} replacing
        the full result.
    """
    _msg = "build_output_file_summary starting"
    log.debug(_msg)
    summary: dict[str, object] = {"output_file": output_file_result.model_dump()}
    _msg = "build_output_file_summary returning"
    log.debug(_msg)
    return summary
