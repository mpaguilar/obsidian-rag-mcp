"""Test UUID, datetime, and date serialization in write_output_file.

Verifies that write_output_file() succeeds when the result dict contains
uuid.UUID, datetime, and date objects, and that the written JSON contains
string representations (not Python repr). Covers both local and S3 paths,
and the defense-in-depth default=str fallback.
"""

import json
import uuid
from datetime import date, datetime, UTC
from pathlib import Path
from unittest.mock import patch

from obsidian_rag.mcp_server.models import OutputFileResult
from obsidian_rag.mcp_server.output_file import (
    OutputFileConfig,
    write_output_file,
)


_S3_CONFIG: dict[str, str] = {
    "type": "s3",
    "endpoint": "http://s3.us-east-1.amazonaws.com",
    "bucket": "mybucket",
    "key": "results.json",
    "access_key_id": "AKIAIOSFODNN7EXAMPLE",
    "secret_access_key": "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
}


def _build_result_with_uuid_datetime_date() -> dict[str, object]:
    """Build a result dict containing uuid, datetime, and date objects."""
    return {
        "id": uuid.uuid4(),
        "created_at": datetime.now(UTC),
        "due_date": date.today(),
        "name": "test document",
    }


def test_write_output_file_local_succeeds_with_uuid_datetime_date(
    tmp_path: Path,
) -> None:
    """Local write succeeds with uuid/datetime/date; file contains string representations."""
    target_path = str(tmp_path / "uuid_out.json")
    original_id = uuid.uuid4()
    original_created_at = datetime.now(UTC)
    original_due_date = date.today()

    result: dict[str, object] = {
        "id": original_id,
        "created_at": original_created_at,
        "due_date": original_due_date,
        "name": "test document",
    }

    config = OutputFileConfig(type="local", path=target_path)
    summary = write_output_file(result, config)

    assert "output_file" in summary
    assert summary["output_file"]["type"] == "local"
    assert summary["output_file"]["path"] == target_path

    text = Path(target_path).read_text(encoding="utf-8")
    parsed = json.loads(text)

    assert isinstance(parsed["id"], str)
    assert parsed["id"] == str(original_id)

    assert isinstance(parsed["created_at"], str)
    # default=str uses str(datetime) which produces "YYYY-MM-DD HH:MM:SS..."
    assert parsed["created_at"] == str(original_created_at)

    assert isinstance(parsed["due_date"], str)
    # default=str uses str(date) which produces "YYYY-MM-DD"
    assert parsed["due_date"] == str(original_due_date)

    assert parsed["name"] == "test document"


def test_write_output_file_s3_succeeds_with_uuid_datetime_date() -> None:
    """S3 write succeeds with uuid/datetime/date; json.dumps uses default=str."""
    original_id = uuid.uuid4()
    original_created_at = datetime.now(UTC)
    original_due_date = date.today()

    result: dict[str, object] = {
        "id": original_id,
        "created_at": original_created_at,
        "due_date": original_due_date,
        "name": "test document",
    }

    captured_result_json: str = ""

    def _mock_write_s3(
        result_json: str, *_args: object, **_kwargs: object
    ) -> OutputFileResult:
        nonlocal captured_result_json
        captured_result_json = result_json
        return OutputFileResult(
            type="s3",
            bucket="mybucket",
            key="results.json",
            bytes=len(result_json.encode("utf-8")),
            item_count=1,
        )

    with patch(
        "obsidian_rag.mcp_server.output_file._write_s3",
        side_effect=_mock_write_s3,
    ):
        config = OutputFileConfig(**_S3_CONFIG)
        summary = write_output_file(result, config)

    assert "output_file" in summary
    assert summary["output_file"]["type"] == "s3"

    parsed = json.loads(captured_result_json)

    assert isinstance(parsed["id"], str)
    assert parsed["id"] == str(original_id)

    assert isinstance(parsed["created_at"], str)
    assert parsed["created_at"] == str(original_created_at)

    assert isinstance(parsed["due_date"], str)
    assert parsed["due_date"] == str(original_due_date)

    assert parsed["name"] == "test document"


def test_write_output_file_local_writes_strings_not_repr(tmp_path: Path) -> None:
    """Written JSON contains canonical strings, not Python repr substrings."""
    target_path = str(tmp_path / "repr_out.json")
    result: dict[str, object] = {
        "id": uuid.uuid4(),
        "created_at": datetime.now(UTC),
    }

    config = OutputFileConfig(type="local", path=target_path)
    write_output_file(result, config)

    text = Path(target_path).read_text(encoding="utf-8")

    assert "UUID('" not in text
    assert "datetime.datetime(" not in text

    parsed = json.loads(text)
    assert isinstance(parsed["id"], str)
    assert len(parsed["id"]) == 36  # canonical hex UUID string
    assert isinstance(parsed["created_at"], str)
    # str(datetime) contains date and time separated by a space, not 'T'
    assert " " in parsed["created_at"]


def test_write_output_file_default_str_catches_arbitrary_object(
    tmp_path: Path,
) -> None:
    """default=str serializes arbitrary object() via str(); no TypeError raised."""
    target_path = str(tmp_path / "arbitrary_out.json")
    arbitrary = object()
    result: dict[str, object] = {"x": arbitrary}

    config = OutputFileConfig(type="local", path=target_path)
    summary = write_output_file(result, config)

    assert "output_file" in summary

    text = Path(target_path).read_text(encoding="utf-8")
    parsed = json.loads(text)

    assert parsed["x"] == str(arbitrary)


def test_write_output_file_local_preserves_already_serializable_values(
    tmp_path: Path,
) -> None:
    """default=str is a no-op for plain serializable types; output unchanged."""
    target_path = str(tmp_path / "plain_out.json")
    result: dict[str, object] = {
        "s": "hello",
        "n": 42,
        "f": 3.14,
        "b": True,
        "none": None,
        "lst": [1, 2, 3],
        "dct": {"a": 1},
    }

    config = OutputFileConfig(type="local", path=target_path)
    write_output_file(result, config)

    text = Path(target_path).read_text(encoding="utf-8")
    parsed = json.loads(text)

    assert parsed["s"] == "hello"
    assert parsed["n"] == 42
    assert parsed["f"] == 3.14
    assert parsed["b"] is True
    assert parsed["none"] is None
    assert parsed["lst"] == [1, 2, 3]
    assert parsed["dct"] == {"a": 1}

    # Verify that default=str did not alter the output for serializable types
    without_default = json.dumps(result)
    assert text == without_default


def test_write_output_file_s3_default_str_arbitrary_object() -> None:
    """S3 branch also uses default=str for arbitrary objects."""
    arbitrary = object()
    result: dict[str, object] = {"x": arbitrary}

    captured_result_json: str = ""

    def _mock_write_s3(
        result_json: str, *_args: object, **_kwargs: object
    ) -> OutputFileResult:
        nonlocal captured_result_json
        captured_result_json = result_json
        return OutputFileResult(
            type="s3",
            bucket="mybucket",
            key="results.json",
            bytes=len(result_json.encode("utf-8")),
            item_count=1,
        )

    with patch(
        "obsidian_rag.mcp_server.output_file._write_s3",
        side_effect=_mock_write_s3,
    ):
        config = OutputFileConfig(**_S3_CONFIG)
        summary = write_output_file(result, config)

    assert "output_file" in summary
    parsed = json.loads(captured_result_json)
    assert parsed["x"] == str(arbitrary)


def test_write_output_file_local_empty_results_list(tmp_path: Path) -> None:
    """Empty result list serializes fine without UUIDs present."""
    target_path = str(tmp_path / "empty_out.json")
    result: dict[str, object] = {"results": [], "total_count": 0}

    config = OutputFileConfig(type="local", path=target_path)
    summary = write_output_file(result, config)

    assert "output_file" in summary

    text = Path(target_path).read_text(encoding="utf-8")
    parsed = json.loads(text)

    assert parsed["results"] == []
    assert parsed["total_count"] == 0
