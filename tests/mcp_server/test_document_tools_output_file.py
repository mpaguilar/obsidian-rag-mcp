"""Tests for document_tools output_file parameter support."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from obsidian_rag.mcp_server.document_tools import (
    _parse_output_file_from_wrapper,
    get_document,
    list_documents,
)
from obsidian_rag.mcp_server.models import OutputFileConfig


# ---------------------------------------------------------------------------
# get_document with output_file
# ---------------------------------------------------------------------------


def test_get_document_with_output_file_local(tmp_path: pytest.TempPathFactory) -> None:
    """output_file config writes to local path, returns summary."""
    target_path = str(tmp_path / "out.json")
    mock_registry = MagicMock()
    mock_tool = MagicMock(return_value={"id": "doc-1", "vault_name": "Personal"})

    with (
        patch(
            "obsidian_rag.mcp_server.document_tools._get_registry",
            return_value=mock_registry,
        ),
        patch(
            "obsidian_rag.mcp_server.document_tools.get_document_tool",
            mock_tool,
        ),
    ):
        result = get_document(
            vault_name="Personal",
            file_path="notes.md",
            output_file={"type": "local", "path": target_path},
        )

    assert "output_file" in result
    assert result["output_file"]["type"] == "local"
    assert result["output_file"]["path"] == target_path
    assert Path(target_path).exists()


def test_get_document_with_output_file_s3() -> None:
    """output_file config writes to S3, returns summary."""
    mock_registry = MagicMock()
    mock_tool = MagicMock(return_value={"id": "doc-1", "vault_name": "Personal"})
    mock_client = MagicMock()

    with (
        patch(
            "obsidian_rag.mcp_server.document_tools._get_registry",
            return_value=mock_registry,
        ),
        patch(
            "obsidian_rag.mcp_server.document_tools.get_document_tool",
            mock_tool,
        ),
        patch("boto3.client", return_value=mock_client),
    ):
        result = get_document(
            vault_name="Personal",
            file_path="notes.md",
            output_file={
                "type": "s3",
                "endpoint": "http://s3.example.com",
                "bucket": "mybucket",
                "key": "results.json",
                "access_key_id": "AKIAIOSFODNN7EXAMPLE",
                "secret_access_key": "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
            },
        )

    assert "output_file" in result
    assert result["output_file"]["type"] == "s3"
    assert result["output_file"]["bucket"] == "mybucket"
    assert result["output_file"]["key"] == "results.json"


def test_get_document_without_output_file() -> None:
    """Unchanged behavior, full result returned."""
    mock_registry = MagicMock()
    mock_tool = MagicMock(return_value={"id": "doc-1", "vault_name": "Personal"})

    with (
        patch(
            "obsidian_rag.mcp_server.document_tools._get_registry",
            return_value=mock_registry,
        ),
        patch(
            "obsidian_rag.mcp_server.document_tools.get_document_tool",
            mock_tool,
        ),
    ):
        result = get_document(vault_name="Personal", file_path="notes.md")

    assert result == {"id": "doc-1", "vault_name": "Personal"}


# ---------------------------------------------------------------------------
# get_document output_file parsing
# ---------------------------------------------------------------------------


def test_get_document_output_file_parse_dict() -> None:
    """Dict output_file parsed correctly."""
    mock_registry = MagicMock()
    mock_tool = MagicMock(return_value={"id": "doc-1"})

    with (
        patch(
            "obsidian_rag.mcp_server.document_tools._get_registry",
            return_value=mock_registry,
        ),
        patch(
            "obsidian_rag.mcp_server.document_tools.get_document_tool",
            mock_tool,
        ),
    ):
        result = get_document(
            document_id="abc-123",
            output_file={"type": "local", "path": "/tmp/out.json"},
        )

    assert "output_file" in result


def test_get_document_output_file_parse_json_string(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """JSON string parsed correctly."""
    target_path = str(tmp_path / "out.json")
    mock_registry = MagicMock()
    mock_tool = MagicMock(return_value={"id": "doc-1"})
    output_file_json = json.dumps({"type": "local", "path": target_path})

    with (
        patch(
            "obsidian_rag.mcp_server.document_tools._get_registry",
            return_value=mock_registry,
        ),
        patch(
            "obsidian_rag.mcp_server.document_tools.get_document_tool",
            mock_tool,
        ),
    ):
        result = get_document(
            document_id="abc-123",
            output_file=output_file_json,
        )

    assert "output_file" in result
    assert result["output_file"]["type"] == "local"


# ---------------------------------------------------------------------------
# list_documents with output_file
# ---------------------------------------------------------------------------


def test_list_documents_with_output_file_local(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """output_file config writes to local path, returns summary."""
    target_path = str(tmp_path / "out.json")
    mock_registry = MagicMock()
    mock_tool = MagicMock(
        return_value={"documents": [{"id": "doc-1"}], "total_count": 1},
    )

    with (
        patch(
            "obsidian_rag.mcp_server.document_tools._get_registry",
            return_value=mock_registry,
        ),
        patch(
            "obsidian_rag.mcp_server.document_tools.list_documents_tool",
            mock_tool,
        ),
    ):
        result = list_documents(
            file_name="notes.md",
            output_file={"type": "local", "path": target_path},
        )

    assert "output_file" in result
    assert result["output_file"]["type"] == "local"
    assert result["output_file"]["path"] == target_path
    assert Path(target_path).exists()


def test_list_documents_with_output_file_s3() -> None:
    """output_file config writes to S3, returns summary."""
    mock_registry = MagicMock()
    mock_tool = MagicMock(
        return_value={"documents": [{"id": "doc-1"}], "total_count": 1},
    )
    mock_client = MagicMock()

    with (
        patch(
            "obsidian_rag.mcp_server.document_tools._get_registry",
            return_value=mock_registry,
        ),
        patch(
            "obsidian_rag.mcp_server.document_tools.list_documents_tool",
            mock_tool,
        ),
        patch("boto3.client", return_value=mock_client),
    ):
        result = list_documents(
            file_name="notes.md",
            output_file={
                "type": "s3",
                "endpoint": "http://s3.example.com",
                "bucket": "mybucket",
                "key": "results.json",
                "access_key_id": "AKIAIOSFODNN7EXAMPLE",
                "secret_access_key": "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
            },
        )

    assert "output_file" in result
    assert result["output_file"]["type"] == "s3"
    assert result["output_file"]["bucket"] == "mybucket"


def test_list_documents_without_output_file() -> None:
    """Unchanged behavior, full result returned."""
    mock_registry = MagicMock()
    mock_tool = MagicMock(
        return_value={"documents": [{"id": "doc-1"}], "total_count": 1},
    )

    with (
        patch(
            "obsidian_rag.mcp_server.document_tools._get_registry",
            return_value=mock_registry,
        ),
        patch(
            "obsidian_rag.mcp_server.document_tools.list_documents_tool",
            mock_tool,
        ),
    ):
        result = list_documents(file_name="notes.md")

    assert result == {"documents": [{"id": "doc-1"}], "total_count": 1}


# ---------------------------------------------------------------------------
# _parse_output_file_from_wrapper unit tests
# ---------------------------------------------------------------------------


def test_parse_output_file_from_wrapper_none() -> None:
    """None returns None."""
    result = _parse_output_file_from_wrapper(None)
    assert result is None


def test_parse_output_file_from_wrapper_model() -> None:
    """OutputFileConfig returned unchanged."""
    config = OutputFileConfig(type="local", path="/tmp/out.json")
    result = _parse_output_file_from_wrapper(config)
    assert result is config


def test_parse_output_file_from_wrapper_dict() -> None:
    """Dict parsed to OutputFileConfig."""
    result = _parse_output_file_from_wrapper({"type": "local", "path": "/tmp/out.json"})
    assert isinstance(result, OutputFileConfig)
    assert result.type == "local"
    assert result.path == "/tmp/out.json"


def test_parse_output_file_from_wrapper_json_string() -> None:
    """JSON string parsed to OutputFileConfig."""
    json_str = '{"type": "local", "path": "/tmp/out.json"}'
    result = _parse_output_file_from_wrapper(json_str)
    assert isinstance(result, OutputFileConfig)
    assert result.type == "local"
    assert result.path == "/tmp/out.json"


def test_parse_output_file_from_wrapper_non_dict_parsed() -> None:
    """JSON string that parses to a list returns None."""
    result = _parse_output_file_from_wrapper("[1, 2, 3]")
    assert result is None


def test_parse_output_file_from_wrapper_invalid_json_raises() -> None:
    """Invalid JSON string raises JSONDecodeError."""
    with pytest.raises(json.JSONDecodeError):
        _parse_output_file_from_wrapper("not-json")
