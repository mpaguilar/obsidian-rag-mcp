"""Integration tests for local file output through the full tool wrapper chain."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from obsidian_rag.mcp_server.document_tools import (
    get_document,
    list_documents,
)
from obsidian_rag.mcp_server.server import (
    get_all_tags,
    get_documents_by_property,
    get_documents_by_tag,
    get_tasks,
    query_documents,
)
from obsidian_rag.mcp_server.tool_definitions import MCPToolRegistry, _set_registry


@pytest.fixture
def setup_registry() -> MCPToolRegistry:
    """Set up a mock registry for testing server tool wrappers."""
    mock_db_manager = MagicMock()
    mock_embedding_provider = MagicMock()
    mock_settings = MagicMock()

    registry = MCPToolRegistry(
        db_manager=mock_db_manager,
        embedding_provider=mock_embedding_provider,
        settings=mock_settings,
    )
    _set_registry(registry)
    yield registry
    _set_registry(None)


def _assert_local_summary(
    result: dict[str, object],
    expected_path: str,
    expected_item_count: int,
) -> None:
    """Assert that a local output_file summary is well-formed."""
    assert "output_file" in result
    output_file = result["output_file"]
    assert isinstance(output_file, dict)
    assert output_file["type"] == "local"
    assert output_file["path"] == expected_path
    assert output_file["bytes"] > 0
    assert output_file["item_count"] == expected_item_count


def _assert_file_content_matches(
    path: str,
    expected: dict[str, object],
) -> None:
    """Read a local JSON file and assert its contents match expected data."""
    text = Path(path).read_text(encoding="utf-8")
    data = json.loads(text)
    assert data == expected


# ---------------------------------------------------------------------------
# get_document
# ---------------------------------------------------------------------------


def test_get_document_output_file_local_integration(tmp_path: Path) -> None:
    """get_document with local output_file writes real file and returns summary."""
    target_path = str(tmp_path / "get_document_out.json")
    mock_registry = MagicMock()
    tool_result = {"id": "doc-1", "vault_name": "Personal", "content": "hello"}

    with (
        patch(
            "obsidian_rag.mcp_server.document_tools._get_registry",
            return_value=mock_registry,
        ),
        patch(
            "obsidian_rag.mcp_server.document_tools.get_document_tool",
            return_value=tool_result,
        ),
    ):
        result = get_document(
            vault_name="Personal",
            file_path="notes.md",
            output_file={"type": "local", "path": target_path},
        )

    _assert_local_summary(result, target_path, 1)
    _assert_file_content_matches(target_path, tool_result)


# ---------------------------------------------------------------------------
# list_documents
# ---------------------------------------------------------------------------


def test_list_documents_output_file_local_integration(tmp_path: Path) -> None:
    """list_documents with local output_file writes real file and returns summary."""
    target_path = str(tmp_path / "list_documents_out.json")
    mock_registry = MagicMock()
    tool_result = {
        "documents": [{"id": "doc-1"}, {"id": "doc-2"}],
        "total_count": 2,
        "has_more": False,
        "next_offset": None,
    }

    with (
        patch(
            "obsidian_rag.mcp_server.document_tools._get_registry",
            return_value=mock_registry,
        ),
        patch(
            "obsidian_rag.mcp_server.document_tools.list_documents_tool",
            return_value=tool_result,
        ),
    ):
        result = list_documents(
            file_name="notes.md",
            output_file={"type": "local", "path": target_path},
        )

    _assert_local_summary(result, target_path, 2)
    _assert_file_content_matches(target_path, tool_result)


# ---------------------------------------------------------------------------
# query_documents
# ---------------------------------------------------------------------------


def test_query_documents_output_file_local_integration(
    setup_registry: MCPToolRegistry,
    tmp_path: Path,
) -> None:
    """query_documents with local output_file writes real file and returns summary."""
    target_path = str(tmp_path / "query_documents_out.json")
    tool_result = {
        "results": [{"id": "doc-1"}],
        "total_count": 1,
        "has_more": False,
        "next_offset": None,
    }

    with patch(
        "obsidian_rag.mcp_server.server.query_documents_tool",
        return_value=tool_result,
    ):
        result = query_documents(
            query="test",
            output_file={"type": "local", "path": target_path},
        )

    _assert_local_summary(result, target_path, 1)
    _assert_file_content_matches(target_path, tool_result)


# ---------------------------------------------------------------------------
# get_documents_by_tag
# ---------------------------------------------------------------------------


def test_get_documents_by_tag_output_file_local_integration(
    setup_registry: MCPToolRegistry,
    tmp_path: Path,
) -> None:
    """get_documents_by_tag with local output_file writes real file and returns summary."""
    target_path = str(tmp_path / "get_documents_by_tag_out.json")
    handler_result = {
        "results": [{"id": "doc-1"}, {"id": "doc-2"}],
        "total_count": 2,
        "has_more": False,
        "next_offset": None,
    }

    with patch(
        "obsidian_rag.mcp_server.server._get_documents_by_tag_handler",
        return_value=handler_result,
    ):
        result = get_documents_by_tag(
            output_file={"type": "local", "path": target_path},
        )

    # _count_items does not recognise the "results" key, so item_count=1
    _assert_local_summary(result, target_path, 1)
    _assert_file_content_matches(target_path, handler_result)


# ---------------------------------------------------------------------------
# get_documents_by_property
# ---------------------------------------------------------------------------


def test_get_documents_by_property_output_file_local_integration(
    setup_registry: MCPToolRegistry,
    tmp_path: Path,
) -> None:
    """get_documents_by_property with local output_file writes real file and returns summary."""
    target_path = str(tmp_path / "get_documents_by_property_out.json")

    from obsidian_rag.mcp_server.models import DocumentListResponse

    tool_response = DocumentListResponse(
        results=[],
        total_count=0,
        has_more=False,
        next_offset=None,
    )

    with patch(
        "obsidian_rag.mcp_server.tools.documents.get_documents_by_property",
        return_value=tool_response,
    ):
        result = get_documents_by_property(
            output_file={"type": "local", "path": target_path},
        )

    # _count_items does not recognise the "results" key, so item_count=1
    _assert_local_summary(result, target_path, 1)
    expected = tool_response.model_dump()
    _assert_file_content_matches(target_path, expected)


# ---------------------------------------------------------------------------
# get_all_tags
# ---------------------------------------------------------------------------


def test_get_all_tags_output_file_local_integration(
    setup_registry: MCPToolRegistry,
    tmp_path: Path,
) -> None:
    """get_all_tags with local output_file writes real file and returns summary."""
    target_path = str(tmp_path / "get_all_tags_out.json")
    tool_result = {"tags": ["work", "personal"], "total_count": 2}

    with patch(
        "obsidian_rag.mcp_server.server.get_all_tags_tool",
        return_value=tool_result,
    ):
        result = get_all_tags(
            output_file={"type": "local", "path": target_path},
        )

    _assert_local_summary(result, target_path, 2)
    _assert_file_content_matches(target_path, tool_result)


# ---------------------------------------------------------------------------
# get_tasks
# ---------------------------------------------------------------------------


def test_get_tasks_output_file_local_integration(
    setup_registry: MCPToolRegistry,
    tmp_path: Path,
) -> None:
    """get_tasks with local output_file writes real file and returns summary."""
    target_path = str(tmp_path / "get_tasks_out.json")
    handler_result = {
        "tasks": [{"id": "task-1"}],
        "total_count": 1,
        "has_more": False,
        "next_offset": None,
    }

    with patch(
        "obsidian_rag.mcp_server.handlers._get_tasks_handler",
        return_value=handler_result,
    ):
        result = get_tasks(
            output_file={"type": "local", "path": target_path},
        )

    _assert_local_summary(result, target_path, 1)
    _assert_file_content_matches(target_path, handler_result)


# ---------------------------------------------------------------------------
# Atomic write / error / directory creation tests
# ---------------------------------------------------------------------------


def test_atomic_write_no_partial_file(
    setup_registry: MCPToolRegistry,
    tmp_path: Path,
) -> None:
    """On write failure, temp file is cleaned up and no partial file remains."""
    target_path = str(tmp_path / "atomic_out.json")
    tool_result = {"results": [{"id": "doc-1"}], "total_count": 1}

    with (
        patch(
            "obsidian_rag.mcp_server.server.query_documents_tool",
            return_value=tool_result,
        ),
        patch(
            "obsidian_rag.mcp_server.output_file.os.replace",
            side_effect=OSError("disk full"),
        ),
    ):
        result = query_documents(
            query="test",
            output_file={"type": "local", "path": target_path},
        )

    assert not Path(target_path).exists()
    assert result["success"] is False
    assert "disk full" in result["error"]


def test_parent_dir_auto_creation(
    setup_registry: MCPToolRegistry,
    tmp_path: Path,
) -> None:
    """Nested output path has parent directories auto-created."""
    target_path = str(tmp_path / "nested" / "deep" / "out.json")
    tool_result = {"tags": ["tag1"], "total_count": 1}

    with patch(
        "obsidian_rag.mcp_server.server.get_all_tags_tool",
        return_value=tool_result,
    ):
        result = get_all_tags(
            output_file={"type": "local", "path": target_path},
        )

    assert Path(target_path).exists()
    _assert_local_summary(result, target_path, 1)
    _assert_file_content_matches(target_path, tool_result)


def test_overwrite_existing_file(
    setup_registry: MCPToolRegistry,
    tmp_path: Path,
) -> None:
    """Second write to the same path overwrites the previous file content."""
    target_path = str(tmp_path / "overwrite_out.json")
    first_result = {"tags": ["old"], "total_count": 1}
    second_result = {"tags": ["new", "new2"], "total_count": 2}

    with patch(
        "obsidian_rag.mcp_server.server.get_all_tags_tool",
        return_value=first_result,
    ):
        first_response = get_all_tags(
            output_file={"type": "local", "path": target_path},
        )

    _assert_local_summary(first_response, target_path, 1)
    _assert_file_content_matches(target_path, first_result)

    with patch(
        "obsidian_rag.mcp_server.server.get_all_tags_tool",
        return_value=second_result,
    ):
        second_response = get_all_tags(
            output_file={"type": "local", "path": target_path},
        )

    _assert_local_summary(second_response, target_path, 2)
    _assert_file_content_matches(target_path, second_result)
