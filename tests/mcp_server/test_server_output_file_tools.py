"""Tests for output_file parameter in MCP server tool wrappers."""

from unittest.mock import MagicMock, patch

import pytest

from obsidian_rag.mcp_server.models import DocumentListResponse
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
    """Set up a mock registry for testing tool wrappers."""
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


def test_query_documents_with_output_file_local(
    setup_registry: MCPToolRegistry,
) -> None:
    """query_documents with local output_file writes result and returns summary."""
    with patch("obsidian_rag.mcp_server.server.query_documents_tool") as mock_tool:
        mock_tool.return_value = {"results": [{"id": "1"}], "total_count": 1}

        with patch("obsidian_rag.mcp_server.server.write_output_file") as mock_write:
            mock_write.return_value = {
                "output_file": {
                    "type": "local",
                    "path": "/tmp/test.json",
                    "bytes": 100,
                    "item_count": 1,
                }
            }

            result = query_documents(
                query="test",
                output_file={"type": "local", "path": "/tmp/test.json"},
            )

            assert result == {
                "output_file": {
                    "type": "local",
                    "path": "/tmp/test.json",
                    "bytes": 100,
                    "item_count": 1,
                }
            }
            mock_write.assert_called_once()
            call_args = mock_write.call_args
            assert call_args[0][0] == {"results": [{"id": "1"}], "total_count": 1}
            assert call_args[0][1].type == "local"
            assert call_args[0][1].path == "/tmp/test.json"


def test_query_documents_with_output_file_s3(setup_registry: MCPToolRegistry) -> None:
    """query_documents with S3 output_file writes result and returns summary."""
    with patch("obsidian_rag.mcp_server.server.query_documents_tool") as mock_tool:
        mock_tool.return_value = {"results": [{"id": "1"}], "total_count": 1}

        with patch("obsidian_rag.mcp_server.server.write_output_file") as mock_write:
            mock_write.return_value = {
                "output_file": {
                    "type": "s3",
                    "bucket": "mybucket",
                    "key": "results.json",
                    "bytes": 100,
                    "item_count": 1,
                }
            }

            result = query_documents(
                query="test",
                output_file={
                    "type": "s3",
                    "endpoint": "http://s3.example.com",
                    "bucket": "mybucket",
                    "key": "results.json",
                    "access_key_id": "AKIAIOSFODNN7EXAMPLE",
                    "secret_access_key": "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
                },
            )

            assert result == {
                "output_file": {
                    "type": "s3",
                    "bucket": "mybucket",
                    "key": "results.json",
                    "bytes": 100,
                    "item_count": 1,
                }
            }
            mock_write.assert_called_once()


def test_query_documents_without_output_file(setup_registry: MCPToolRegistry) -> None:
    """query_documents without output_file returns raw result unchanged."""
    with patch("obsidian_rag.mcp_server.server.query_documents_tool") as mock_tool:
        mock_tool.return_value = {"results": [], "total_count": 0}

        with patch("obsidian_rag.mcp_server.server.write_output_file") as mock_write:
            result = query_documents(query="test")

            assert result == {"results": [], "total_count": 0}
            mock_write.assert_not_called()


def test_get_documents_by_tag_with_output_file_local(
    setup_registry: MCPToolRegistry,
) -> None:
    """get_documents_by_tag with local output_file writes result and returns summary."""
    with patch(
        "obsidian_rag.mcp_server.server._get_documents_by_tag_handler"
    ) as mock_handler:
        mock_handler.return_value = {
            "results": [{"id": "1"}],
            "total_count": 1,
            "has_more": False,
            "next_offset": None,
        }

        with patch("obsidian_rag.mcp_server.server.write_output_file") as mock_write:
            mock_write.return_value = {
                "output_file": {
                    "type": "local",
                    "path": "/tmp/test.json",
                    "bytes": 100,
                    "item_count": 1,
                }
            }

            result = get_documents_by_tag(
                output_file={"type": "local", "path": "/tmp/test.json"},
            )

            assert result == {
                "output_file": {
                    "type": "local",
                    "path": "/tmp/test.json",
                    "bytes": 100,
                    "item_count": 1,
                }
            }
            mock_write.assert_called_once()


def test_get_documents_by_tag_without_output_file(
    setup_registry: MCPToolRegistry,
) -> None:
    """get_documents_by_tag without output_file returns raw result unchanged."""
    with patch(
        "obsidian_rag.mcp_server.server._get_documents_by_tag_handler"
    ) as mock_handler:
        mock_handler.return_value = {
            "results": [],
            "total_count": 0,
            "has_more": False,
            "next_offset": None,
        }

        with patch("obsidian_rag.mcp_server.server.write_output_file") as mock_write:
            result = get_documents_by_tag()

            assert result == {
                "results": [],
                "total_count": 0,
                "has_more": False,
                "next_offset": None,
            }
            mock_write.assert_not_called()


def test_get_documents_by_property_with_output_file_local(
    setup_registry: MCPToolRegistry,
) -> None:
    """get_documents_by_property with local output_file writes result and returns summary."""
    with patch(
        "obsidian_rag.mcp_server.tools.documents.get_documents_by_property"
    ) as mock_tool:
        mock_response = DocumentListResponse(
            results=[],
            total_count=0,
            has_more=False,
            next_offset=None,
        )
        mock_tool.return_value = mock_response

        with patch("obsidian_rag.mcp_server.server.write_output_file") as mock_write:
            mock_write.return_value = {
                "output_file": {
                    "type": "local",
                    "path": "/tmp/test.json",
                    "bytes": 100,
                    "item_count": 0,
                }
            }

            result = get_documents_by_property(
                output_file={"type": "local", "path": "/tmp/test.json"},
            )

            assert result == {
                "output_file": {
                    "type": "local",
                    "path": "/tmp/test.json",
                    "bytes": 100,
                    "item_count": 0,
                }
            }
            mock_write.assert_called_once()


def test_get_documents_by_property_without_output_file(
    setup_registry: MCPToolRegistry,
) -> None:
    """get_documents_by_property without output_file returns raw result unchanged."""
    with patch(
        "obsidian_rag.mcp_server.tools.documents.get_documents_by_property"
    ) as mock_tool:
        mock_response = DocumentListResponse(
            results=[],
            total_count=0,
            has_more=False,
            next_offset=None,
        )
        mock_tool.return_value = mock_response

        with patch("obsidian_rag.mcp_server.server.write_output_file") as mock_write:
            result = get_documents_by_property()

            assert result == {
                "results": [],
                "total_count": 0,
                "has_more": False,
                "next_offset": None,
            }
            mock_write.assert_not_called()


def test_get_all_tags_with_output_file_local(setup_registry: MCPToolRegistry) -> None:
    """get_all_tags with local output_file writes result and returns summary."""
    with patch("obsidian_rag.mcp_server.server.get_all_tags_tool") as mock_tool:
        mock_tool.return_value = {"tags": ["work", "personal"], "total_count": 2}

        with patch("obsidian_rag.mcp_server.server.write_output_file") as mock_write:
            mock_write.return_value = {
                "output_file": {
                    "type": "local",
                    "path": "/tmp/test.json",
                    "bytes": 100,
                    "item_count": 2,
                }
            }

            result = get_all_tags(
                output_file={"type": "local", "path": "/tmp/test.json"},
            )

            assert result == {
                "output_file": {
                    "type": "local",
                    "path": "/tmp/test.json",
                    "bytes": 100,
                    "item_count": 2,
                }
            }
            mock_write.assert_called_once()


def test_get_all_tags_without_output_file(setup_registry: MCPToolRegistry) -> None:
    """get_all_tags without output_file returns raw result unchanged."""
    with patch("obsidian_rag.mcp_server.server.get_all_tags_tool") as mock_tool:
        mock_tool.return_value = {"tags": [], "total_count": 0}

        with patch("obsidian_rag.mcp_server.server.write_output_file") as mock_write:
            result = get_all_tags()

            assert result == {"tags": [], "total_count": 0}
            mock_write.assert_not_called()


def test_get_tasks_with_output_file_local(setup_registry: MCPToolRegistry) -> None:
    """get_tasks with local output_file writes result and returns summary."""
    with patch("obsidian_rag.mcp_server.handlers._get_tasks_handler") as mock_handler:
        mock_handler.return_value = {
            "tasks": [{"id": "1"}],
            "total_count": 1,
            "has_more": False,
            "next_offset": None,
        }

        with patch("obsidian_rag.mcp_server.server.write_output_file") as mock_write:
            mock_write.return_value = {
                "output_file": {
                    "type": "local",
                    "path": "/tmp/test.json",
                    "bytes": 100,
                    "item_count": 1,
                }
            }

            result = get_tasks(
                output_file={"type": "local", "path": "/tmp/test.json"},
            )

            assert result == {
                "output_file": {
                    "type": "local",
                    "path": "/tmp/test.json",
                    "bytes": 100,
                    "item_count": 1,
                }
            }
            mock_write.assert_called_once()


def test_get_tasks_with_output_file_s3(setup_registry: MCPToolRegistry) -> None:
    """get_tasks with S3 output_file writes result and returns summary."""
    with patch("obsidian_rag.mcp_server.handlers._get_tasks_handler") as mock_handler:
        mock_handler.return_value = {
            "tasks": [{"id": "1"}],
            "total_count": 1,
            "has_more": False,
            "next_offset": None,
        }

        with patch("obsidian_rag.mcp_server.server.write_output_file") as mock_write:
            mock_write.return_value = {
                "output_file": {
                    "type": "s3",
                    "bucket": "mybucket",
                    "key": "results.json",
                    "bytes": 100,
                    "item_count": 1,
                }
            }

            result = get_tasks(
                output_file={
                    "type": "s3",
                    "endpoint": "http://s3.example.com",
                    "bucket": "mybucket",
                    "key": "results.json",
                    "access_key_id": "AKIAIOSFODNN7EXAMPLE",
                    "secret_access_key": "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
                },
            )

            assert result == {
                "output_file": {
                    "type": "s3",
                    "bucket": "mybucket",
                    "key": "results.json",
                    "bytes": 100,
                    "item_count": 1,
                }
            }
            mock_write.assert_called_once()


def test_get_tasks_without_output_file(setup_registry: MCPToolRegistry) -> None:
    """get_tasks without output_file returns raw result unchanged."""
    with patch("obsidian_rag.mcp_server.handlers._get_tasks_handler") as mock_handler:
        mock_handler.return_value = {
            "tasks": [],
            "total_count": 0,
            "has_more": False,
            "next_offset": None,
        }

        with patch("obsidian_rag.mcp_server.server.write_output_file") as mock_write:
            result = get_tasks()

            assert result == {
                "tasks": [],
                "total_count": 0,
                "has_more": False,
                "next_offset": None,
            }
            mock_write.assert_not_called()


def test_output_file_validation_error_propagates(
    setup_registry: MCPToolRegistry,
) -> None:
    """ValueError from output_file validation is propagated to caller."""
    with patch("obsidian_rag.mcp_server.server.query_documents_tool") as mock_tool:
        mock_tool.return_value = {"results": [], "total_count": 0}

        with pytest.raises(ValueError, match="must be under /tmp/"):
            query_documents(
                query="test",
                output_file={"type": "local", "path": "/etc/passwd"},
            )


def test_output_file_write_error_returns_error_dict(
    setup_registry: MCPToolRegistry,
) -> None:
    """write_output_file returning error dict is passed through by wrapper."""
    with patch("obsidian_rag.mcp_server.server.query_documents_tool") as mock_tool:
        mock_tool.return_value = {"results": [], "total_count": 0}

        with patch("obsidian_rag.mcp_server.server.write_output_file") as mock_write:
            mock_write.return_value = {"success": False, "error": "disk full"}

            result = query_documents(
                query="test",
                output_file={"type": "local", "path": "/tmp/test.json"},
            )

            assert result == {"success": False, "error": "disk full"}
