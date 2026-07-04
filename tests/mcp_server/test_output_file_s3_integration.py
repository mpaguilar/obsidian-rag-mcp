"""Integration tests for S3 writes through MCP tool wrappers."""

from unittest.mock import MagicMock, patch

import pytest

from obsidian_rag.mcp_server.document_tools import (
    get_document,
    list_documents,
)
from obsidian_rag.mcp_server.server import (
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


_S3_CONFIG: dict[str, str] = {
    "type": "s3",
    "endpoint": "http://s3.example.com",
    "bucket": "mybucket",
    "key": "results.json",
    "access_key_id": "AKIAIOSFODNN7EXAMPLE",
    "secret_access_key": "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
}


def _assert_s3_summary(result: dict[str, object], expected_item_count: int) -> None:
    """Assert result contains the expected S3 output_file summary structure."""
    assert "output_file" in result
    output_file = result["output_file"]
    assert isinstance(output_file, dict)
    assert output_file["type"] == "s3"
    assert output_file["bucket"] == "mybucket"
    assert output_file["key"] == "results.json"
    assert output_file["bytes"] > 0
    assert output_file["item_count"] == expected_item_count


def test_get_document_output_file_s3_integration(
    setup_registry: MCPToolRegistry,
) -> None:
    """get_document with S3 output_file writes result via PutObject and returns summary."""
    mock_client = MagicMock()
    with patch("boto3.client", return_value=mock_client) as mock_boto3:
        with patch(
            "obsidian_rag.mcp_server.document_tools.get_document_tool"
        ) as mock_tool:
            mock_tool.return_value = {
                "id": "doc-1",
                "vault_name": "Personal",
                "content": "test content",
            }

            result = get_document(
                vault_name="Personal",
                file_path="notes.md",
                output_file=_S3_CONFIG,
            )

            mock_boto3.assert_called_once()
            mock_client.put_object.assert_called_once()
            _assert_s3_summary(result, 1)


def test_list_documents_output_file_s3_integration(
    setup_registry: MCPToolRegistry,
) -> None:
    """list_documents with S3 output_file writes result via PutObject and returns summary."""
    mock_client = MagicMock()
    with patch("boto3.client", return_value=mock_client) as mock_boto3:
        with patch(
            "obsidian_rag.mcp_server.document_tools.list_documents_tool"
        ) as mock_tool:
            mock_tool.return_value = {
                "documents": [
                    {"id": "doc-1", "file_name": "a.md"},
                    {"id": "doc-2", "file_name": "b.md"},
                ],
                "total_count": 2,
            }

            result = list_documents(
                file_name="notes.md",
                output_file=_S3_CONFIG,
            )

            mock_boto3.assert_called_once()
            mock_client.put_object.assert_called_once()
            _assert_s3_summary(result, 2)


def test_query_documents_output_file_s3_integration(
    setup_registry: MCPToolRegistry,
) -> None:
    """query_documents with S3 output_file writes result via PutObject and returns summary."""
    mock_client = MagicMock()
    with patch("boto3.client", return_value=mock_client) as mock_boto3:
        with patch("obsidian_rag.mcp_server.server.query_documents_tool") as mock_tool:
            mock_tool.return_value = {
                "documents": [
                    {"id": "doc-1", "similarity_score": 0.9},
                    {"id": "doc-2", "similarity_score": 0.8},
                    {"id": "doc-3", "similarity_score": 0.7},
                ],
                "total_count": 3,
            }

            result = query_documents(
                query="test query",
                output_file=_S3_CONFIG,
            )

            mock_boto3.assert_called_once()
            mock_client.put_object.assert_called_once()
            _assert_s3_summary(result, 3)


def test_get_tasks_output_file_s3_integration(
    setup_registry: MCPToolRegistry,
) -> None:
    """get_tasks with S3 output_file writes result via PutObject and returns summary."""
    mock_client = MagicMock()
    with patch("boto3.client", return_value=mock_client) as mock_boto3:
        with patch(
            "obsidian_rag.mcp_server.handlers._get_tasks_handler"
        ) as mock_handler:
            mock_handler.return_value = {
                "tasks": [
                    {"id": "task-1", "status": "not_completed"},
                    {"id": "task-2", "status": "completed"},
                ],
                "total_count": 2,
            }

            result = get_tasks(
                status=["not_completed", "completed"],
                output_file=_S3_CONFIG,
            )

            mock_boto3.assert_called_once()
            mock_client.put_object.assert_called_once()
            _assert_s3_summary(result, 2)


def test_s3_put_object_parameters(setup_registry: MCPToolRegistry) -> None:
    """Verify PutObject receives correct Bucket, Key, Body, and ContentType."""
    mock_client = MagicMock()
    with patch("boto3.client", return_value=mock_client) as mock_boto3:
        with patch("obsidian_rag.mcp_server.server.query_documents_tool") as mock_tool:
            mock_tool.return_value = {
                "documents": [{"id": "doc-1"}],
                "total_count": 1,
            }

            query_documents(
                query="test",
                output_file=_S3_CONFIG,
            )

            mock_boto3.assert_called_once()
            call_kwargs = mock_client.put_object.call_args.kwargs
            assert call_kwargs["Bucket"] == "mybucket"
            assert call_kwargs["Key"] == "results.json"
            assert (
                call_kwargs["Body"]
                == b'{"documents": [{"id": "doc-1"}], "total_count": 1}'
            )
            assert call_kwargs["ContentType"] == "application/json"


def test_s3_client_configuration(setup_registry: MCPToolRegistry) -> None:
    """Verify boto3.client configured with correct endpoint_url, credentials, timeouts."""
    mock_client = MagicMock()
    with patch("boto3.client", return_value=mock_client) as mock_boto3:
        with patch("obsidian_rag.mcp_server.server.query_documents_tool") as mock_tool:
            mock_tool.return_value = {"documents": [], "total_count": 0}

            query_documents(
                query="test",
                output_file=_S3_CONFIG,
            )

            mock_boto3.assert_called_once()
            call_args = mock_boto3.call_args
            assert call_args[0][0] == "s3"
            assert call_args.kwargs["endpoint_url"] == "http://s3.example.com"
            assert call_args.kwargs["aws_access_key_id"] == "AKIAIOSFODNN7EXAMPLE"
            assert (
                call_args.kwargs["aws_secret_access_key"]
                == "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
            )
            assert call_args.kwargs["config"] is not None
            assert call_args.kwargs["config"].connect_timeout == 10
            assert call_args.kwargs["config"].read_timeout == 30
            assert call_args.kwargs["config"].s3["addressing_style"] == "virtual"
