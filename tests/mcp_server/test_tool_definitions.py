"""Tests for MCP tool definitions module."""

from unittest.mock import MagicMock, patch

import pytest

from obsidian_rag.mcp_server.handlers import QueryFilterParams
from obsidian_rag.mcp_server.tool_definitions import (
    delete_vault_tool,
    get_vault_tool,
    query_documents_tool,
    update_vault_tool,
)
from obsidian_rag.mcp_server.tools.vaults_params import VaultUpdateParams


class TestQueryDocumentsTool:
    """Tests for query_documents_tool function."""

    @patch("obsidian_rag.mcp_server.tools.documents.query_documents")
    def test_query_text_passed_to_impl(
        self,
        mock_query_documents,
    ):
        """Test that query text is passed to query_documents."""
        from obsidian_rag.mcp_server.tools.documents_params import (
            PaginationParams,
        )

        # Setup mocks
        mock_db_manager = MagicMock()
        mock_embedding_provider = MagicMock()
        mock_embedding_provider.generate_embedding.return_value = [0.1] * 1536
        mock_session = MagicMock()
        mock_db_manager.get_session.return_value.__enter__ = MagicMock(
            return_value=mock_session
        )
        mock_db_manager.get_session.return_value.__exit__ = MagicMock(return_value=None)

        from obsidian_rag.mcp_server.models import DocumentListResponse

        mock_query_documents.return_value = DocumentListResponse(
            results=[],
            total_count=0,
            has_more=False,
            next_offset=None,
        )

        test_query = "semantic search query"
        query_documents_tool(
            db_manager=mock_db_manager,
            embedding_provider=mock_embedding_provider,
            query=test_query,
            filters=QueryFilterParams(
                include_properties=None,
                exclude_properties=None,
                include_tags=None,
                exclude_tags=None,
            ),
            pagination=PaginationParams(limit=20, offset=0),
            use_chunks=True,
            rerank=True,
        )

        # Verify query_documents was called with query_text
        mock_query_documents.assert_called_once()
        call_kwargs = mock_query_documents.call_args.kwargs
        assert call_kwargs.get("query_text") == test_query, (
            f"Expected query_text '{test_query}', got '{call_kwargs.get('query_text')}'"
        )

    def test_raises_error_without_embedding_provider(self):
        """Test that error is raised when embedding provider is None."""
        mock_db_manager = MagicMock()

        with pytest.raises(RuntimeError) as exc_info:
            query_documents_tool(
                db_manager=mock_db_manager,
                embedding_provider=None,
                query="test query",
            )

        assert "Embedding provider not configured" in str(exc_info.value)


class TestGetVaultTool:
    """Tests for get_vault_tool function."""

    @patch("obsidian_rag.mcp_server.tool_definitions._get_vault_handler")
    def test_get_vault_tool_delegates_to_handler(self, mock_handler):
        """Test that get_vault_tool delegates to _get_vault_handler."""
        mock_db_manager = MagicMock()
        expected_doc_count = 5
        mock_handler.return_value = {
            "id": "123",
            "name": "TestVault",
            "document_count": expected_doc_count,
        }

        result = get_vault_tool(
            db_manager=mock_db_manager,
            name="TestVault",
        )

        mock_handler.assert_called_once_with(
            mock_db_manager,
            name="TestVault",
            vault_id=None,
        )
        assert result["name"] == "TestVault"
        assert result["document_count"] == expected_doc_count

    @patch("obsidian_rag.mcp_server.tool_definitions._get_vault_handler")
    def test_get_vault_tool_with_vault_id(self, mock_handler):
        """Test get_vault_tool with vault_id parameter."""
        mock_db_manager = MagicMock()
        mock_handler.return_value = {
            "id": "abc-123",
            "name": "MyVault",
            "document_count": 10,
        }

        result = get_vault_tool(
            db_manager=mock_db_manager,
            vault_id="abc-123",
        )

        mock_handler.assert_called_once_with(
            mock_db_manager,
            name=None,
            vault_id="abc-123",
        )
        assert result["id"] == "abc-123"

    @patch("obsidian_rag.mcp_server.tool_definitions._get_vault_handler")
    def test_get_vault_tool_name_preferred_over_id(self, mock_handler):
        """Test that name takes precedence when both provided."""
        mock_db_manager = MagicMock()
        mock_handler.return_value = {"name": "ByName"}

        result = get_vault_tool(
            db_manager=mock_db_manager,
            name="ByName",
            vault_id="some-id",
        )

        mock_handler.assert_called_once_with(
            mock_db_manager,
            name="ByName",
            vault_id="some-id",
        )
        assert result["name"] == "ByName"


class TestUpdateVaultTool:
    """Tests for update_vault_tool function."""

    @patch("obsidian_rag.mcp_server.tool_definitions._update_vault_handler")
    def test_update_vault_tool_delegates_to_handler(self, mock_handler):
        """Test that update_vault_tool delegates to _update_vault_handler."""
        mock_db_manager = MagicMock()
        mock_handler.return_value = {
            "id": "123",
            "name": "TestVault",
            "description": "Updated desc",
        }

        params = VaultUpdateParams(
            name="TestVault",
            description="Updated desc",
        )

        result = update_vault_tool(
            db_manager=mock_db_manager,
            params=params,
        )

        mock_handler.assert_called_once_with(mock_db_manager, params)
        assert result["description"] == "Updated desc"

    @patch("obsidian_rag.mcp_server.tool_definitions._update_vault_handler")
    def test_update_vault_tool_returns_error_dict(self, mock_handler):
        """Test update_vault_tool returns error dict when force required."""
        mock_db_manager = MagicMock()
        mock_handler.return_value = {
            "success": False,
            "error": "Changing container_path requires force=True",
        }

        params = VaultUpdateParams(
            name="TestVault",
            container_path="/new/path",
        )

        result = update_vault_tool(
            db_manager=mock_db_manager,
            params=params,
        )

        assert result["success"] is False
        assert "force=True" in result["error"]

    @patch("obsidian_rag.mcp_server.tool_definitions._update_vault_handler")
    def test_update_vault_tool_with_force(self, mock_handler):
        """Test update_vault_tool with force=True."""
        mock_db_manager = MagicMock()
        mock_handler.return_value = {
            "id": "123",
            "name": "TestVault",
            "container_path": "/new/path",
        }

        params = VaultUpdateParams(
            name="TestVault",
            container_path="/new/path",
            force=True,
        )

        result = update_vault_tool(
            db_manager=mock_db_manager,
            params=params,
        )

        mock_handler.assert_called_once_with(mock_db_manager, params)
        assert result["container_path"] == "/new/path"


class TestDeleteVaultTool:
    """Tests for delete_vault_tool function."""

    @patch("obsidian_rag.mcp_server.tool_definitions._delete_vault_handler")
    def test_delete_vault_tool_delegates_to_handler(self, mock_handler):
        """Test that delete_vault_tool delegates to _delete_vault_handler."""
        mock_db_manager = MagicMock()
        expected_docs_deleted = 5
        mock_handler.return_value = {
            "success": True,
            "name": "TestVault",
            "documents_deleted": expected_docs_deleted,
            "tasks_deleted": 10,
            "chunks_deleted": 20,
        }

        result = delete_vault_tool(
            db_manager=mock_db_manager,
            name="TestVault",
            confirm=True,
        )

        mock_handler.assert_called_once_with(
            mock_db_manager,
            name="TestVault",
            confirm=True,
        )
        assert result["success"] is True
        assert result["documents_deleted"] == expected_docs_deleted

    @patch("obsidian_rag.mcp_server.tool_definitions._delete_vault_handler")
    def test_delete_vault_tool_not_confirmed(self, mock_handler):
        """Test delete_vault_tool returns error when not confirmed."""
        mock_db_manager = MagicMock()
        mock_handler.return_value = {
            "success": False,
            "error": "confirm=True is required",
        }

        result = delete_vault_tool(
            db_manager=mock_db_manager,
            name="TestVault",
            confirm=False,
        )

        mock_handler.assert_called_once_with(
            mock_db_manager,
            name="TestVault",
            confirm=False,
        )
        assert result["success"] is False

    @patch("obsidian_rag.mcp_server.tool_definitions._delete_vault_handler")
    def test_delete_vault_tool_vault_not_found(self, mock_handler):
        """Test delete_vault_tool returns error when vault not found."""
        mock_db_manager = MagicMock()
        mock_handler.return_value = {
            "success": False,
            "error": "Vault 'NonExistent' not found",
        }

        result = delete_vault_tool(
            db_manager=mock_db_manager,
            name="NonExistent",
            confirm=True,
        )

        assert result["success"] is False
        assert "not found" in result["error"]
