"""End-to-end integration tests for document retrieval MCP tools."""

from unittest.mock import Mock, patch

import uuid as uuid_module

# Import the server-level wrappers
from obsidian_rag.mcp_server.document_tools import (
    get_document,
    list_documents,
)


class TestGetDocumentMCPTool:
    """End-to-end tests for get_document MCP tool."""

    @patch("obsidian_rag.mcp_server.document_tools._get_registry")
    @patch("obsidian_rag.mcp_server.document_tools.get_document_tool")
    def test_get_document_mcp_tool_by_vault_path(self, mock_tool, mock_registry):
        """Call server.py get_document() with vault+path."""
        mock_registry.return_value.db_manager = Mock()
        mock_tool.return_value = {
            "id": str(uuid_module.uuid4()),
            "vault_name": "Personal",
            "file_path": "notes.md",
            "content": "test content",
        }
        result = get_document(vault_name="Personal", file_path="notes.md")
        mock_tool.assert_called_once_with(
            mock_registry.return_value.db_manager,
            vault_name="Personal",
            file_path="notes.md",
            document_id=None,
            include_content=True,
        )
        assert result["vault_name"] == "Personal"
        assert result["file_path"] == "notes.md"

    @patch("obsidian_rag.mcp_server.document_tools._get_registry")
    @patch("obsidian_rag.mcp_server.document_tools.get_document_tool")
    def test_get_document_mcp_tool_by_id(self, mock_tool, mock_registry):
        """Call server.py get_document() with document_id."""
        mock_registry.return_value.db_manager = Mock()
        doc_id = str(uuid_module.uuid4())
        mock_tool.return_value = {"id": doc_id, "vault_name": "Personal"}
        result = get_document(document_id=doc_id)
        mock_tool.assert_called_once_with(
            mock_registry.return_value.db_manager,
            vault_name=None,
            file_path=None,
            document_id=doc_id,
            include_content=True,
        )
        assert result["id"] == doc_id

    @patch("obsidian_rag.mcp_server.document_tools._get_registry")
    @patch("obsidian_rag.mcp_server.document_tools.get_document_tool")
    def test_get_document_mcp_tool_no_params(self, mock_tool, mock_registry):
        """Returns error dict."""
        mock_registry.return_value.db_manager = Mock()
        mock_tool.return_value = {
            "success": False,
            "error": "Must provide either document_id, or vault_name and file_path",
        }
        result = get_document()
        assert result["success"] is False
        assert "error" in result

    @patch("obsidian_rag.mcp_server.document_tools._get_registry")
    @patch("obsidian_rag.mcp_server.document_tools.get_document_tool")
    def test_get_document_mcp_tool_path_without_vault(self, mock_tool, mock_registry):
        """Returns error dict."""
        mock_registry.return_value.db_manager = Mock()
        mock_tool.return_value = {
            "success": False,
            "error": "vault_name is required when using file_path",
        }
        result = get_document(file_path="notes.md")
        assert result["success"] is False
        assert "vault_name" in result["error"]

    @patch("obsidian_rag.mcp_server.document_tools._get_registry")
    @patch("obsidian_rag.mcp_server.document_tools.get_document_tool")
    def test_get_document_mcp_tool_not_found(self, mock_tool, mock_registry):
        """Returns error dict."""
        mock_registry.return_value.db_manager = Mock()
        mock_tool.return_value = {"success": False, "error": "Document not found"}
        result = get_document(vault_name="Personal", file_path="missing.md")
        assert result["success"] is False


class TestListDocumentsMCPTool:
    """End-to-end tests for list_documents MCP tool."""

    @patch("obsidian_rag.mcp_server.document_tools._get_registry")
    @patch("obsidian_rag.mcp_server.document_tools.list_documents_tool")
    def test_list_documents_mcp_tool_by_name(self, mock_tool, mock_registry):
        """Call server.py list_documents() with name."""
        mock_registry.return_value.db_manager = Mock()
        mock_tool.return_value = {
            "documents": [{"id": str(uuid_module.uuid4()), "file_name": "notes.md"}],
            "total_count": 1,
            "has_more": False,
        }
        result = list_documents(file_name="notes.md")
        mock_tool.assert_called_once_with(
            mock_registry.return_value.db_manager,
            file_name="notes.md",
            vault_name=None,
            limit=20,
            offset=0,
        )
        assert result["total_count"] == 1

    @patch("obsidian_rag.mcp_server.document_tools._get_registry")
    @patch("obsidian_rag.mcp_server.document_tools.list_documents_tool")
    def test_list_documents_mcp_tool_by_name_with_vault(self, mock_tool, mock_registry):
        """Scoped to vault."""
        mock_registry.return_value.db_manager = Mock()
        mock_tool.return_value = {"documents": [], "total_count": 0, "has_more": False}
        list_documents(file_name="notes.md", vault_name="Personal")
        mock_tool.assert_called_once_with(
            mock_registry.return_value.db_manager,
            file_name="notes.md",
            vault_name="Personal",
            limit=20,
            offset=0,
        )

    @patch("obsidian_rag.mcp_server.document_tools._get_registry")
    @patch("obsidian_rag.mcp_server.document_tools.list_documents_tool")
    def test_list_documents_mcp_tool_no_name(self, mock_tool, mock_registry):
        """Returns error dict."""
        mock_registry.return_value.db_manager = Mock()
        mock_tool.return_value = {
            "success": False,
            "error": "Must provide at least file_name",
        }
        result = list_documents()
        assert result["success"] is False
        assert "file_name" in result["error"]

    @patch("obsidian_rag.mcp_server.document_tools._get_registry")
    @patch("obsidian_rag.mcp_server.document_tools.list_documents_tool")
    def test_list_documents_mcp_tool_empty_results(self, mock_tool, mock_registry):
        """Returns empty list."""
        mock_registry.return_value.db_manager = Mock()
        mock_tool.return_value = {"documents": [], "total_count": 0, "has_more": False}
        result = list_documents(file_name="nonexistent.md")
        assert result["documents"] == []
        assert result["total_count"] == 0


class TestToolRegistration:
    """Verify tools are registered in _register_tools."""

    def test_get_document_tool_registered(self):
        """Verify get_document is registered."""
        from obsidian_rag.mcp_server.server import _register_tools

        mcp = Mock()
        registered = []

        def capture_tool(func):
            """Decorator that captures the registered function."""
            registered.append(func)
            return func

        mcp.tool.return_value = capture_tool

        _register_tools(mcp)

        tool_names = [tool.__name__ for tool in registered if hasattr(tool, "__name__")]
        assert "get_document" in tool_names

    def test_list_documents_tool_registered(self):
        """Verify list_documents is registered."""
        from obsidian_rag.mcp_server.server import _register_tools

        mcp = Mock()
        registered = []

        def capture_tool(func):
            """Decorator that captures the registered function."""
            registered.append(func)
            return func

        mcp.tool.return_value = capture_tool

        _register_tools(mcp)

        tool_names = [tool.__name__ for tool in registered if hasattr(tool, "__name__")]
        assert "list_documents" in tool_names
