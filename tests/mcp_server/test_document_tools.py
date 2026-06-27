"""Tests for document_tools.py MCP tool wrappers."""

from unittest.mock import Mock, patch

from obsidian_rag.mcp_server.document_tools import (
    get_document,
    list_documents,
)


class TestGetDocumentWrapper:
    """Test get_document wrapper delegates to tool."""

    @patch("obsidian_rag.mcp_server.document_tools._get_registry")
    @patch("obsidian_rag.mcp_server.document_tools.get_document_tool")
    def test_get_document_wrapper_delegates_to_tool(
        self,
        mock_tool: Mock,
        mock_registry: Mock,
    ) -> None:
        """Verifies registry access and delegation."""
        mock_registry.return_value.db_manager = Mock()
        mock_tool.return_value = {"id": "doc-1", "vault_name": "Personal"}

        result = get_document()

        mock_registry.assert_called_once()
        mock_tool.assert_called_once_with(
            mock_registry.return_value.db_manager,
            vault_name=None,
            file_path=None,
            document_id=None,
            include_content=True,
        )
        assert result == {"id": "doc-1", "vault_name": "Personal"}

    @patch("obsidian_rag.mcp_server.document_tools._get_registry")
    @patch("obsidian_rag.mcp_server.document_tools.get_document_tool")
    def test_get_document_wrapper_with_vault_and_path(
        self,
        mock_tool: Mock,
        mock_registry: Mock,
    ) -> None:
        """Passes vault_name and file_path."""
        mock_registry.return_value.db_manager = Mock()
        mock_tool.return_value = {"id": "doc-1"}

        get_document(vault_name="Personal", file_path="notes.md")

        mock_tool.assert_called_once_with(
            mock_registry.return_value.db_manager,
            vault_name="Personal",
            file_path="notes.md",
            document_id=None,
            include_content=True,
        )

    @patch("obsidian_rag.mcp_server.document_tools._get_registry")
    @patch("obsidian_rag.mcp_server.document_tools.get_document_tool")
    def test_get_document_wrapper_with_document_id(
        self,
        mock_tool: Mock,
        mock_registry: Mock,
    ) -> None:
        """Passes document_id."""
        mock_registry.return_value.db_manager = Mock()
        mock_tool.return_value = {"id": "doc-1"}

        get_document(document_id="abc-123")

        mock_tool.assert_called_once_with(
            mock_registry.return_value.db_manager,
            vault_name=None,
            file_path=None,
            document_id="abc-123",
            include_content=True,
        )


class TestListDocumentsWrapper:
    """Test list_documents wrapper delegates to tool."""

    @patch("obsidian_rag.mcp_server.document_tools._get_registry")
    @patch("obsidian_rag.mcp_server.document_tools.list_documents_tool")
    def test_list_documents_wrapper_delegates_to_tool(
        self,
        mock_tool: Mock,
        mock_registry: Mock,
    ) -> None:
        """Verifies registry access and delegation."""
        mock_registry.return_value.db_manager = Mock()
        mock_tool.return_value = {"documents": [], "total_count": 0}

        result = list_documents()

        mock_registry.assert_called_once()
        mock_tool.assert_called_once_with(
            mock_registry.return_value.db_manager,
            file_name=None,
            vault_name=None,
            limit=20,
            offset=0,
            include_content=True,
        )
        assert result == {"documents": [], "total_count": 0}

    @patch("obsidian_rag.mcp_server.document_tools._get_registry")
    @patch("obsidian_rag.mcp_server.document_tools.list_documents_tool")
    def test_list_documents_wrapper_with_file_name(
        self,
        mock_tool: Mock,
        mock_registry: Mock,
    ) -> None:
        """Passes file_name."""
        mock_registry.return_value.db_manager = Mock()
        mock_tool.return_value = {"documents": []}

        list_documents(file_name="notes.md")

        mock_tool.assert_called_once_with(
            mock_registry.return_value.db_manager,
            file_name="notes.md",
            vault_name=None,
            limit=20,
            offset=0,
            include_content=True,
        )

    @patch("obsidian_rag.mcp_server.document_tools._get_registry")
    @patch("obsidian_rag.mcp_server.document_tools.list_documents_tool")
    def test_list_documents_wrapper_with_vault_scope(
        self,
        mock_tool: Mock,
        mock_registry: Mock,
    ) -> None:
        """Passes file_name and vault_name."""
        mock_registry.return_value.db_manager = Mock()
        mock_tool.return_value = {"documents": []}

        list_documents(file_name="notes.md", vault_name="Personal")

        mock_tool.assert_called_once_with(
            mock_registry.return_value.db_manager,
            file_name="notes.md",
            vault_name="Personal",
            limit=20,
            offset=0,
            include_content=True,
        )

    @patch("obsidian_rag.mcp_server.document_tools._get_registry")
    @patch("obsidian_rag.mcp_server.document_tools.list_documents_tool")
    def test_list_documents_wrapper_pagination(
        self,
        mock_tool: Mock,
        mock_registry: Mock,
    ) -> None:
        """Passes limit and offset."""
        mock_registry.return_value.db_manager = Mock()
        mock_tool.return_value = {"documents": []}

        list_documents(file_name="notes.md", limit=10, offset=5)

        mock_tool.assert_called_once_with(
            mock_registry.return_value.db_manager,
            file_name="notes.md",
            vault_name=None,
            limit=10,
            offset=5,
            include_content=True,
        )
