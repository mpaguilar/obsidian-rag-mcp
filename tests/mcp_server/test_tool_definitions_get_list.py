"""Tests for get_document_tool and list_documents_tool in tool_definitions.py."""

from unittest.mock import Mock, patch

import pytest

from obsidian_rag.mcp_server.tool_definitions import (
    get_document_tool,
    list_documents_tool,
)


class TestGetDocumentTool:
    """Test get_document_tool delegates to handler."""

    @patch("obsidian_rag.mcp_server.tool_definitions._get_document_handler")
    @patch("obsidian_rag.mcp_server.tool_definitions.GetDocumentHandlerParams")
    def test_get_document_tool_delegates_to_handler(
        self,
        mock_params_cls: Mock,
        mock_handler: Mock,
    ) -> None:
        """Creates params, calls handler."""
        mock_db = Mock()
        mock_params = Mock()
        mock_params_cls.return_value = mock_params
        mock_handler.return_value = {"id": "doc-1"}

        result = get_document_tool(mock_db)

        mock_params_cls.assert_called_once_with(
            db_manager=mock_db,
            vault_name=None,
            file_path=None,
            document_id=None,
            include_content=True,
        )
        mock_handler.assert_called_once_with(mock_params)
        assert result == {"id": "doc-1"}

    @patch("obsidian_rag.mcp_server.tool_definitions._get_document_handler")
    @patch("obsidian_rag.mcp_server.tool_definitions.GetDocumentHandlerParams")
    def test_get_document_tool_with_vault_and_path(
        self,
        mock_params_cls: Mock,
        _mock_handler: Mock,
    ) -> None:
        """Passes vault_name and file_path."""
        mock_db = Mock()
        mock_params = Mock()
        mock_params_cls.return_value = mock_params

        get_document_tool(mock_db, vault_name="Personal", file_path="notes.md")

        mock_params_cls.assert_called_once_with(
            db_manager=mock_db,
            vault_name="Personal",
            file_path="notes.md",
            document_id=None,
            include_content=True,
        )

    @patch("obsidian_rag.mcp_server.tool_definitions._get_document_handler")
    @patch("obsidian_rag.mcp_server.tool_definitions.GetDocumentHandlerParams")
    def test_get_document_tool_with_document_id(
        self,
        mock_params_cls: Mock,
        _mock_handler: Mock,
    ) -> None:
        """Passes document_id."""
        mock_db = Mock()
        mock_params = Mock()
        mock_params_cls.return_value = mock_params

        get_document_tool(mock_db, document_id="abc-123")

        mock_params_cls.assert_called_once_with(
            db_manager=mock_db,
            vault_name=None,
            file_path=None,
            document_id="abc-123",
            include_content=True,
        )


class TestListDocumentsTool:
    """Test list_documents_tool delegates to handler."""

    @patch("obsidian_rag.mcp_server.tool_definitions._list_documents_handler")
    @patch("obsidian_rag.mcp_server.tool_definitions.ListDocumentsHandlerParams")
    def test_list_documents_tool_delegates_to_handler(
        self,
        mock_params_cls: Mock,
        mock_handler: Mock,
    ) -> None:
        """Creates params, calls handler."""
        mock_db = Mock()
        mock_params = Mock()
        mock_params_cls.return_value = mock_params
        mock_handler.return_value = {"documents": []}

        result = list_documents_tool(mock_db)

        mock_params_cls.assert_called_once_with(
            db_manager=mock_db,
            file_name=None,
            vault_name=None,
            limit=20,
            offset=0,
        )
        mock_handler.assert_called_once_with(mock_params)
        assert result == {"documents": []}

    @patch("obsidian_rag.mcp_server.tool_definitions._list_documents_handler")
    @patch("obsidian_rag.mcp_server.tool_definitions.ListDocumentsHandlerParams")
    def test_list_documents_tool_with_file_name(
        self,
        mock_params_cls: Mock,
        _mock_handler: Mock,
    ) -> None:
        """Passes file_name."""
        mock_db = Mock()
        mock_params = Mock()
        mock_params_cls.return_value = mock_params

        list_documents_tool(mock_db, file_name="notes.md")

        mock_params_cls.assert_called_once_with(
            db_manager=mock_db,
            file_name="notes.md",
            vault_name=None,
            limit=20,
            offset=0,
        )

    @patch("obsidian_rag.mcp_server.tool_definitions._list_documents_handler")
    @patch("obsidian_rag.mcp_server.tool_definitions.ListDocumentsHandlerParams")
    def test_list_documents_tool_with_vault_scope(
        self,
        mock_params_cls: Mock,
        _mock_handler: Mock,
    ) -> None:
        """Passes vault_name."""
        mock_db = Mock()
        mock_params = Mock()
        mock_params_cls.return_value = mock_params

        list_documents_tool(mock_db, file_name="notes.md", vault_name="Personal")

        mock_params_cls.assert_called_once_with(
            db_manager=mock_db,
            file_name="notes.md",
            vault_name="Personal",
            limit=20,
            offset=0,
        )

    @patch("obsidian_rag.mcp_server.tool_definitions._list_documents_handler")
    @patch("obsidian_rag.mcp_server.tool_definitions.ListDocumentsHandlerParams")
    def test_list_documents_tool_pagination(
        self,
        mock_params_cls: Mock,
        _mock_handler: Mock,
    ) -> None:
        """Passes limit and offset."""
        mock_db = Mock()
        mock_params = Mock()
        mock_params_cls.return_value = mock_params

        list_documents_tool(mock_db, file_name="notes.md", limit=10, offset=5)

        mock_params_cls.assert_called_once_with(
            db_manager=mock_db,
            file_name="notes.md",
            vault_name=None,
            limit=10,
            offset=5,
        )

    def test_list_documents_tool_rejects_include_content(self) -> None:
        """Passing include_content to list_documents_tool raises TypeError."""
        mock_db = Mock()

        with pytest.raises(TypeError):
            list_documents_tool(mock_db, file_name="notes.md", include_content=True)
