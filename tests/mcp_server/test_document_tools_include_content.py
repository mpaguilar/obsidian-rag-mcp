"""Tests for document_tools.py include_content parameter."""

import pytest
from unittest.mock import Mock, patch

from obsidian_rag.mcp_server.document_tools import (
    get_document,
    list_documents,
)


class TestGetDocumentWrapperIncludeContent:
    """Test get_document wrapper passes include_content."""

    @patch("obsidian_rag.mcp_server.document_tools._get_registry")
    @patch("obsidian_rag.mcp_server.document_tools.get_document_tool")
    def test_get_document_wrapper_include_content_true(
        self,
        mock_tool: Mock,
        mock_registry: Mock,
    ) -> None:
        """Verifies include_content=True passed by default."""
        mock_registry.return_value.db_manager = Mock()
        mock_tool.return_value = {"id": "doc-1", "content": "hello"}

        result = get_document(document_id="abc-123")

        mock_tool.assert_called_once_with(
            mock_registry.return_value.db_manager,
            vault_name=None,
            file_path=None,
            document_id="abc-123",
            include_content=True,
        )
        assert result == {"id": "doc-1", "content": "hello"}

    @patch("obsidian_rag.mcp_server.document_tools._get_registry")
    @patch("obsidian_rag.mcp_server.document_tools.get_document_tool")
    def test_get_document_wrapper_include_content_false(
        self,
        mock_tool: Mock,
        mock_registry: Mock,
    ) -> None:
        """Verifies include_content=False is passed through."""
        mock_registry.return_value.db_manager = Mock()
        mock_tool.return_value = {"id": "doc-1", "content": ""}

        result = get_document(document_id="abc-123", include_content=False)

        mock_tool.assert_called_once_with(
            mock_registry.return_value.db_manager,
            vault_name=None,
            file_path=None,
            document_id="abc-123",
            include_content=False,
        )
        assert result == {"id": "doc-1", "content": ""}


class TestListDocumentsWrapperIncludeContent:
    """Test list_documents wrapper rejects include_content (API break)."""

    @patch("obsidian_rag.mcp_server.document_tools._get_registry")
    @patch("obsidian_rag.mcp_server.document_tools.list_documents_tool")
    def test_list_documents_wrapper_rejects_include_content_kwarg(
        self,
        mock_tool: Mock,
        mock_registry: Mock,
    ) -> None:
        """Verifies include_content kwarg raises TypeError (REQ-012 API break)."""
        mock_registry.return_value.db_manager = Mock()
        mock_tool.return_value = {"documents": [{"id": "doc-1"}]}

        with pytest.raises(TypeError):
            list_documents(file_name="notes.md", include_content=False)
