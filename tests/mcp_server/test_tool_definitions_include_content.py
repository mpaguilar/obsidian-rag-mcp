"""Tests for include_content parameter in tool_definitions.py."""

from unittest.mock import MagicMock, patch

import pytest

from obsidian_rag.mcp_server.handlers import QueryFilterParams
from obsidian_rag.mcp_server.tool_definitions import (
    get_document_tool,
    list_documents_tool,
    query_documents_tool,
)


class TestQueryDocumentsToolIncludeContent:
    """Tests for query_documents_tool include_content parameter."""

    @patch("obsidian_rag.mcp_server.tools.documents.query_documents")
    def test_query_documents_tool_include_content_true(self, mock_query_documents):
        """Test that include_content=True is passed to query_documents."""
        from obsidian_rag.mcp_server.tools.documents_params import PaginationParams

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

        query_documents_tool(
            db_manager=mock_db_manager,
            embedding_provider=mock_embedding_provider,
            query="test query",
            filters=QueryFilterParams(
                include_properties=None,
                exclude_properties=None,
                include_tags=None,
                exclude_tags=None,
            ),
            pagination=PaginationParams(limit=20, offset=0),
            include_content=True,
        )

        mock_query_documents.assert_called_once()
        call_kwargs = mock_query_documents.call_args.kwargs
        assert call_kwargs.get("include_content") is True

    @patch("obsidian_rag.mcp_server.tools.documents.query_documents")
    def test_query_documents_tool_include_content_false(self, mock_query_documents):
        """Test that include_content=False is passed to query_documents."""
        from obsidian_rag.mcp_server.tools.documents_params import PaginationParams

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

        query_documents_tool(
            db_manager=mock_db_manager,
            embedding_provider=mock_embedding_provider,
            query="test query",
            filters=QueryFilterParams(
                include_properties=None,
                exclude_properties=None,
                include_tags=None,
                exclude_tags=None,
            ),
            pagination=PaginationParams(limit=20, offset=0),
            include_content=False,
        )

        mock_query_documents.assert_called_once()
        call_kwargs = mock_query_documents.call_args.kwargs
        assert call_kwargs.get("include_content") is False


class TestGetDocumentToolIncludeContent:
    """Tests for get_document_tool include_content parameter."""

    @patch("obsidian_rag.mcp_server.tool_definitions._get_document_handler")
    @patch("obsidian_rag.mcp_server.tool_definitions.GetDocumentHandlerParams")
    def test_get_document_tool_include_content_true(
        self,
        mock_params_cls: MagicMock,
        mock_handler: MagicMock,
    ) -> None:
        """Test that include_content=True is passed to GetDocumentHandlerParams."""
        mock_db = MagicMock()
        mock_params = MagicMock()
        mock_params_cls.return_value = mock_params
        mock_handler.return_value = {"id": "doc-1"}

        result = get_document_tool(mock_db, document_id="abc-123", include_content=True)

        mock_params_cls.assert_called_once_with(
            db_manager=mock_db,
            vault_name=None,
            file_path=None,
            document_id="abc-123",
            include_content=True,
        )
        mock_handler.assert_called_once_with(mock_params)
        assert result == {"id": "doc-1"}

    @patch("obsidian_rag.mcp_server.tool_definitions._get_document_handler")
    @patch("obsidian_rag.mcp_server.tool_definitions.GetDocumentHandlerParams")
    def test_get_document_tool_include_content_false(
        self,
        mock_params_cls: MagicMock,
        mock_handler: MagicMock,
    ) -> None:
        """Test that include_content=False is passed to GetDocumentHandlerParams."""
        mock_db = MagicMock()
        mock_params = MagicMock()
        mock_params_cls.return_value = mock_params
        mock_handler.return_value = {"id": "doc-1"}

        result = get_document_tool(
            mock_db, document_id="abc-123", include_content=False
        )

        mock_params_cls.assert_called_once_with(
            db_manager=mock_db,
            vault_name=None,
            file_path=None,
            document_id="abc-123",
            include_content=False,
        )
        mock_handler.assert_called_once_with(mock_params)
        assert result == {"id": "doc-1"}


@patch("obsidian_rag.mcp_server.tool_definitions._list_documents_handler")
@patch("obsidian_rag.mcp_server.tool_definitions.ListDocumentsHandlerParams")
def test_list_documents_tool_hardcodes_include_content_false(
    mock_params_cls: MagicMock,
    mock_handler: MagicMock,
) -> None:
    """Assert ListDocumentsHandlerParams is constructed without include_content."""
    mock_db = MagicMock()
    mock_params = MagicMock()
    mock_params_cls.return_value = mock_params
    mock_handler.return_value = {"documents": []}

    result = list_documents_tool(mock_db, file_name="notes.md")

    mock_params_cls.assert_called_once_with(
        db_manager=mock_db,
        file_name="notes.md",
        vault_name=None,
        limit=20,
        offset=0,
    )
    mock_handler.assert_called_once_with(mock_params)
    assert result == {"documents": []}


def test_list_documents_tool_rejects_include_content_kwarg() -> None:
    """Calling list_documents_tool with include_content raises TypeError."""
    mock_db = MagicMock()

    with pytest.raises(TypeError):
        list_documents_tool(mock_db, file_name="notes.md", include_content=True)
