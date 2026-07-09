"""Tests that handlers pass include_content through to tool implementations."""

from unittest.mock import MagicMock, patch

from obsidian_rag.mcp_server.handlers import (
    DocumentTagParams,
    GetDocumentHandlerParams,
    _get_documents_by_property_handler,
    _get_documents_by_tag_handler,
    _get_document_handler,
    _get_tasks_handler,
    _list_documents_handler,
)
from obsidian_rag.mcp_server.tools.tasks_params import GetTasksRequest


def test_get_documents_by_tag_handler_passes_include_content_false():
    """_get_documents_by_tag_handler should hardcode include_content=False regardless of params."""
    mock_db_manager = MagicMock()
    mock_session = MagicMock()
    mock_db_manager.get_session.return_value.__enter__.return_value = mock_session
    mock_db_manager.get_session.return_value.__exit__.return_value = False

    with patch(
        "obsidian_rag.mcp_server.handlers.get_documents_by_tag_tool"
    ) as mock_tool:
        mock_tool.return_value.model_dump.return_value = {
            "results": [],
            "total_count": 0,
            "has_more": False,
            "next_offset": None,
        }

        params: DocumentTagParams = {
            "include_tags": ["work"],
            "match_mode": "all",
            "vault_name": "Personal",
            "limit": 20,
            "offset": 0,
        }

        _get_documents_by_tag_handler(mock_db_manager, params)

        assert "include_content" not in mock_tool.call_args.kwargs


def test_get_document_handler_passes_include_content_true():
    """_get_document_handler should pass include_content=True by default."""
    mock_db_manager = MagicMock()
    mock_session = MagicMock()
    mock_db_manager.get_session.return_value.__enter__.return_value = mock_session
    mock_db_manager.get_session.return_value.__exit__.return_value = False

    with patch("obsidian_rag.mcp_server.tools.documents.get_document") as mock_impl:
        mock_impl.return_value.model_dump.return_value = {
            "id": "test-id",
            "content": "content",
        }

        params = GetDocumentHandlerParams(
            db_manager=mock_db_manager,
            document_id="test-id",
        )

        _get_document_handler(params)

        assert mock_impl.call_args.kwargs["include_content"] is True


def test_get_document_handler_passes_include_content_false():
    """_get_document_handler should pass include_content=False when set."""
    mock_db_manager = MagicMock()
    mock_session = MagicMock()
    mock_db_manager.get_session.return_value.__enter__.return_value = mock_session
    mock_db_manager.get_session.return_value.__exit__.return_value = False

    with patch("obsidian_rag.mcp_server.tools.documents.get_document") as mock_impl:
        mock_impl.return_value.model_dump.return_value = {
            "id": "test-id",
            "content": "",
        }

        params = GetDocumentHandlerParams(
            db_manager=mock_db_manager,
            document_id="test-id",
            include_content=False,
        )

        _get_document_handler(params)

        assert mock_impl.call_args.kwargs["include_content"] is False


def test_list_documents_handler_passes_include_content_false():
    """_list_documents_handler should hardcode include_content=False regardless of params."""
    mock_db_manager = MagicMock()
    mock_session = MagicMock()
    mock_db_manager.get_session.return_value.__enter__.return_value = mock_session
    mock_db_manager.get_session.return_value.__exit__.return_value = False

    with patch("obsidian_rag.mcp_server.tools.documents.list_documents") as mock_impl:
        mock_impl.return_value.model_dump.return_value = {
            "results": [],
            "total_count": 0,
            "has_more": False,
            "next_offset": None,
        }

        from obsidian_rag.mcp_server.handlers import ListDocumentsHandlerParams

        params = ListDocumentsHandlerParams(
            db_manager=mock_db_manager,
            file_name="test.md",
        )

        _list_documents_handler(params)

        assert "include_content" not in mock_impl.call_args.kwargs


def test_get_documents_by_property_handler_hardcodes_include_content_false():
    """_get_documents_by_property_handler should hardcode PaginationParams with include_content=False and NOT pass standalone include_content kwarg."""
    mock_db_manager = MagicMock()
    mock_session = MagicMock()
    mock_db_manager.get_session.return_value.__enter__.return_value = mock_session
    mock_db_manager.get_session.return_value.__exit__.return_value = False

    with patch(
        "obsidian_rag.mcp_server.tools.documents.get_documents_by_property"
    ) as mock_tool:
        mock_tool.return_value.model_dump.return_value = {
            "results": [],
            "total_count": 0,
            "has_more": False,
            "next_offset": None,
        }

        _get_documents_by_property_handler(
            db_manager=mock_db_manager,
            property_filters=None,
            tag_filter=None,
            vault_name=None,
            limit=10,
            offset=0,
        )

        kwargs = mock_tool.call_args.kwargs
        assert "include_content" not in kwargs
        pagination = kwargs["pagination"]
        assert pagination.include_content is False


def test_get_tasks_handler_passes_include_content_through_filters():
    """_get_tasks_handler should pass include_content through GetTasksFilterParams."""
    mock_db_manager = MagicMock()
    mock_session = MagicMock()
    mock_db_manager.get_session.return_value.__enter__.return_value = mock_session
    mock_db_manager.get_session.return_value.__exit__.return_value = False

    with patch("obsidian_rag.mcp_server.handlers.get_tasks_tool") as mock_tool:
        mock_tool.return_value.model_dump.return_value = {
            "results": [],
            "total_count": 0,
            "has_more": False,
            "next_offset": None,
        }

        request = GetTasksRequest(
            status=["not_completed"],
            include_content=False,
            limit=10,
            offset=5,
        )

        _get_tasks_handler(
            db_manager=mock_db_manager,
            request=request,
        )

        call_args = mock_tool.call_args
        filters = call_args.kwargs["filters"]
        assert filters.include_content is False
