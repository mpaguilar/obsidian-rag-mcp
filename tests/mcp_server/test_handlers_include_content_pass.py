"""Tests that handlers pass include_content through to tool implementations."""

from unittest.mock import MagicMock, patch

from obsidian_rag.mcp_server.handlers import (
    DocumentTagParams,
    GetDocumentHandlerParams,
    ListDocumentsHandlerParams,
    _get_documents_by_tag_handler,
    _get_document_handler,
    _get_tasks_handler,
    _list_documents_handler,
)
from obsidian_rag.mcp_server.tools.tasks_params import GetTasksRequest


def test_get_documents_by_tag_handler_passes_include_content_true():
    """_get_documents_by_tag_handler should pass include_content=True by default."""
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
            "include_content": True,
        }

        _get_documents_by_tag_handler(mock_db_manager, params)

        assert mock_tool.call_args.kwargs["include_content"] is True


def test_get_documents_by_tag_handler_passes_include_content_false():
    """_get_documents_by_tag_handler should pass include_content=False when set."""
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
            "include_content": False,
        }

        _get_documents_by_tag_handler(mock_db_manager, params)

        assert mock_tool.call_args.kwargs["include_content"] is False


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


def test_list_documents_handler_passes_include_content_true():
    """_list_documents_handler should pass include_content=True by default."""
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

        params = ListDocumentsHandlerParams(
            db_manager=mock_db_manager,
            file_name="test.md",
        )

        _list_documents_handler(params)

        assert mock_impl.call_args.kwargs["include_content"] is True


def test_list_documents_handler_passes_include_content_false():
    """_list_documents_handler should pass include_content=False when set."""
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

        params = ListDocumentsHandlerParams(
            db_manager=mock_db_manager,
            file_name="test.md",
            include_content=False,
        )

        _list_documents_handler(params)

        assert mock_impl.call_args.kwargs["include_content"] is False


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
