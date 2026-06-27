"""Tests for include_content parameter in handler dataclasses and TypedDict."""

from unittest.mock import MagicMock, patch

from obsidian_rag.mcp_server.handlers import (
    DocumentTagParams,
    GetDocumentHandlerParams,
    ListDocumentsHandlerParams,
    _get_tasks_handler,
)
from obsidian_rag.mcp_server.tools.tasks_params import GetTasksRequest


def test_document_tag_params_has_include_content():
    """DocumentTagParams TypedDict should include include_content key."""
    params: DocumentTagParams = {
        "include_tags": ["work"],
        "exclude_tags": ["blocked"],
        "match_mode": "all",
        "vault_name": "Personal",
        "limit": 20,
        "offset": 0,
        "include_content": False,
    }
    assert params["include_content"] is False


def test_get_document_handler_params_default_include_content_true():
    """GetDocumentHandlerParams should default include_content to True."""
    db_manager = MagicMock()
    params = GetDocumentHandlerParams(db_manager=db_manager)
    assert params.include_content is True


def test_get_document_handler_params_include_content_false():
    """GetDocumentHandlerParams should accept include_content=False."""
    db_manager = MagicMock()
    params = GetDocumentHandlerParams(db_manager=db_manager, include_content=False)
    assert params.include_content is False


def test_list_documents_handler_params_default_include_content_true():
    """ListDocumentsHandlerParams should default include_content to True."""
    db_manager = MagicMock()
    params = ListDocumentsHandlerParams(db_manager=db_manager)
    assert params.include_content is True


def test_list_documents_handler_params_include_content_false():
    """ListDocumentsHandlerParams should accept include_content=False."""
    db_manager = MagicMock()
    params = ListDocumentsHandlerParams(db_manager=db_manager, include_content=False)
    assert params.include_content is False


def test_get_tasks_request_default_include_content_true():
    """GetTasksRequest should default include_content to True."""
    request = GetTasksRequest()
    assert request.include_content is True


def test_get_tasks_request_accepts_include_content_false():
    """GetTasksRequest should accept include_content=False."""
    request = GetTasksRequest(include_content=False)
    assert request.include_content is False


def test_get_tasks_handler_passes_include_content_through_filters():
    """_get_tasks_handler should pass include_content through to GetTasksFilterParams."""
    mock_db_manager = MagicMock()
    mock_session = MagicMock()
    mock_db_manager.get_session.return_value.__enter__.return_value = mock_session
    mock_db_manager.get_session.return_value.__exit__.return_value = False

    with patch("obsidian_rag.mcp_server.handlers.get_tasks_tool") as mock_get_tasks:
        mock_get_tasks.return_value.model_dump.return_value = {
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

        call_args = mock_get_tasks.call_args
        filters = call_args.kwargs["filters"]
        assert filters.include_content is False
        assert filters.status == ["not_completed"]
        assert filters.limit == 10
        assert filters.offset == 5
