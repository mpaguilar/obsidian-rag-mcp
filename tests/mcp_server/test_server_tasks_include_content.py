"""Tests that get_tasks() server.py wrapper correctly threads include_content through to GetTasksRequest."""

from unittest.mock import MagicMock, patch

from obsidian_rag.mcp_server.handlers import GetTasksRequest, GetTasksToolInput


@patch("obsidian_rag.mcp_server.handlers._get_tasks_handler")
@patch("obsidian_rag.mcp_server.server._get_registry")
def test_get_tasks_wrapper_include_content_default_true(
    mock_get_registry: MagicMock,
    mock_handler: MagicMock,
) -> None:
    """Default GetTasksToolInput should pass include_content=True to GetTasksRequest."""
    mock_registry = MagicMock()
    mock_get_registry.return_value = mock_registry
    mock_handler.return_value = {"results": [], "total_count": 0}

    params = GetTasksToolInput()
    assert params.include_content is True

    from obsidian_rag.mcp_server.server import get_tasks

    get_tasks(params=params)

    call_args = mock_handler.call_args
    request = call_args.kwargs["request"]
    assert isinstance(request, GetTasksRequest)
    assert request.include_content is True


@patch("obsidian_rag.mcp_server.handlers._get_tasks_handler")
@patch("obsidian_rag.mcp_server.server._get_registry")
def test_get_tasks_wrapper_include_content_false(
    mock_get_registry: MagicMock,
    mock_handler: MagicMock,
) -> None:
    """Explicitly setting include_content=False should pass it to GetTasksRequest."""
    mock_registry = MagicMock()
    mock_get_registry.return_value = mock_registry
    mock_handler.return_value = {"results": [], "total_count": 0}

    params = GetTasksToolInput(include_content=False)
    assert params.include_content is False

    from obsidian_rag.mcp_server.server import get_tasks

    get_tasks(params=params)

    call_args = mock_handler.call_args
    request = call_args.kwargs["request"]
    assert isinstance(request, GetTasksRequest)
    assert request.include_content is False


@patch("obsidian_rag.mcp_server.handlers._get_tasks_handler")
@patch("obsidian_rag.mcp_server.server._get_registry")
def test_get_tasks_wrapper_include_content_via_json_string(
    mock_get_registry: MagicMock,
    mock_handler: MagicMock,
) -> None:
    """JSON string input with include_content=false should pass False to GetTasksRequest."""
    mock_registry = MagicMock()
    mock_get_registry.return_value = mock_registry
    mock_handler.return_value = {"results": [], "total_count": 0}

    json_params = '{"include_content": false}'

    from obsidian_rag.mcp_server.server import get_tasks

    get_tasks(params=json_params)

    call_args = mock_handler.call_args
    request = call_args.kwargs["request"]
    assert isinstance(request, GetTasksRequest)
    assert request.include_content is False
