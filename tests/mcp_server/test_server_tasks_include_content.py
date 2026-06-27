"""Tests that get_tasks() server wrapper threads include_content to GetTasksRequest."""

from unittest.mock import MagicMock, patch

from obsidian_rag.mcp_server.handlers import GetTasksRequest


@patch("obsidian_rag.mcp_server.handlers._get_tasks_handler")
@patch("obsidian_rag.mcp_server.server._get_registry")
def test_get_tasks_wrapper_include_content_default_true(
    mock_get_registry: MagicMock,
    mock_handler: MagicMock,
) -> None:
    """Default include_content should be True in GetTasksRequest."""
    mock_registry = MagicMock()
    mock_get_registry.return_value = mock_registry
    mock_handler.return_value = {"results": [], "total_count": 0}

    from obsidian_rag.mcp_server.server import get_tasks

    get_tasks()

    request = mock_handler.call_args.kwargs["request"]
    assert isinstance(request, GetTasksRequest)
    assert request.include_content is True


@patch("obsidian_rag.mcp_server.handlers._get_tasks_handler")
@patch("obsidian_rag.mcp_server.server._get_registry")
def test_get_tasks_wrapper_include_content_false(
    mock_get_registry: MagicMock,
    mock_handler: MagicMock,
) -> None:
    """Explicit include_content=False should reach GetTasksRequest."""
    mock_registry = MagicMock()
    mock_get_registry.return_value = mock_registry
    mock_handler.return_value = {"results": [], "total_count": 0}

    from obsidian_rag.mcp_server.server import get_tasks

    get_tasks(include_content=False)

    request = mock_handler.call_args.kwargs["request"]
    assert isinstance(request, GetTasksRequest)
    assert request.include_content is False


@patch("obsidian_rag.mcp_server.handlers._get_tasks_handler")
@patch("obsidian_rag.mcp_server.server._get_registry")
def test_get_tasks_wrapper_include_content_via_json_filter(
    mock_get_registry: MagicMock,
    mock_handler: MagicMock,
) -> None:
    """JSON string filter with include_content=false should pass False to GetTasksRequest."""
    mock_registry = MagicMock()
    mock_get_registry.return_value = mock_registry
    mock_handler.return_value = {"results": [], "total_count": 0}

    from obsidian_rag.mcp_server.server import get_tasks

    get_tasks(
        tag_filters='{"include_tags": ["work"]}',
        include_content=False,
    )

    request = mock_handler.call_args.kwargs["request"]
    assert isinstance(request, GetTasksRequest)
    assert request.tag_filters.include_tags == ["work"]
    assert request.include_content is False
