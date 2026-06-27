"""Tests for get_tasks flat parameter handling."""

from unittest.mock import MagicMock, patch

import pytest

from obsidian_rag.mcp_server.handlers import TagFilterStrings, TaskDateFilterStrings


@pytest.fixture
def patched_handler():
    """Patch _get_tasks_handler and _get_registry."""
    with (
        patch("obsidian_rag.mcp_server.handlers._get_tasks_handler") as mock_handler,
        patch("obsidian_rag.mcp_server.server._get_registry") as mock_registry,
    ):
        mock_registry.return_value = MagicMock()
        mock_handler.return_value = {
            "results": [],
            "total_count": 0,
            "has_more": False,
            "next_offset": None,
        }
        yield mock_handler


def test_get_tasks_passes_status(patched_handler: MagicMock) -> None:
    """status list reaches the handler request."""
    from obsidian_rag.mcp_server.server import get_tasks

    get_tasks(status=["not_completed", "in_progress"])

    request = patched_handler.call_args.kwargs["request"]
    assert request.status == ["not_completed", "in_progress"]


def test_get_tasks_passes_tag_filters(patched_handler: MagicMock) -> None:
    """TagFilterStrings reaches the handler request."""
    from obsidian_rag.mcp_server.server import get_tasks

    tag_filters = TagFilterStrings(
        include_tags=["work", "urgent"],
        exclude_tags=["blocked"],
        match_mode="all",
    )
    get_tasks(tag_filters=tag_filters)

    request = patched_handler.call_args.kwargs["request"]
    assert request.tag_filters is tag_filters
    assert request.tag_filters.include_tags == ["work", "urgent"]
    assert request.tag_filters.exclude_tags == ["blocked"]


def test_get_tasks_passes_date_filters(patched_handler: MagicMock) -> None:
    """TaskDateFilterStrings reaches the handler request."""
    from obsidian_rag.mcp_server.server import get_tasks

    date_filters = TaskDateFilterStrings(
        due_after="2026-01-01",
        due_before="2026-12-31",
        match_mode="any",
    )
    get_tasks(date_filters=date_filters)

    request = patched_handler.call_args.kwargs["request"]
    assert request.date_filters is date_filters
    assert request.date_filters.due_after == "2026-01-01"
    assert request.date_filters.match_mode == "any"


def test_get_tasks_passes_priority(patched_handler: MagicMock) -> None:
    """priority list reaches the handler request."""
    from obsidian_rag.mcp_server.server import get_tasks

    get_tasks(priority=["high", "highest"])

    request = patched_handler.call_args.kwargs["request"]
    assert request.priority == ["high", "highest"]


def test_get_tasks_defaults(patched_handler: MagicMock) -> None:
    """No arguments produces default filters."""
    from obsidian_rag.mcp_server.server import get_tasks

    get_tasks()

    request = patched_handler.call_args.kwargs["request"]
    assert request.status is None
    assert isinstance(request.tag_filters, TagFilterStrings)
    assert isinstance(request.date_filters, TaskDateFilterStrings)
    assert request.priority is None
    assert request.include_content is True
    assert request.limit == 20
    assert request.offset == 0
