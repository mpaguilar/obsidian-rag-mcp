"""Tests for get_tasks flat parameter handling."""

from unittest.mock import MagicMock, patch

import pytest

from obsidian_rag.mcp_server.handlers import TagFilterStrings, TaskDateFilterStrings
from obsidian_rag.mcp_server.models import PropertyFilter


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
    assert request.inline_filters is None


def test_get_tasks_passes_inline_filters_list(patched_handler: MagicMock) -> None:
    """List of PropertyFilter objects reaches the handler request."""
    from obsidian_rag.mcp_server.server import get_tasks

    inline_filters = [
        PropertyFilter(path="vendor", operator="equals", value="Amazon"),
        PropertyFilter(path="status", operator="in", value=["active", "pending"]),
    ]
    get_tasks(inline_filters=inline_filters)

    request = patched_handler.call_args.kwargs["request"]
    assert request.inline_filters is inline_filters
    assert len(request.inline_filters) == 2
    assert request.inline_filters[0].path == "vendor"
    assert request.inline_filters[0].operator == "equals"
    assert request.inline_filters[0].value == "Amazon"
    assert request.inline_filters[1].path == "status"
    assert request.inline_filters[1].operator == "in"
    assert request.inline_filters[1].value == ["active", "pending"]


def test_get_tasks_passes_inline_filters_dict(patched_handler: MagicMock) -> None:
    """Dict input is parsed into a list of PropertyFilter."""
    from obsidian_rag.mcp_server.server import get_tasks

    get_tasks(
        inline_filters={"path": "vendor", "operator": "equals", "value": "Amazon"}
    )

    request = patched_handler.call_args.kwargs["request"]
    assert request.inline_filters is not None
    assert len(request.inline_filters) == 1
    assert request.inline_filters[0].path == "vendor"
    assert request.inline_filters[0].operator == "equals"
    assert request.inline_filters[0].value == "Amazon"


def test_get_tasks_passes_inline_filters_json_str(patched_handler: MagicMock) -> None:
    """JSON string input is parsed into a list of PropertyFilter."""
    from obsidian_rag.mcp_server.server import get_tasks

    get_tasks(
        inline_filters='{"path": "vendor", "operator": "equals", "value": "Amazon"}'
    )

    request = patched_handler.call_args.kwargs["request"]
    assert request.inline_filters is not None
    assert len(request.inline_filters) == 1
    assert request.inline_filters[0].path == "vendor"
    assert request.inline_filters[0].operator == "equals"
    assert request.inline_filters[0].value == "Amazon"


def test_get_tasks_passes_inline_filters_json_list_str(
    patched_handler: MagicMock,
) -> None:
    """JSON list string input is parsed into multiple PropertyFilter objects."""
    from obsidian_rag.mcp_server.server import get_tasks

    get_tasks(
        inline_filters='[{"path": "vendor", "operator": "equals", "value": "Amazon"}, {"path": "priority", "operator": "equals", "value": "high"}]'
    )

    request = patched_handler.call_args.kwargs["request"]
    assert request.inline_filters is not None
    assert len(request.inline_filters) == 2
    assert request.inline_filters[0].path == "vendor"
    assert request.inline_filters[0].operator == "equals"
    assert request.inline_filters[0].value == "Amazon"
    assert request.inline_filters[1].path == "priority"
    assert request.inline_filters[1].operator == "equals"
    assert request.inline_filters[1].value == "high"


def test_get_tasks_inline_filters_none(patched_handler: MagicMock) -> None:
    """Explicit None for inline_filters stays None in the request."""
    from obsidian_rag.mcp_server.server import get_tasks

    get_tasks(inline_filters=None)

    request = patched_handler.call_args.kwargs["request"]
    assert request.inline_filters is None


def test_get_tasks_inline_filters_invalid_json_scalar(
    patched_handler: MagicMock,
) -> None:
    """JSON string parsing to a scalar returns None for inline_filters."""
    from obsidian_rag.mcp_server.server import get_tasks

    get_tasks(inline_filters='"just_a_string"')

    request = patched_handler.call_args.kwargs["request"]
    assert request.inline_filters is None
