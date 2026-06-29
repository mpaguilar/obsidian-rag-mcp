"""Tests for get_tasks server wrapper and filter parsing helpers."""

from unittest.mock import MagicMock, patch

import pytest

from obsidian_rag.mcp_server.handlers import TagFilterStrings, TaskDateFilterStrings
from obsidian_rag.mcp_server.models import PropertyFilter
from obsidian_rag.mcp_server.server import (
    _parse_date_filters,
    _parse_inline_filters,
    _parse_tag_filters,
    get_tasks,
)


@pytest.fixture
def mock_handler_response():
    """Return a consistent successful handler response."""
    return {
        "results": [],
        "total_count": 0,
        "has_more": False,
        "next_offset": None,
    }


@pytest.fixture
def patched_handler(mock_handler_response):
    """Patch _get_tasks_handler and yield the mock."""
    with (
        patch("obsidian_rag.mcp_server.handlers._get_tasks_handler") as mock_handler,
        patch("obsidian_rag.mcp_server.server._get_registry") as mock_registry,
    ):
        mock_registry_instance = MagicMock()
        mock_registry.return_value = mock_registry_instance
        mock_handler.return_value = mock_handler_response
        yield mock_handler


def test_parse_tag_filters_from_json_string():
    """JSON string is parsed into TagFilterStrings."""
    json_input = (
        '{"include_tags": ["work"], "exclude_tags": ["blocked"], "match_mode": "any"}'
    )
    result = _parse_tag_filters(json_input)
    assert isinstance(result, TagFilterStrings)
    assert result.include_tags == ["work"]
    assert result.exclude_tags == ["blocked"]
    assert result.match_mode == "any"


def test_parse_tag_filters_from_dict():
    """Dict input is converted to TagFilterStrings."""
    result = _parse_tag_filters({"include_tags": ["work"], "match_mode": "all"})
    assert isinstance(result, TagFilterStrings)
    assert result.include_tags == ["work"]
    assert result.match_mode == "all"


def test_parse_tag_filters_from_dataclass():
    """TagFilterStrings dataclass is returned unchanged."""
    tag_filters = TagFilterStrings(include_tags=["work"])
    result = _parse_tag_filters(tag_filters)
    assert result is tag_filters


def test_parse_tag_filters_from_none():
    """None returns default TagFilterStrings."""
    result = _parse_tag_filters(None)
    assert isinstance(result, TagFilterStrings)
    assert result.include_tags is None
    assert result.exclude_tags is None
    assert result.match_mode == "all"


def test_parse_tag_filters_from_empty_string():
    """Empty JSON string returns default TagFilterStrings."""
    result = _parse_tag_filters("")
    assert isinstance(result, TagFilterStrings)
    assert result.include_tags is None


def test_parse_date_filters_from_json_string():
    """JSON string is parsed into TaskDateFilterStrings."""
    json_input = (
        '{"due_after": "2026-01-01", "due_before": "2026-12-31", "match_mode": "any"}'
    )
    result = _parse_date_filters(json_input)
    assert isinstance(result, TaskDateFilterStrings)
    assert result.due_after == "2026-01-01"
    assert result.due_before == "2026-12-31"
    assert result.match_mode == "any"


def test_parse_date_filters_from_dict():
    """Dict input is converted to TaskDateFilterStrings."""
    result = _parse_date_filters({"scheduled_after": "2026-01-01", "match_mode": "all"})
    assert isinstance(result, TaskDateFilterStrings)
    assert result.scheduled_after == "2026-01-01"
    assert result.match_mode == "all"


def test_parse_date_filters_from_dataclass():
    """TaskDateFilterStrings dataclass is returned unchanged."""
    date_filters = TaskDateFilterStrings(due_after="2026-01-01")
    result = _parse_date_filters(date_filters)
    assert result is date_filters


def test_parse_date_filters_from_none():
    """None returns default TaskDateFilterStrings."""
    result = _parse_date_filters(None)
    assert isinstance(result, TaskDateFilterStrings)
    assert result.due_after is None
    assert result.match_mode == "all"


def test_parse_date_filters_from_empty_string():
    """Empty JSON string returns default TaskDateFilterStrings."""
    result = _parse_date_filters("")
    assert isinstance(result, TaskDateFilterStrings)
    assert result.due_after is None


def test_parse_inline_filters_from_json_string_list():
    """JSON string list is parsed into list[PropertyFilter]."""
    json_input = '[{"path": "vendor", "operator": "equals", "value": "Amazon"}]'
    result = _parse_inline_filters(json_input)
    assert isinstance(result, list)
    assert len(result) == 1
    assert isinstance(result[0], PropertyFilter)
    assert result[0].path == "vendor"
    assert result[0].operator == "equals"
    assert result[0].value == "Amazon"


def test_parse_inline_filters_from_json_string_dict():
    """JSON string dict is parsed into list[PropertyFilter] with one item."""
    json_input = '{"path": "status", "operator": "equals", "value": "active"}'
    result = _parse_inline_filters(json_input)
    assert isinstance(result, list)
    assert len(result) == 1
    assert isinstance(result[0], PropertyFilter)
    assert result[0].path == "status"
    assert result[0].operator == "equals"
    assert result[0].value == "active"


def test_parse_inline_filters_from_dict():
    """Dict input is converted to list[PropertyFilter] with one item."""
    result = _parse_inline_filters(
        {"path": "priority", "operator": "equals", "value": "high"}
    )
    assert isinstance(result, list)
    assert len(result) == 1
    assert isinstance(result[0], PropertyFilter)
    assert result[0].path == "priority"
    assert result[0].operator == "equals"
    assert result[0].value == "high"


def test_parse_inline_filters_from_property_filter_list():
    """list[PropertyFilter] is returned unchanged."""
    filters = [PropertyFilter(path="type", operator="equals", value="bug")]
    result = _parse_inline_filters(filters)
    assert result is filters


def test_parse_inline_filters_from_none():
    """None returns None."""
    result = _parse_inline_filters(None)
    assert result is None


def test_parse_inline_filters_from_empty_string():
    """Empty JSON string returns None."""
    result = _parse_inline_filters("")
    assert result is None


def test_parse_inline_filters_from_unexpected_type():
    """Unexpected type returns None defensively."""
    result = _parse_inline_filters(42)
    assert result is None


def test_get_tasks_passes_flat_parameters(patched_handler):
    """get_tasks passes flat keyword args to the handler."""
    mock_handler = patched_handler

    result = get_tasks(
        status=["not_completed"],
        tag_filters={"include_tags": ["work"], "match_mode": "all"},
        date_filters={"due_after": "2026-01-01"},
        priority=["high"],
        include_content=False,
        limit=10,
        offset=5,
    )

    assert result == {
        "results": [],
        "total_count": 0,
        "has_more": False,
        "next_offset": None,
    }
    mock_handler.assert_called_once()
    request = mock_handler.call_args.kwargs["request"]
    assert request.status == ["not_completed"]
    assert request.tag_filters.include_tags == ["work"]
    assert request.tag_filters.match_mode == "all"
    assert request.date_filters.due_after == "2026-01-01"
    assert request.priority == ["high"]
    assert request.include_content is False
    assert request.limit == 10
    assert request.offset == 5


def test_get_tasks_accepts_tag_filters_json_string(patched_handler):
    """get_tasks accepts tag_filters as a JSON string."""
    mock_handler = patched_handler

    result = get_tasks(
        status=["completed"],
        tag_filters='{"include_tags": ["work"], "match_mode": "any"}',
    )

    assert result["total_count"] == 0
    request = mock_handler.call_args.kwargs["request"]
    assert request.status == ["completed"]
    assert request.tag_filters.include_tags == ["work"]
    assert request.tag_filters.match_mode == "any"


def test_get_tasks_accepts_date_filters_json_string(patched_handler):
    """get_tasks accepts date_filters as a JSON string."""
    mock_handler = patched_handler

    result = get_tasks(
        date_filters='{"due_before": "2026-12-31", "match_mode": "all"}',
    )

    assert result["total_count"] == 0
    request = mock_handler.call_args.kwargs["request"]
    assert request.date_filters.due_before == "2026-12-31"
    assert request.date_filters.match_mode == "all"


def test_get_tasks_accepts_tag_filters_dataclass(patched_handler):
    """get_tasks accepts a TagFilterStrings dataclass directly."""
    mock_handler = patched_handler

    tag_filters = TagFilterStrings(include_tags=["urgent"], match_mode="all")
    result = get_tasks(tag_filters=tag_filters)

    assert result["total_count"] == 0
    request = mock_handler.call_args.kwargs["request"]
    assert request.tag_filters is tag_filters


def test_get_tasks_accepts_date_filters_dataclass(patched_handler):
    """get_tasks accepts a TaskDateFilterStrings dataclass directly."""
    mock_handler = patched_handler

    date_filters = TaskDateFilterStrings(due_after="2026-01-01")
    result = get_tasks(date_filters=date_filters)

    assert result["total_count"] == 0
    request = mock_handler.call_args.kwargs["request"]
    assert request.date_filters is date_filters


def test_get_tasks_default_filters(patched_handler):
    """get_tasks uses default filters when none are provided."""
    mock_handler = patched_handler

    result = get_tasks()

    assert result["total_count"] == 0
    request = mock_handler.call_args.kwargs["request"]
    assert request.status is None
    assert isinstance(request.tag_filters, TagFilterStrings)
    assert isinstance(request.date_filters, TaskDateFilterStrings)
    assert request.include_content is True
    assert request.limit == 20
    assert request.offset == 0


def test_get_tasks_include_content_false(patched_handler):
    """get_tasks passes include_content=False to the handler."""
    mock_handler = patched_handler

    get_tasks(include_content=False)

    request = mock_handler.call_args.kwargs["request"]
    assert request.include_content is False


def test_get_tasks_limit_and_offset(patched_handler):
    """get_tasks passes limit and offset to the handler."""
    mock_handler = patched_handler

    get_tasks(limit=50, offset=10)

    request = mock_handler.call_args.kwargs["request"]
    assert request.limit == 50
    assert request.offset == 10


def test_get_tasks_passes_inline_filters_dict(patched_handler):
    """get_tasks passes inline_filters dict to the handler."""
    mock_handler = patched_handler

    result = get_tasks(
        status=["not_completed"],
        inline_filters={"path": "vendor", "operator": "equals", "value": "Amazon"},
    )

    assert result["total_count"] == 0
    request = mock_handler.call_args.kwargs["request"]
    assert request.status == ["not_completed"]
    assert isinstance(request.inline_filters, list)
    assert len(request.inline_filters) == 1
    assert request.inline_filters[0].path == "vendor"
    assert request.inline_filters[0].operator == "equals"
    assert request.inline_filters[0].value == "Amazon"


def test_get_tasks_accepts_inline_filters_json_string(patched_handler):
    """get_tasks accepts inline_filters as a JSON string."""
    mock_handler = patched_handler

    result = get_tasks(
        inline_filters='[{"path": "priority", "operator": "equals", "value": "high"}]',
    )

    assert result["total_count"] == 0
    request = mock_handler.call_args.kwargs["request"]
    assert isinstance(request.inline_filters, list)
    assert len(request.inline_filters) == 1
    assert request.inline_filters[0].path == "priority"
    assert request.inline_filters[0].operator == "equals"
    assert request.inline_filters[0].value == "high"


def test_get_tasks_accepts_inline_filters_property_filter_list(patched_handler):
    """get_tasks accepts a list of PropertyFilter objects directly."""
    mock_handler = patched_handler

    inline_filters = [
        PropertyFilter(path="type", operator="equals", value="bug"),
        PropertyFilter(path="severity", operator="equals", value="critical"),
    ]
    result = get_tasks(inline_filters=inline_filters)

    assert result["total_count"] == 0
    request = mock_handler.call_args.kwargs["request"]
    assert request.inline_filters is inline_filters
    assert len(request.inline_filters) == 2


def test_get_tasks_default_inline_filters_is_none(patched_handler):
    """get_tasks uses default inline_filters=None when none provided."""
    mock_handler = patched_handler

    result = get_tasks()

    assert result["total_count"] == 0
    request = mock_handler.call_args.kwargs["request"]
    assert request.inline_filters is None
