"""Tests for get_tasks server wrapper and filter parsing helpers."""

from unittest.mock import MagicMock, patch

import pytest

from obsidian_rag.mcp_server.handlers import TagFilterStrings, TaskDateFilterStrings
from obsidian_rag.mcp_server.server import (
    _parse_date_filters,
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
