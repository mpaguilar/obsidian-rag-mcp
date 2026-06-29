"""Tests for GetTasksFilterParams inline_filters field."""

from obsidian_rag.mcp_server.models import PropertyFilter
from obsidian_rag.mcp_server.tools.tasks_params import GetTasksFilterParams


def test_get_tasks_filter_params_has_inline_filters() -> None:
    """Field exists."""
    params = GetTasksFilterParams()
    assert hasattr(params, "inline_filters")


def test_get_tasks_filter_params_inline_filters_default_none() -> None:
    """Default is None."""
    params = GetTasksFilterParams()
    assert params.inline_filters is None


def test_get_tasks_filter_params_inline_filters_with_list() -> None:
    """Can set list of PropertyFilter."""
    filter_value = PropertyFilter(path="status", operator="equals", value="completed")
    params = GetTasksFilterParams(inline_filters=[filter_value])
    assert params.inline_filters == [filter_value]
