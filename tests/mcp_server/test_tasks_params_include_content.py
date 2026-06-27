"""Tests for include_content field in task parameter dataclasses."""

from obsidian_rag.mcp_server.tools.tasks_params import (
    GetTasksFilterParams,
    GetTasksRequest,
)


def test_get_tasks_filter_params_default_include_content_true() -> None:
    """GetTasksFilterParams defaults include_content to True."""
    params = GetTasksFilterParams()
    assert params.include_content is True


def test_get_tasks_filter_params_include_content_false() -> None:
    """GetTasksFilterParams accepts include_content=False."""
    params = GetTasksFilterParams(include_content=False)
    assert params.include_content is False


def test_get_tasks_request_default_include_content_true() -> None:
    """GetTasksRequest defaults include_content to True."""
    request = GetTasksRequest()
    assert request.include_content is True


def test_get_tasks_request_include_content_false() -> None:
    """GetTasksRequest accepts include_content=False."""
    request = GetTasksRequest(include_content=False)
    assert request.include_content is False
