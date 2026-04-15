"""Tests for tasks_params module."""

from datetime import date

from obsidian_rag.mcp_server.tools.tasks_params import GetTasksFilterParams


class TestGetTasksFilterParams:
    """Tests for GetTasksFilterParams dataclass."""

    def test_default_values(self):
        """Test default parameter values."""
        params = GetTasksFilterParams()

        assert params.status is None
        assert params.due_after is None
        assert params.due_before is None
        assert params.scheduled_after is None
        assert params.scheduled_before is None
        assert params.completion_after is None
        assert params.completion_before is None
        assert params.priority is None
        assert params.limit == 20
        assert params.offset == 0

    def test_custom_values(self):
        """Test setting custom values."""
        params = GetTasksFilterParams(
            status=["not_completed", "in_progress"],
            due_after=date(2026, 1, 1),
            due_before=date(2026, 12, 31),
            scheduled_after=date(2026, 1, 1),
            scheduled_before=date(2026, 6, 30),
            completion_after=date(2026, 1, 1),
            completion_before=date(2026, 3, 31),
            priority=["high", "highest"],
            limit=50,
            offset=10,
        )

        assert params.status == ["not_completed", "in_progress"]
        assert params.due_after == date(2026, 1, 1)
        assert params.due_before == date(2026, 12, 31)
        assert params.scheduled_after == date(2026, 1, 1)
        assert params.scheduled_before == date(2026, 6, 30)
        assert params.completion_after == date(2026, 1, 1)
        assert params.completion_before == date(2026, 3, 31)
        assert params.priority == ["high", "highest"]
        assert params.limit == 50
        assert params.offset == 10

    def test_default_date_match_mode(self):
        """Test that date_match_mode defaults to 'all'."""
        params = GetTasksFilterParams()
        assert params.date_match_mode == "all"

    def test_explicit_all_mode(self):
        """Test explicit 'all' date_match_mode."""
        params = GetTasksFilterParams(date_match_mode="all")
        assert params.date_match_mode == "all"

    def test_any_mode(self):
        """Test 'any' date_match_mode."""
        params = GetTasksFilterParams(date_match_mode="any")
        assert params.date_match_mode == "any"

    def test_date_match_mode_with_date_filters(self):
        """Test GetTasksFilterParams with date filters and match mode."""
        params = GetTasksFilterParams(
            due_after=date(2026, 3, 1),
            due_before=date(2026, 3, 31),
            scheduled_after=date(2026, 3, 15),
            date_match_mode="any",
        )
        assert params.due_after == date(2026, 3, 1)
        assert params.due_before == date(2026, 3, 31)
        assert params.scheduled_after == date(2026, 3, 15)
        assert params.date_match_mode == "any"

    def test_default_tag_filter_values(self):
        """Test default values for new tag filter fields."""
        params = GetTasksFilterParams()
        assert params.include_tags is None
        assert params.exclude_tags is None
        assert params.tag_match_mode == "all"

    def test_include_tags_all_mode(self):
        """Test include_tags with 'all' match mode."""
        params = GetTasksFilterParams(
            include_tags=["work", "urgent"],
            tag_match_mode="all",
        )
        assert params.include_tags == ["work", "urgent"]
        assert params.tag_match_mode == "all"

    def test_include_tags_any_mode(self):
        """Test include_tags with 'any' match mode."""
        params = GetTasksFilterParams(
            include_tags=["work", "personal"],
            tag_match_mode="any",
        )
        assert params.include_tags == ["work", "personal"]
        assert params.tag_match_mode == "any"

    def test_exclude_tags(self):
        """Test exclude_tags field."""
        params = GetTasksFilterParams(exclude_tags=["blocked", "waiting"])
        assert params.exclude_tags == ["blocked", "waiting"]

    def test_combined_tag_filters(self):
        """Test combining include_tags, exclude_tags, and tag_match_mode."""
        params = GetTasksFilterParams(
            include_tags=["work"],
            exclude_tags=["blocked"],
            tag_match_mode="all",
        )
        assert params.include_tags == ["work"]
        assert params.exclude_tags == ["blocked"]
        assert params.tag_match_mode == "all"


class TestGetTasksRequest:
    """Tests for GetTasksRequest dataclass."""

    def test_default_values(self):
        """Test default parameter values."""
        from obsidian_rag.mcp_server.tools.tasks_params import GetTasksRequest

        request = GetTasksRequest()

        assert request.status is None
        assert request.tag_filters is None
        assert request.date_filters is None
        assert request.priority is None
        assert request.limit == 20
        assert request.offset == 0

    def test_with_tag_filters(self):
        """Test with tag_filters parameter."""
        from obsidian_rag.mcp_server.handlers import TagFilterStrings
        from obsidian_rag.mcp_server.tools.tasks_params import GetTasksRequest

        tag_filters = TagFilterStrings(
            include_tags=["work", "urgent"],
            exclude_tags=["blocked"],
            match_mode="all",
        )

        request = GetTasksRequest(
            status=["not_completed"],
            tag_filters=tag_filters,
        )

        assert request.status == ["not_completed"]
        assert request.tag_filters is not None
        assert request.tag_filters.include_tags == ["work", "urgent"]
        assert request.tag_filters.exclude_tags == ["blocked"]
        assert request.tag_filters.match_mode == "all"

    def test_with_date_filters(self):
        """Test with date_filters parameter."""
        from obsidian_rag.mcp_server.handlers import TaskDateFilterStrings
        from obsidian_rag.mcp_server.tools.tasks_params import GetTasksRequest

        date_filters = TaskDateFilterStrings(
            due_after="2026-01-01",
            due_before="2026-12-31",
            match_mode="any",
        )

        request = GetTasksRequest(
            date_filters=date_filters,
        )

        assert request.date_filters is not None
        assert request.date_filters.due_after == "2026-01-01"
        assert request.date_filters.match_mode == "any"

    def test_with_both_filter_types(self):
        """Test with both tag and date filters."""
        from obsidian_rag.mcp_server.handlers import (
            TagFilterStrings,
            TaskDateFilterStrings,
        )
        from obsidian_rag.mcp_server.tools.tasks_params import GetTasksRequest

        tag_filters = TagFilterStrings(
            include_tags=["work"],
            match_mode="all",
        )
        date_filters = TaskDateFilterStrings(
            due_before="2026-03-31",
            match_mode="all",
        )

        request = GetTasksRequest(
            tag_filters=tag_filters,
            date_filters=date_filters,
        )

        assert request.tag_filters is not None
        assert request.tag_filters.include_tags == ["work"]
        assert request.date_filters is not None
        assert request.date_filters.due_before == "2026-03-31"
