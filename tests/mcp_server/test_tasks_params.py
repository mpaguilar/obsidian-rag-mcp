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
        assert params.tags is None
        assert params.priority is None
        assert params.include_completed is True
        assert params.include_cancelled is False
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
            tags=["work", "urgent"],
            priority=["high", "highest"],
            include_completed=False,
            include_cancelled=True,
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
        assert params.tags == ["work", "urgent"]
        assert params.priority == ["high", "highest"]
        assert params.include_completed is False
        assert params.include_cancelled is True
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
