"""Tests for task parsing module."""

import pytest

from obsidian_rag.database.models import TaskPriority, TaskStatus
from obsidian_rag.parsing.tasks import (
    _map_checkbox_status,
    _map_priority,
    _obsidian_to_rrule,
    _parse_date,
    parse_task_line,
    parse_tasks_from_content,
)


class TestMapCheckboxStatus:
    """Test cases for _map_checkbox_status function."""

    def test_space_is_not_completed(self):
        """Test that space maps to not_completed."""
        assert _map_checkbox_status(" ") == TaskStatus.NOT_COMPLETED.value

    def test_x_is_completed(self):
        """Test that x maps to completed."""
        assert _map_checkbox_status("x") == TaskStatus.COMPLETED.value

    def test_slash_is_in_progress(self):
        """Test that / maps to in_progress."""
        assert _map_checkbox_status("/") == TaskStatus.IN_PROGRESS.value

    def test_dash_is_cancelled(self):
        """Test that - maps to cancelled."""
        assert _map_checkbox_status("-") == TaskStatus.CANCELLED.value

    def test_unknown_defaults_to_not_completed(self):
        """Test that unknown checkbox defaults to not_completed."""
        assert _map_checkbox_status("?") == TaskStatus.NOT_COMPLETED.value


class TestMapPriority:
    """Test cases for _map_priority function."""

    def test_highest(self):
        """Test highest priority mapping."""
        assert _map_priority("highest") == TaskPriority.HIGHEST.value

    def test_high(self):
        """Test high priority mapping."""
        assert _map_priority("high") == TaskPriority.HIGH.value

    def test_normal(self):
        """Test normal priority mapping."""
        assert _map_priority("normal") == TaskPriority.NORMAL.value

    def test_low(self):
        """Test low priority mapping."""
        assert _map_priority("low") == TaskPriority.LOW.value

    def test_lowest(self):
        """Test lowest priority mapping."""
        assert _map_priority("lowest") == TaskPriority.LOWEST.value

    def test_case_insensitive(self):
        """Test that priority mapping is case insensitive."""
        assert _map_priority("HIGH") == TaskPriority.HIGH.value
        assert _map_priority("High") == TaskPriority.HIGH.value

    def test_unknown_defaults_to_normal(self):
        """Test that unknown priority defaults to normal."""
        assert _map_priority("unknown") == TaskPriority.NORMAL.value


class TestParseDate:
    """Test cases for _parse_date function."""

    def test_iso_format(self):
        """Test parsing ISO format date."""
        result = _parse_date("2024-03-15")
        assert result is not None
        assert result.year == 2024
        assert result.month == 3
        assert result.day == 15

    def test_slash_format(self):
        """Test parsing slash-separated date."""
        result = _parse_date("2024/03/15")
        assert result is not None
        assert result.year == 2024
        assert result.month == 3
        assert result.day == 15

    def test_invalid_date_returns_none(self):
        """Test that invalid date returns None."""
        assert _parse_date("not-a-date") is None
        assert _parse_date("2024-13-45") is None


class TestObsidianToRrule:
    """Test cases for _obsidian_to_rrule function."""

    def test_every_day(self):
        """Test 'every day' pattern."""
        result = _obsidian_to_rrule("every day")
        assert result == "FREQ=DAILY"

    def test_every_week(self):
        """Test 'every week' pattern."""
        result = _obsidian_to_rrule("every week")
        assert result == "FREQ=WEEKLY"

    def test_every_month(self):
        """Test 'every month' pattern."""
        result = _obsidian_to_rrule("every month")
        assert result == "FREQ=MONTHLY"

    def test_every_year(self):
        """Test 'every year' pattern."""
        result = _obsidian_to_rrule("every year")
        assert result == "FREQ=YEARLY"

    def test_every_n_days(self):
        """Test 'every N days' pattern."""
        result = _obsidian_to_rrule("every 3 days")
        assert result == "FREQ=DAILY;INTERVAL=3"

    def test_every_n_weeks(self):
        """Test 'every N weeks' pattern."""
        result = _obsidian_to_rrule("every 2 weeks")
        assert result == "FREQ=WEEKLY;INTERVAL=2"

    def test_unknown_pattern_returns_none(self):
        """Test that unknown patterns return None."""
        assert _obsidian_to_rrule("sometimes") is None


class TestParseTaskLine:
    """Test cases for parse_task_line function."""

    def test_parse_simple_task(self):
        """Test parsing a simple task."""
        line = "- [ ] A simple task"
        result = parse_task_line(line)

        assert result is not None
        assert result.status == TaskStatus.NOT_COMPLETED.value
        assert result.description == "A simple task"

    def test_parse_completed_task(self):
        """Test parsing a completed task."""
        line = "- [x] A completed task"
        result = parse_task_line(line)

        assert result is not None
        assert result.status == TaskStatus.COMPLETED.value

    def test_parse_task_with_tags(self):
        """Test parsing task with inline tags."""
        line = "- [ ] Task with #tag1 and #tag2"
        result = parse_task_line(line)

        assert result is not None
        assert result.tags == ["tag1", "tag2"]
        assert "tag1" not in result.description
        assert "tag2" not in result.description

    def test_parse_task_with_priority(self):
        """Test parsing task with priority."""
        line = "- [ ] Important task [priority:: high]"
        result = parse_task_line(line)

        assert result is not None
        assert result.priority == TaskPriority.HIGH.value

    def test_parse_task_with_due_date(self):
        """Test parsing task with due date."""
        line = "- [ ] Task with due date [due:: 2024-03-15]"
        result = parse_task_line(line)

        assert result is not None
        assert result.due is not None
        assert result.due.year == 2024
        assert result.due.month == 3
        assert result.due.day == 15

    def test_non_task_line_returns_none(self):
        """Test that non-task lines return None."""
        line = "Just some regular text"
        result = parse_task_line(line)

        assert result is None

    def test_parse_task_with_repeat(self):
        """Test parsing task with recurrence."""
        line = "- [ ] Daily task [repeat:: every day]"
        result = parse_task_line(line)

        assert result is not None
        assert result.repeat == "FREQ=DAILY"

    def test_parse_all_status_types(self):
        """Test parsing all four status types."""
        result1 = parse_task_line("- [ ] Task")
        assert result1 is not None
        assert result1.status == TaskStatus.NOT_COMPLETED.value

        result2 = parse_task_line("- [x] Task")
        assert result2 is not None
        assert result2.status == TaskStatus.COMPLETED.value

        result3 = parse_task_line("- [/] Task")
        assert result3 is not None
        assert result3.status == TaskStatus.IN_PROGRESS.value

        result4 = parse_task_line("- [-] Task")
        assert result4 is not None
        assert result4.status == TaskStatus.CANCELLED.value


class TestRruleValidation:
    """Test cases for rrule validation."""

    def test_validate_rrule_valid(self):
        """Test validating a valid rrule pattern."""
        from obsidian_rag.parsing.tasks import _validate_rrule

        assert _validate_rrule("FREQ=DAILY") is True
        assert _validate_rrule("FREQ=WEEKLY;INTERVAL=2") is True

    def test_validate_rrule_invalid(self):
        """Test validating an invalid rrule pattern."""
        from obsidian_rag.parsing.tasks import _validate_rrule

        assert _validate_rrule("invalid") is False
        assert _validate_rrule("FREQ=INVALID") is False


class TestProcessRepeatValue:
    """Test cases for repeat value processing."""

    def test_process_repeat_with_valid_rrule(self):
        """Test processing repeat value that produces valid rrule."""
        from obsidian_rag.parsing.tasks import _process_repeat_value

        standard_fields: dict = {}
        _process_repeat_value("every day", standard_fields)

        assert standard_fields["repeat"] == "FREQ=DAILY"

    def test_process_repeat_with_invalid_rrule(self):
        """Test processing repeat value that produces invalid rrule."""
        from obsidian_rag.parsing.tasks import _process_repeat_value

        standard_fields: dict = {}
        _process_repeat_value("invalid pattern", standard_fields)

        # Falls back to original value
        assert standard_fields["repeat"] == "invalid pattern"


class TestProcessStandardField:
    """Test cases for _process_standard_field function."""

    def test_process_date_fields(self):
        """Test processing date standard fields."""
        from obsidian_rag.parsing.tasks import _process_standard_field

        standard_fields: dict = {}
        result = _process_standard_field("due", "2024-03-15", standard_fields)

        assert result is True
        assert standard_fields["due"] is not None

    def test_process_priority_field(self):
        """Test processing priority field."""
        from obsidian_rag.parsing.tasks import _process_standard_field

        standard_fields: dict = {}
        result = _process_standard_field("priority", "high", standard_fields)

        assert result is True
        assert standard_fields["priority"] == "high"

    def test_process_repeat_field(self):
        """Test processing repeat field."""
        from obsidian_rag.parsing.tasks import _process_standard_field

        standard_fields: dict = {}
        result = _process_standard_field("repeat", "every day", standard_fields)

        assert result is True
        assert standard_fields["repeat"] == "FREQ=DAILY"

    def test_process_unknown_field_returns_false(self):
        """Test that unknown fields return False."""
        from obsidian_rag.parsing.tasks import _process_standard_field

        standard_fields: dict = {}
        result = _process_standard_field("unknown", "value", standard_fields)

        assert result is False


class TestProcessMetadataKey:
    """Test cases for _process_metadata_key function."""

    def test_process_standard_key(self):
        """Test processing a standard metadata key."""
        from obsidian_rag.parsing.tasks import _process_metadata_key

        standard_fields: dict = {}
        custom_metadata: dict = {}
        _process_metadata_key("priority", "high", standard_fields, custom_metadata)

        assert standard_fields["priority"] == "high"
        assert "priority" not in custom_metadata

    def test_process_custom_key(self):
        """Test processing a custom metadata key."""
        from obsidian_rag.parsing.tasks import _process_metadata_key

        standard_fields: dict = {}
        custom_metadata: dict = {}
        _process_metadata_key("custom_key", "value", standard_fields, custom_metadata)

        assert custom_metadata["custom_key"] == "value"


class TestParseTasksFromContent:
    """Test cases for parse_tasks_from_content function."""

    def test_parse_multiple_tasks(self):
        """Test parsing multiple tasks from content."""
        content = """# Tasks

- [ ] First task
- [x] Second task
- [ ] Third task

Some other text.

- [/] Fourth task
"""
        results = parse_tasks_from_content(content)

        assert len(results) == 4
        assert results[0][0] == 3  # Line 3
        assert results[0][1].description == "First task"
        assert results[1][1].description == "Second task"

    def test_empty_content_returns_empty_list(self):
        """Test that empty content returns empty list."""
        results = parse_tasks_from_content("")
        assert results == []

    def test_no_tasks_returns_empty_list(self):
        """Test that content without tasks returns empty list."""
        content = "Just some text\nMore text"
        results = parse_tasks_from_content(content)
        assert results == []

    def test_respects_max_tasks_limit(self, caplog):
        """Test that max tasks limit is respected."""
        # Create content with more than 10000 tasks
        lines = [f"- [ ] Task {i}" for i in range(10005)]
        content = "\n".join(lines)

        with caplog.at_level("WARNING"):
            results = parse_tasks_from_content(content)

        assert len(results) == 10000
        assert "Reached maximum task limit" in caplog.text
