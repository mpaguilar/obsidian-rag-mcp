"""Tests for task handler functions and task-related validation."""

import json
from unittest.mock import MagicMock, patch

import pytest
from pydantic import TypeAdapter, ValidationError

from obsidian_rag.mcp_server.handlers import (
    AnnotatedGetTasksInput,
    GetTasksRequest,
    GetTasksToolInput,
    TagFilterStrings,
    TaskDateFilterStrings,
    _get_tasks_handler,
)


class TestGetTasksHandler:
    """Tests for _get_tasks_handler function."""

    @patch("obsidian_rag.mcp_server.handlers.get_tasks_tool")
    @patch("obsidian_rag.mcp_server.handlers.parse_iso_date")
    def test_handler_parses_dates(self, mock_parse_date, mock_get_tasks):
        """Test that handler parses date parameters."""
        from datetime import date

        mock_db_manager = MagicMock()
        mock_session = MagicMock()
        mock_db_manager.get_session.return_value.__enter__ = MagicMock(
            return_value=mock_session
        )
        mock_db_manager.get_session.return_value.__exit__ = MagicMock(
            return_value=False
        )

        mock_parse_date.side_effect = [
            date(2026, 1, 1),  # due_after
            date(2026, 12, 31),  # due_before
            None,  # scheduled_after
            None,  # scheduled_before
            None,  # completion_after
            None,  # completion_before
        ]

        mock_get_tasks.return_value = MagicMock(model_dump=lambda: {"results": []})

        date_filters = TaskDateFilterStrings(
            due_after="2026-01-01",
            due_before="2026-12-31",
        )

        request = GetTasksRequest(
            date_filters=date_filters,
        )

        _get_tasks_handler(
            db_manager=mock_db_manager,
            request=request,
        )

        assert mock_parse_date.call_count == 6
        mock_get_tasks.assert_called_once()

    @patch("obsidian_rag.mcp_server.handlers.get_tasks_tool")
    def test_handler_with_no_dates(self, mock_get_tasks):
        """Test handler with no date parameters."""
        mock_db_manager = MagicMock()
        mock_session = MagicMock()
        mock_db_manager.get_session.return_value.__enter__ = MagicMock(
            return_value=mock_session
        )
        mock_db_manager.get_session.return_value.__exit__ = MagicMock(
            return_value=False
        )

        mock_get_tasks.return_value = MagicMock(model_dump=lambda: {"results": []})

        request = GetTasksRequest(
            status=["not_completed"],
            limit=10,
        )

        _get_tasks_handler(
            db_manager=mock_db_manager,
            request=request,
        )

        mock_get_tasks.assert_called_once()
        call_args = mock_get_tasks.call_args
        assert call_args.kwargs["filters"].status == ["not_completed"]
        assert call_args.kwargs["filters"].limit == 10


class TestTaskDateFilterStrings:
    """Tests for TaskDateFilterStrings dataclass."""

    def test_default_match_mode(self):
        """Test that match_mode defaults to 'all'."""
        filters = TaskDateFilterStrings()
        assert filters.match_mode == "all"

    def test_explicit_all_mode(self):
        """Test explicit 'all' mode."""
        filters = TaskDateFilterStrings(match_mode="all")
        assert filters.match_mode == "all"

    def test_any_mode(self):
        """Test 'any' mode."""
        filters = TaskDateFilterStrings(match_mode="any")
        assert filters.match_mode == "any"

    def test_with_date_strings(self):
        """Test with date strings and match mode."""
        filters = TaskDateFilterStrings(
            due_after="2026-03-01",
            due_before="2026-03-31",
            match_mode="any",
        )
        assert filters.due_after == "2026-03-01"
        assert filters.due_before == "2026-03-31"
        assert filters.match_mode == "any"


class TestGetTasksHandlerDateMatchMode:
    """Tests for _get_tasks_handler with date match_mode."""

    def test_handler_passes_match_mode_all(self):
        """Test that handler passes 'all' match_mode to filter params."""
        mock_db_manager = MagicMock()
        mock_session = MagicMock()
        mock_db_manager.get_session.return_value.__enter__.return_value = mock_session
        mock_db_manager.get_session.return_value.__exit__.return_value = False

        date_filters = TaskDateFilterStrings(
            due_after="2026-03-01",
            match_mode="all",
        )

        with patch("obsidian_rag.mcp_server.handlers.get_tasks_tool") as mock_get_tasks:
            mock_get_tasks.return_value.model_dump.return_value = {
                "results": [],
                "total_count": 0,
                "has_more": False,
                "next_offset": None,
            }

            request = GetTasksRequest(
                date_filters=date_filters,
            )

            _get_tasks_handler(
                db_manager=mock_db_manager,
                request=request,
            )

            # Verify get_tasks was called with correct match_mode
            call_args = mock_get_tasks.call_args
            filter_params = call_args.kwargs["filters"]
            assert filter_params.date_match_mode == "all"

    def test_handler_passes_match_mode_any(self):
        """Test that handler passes 'any' match_mode to filter params."""
        mock_db_manager = MagicMock()
        mock_session = MagicMock()
        mock_db_manager.get_session.return_value.__enter__.return_value = mock_session
        mock_db_manager.get_session.return_value.__exit__.return_value = False

        date_filters = TaskDateFilterStrings(
            due_after="2026-03-01",
            scheduled_before="2026-03-31",
            match_mode="any",
        )

        with patch("obsidian_rag.mcp_server.handlers.get_tasks_tool") as mock_get_tasks:
            mock_get_tasks.return_value.model_dump.return_value = {
                "results": [],
                "total_count": 0,
                "has_more": False,
                "next_offset": None,
            }

            request = GetTasksRequest(
                date_filters=date_filters,
            )

            _get_tasks_handler(
                db_manager=mock_db_manager,
                request=request,
            )

            # Verify get_tasks was called with correct match_mode
            call_args = mock_get_tasks.call_args
            filter_params = call_args.kwargs["filters"]
            assert filter_params.date_match_mode == "any"

    def test_handler_defaults_to_all_mode(self):
        """Test that handler defaults to 'all' mode when not specified."""
        mock_db_manager = MagicMock()
        mock_session = MagicMock()
        mock_db_manager.get_session.return_value.__enter__.return_value = mock_session
        mock_db_manager.get_session.return_value.__exit__.return_value = False

        # No date_filters provided
        with patch("obsidian_rag.mcp_server.handlers.get_tasks_tool") as mock_get_tasks:
            mock_get_tasks.return_value.model_dump.return_value = {
                "results": [],
                "total_count": 0,
                "has_more": False,
                "next_offset": None,
            }

            request = GetTasksRequest()

            _get_tasks_handler(
                db_manager=mock_db_manager,
                request=request,
            )

            # Verify get_tasks was called with default 'all' mode
            call_args = mock_get_tasks.call_args
            filter_params = call_args.kwargs["filters"]
            assert filter_params.date_match_mode == "all"


class TestTagFilterStrings:
    """Tests for TagFilterStrings dataclass."""

    def test_default_values(self):
        """Test default parameter values."""
        filters = TagFilterStrings()
        assert filters.include_tags is None
        assert filters.exclude_tags is None
        assert filters.match_mode == "all"

    def test_explicit_all_mode(self):
        """Test explicit 'all' mode."""
        filters = TagFilterStrings(match_mode="all")
        assert filters.match_mode == "all"

    def test_any_mode(self):
        """Test 'any' mode."""
        filters = TagFilterStrings(match_mode="any")
        assert filters.match_mode == "any"

    def test_with_include_tags(self):
        """Test with include_tags."""
        filters = TagFilterStrings(
            include_tags=["work", "urgent"],
            match_mode="all",
        )
        assert filters.include_tags == ["work", "urgent"]
        assert filters.match_mode == "all"

    def test_with_exclude_tags(self):
        """Test with exclude_tags."""
        filters = TagFilterStrings(
            exclude_tags=["blocked"],
            match_mode="any",
        )
        assert filters.exclude_tags == ["blocked"]

    def test_with_both_tag_types(self):
        """Test with both include and exclude tags."""
        filters = TagFilterStrings(
            include_tags=["work"],
            exclude_tags=["blocked"],
            match_mode="all",
        )
        assert filters.include_tags == ["work"]
        assert filters.exclude_tags == ["blocked"]
        assert filters.match_mode == "all"


class TestGetTasksHandlerAdditional:
    """Tests for _get_tasks_handler function (additional tests)."""

    @patch("obsidian_rag.mcp_server.handlers.get_tasks_tool")
    @patch("obsidian_rag.mcp_server.handlers.parse_iso_date")
    def test_handler_parses_dates(self, mock_parse_date, mock_get_tasks):
        """Test that handler parses date parameters."""
        from datetime import date

        mock_db_manager = MagicMock()
        mock_session = MagicMock()
        mock_db_manager.get_session.return_value.__enter__ = MagicMock(
            return_value=mock_session
        )
        mock_db_manager.get_session.return_value.__exit__ = MagicMock(
            return_value=False
        )

        mock_parse_date.side_effect = [
            date(2026, 1, 1),  # due_after
            date(2026, 12, 31),  # due_before
            None,  # scheduled_after
            None,  # scheduled_before
            None,  # completion_after
            None,  # completion_before
        ]

        mock_get_tasks.return_value = MagicMock(model_dump=lambda: {"results": []})

        date_filters = TaskDateFilterStrings(
            due_after="2026-01-01",
            due_before="2026-12-31",
        )

        request = GetTasksRequest(
            date_filters=date_filters,
        )

        _get_tasks_handler(
            db_manager=mock_db_manager,
            request=request,
        )

        assert mock_parse_date.call_count == 6
        mock_get_tasks.assert_called_once()

    @patch("obsidian_rag.mcp_server.handlers.get_tasks_tool")
    def test_handler_with_no_dates(self, mock_get_tasks):
        """Test handler with no date parameters."""
        mock_db_manager = MagicMock()
        mock_session = MagicMock()
        mock_db_manager.get_session.return_value.__enter__ = MagicMock(
            return_value=mock_session
        )
        mock_db_manager.get_session.return_value.__exit__ = MagicMock(
            return_value=False
        )

        mock_get_tasks.return_value = MagicMock(model_dump=lambda: {"results": []})

        request = GetTasksRequest(
            status=["not_completed"],
            limit=10,
        )

        _get_tasks_handler(
            db_manager=mock_db_manager,
            request=request,
        )

        mock_get_tasks.assert_called_once()
        call_args = mock_get_tasks.call_args
        assert call_args.kwargs["filters"].status == ["not_completed"]
        assert call_args.kwargs["filters"].limit == 10

    @patch("obsidian_rag.mcp_server.handlers.get_tasks_tool")
    def test_handler_with_tag_filters(self, mock_get_tasks):
        """Test handler with tag_filters parameter."""
        mock_db_manager = MagicMock()
        mock_session = MagicMock()
        mock_db_manager.get_session.return_value.__enter__ = MagicMock(
            return_value=mock_session
        )
        mock_db_manager.get_session.return_value.__exit__ = MagicMock(
            return_value=False
        )

        mock_get_tasks.return_value = MagicMock(model_dump=lambda: {"results": []})

        tag_filters = TagFilterStrings(
            include_tags=["work", "urgent"],
            exclude_tags=["blocked"],
            match_mode="all",
        )

        request = GetTasksRequest(
            tag_filters=tag_filters,
            status=["not_completed"],
        )

        _get_tasks_handler(
            db_manager=mock_db_manager,
            request=request,
        )

        mock_get_tasks.assert_called_once()
        call_args = mock_get_tasks.call_args
        assert call_args.kwargs["filters"].include_tags == ["work", "urgent"]
        assert call_args.kwargs["filters"].exclude_tags == ["blocked"]
        assert call_args.kwargs["filters"].tag_match_mode == "all"


class TestAnnotatedGetTasksInput:
    """Tests for AnnotatedGetTasksInput type with BeforeValidator."""

    def test_accepts_dict_input(self):
        """Test that AnnotatedGetTasksInput accepts dict input."""
        adapter = TypeAdapter(AnnotatedGetTasksInput)
        input_data = {"status": ["not_completed"], "limit": 10}

        result = adapter.validate_python(input_data)

        assert isinstance(result, GetTasksToolInput)
        assert result.status == ["not_completed"]
        assert result.limit == 10

    def test_accepts_json_string_input(self):
        """Test that AnnotatedGetTasksInput accepts JSON string input."""
        adapter = TypeAdapter(AnnotatedGetTasksInput)
        json_str = '{"status": ["not_completed"], "limit": 10}'

        result = adapter.validate_python(json_str)

        assert isinstance(result, GetTasksToolInput)
        assert result.status == ["not_completed"]
        assert result.limit == 10

    def test_accepts_none_input(self):
        """Test that AnnotatedGetTasksInput accepts None input."""
        adapter = TypeAdapter(AnnotatedGetTasksInput)

        result = adapter.validate_python(None)

        assert result is None

    def test_accepts_empty_string_input(self):
        """Test that AnnotatedGetTasksInput treats empty string as None."""
        adapter = TypeAdapter(AnnotatedGetTasksInput)

        result = adapter.validate_python("")

        assert result is None

    def test_rejects_invalid_json_string(self):
        """Test that AnnotatedGetTasksInput rejects invalid JSON."""
        adapter = TypeAdapter(AnnotatedGetTasksInput)
        invalid_json = "{not valid}"

        with pytest.raises(ValidationError):
            adapter.validate_python(invalid_json)

    def test_nested_tag_filters_in_json_string(self):
        """Test that nested tag_filters work in JSON string."""
        adapter = TypeAdapter(AnnotatedGetTasksInput)
        json_str = json.dumps(
            {
                "status": ["not_completed"],
                "tag_filters": {
                    "include_tags": ["work", "urgent"],
                    "exclude_tags": ["blocked"],
                    "match_mode": "all",
                },
                "limit": 20,
            }
        )

        result = adapter.validate_python(json_str)

        assert isinstance(result, GetTasksToolInput)
        assert result.status == ["not_completed"]
        assert result.tag_filters is not None
        assert result.tag_filters.include_tags == ["work", "urgent"]
        assert result.tag_filters.exclude_tags == ["blocked"]
        assert result.tag_filters.match_mode == "all"

    def test_nested_date_filters_in_json_string(self):
        """Test that nested date_filters work in JSON string."""
        adapter = TypeAdapter(AnnotatedGetTasksInput)
        json_str = json.dumps(
            {
                "date_filters": {
                    "due_after": "2026-01-01",
                    "due_before": "2026-12-31",
                    "match_mode": "any",
                }
            }
        )

        result = adapter.validate_python(json_str)

        assert isinstance(result, GetTasksToolInput)
        assert result.date_filters is not None
        assert result.date_filters.due_after == "2026-01-01"
        assert result.date_filters.due_before == "2026-12-31"
        assert result.date_filters.match_mode == "any"


class TestAnnotatedGetTasksInputEdgeCases:
    """Edge case tests for AnnotatedGetTasksInput."""

    def test_all_optional_fields_omitted(self):
        """Test that all optional fields can be omitted."""
        adapter = TypeAdapter(AnnotatedGetTasksInput)

        # Empty JSON object should create default GetTasksToolInput
        result = adapter.validate_python("{}")

        assert isinstance(result, GetTasksToolInput)
        assert result.status is None
        assert result.limit == 20  # Default value
        assert result.offset == 0  # Default value

    def test_ignores_unknown_fields(self):
        """Test that unknown fields are ignored (dataclass behavior)."""
        adapter = TypeAdapter(AnnotatedGetTasksInput)
        json_with_unknown = '{"status": ["not_completed"], "unknown_field": "value"}'

        # Dataclasses ignore unknown fields rather than rejecting them
        result = adapter.validate_python(json_with_unknown)

        assert isinstance(result, GetTasksToolInput)
        assert result.status == ["not_completed"]

    def test_rejects_wrong_type_for_field(self):
        """Test that wrong types are rejected."""
        adapter = TypeAdapter(AnnotatedGetTasksInput)
        json_wrong_type = '{"limit": "not a number"}'

        with pytest.raises(ValidationError):
            adapter.validate_python(json_wrong_type)
