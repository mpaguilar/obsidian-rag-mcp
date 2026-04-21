"""Tests for _get_tasks_handler function."""

import json
from unittest.mock import MagicMock, patch

import pytest
from pydantic import TypeAdapter, ValidationError

from obsidian_rag.mcp_server.handlers import (
    GetTasksRequest,
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


class TestParseJsonStr:
    """Tests for parse_json_str helper function."""

    def test_parse_valid_json_string(self):
        """Test parsing a valid JSON string."""
        from obsidian_rag.mcp_server.handlers import parse_json_str

        json_str = '{"include_tags": ["work"], "match_mode": "any"}'
        result = parse_json_str(json_str)

        assert result == {"include_tags": ["work"], "match_mode": "any"}

    def test_parse_invalid_json_string_raises_error(self):
        """Test that invalid JSON raises ValidationError."""
        from obsidian_rag.mcp_server.handlers import parse_json_str

        invalid_json = "{not valid json}"

        with pytest.raises(json.JSONDecodeError):
            parse_json_str(invalid_json)

    def test_parse_empty_string_returns_none(self):
        """Test that empty string returns None."""
        from obsidian_rag.mcp_server.handlers import parse_json_str

        result = parse_json_str("")
        assert result is None

    def test_parse_whitespace_string_returns_none(self):
        """Test that whitespace-only string returns None."""
        from obsidian_rag.mcp_server.handlers import parse_json_str

        result = parse_json_str("   ")
        assert result is None

    def test_pass_through_dict_unchanged(self):
        """Test that dict is passed through unchanged."""
        from obsidian_rag.mcp_server.handlers import parse_json_str

        input_dict = {"include_tags": ["work"], "match_mode": "any"}
        result = parse_json_str(input_dict)

        assert result is input_dict

    def test_pass_through_none_unchanged(self):
        """Test that None is passed through unchanged."""
        from obsidian_rag.mcp_server.handlers import parse_json_str

        result = parse_json_str(None)
        assert result is None


class TestAnnotatedQueryFilter:
    """Tests for AnnotatedQueryFilter type with BeforeValidator."""

    def test_accepts_dict_input(self):
        """Test that AnnotatedQueryFilter accepts dict input."""
        from obsidian_rag.mcp_server.handlers import (
            AnnotatedQueryFilter,
            QueryFilterParams,
        )

        adapter = TypeAdapter(AnnotatedQueryFilter)
        input_data = {"include_tags": ["work"], "match_mode": "any"}

        result = adapter.validate_python(input_data)

        assert isinstance(result, QueryFilterParams)
        assert result.include_tags == ["work"]
        assert result.match_mode == "any"

    def test_accepts_json_string_input(self):
        """Test that AnnotatedQueryFilter accepts JSON string input."""
        from obsidian_rag.mcp_server.handlers import (
            AnnotatedQueryFilter,
            QueryFilterParams,
        )

        adapter = TypeAdapter(AnnotatedQueryFilter)
        json_str = '{"include_tags": ["work"], "match_mode": "any"}'

        result = adapter.validate_python(json_str)

        assert isinstance(result, QueryFilterParams)
        assert result.include_tags == ["work"]
        assert result.match_mode == "any"

    def test_accepts_none_input(self):
        """Test that AnnotatedQueryFilter accepts None input."""
        from obsidian_rag.mcp_server.handlers import AnnotatedQueryFilter

        adapter = TypeAdapter(AnnotatedQueryFilter)

        result = adapter.validate_python(None)

        assert result is None

    def test_accepts_empty_string_input(self):
        """Test that AnnotatedQueryFilter treats empty string as None."""
        from obsidian_rag.mcp_server.handlers import AnnotatedQueryFilter

        adapter = TypeAdapter(AnnotatedQueryFilter)

        result = adapter.validate_python("")

        assert result is None

    def test_rejects_invalid_json_string(self):
        """Test that AnnotatedQueryFilter rejects invalid JSON."""
        from obsidian_rag.mcp_server.handlers import AnnotatedQueryFilter

        adapter = TypeAdapter(AnnotatedQueryFilter)
        invalid_json = "{not valid}"

        with pytest.raises(ValidationError):
            adapter.validate_python(invalid_json)


class TestAnnotatedGetTasksInput:
    """Tests for AnnotatedGetTasksInput type with BeforeValidator."""

    def test_accepts_dict_input(self):
        """Test that AnnotatedGetTasksInput accepts dict input."""
        from obsidian_rag.mcp_server.handlers import (
            AnnotatedGetTasksInput,
            GetTasksToolInput,
        )

        adapter = TypeAdapter(AnnotatedGetTasksInput)
        input_data = {"status": ["not_completed"], "limit": 10}

        result = adapter.validate_python(input_data)

        assert isinstance(result, GetTasksToolInput)
        assert result.status == ["not_completed"]
        assert result.limit == 10

    def test_accepts_json_string_input(self):
        """Test that AnnotatedGetTasksInput accepts JSON string input."""
        from obsidian_rag.mcp_server.handlers import (
            AnnotatedGetTasksInput,
            GetTasksToolInput,
        )

        adapter = TypeAdapter(AnnotatedGetTasksInput)
        json_str = '{"status": ["not_completed"], "limit": 10}'

        result = adapter.validate_python(json_str)

        assert isinstance(result, GetTasksToolInput)
        assert result.status == ["not_completed"]
        assert result.limit == 10

    def test_accepts_none_input(self):
        """Test that AnnotatedGetTasksInput accepts None input."""
        from obsidian_rag.mcp_server.handlers import AnnotatedGetTasksInput

        adapter = TypeAdapter(AnnotatedGetTasksInput)

        result = adapter.validate_python(None)

        assert result is None

    def test_accepts_empty_string_input(self):
        """Test that AnnotatedGetTasksInput treats empty string as None."""
        from obsidian_rag.mcp_server.handlers import AnnotatedGetTasksInput

        adapter = TypeAdapter(AnnotatedGetTasksInput)

        result = adapter.validate_python("")

        assert result is None

    def test_rejects_invalid_json_string(self):
        """Test that AnnotatedGetTasksInput rejects invalid JSON."""
        from obsidian_rag.mcp_server.handlers import AnnotatedGetTasksInput

        adapter = TypeAdapter(AnnotatedGetTasksInput)
        invalid_json = "{not valid}"

        with pytest.raises(ValidationError):
            adapter.validate_python(invalid_json)

    def test_nested_tag_filters_in_json_string(self):
        """Test that nested tag_filters work in JSON string."""
        from obsidian_rag.mcp_server.handlers import (
            AnnotatedGetTasksInput,
            GetTasksToolInput,
        )

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
        from obsidian_rag.mcp_server.handlers import (
            AnnotatedGetTasksInput,
            GetTasksToolInput,
        )

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


class TestParseJsonStrComplex:
    """Additional tests for parse_json_str with complex inputs."""

    def test_parse_nested_object(self):
        """Test parsing JSON string with nested objects."""
        from obsidian_rag.mcp_server.handlers import parse_json_str

        json_str = '{"outer": {"inner": [1, 2, 3]}, "simple": "value"}'
        result = parse_json_str(json_str)

        assert result == {"outer": {"inner": [1, 2, 3]}, "simple": "value"}

    def test_parse_array(self):
        """Test parsing JSON string with array at root."""
        from obsidian_rag.mcp_server.handlers import parse_json_str

        json_str = '["item1", "item2", "item3"]'
        result = parse_json_str(json_str)

        assert result == ["item1", "item2", "item3"]

    def test_parse_boolean(self):
        """Test parsing JSON string with boolean value."""
        from obsidian_rag.mcp_server.handlers import parse_json_str

        json_str = "true"
        result = parse_json_str(json_str)

        assert result is True

    def test_parse_number(self):
        """Test parsing JSON string with number value."""
        from obsidian_rag.mcp_server.handlers import parse_json_str

        json_str = "42"
        result = parse_json_str(json_str)

        assert result == 42

    def test_parse_string(self):
        """Test parsing JSON string with string value."""
        from obsidian_rag.mcp_server.handlers import parse_json_str

        json_str = '"just a string"'
        result = parse_json_str(json_str)

        assert result == "just a string"

    def test_pass_through_list_unchanged(self):
        """Test that list is passed through unchanged."""
        from obsidian_rag.mcp_server.handlers import parse_json_str

        input_list = [1, 2, 3]
        result = parse_json_str(input_list)

        assert result is input_list

    def test_pass_through_int_unchanged(self):
        """Test that int is passed through unchanged."""
        from obsidian_rag.mcp_server.handlers import parse_json_str

        result = parse_json_str(42)
        assert result == 42

    def test_pass_through_bool_unchanged(self):
        """Test that bool is passed through unchanged."""
        from obsidian_rag.mcp_server.handlers import parse_json_str

        result = parse_json_str(True)
        assert result is True


class TestAnnotatedQueryFilterEdgeCases:
    """Edge case tests for AnnotatedQueryFilter."""

    def test_rejects_partial_json(self):
        """Test that partial JSON is rejected."""
        from obsidian_rag.mcp_server.handlers import AnnotatedQueryFilter

        adapter = TypeAdapter(AnnotatedQueryFilter)
        partial_json = '{"include_tags": ["work"]'  # Missing closing brace

        with pytest.raises(ValidationError):
            adapter.validate_python(partial_json)

    def test_rejects_trailing_comma(self):
        """Test that JSON with trailing comma is rejected."""
        from obsidian_rag.mcp_server.handlers import AnnotatedQueryFilter

        adapter = TypeAdapter(AnnotatedQueryFilter)
        json_with_trailing = '{"include_tags": ["work"],}'

        with pytest.raises(ValidationError):
            adapter.validate_python(json_with_trailing)

    def test_rejects_single_quotes(self):
        """Test that JSON with single quotes is rejected."""
        from obsidian_rag.mcp_server.handlers import AnnotatedQueryFilter

        adapter = TypeAdapter(AnnotatedQueryFilter)
        single_quotes = "{'include_tags': ['work']}"

        with pytest.raises(ValidationError):
            adapter.validate_python(single_quotes)

    def test_accepts_unicode_in_json(self):
        """Test that JSON with unicode characters is accepted."""
        from obsidian_rag.mcp_server.handlers import (
            AnnotatedQueryFilter,
            QueryFilterParams,
        )

        adapter = TypeAdapter(AnnotatedQueryFilter)
        json_with_unicode = '{"include_tags": ["café", "naïve"]}'

        result = adapter.validate_python(json_with_unicode)

        assert isinstance(result, QueryFilterParams)
        assert result.include_tags == ["café", "naïve"]

    def test_accepts_whitespace_around_json(self):
        """Test that JSON with surrounding whitespace is accepted."""
        from obsidian_rag.mcp_server.handlers import (
            AnnotatedQueryFilter,
            QueryFilterParams,
        )

        adapter = TypeAdapter(AnnotatedQueryFilter)
        json_with_whitespace = '  \n  {"include_tags": ["work"]}  \n  '

        result = adapter.validate_python(json_with_whitespace)

        assert isinstance(result, QueryFilterParams)
        assert result.include_tags == ["work"]


class TestValidationErrorMessages:
    """Tests for validation error message quality."""

    def test_invalid_json_error_message(self):
        """Test that invalid JSON provides helpful error message."""
        from obsidian_rag.mcp_server.handlers import AnnotatedQueryFilter

        adapter = TypeAdapter(AnnotatedQueryFilter)
        invalid_json = "{not valid json}"

        with pytest.raises(ValidationError) as exc_info:
            adapter.validate_python(invalid_json)

        error_str = str(exc_info.value)
        # Should indicate JSON parsing failed
        assert "JSON" in error_str or "json" in error_str.lower()

    def test_validation_error_includes_location(self):
        """Test that validation error includes field location."""
        from obsidian_rag.mcp_server.handlers import AnnotatedQueryFilter

        adapter = TypeAdapter(AnnotatedQueryFilter)
        # match_mode has wrong type
        json_wrong_type = '{"match_mode": 123}'

        with pytest.raises(ValidationError) as exc_info:
            adapter.validate_python(json_wrong_type)

        error_str = str(exc_info.value)
        # Should mention match_mode field
        assert "match_mode" in error_str


class TestAnnotatedGetTasksInputEdgeCases:
    """Edge case tests for AnnotatedGetTasksInput."""

    def test_all_optional_fields_omitted(self):
        """Test that all optional fields can be omitted."""
        from obsidian_rag.mcp_server.handlers import (
            AnnotatedGetTasksInput,
            GetTasksToolInput,
        )

        adapter = TypeAdapter(AnnotatedGetTasksInput)

        # Empty JSON object should create default GetTasksToolInput
        result = adapter.validate_python("{}")

        assert isinstance(result, GetTasksToolInput)
        assert result.status is None
        assert result.limit == 20  # Default value
        assert result.offset == 0  # Default value

    def test_ignores_unknown_fields(self):
        """Test that unknown fields are ignored (dataclass behavior)."""
        from obsidian_rag.mcp_server.handlers import (
            AnnotatedGetTasksInput,
            GetTasksToolInput,
        )

        adapter = TypeAdapter(AnnotatedGetTasksInput)
        json_with_unknown = '{"status": ["not_completed"], "unknown_field": "value"}'

        # Dataclasses ignore unknown fields rather than rejecting them
        result = adapter.validate_python(json_with_unknown)

        assert isinstance(result, GetTasksToolInput)
        assert result.status == ["not_completed"]

    def test_rejects_wrong_type_for_field(self):
        """Test that wrong types are rejected."""
        from obsidian_rag.mcp_server.handlers import AnnotatedGetTasksInput

        adapter = TypeAdapter(AnnotatedGetTasksInput)
        json_wrong_type = '{"limit": "not a number"}'

        with pytest.raises(ValidationError):
            adapter.validate_python(json_wrong_type)


class TestJsonStringSpecialCharacters:
    """Tests for JSON strings with special characters."""

    def test_escapes_quotes(self):
        """Test JSON with escaped quotes."""
        from obsidian_rag.mcp_server.handlers import parse_json_str

        json_str = '{"description": "He said \\"hello\\""}'
        result = parse_json_str(json_str)

        assert result == {"description": 'He said "hello"'}

    def test_escapes_backslash(self):
        """Test JSON with escaped backslash."""
        from obsidian_rag.mcp_server.handlers import parse_json_str

        json_str = '{"path": "C:\\\\Users\\\\test"}'
        result = parse_json_str(json_str)

        assert result == {"path": "C:\\Users\\test"}

    def test_escapes_newline(self):
        """Test JSON with escaped newline."""
        from obsidian_rag.mcp_server.handlers import parse_json_str

        json_str = '{"text": "line1\\nline2"}'
        result = parse_json_str(json_str)

        assert result == {"text": "line1\nline2"}

    def test_null_value(self):
        """Test JSON with explicit null value."""
        from obsidian_rag.mcp_server.handlers import parse_json_str

        json_str = '{"field": null}'
        result = parse_json_str(json_str)

        assert result == {"field": None}


class TestGetVaultHandler:
    """Tests for _get_vault_handler function."""

    @patch("obsidian_rag.mcp_server.handlers.get_vault")
    def test_get_vault_handler_by_name(self, mock_get_vault):
        """Handler returns model_dump when getting vault by name."""
        mock_db_manager = MagicMock()
        mock_session = MagicMock()
        mock_db_manager.get_session.return_value.__enter__.return_value = mock_session
        mock_db_manager.get_session.return_value.__exit__.return_value = False

        mock_vault_response = MagicMock()
        mock_vault_response.model_dump.return_value = {
            "id": "vault-123",
            "name": "TestVault",
            "document_count": 5,
        }
        mock_get_vault.return_value = mock_vault_response

        from obsidian_rag.mcp_server.handlers import _get_vault_handler

        result = _get_vault_handler(
            db_manager=mock_db_manager,
            name="TestVault",
        )

        mock_get_vault.assert_called_once_with(
            session=mock_session,
            name="TestVault",
            vault_id=None,
        )
        assert result == {"id": "vault-123", "name": "TestVault", "document_count": 5}

    @patch("obsidian_rag.mcp_server.handlers.get_vault")
    def test_get_vault_handler_by_id(self, mock_get_vault):
        """Handler returns model_dump when getting vault by ID."""
        mock_db_manager = MagicMock()
        mock_session = MagicMock()
        mock_db_manager.get_session.return_value.__enter__.return_value = mock_session
        mock_db_manager.get_session.return_value.__exit__.return_value = False

        mock_vault_response = MagicMock()
        mock_vault_response.model_dump.return_value = {
            "id": "vault-456",
            "name": "AnotherVault",
            "document_count": 10,
        }
        mock_get_vault.return_value = mock_vault_response

        from obsidian_rag.mcp_server.handlers import _get_vault_handler

        result = _get_vault_handler(
            db_manager=mock_db_manager,
            vault_id="vault-456",
        )

        mock_get_vault.assert_called_once_with(
            session=mock_session,
            name=None,
            vault_id="vault-456",
        )
        assert result == {
            "id": "vault-456",
            "name": "AnotherVault",
            "document_count": 10,
        }

    @patch("obsidian_rag.mcp_server.handlers.get_vault")
    def test_get_vault_handler_not_found(self, mock_get_vault):
        """Handler catches ValueError and returns error dict."""
        mock_db_manager = MagicMock()
        mock_session = MagicMock()
        mock_db_manager.get_session.return_value.__enter__.return_value = mock_session
        mock_db_manager.get_session.return_value.__exit__.return_value = False

        mock_get_vault.side_effect = ValueError("Vault 'NonExistent' not found")

        from obsidian_rag.mcp_server.handlers import _get_vault_handler

        result = _get_vault_handler(
            db_manager=mock_db_manager,
            name="NonExistent",
        )

        assert result == {"success": False, "error": "Vault 'NonExistent' not found"}


class TestUpdateVaultHandler:
    """Tests for _update_vault_handler function."""

    @patch("obsidian_rag.mcp_server.handlers.update_vault")
    def test_update_vault_handler_success(self, mock_update_vault):
        """Handler returns model_dump on successful update."""
        mock_db_manager = MagicMock()
        mock_session = MagicMock()
        mock_db_manager.get_session.return_value.__enter__.return_value = mock_session
        mock_db_manager.get_session.return_value.__exit__.return_value = False

        mock_vault_response = MagicMock()
        mock_vault_response.model_dump.return_value = {
            "id": "vault-123",
            "name": "TestVault",
            "description": "Updated description",
        }
        mock_update_vault.return_value = mock_vault_response

        from obsidian_rag.mcp_server.handlers import _update_vault_handler
        from obsidian_rag.mcp_server.tools.vaults_params import VaultUpdateParams

        params = VaultUpdateParams(
            name="TestVault",
            description="Updated description",
        )

        result = _update_vault_handler(
            db_manager=mock_db_manager,
            params=params,
        )

        mock_update_vault.assert_called_once_with(
            session=mock_session,
            params=params,
        )
        assert result == {
            "id": "vault-123",
            "name": "TestVault",
            "description": "Updated description",
        }

    @patch("obsidian_rag.mcp_server.handlers.update_vault")
    def test_update_vault_handler_force_required(self, mock_update_vault):
        """Handler returns error dict when container_path change requires force."""
        mock_db_manager = MagicMock()
        mock_session = MagicMock()
        mock_db_manager.get_session.return_value.__enter__.return_value = mock_session
        mock_db_manager.get_session.return_value.__exit__.return_value = False

        error_dict = {
            "success": False,
            "error": "Changing container_path will delete all documents, tasks, and chunks for this vault and require re-ingestion. Set force=True to confirm.",
        }
        mock_update_vault.return_value = error_dict

        from obsidian_rag.mcp_server.handlers import _update_vault_handler
        from obsidian_rag.mcp_server.tools.vaults_params import VaultUpdateParams

        params = VaultUpdateParams(
            name="TestVault",
            container_path="/new/path",
            force=False,
        )

        result = _update_vault_handler(
            db_manager=mock_db_manager,
            params=params,
        )

        assert result == error_dict

    @patch("obsidian_rag.mcp_server.handlers.update_vault")
    def test_update_vault_handler_not_found(self, mock_update_vault):
        """Handler catches ValueError and returns error dict."""
        mock_db_manager = MagicMock()
        mock_session = MagicMock()
        mock_db_manager.get_session.return_value.__enter__.return_value = mock_session
        mock_db_manager.get_session.return_value.__exit__.return_value = False

        mock_update_vault.side_effect = ValueError("Vault 'NonExistent' not found")

        from obsidian_rag.mcp_server.handlers import _update_vault_handler
        from obsidian_rag.mcp_server.tools.vaults_params import VaultUpdateParams

        params = VaultUpdateParams(name="NonExistent")

        result = _update_vault_handler(
            db_manager=mock_db_manager,
            params=params,
        )

        assert result == {"success": False, "error": "Vault 'NonExistent' not found"}


class TestDeleteVaultHandler:
    """Tests for _delete_vault_handler function."""

    @patch("obsidian_rag.mcp_server.handlers.delete_vault")
    def test_delete_vault_handler_success(self, mock_delete_vault):
        """Handler returns success dict when deletion confirmed."""
        mock_db_manager = MagicMock()
        mock_session = MagicMock()
        mock_db_manager.get_session.return_value.__enter__.return_value = mock_session
        mock_db_manager.get_session.return_value.__exit__.return_value = False

        success_dict = {
            "success": True,
            "name": "TestVault",
            "id": "vault-123",
            "documents_deleted": 5,
            "tasks_deleted": 10,
            "chunks_deleted": 20,
            "warning": "Vault config entry still exists.",
        }
        mock_delete_vault.return_value = success_dict

        from obsidian_rag.mcp_server.handlers import _delete_vault_handler

        result = _delete_vault_handler(
            db_manager=mock_db_manager,
            name="TestVault",
            confirm=True,
        )

        mock_delete_vault.assert_called_once_with(
            session=mock_session,
            name="TestVault",
            confirm=True,
        )
        assert result == success_dict

    @patch("obsidian_rag.mcp_server.handlers.delete_vault")
    def test_delete_vault_handler_not_confirmed(self, mock_delete_vault):
        """Handler returns error dict when confirm is False."""
        mock_db_manager = MagicMock()
        mock_session = MagicMock()
        mock_db_manager.get_session.return_value.__enter__.return_value = mock_session
        mock_db_manager.get_session.return_value.__exit__.return_value = False

        error_dict = {
            "success": False,
            "error": "confirm=True is required to delete a vault. This action is irreversible and will cascade-delete all associated documents, tasks, and chunks.",
        }
        mock_delete_vault.return_value = error_dict

        from obsidian_rag.mcp_server.handlers import _delete_vault_handler

        result = _delete_vault_handler(
            db_manager=mock_db_manager,
            name="TestVault",
            confirm=False,
        )

        assert result == error_dict

    @patch("obsidian_rag.mcp_server.handlers.delete_vault")
    def test_delete_vault_handler_not_found(self, mock_delete_vault):
        """Handler catches ValueError and returns error dict."""
        mock_db_manager = MagicMock()
        mock_session = MagicMock()
        mock_db_manager.get_session.return_value.__enter__.return_value = mock_session
        mock_db_manager.get_session.return_value.__exit__.return_value = False

        mock_delete_vault.side_effect = ValueError("Vault 'NonExistent' not found")

        from obsidian_rag.mcp_server.handlers import _delete_vault_handler

        result = _delete_vault_handler(
            db_manager=mock_db_manager,
            name="NonExistent",
            confirm=True,
        )

        assert result == {"success": False, "error": "Vault 'NonExistent' not found"}
