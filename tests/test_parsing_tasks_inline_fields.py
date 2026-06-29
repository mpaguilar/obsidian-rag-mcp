"""Tests for _extract_task_metadata well-known field inclusion in inline_fields dict."""

from obsidian_rag.database.models import TaskPriority
from obsidian_rag.parsing.tasks import (
    _extract_task_metadata,
    _process_metadata_key,
    parse_task_line,
)


class TestExtractTaskMetadataInlineFields:
    """Test cases for well-known fields included in inline_fields dict."""

    def test_extract_task_metadata_includes_well_known_fields_in_inline_fields(self):
        """Test that well-known fields appear in both standard_fields and inline_fields."""
        task_text = "[priority:: high] [due:: 2024-03-15] [custom:: value]"
        standard_fields, inline_fields = _extract_task_metadata(task_text)

        assert inline_fields == {
            "priority": "high",
            "due": "2024-03-15",
            "custom": "value",
        }
        assert standard_fields["priority"] == TaskPriority.HIGH.value
        assert standard_fields["due"] is not None

    def test_extract_task_metadata_priority_in_inline_fields(self):
        """Test that priority field is included in inline_fields."""
        task_text = "[priority:: high]"
        _standard, inline = _extract_task_metadata(task_text)

        assert inline == {"priority": "high"}

    def test_extract_task_metadata_scheduled_in_inline_fields(self):
        """Test that scheduled field is included in inline_fields."""
        task_text = "[scheduled:: 2024-04-20]"
        _standard, inline = _extract_task_metadata(task_text)

        assert inline == {"scheduled": "2024-04-20"}

    def test_extract_task_metadata_completion_in_inline_fields(self):
        """Test that completion field is included in inline_fields."""
        task_text = "[completion:: 2024-05-10]"
        _standard, inline = _extract_task_metadata(task_text)

        assert inline == {"completion": "2024-05-10"}

    def test_extract_task_metadata_repeat_in_inline_fields(self):
        """Test that repeat field is included in inline_fields."""
        task_text = "[repeat:: every day]"
        _standard, inline = _extract_task_metadata(task_text)

        assert inline == {"repeat": "every day"}

    def test_extract_task_metadata_mixed_fields(self):
        """Test mixed well-known and custom fields all appear in inline_fields."""
        task_text = (
            "[priority:: high] [due:: 2024-03-15] [scheduled:: 2024-04-20]"
            " [repeat:: every week] [custom_field:: custom value]"
        )
        standard_fields, inline_fields = _extract_task_metadata(task_text)

        assert inline_fields == {
            "priority": "high",
            "due": "2024-03-15",
            "scheduled": "2024-04-20",
            "repeat": "every week",
            "custom_field": "custom value",
        }
        assert standard_fields["priority"] == TaskPriority.HIGH.value
        assert standard_fields["due"] is not None
        assert standard_fields["scheduled"] is not None
        assert standard_fields["repeat"] == "FREQ=WEEKLY"

    def test_extract_task_metadata_no_fields(self):
        """Test that empty metadata produces empty inline_fields."""
        task_text = "No metadata here"
        _standard, inline = _extract_task_metadata(task_text)

        assert inline == {}

    def test_parse_task_line_inline_fields_includes_well_known(self):
        """Test that parse_task_line returns inline_fields with well-known fields."""
        line = "- [ ] Task [priority:: high] [due:: 2024-03-15]"
        result = parse_task_line(line)

        assert result is not None
        assert result.inline_fields is not None
        assert result.inline_fields == {
            "priority": "high",
            "due": "2024-03-15",
        }


class TestProcessMetadataKeyInlineFields:
    """Test cases for _process_metadata_key inline field inclusion."""

    def test_process_metadata_key_includes_well_known_in_inline_fields(self):
        """Test that well-known keys are added to inline_fields."""
        standard_fields: dict = {}
        inline_fields: dict = {}
        _process_metadata_key("priority", "high", standard_fields, inline_fields)

        assert inline_fields == {"priority": "high"}

    def test_process_metadata_key_includes_custom_in_inline_fields(self):
        """Test that custom keys are added to inline_fields."""
        standard_fields: dict = {}
        inline_fields: dict = {}
        _process_metadata_key("custom", "value", standard_fields, inline_fields)

        assert inline_fields == {"custom": "value"}


class TestExtractTaskMetadataInlineFieldsEdgeCases:
    """Edge case tests for inline_fields extraction."""

    def test_inline_fields_empty_value(self):
        """Test that empty value after :: is not captured by the regex."""
        task_text = "[empty::]"
        _standard, inline = _extract_task_metadata(task_text)

        assert inline == {}

    def test_inline_fields_all_well_known_no_custom(self):
        """Test that all well-known fields appear with no custom fields."""
        task_text = (
            "[priority:: high] [due:: 2024-03-15] [scheduled:: 2024-04-20]"
            " [repeat:: every day] [completion:: 2024-05-10]"
        )
        standard_fields, inline_fields = _extract_task_metadata(task_text)

        assert inline_fields == {
            "priority": "high",
            "due": "2024-03-15",
            "scheduled": "2024-04-20",
            "repeat": "every day",
            "completion": "2024-05-10",
        }
        assert standard_fields["priority"] == TaskPriority.HIGH.value
        assert standard_fields["due"] is not None
        assert standard_fields["scheduled"] is not None
        assert standard_fields["repeat"] is not None
        assert standard_fields["completion"] is not None

    def test_inline_fields_preserves_original_key_casing(self):
        """Test that original key casing is preserved in inline_fields."""
        task_text = "[CustomKey:: value] [DUE:: 2024-06-01]"
        _standard, inline = _extract_task_metadata(task_text)

        assert inline == {"CustomKey": "value", "DUE": "2024-06-01"}

    def test_inline_fields_task_with_no_metadata(self):
        """Test that a task line with no metadata produces empty inline_fields."""
        line = "- [ ] Just a plain task"
        result = parse_task_line(line)

        assert result is not None
        assert result.inline_fields is not None
        assert result.inline_fields == {}
