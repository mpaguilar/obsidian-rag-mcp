"""Tests for _parse_inline_filters helper in server.py."""

import json

import pytest

from obsidian_rag.mcp_server.models import PropertyFilter
from obsidian_rag.mcp_server.server import _parse_inline_filters


def test_parse_inline_filters_none():
    """Test that None input returns None."""
    result = _parse_inline_filters(None)
    assert result is None


def test_parse_inline_filters_list_of_property_filter():
    """Test that a list of PropertyFilter objects is returned as-is."""
    filters = [
        PropertyFilter(path="author", operator="equals", value="Alice"),
        PropertyFilter(path="status", operator="contains", value="draft"),
    ]
    result = _parse_inline_filters(filters)
    assert result == filters
    assert result is filters


def test_parse_inline_filters_json_string_single():
    """Test parsing a JSON string containing a single filter dict."""
    json_str = '{"path": "author", "operator": "equals", "value": "Alice"}'
    result = _parse_inline_filters(json_str)
    assert result is not None
    assert len(result) == 1
    assert isinstance(result[0], PropertyFilter)
    assert result[0].path == "author"
    assert result[0].operator == "equals"
    assert result[0].value == "Alice"


def test_parse_inline_filters_json_string_list():
    """Test parsing a JSON string containing a list of filter dicts."""
    json_str = (
        '[{"path": "author", "operator": "equals", "value": "Alice"},'
        ' {"path": "status", "operator": "contains", "value": "draft"}]'
    )
    result = _parse_inline_filters(json_str)
    assert result is not None
    assert len(result) == 2
    assert isinstance(result[0], PropertyFilter)
    assert result[0].path == "author"
    assert isinstance(result[1], PropertyFilter)
    assert result[1].path == "status"


def test_parse_inline_filters_dict():
    """Test parsing a dict input into a single PropertyFilter."""
    filter_dict = {"path": "author", "operator": "equals", "value": "Alice"}
    result = _parse_inline_filters(filter_dict)
    assert result is not None
    assert len(result) == 1
    assert isinstance(result[0], PropertyFilter)
    assert result[0].path == "author"
    assert result[0].operator == "equals"
    assert result[0].value == "Alice"


def test_parse_inline_filters_empty_string():
    """Test that an empty string returns None."""
    result = _parse_inline_filters("")
    assert result is None


def test_parse_inline_filters_whitespace_string():
    """Test that a whitespace-only string returns None."""
    result = _parse_inline_filters("   ")
    assert result is None


def test_parse_inline_filters_invalid_json():
    """Test that invalid JSON string raises JSONDecodeError."""
    with pytest.raises(json.JSONDecodeError):
        _parse_inline_filters("{invalid json}")


def test_parse_inline_filters_unsupported_type():
    """Test that an unsupported type (e.g., int) returns None."""
    result = _parse_inline_filters(123)
    assert result is None


def test_parse_inline_filters_json_string_scalar():
    """Test that a JSON string parsing to a scalar returns None."""
    result = _parse_inline_filters("123")
    assert result is None
