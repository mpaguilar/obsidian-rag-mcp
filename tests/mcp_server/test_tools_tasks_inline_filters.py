"""Tests for tasks_inline_filters.py JSONB filtering on Task.inline_fields."""

from unittest.mock import MagicMock

import pytest

from obsidian_rag.mcp_server.models import PropertyFilter
from obsidian_rag.mcp_server.tools.tasks_inline_filters import (
    apply_inline_field_filter,
    build_inline_equals_condition,
    build_inline_exists_condition,
    build_inline_in_condition,
    build_inline_like_condition,
    build_inline_regex_condition,
    get_inline_field_path_expression,
    validate_inline_field_path,
    validate_inline_filter,
    validate_inline_filters,
)


def test_get_inline_field_path_expression_single_key() -> None:
    """Test path expression for a flat inline field key."""
    result = get_inline_field_path_expression("vendor")
    assert result == "inline_fields->>'vendor'"


def test_get_inline_field_path_expression_rejects_dots() -> None:
    """Test that dotted paths are rejected for inline fields."""
    with pytest.raises(ValueError, match="Inline field paths must be flat keys"):
        get_inline_field_path_expression("vendor.name")


def test_validate_inline_field_path_empty() -> None:
    """Test empty inline field path validation."""
    with pytest.raises(ValueError, match="Inline field path cannot be empty"):
        validate_inline_field_path("")


def test_validate_inline_field_path_dots() -> None:
    """Test dotted inline field path validation."""
    with pytest.raises(ValueError, match="Inline field paths must be flat keys"):
        validate_inline_field_path("vendor.name")


def test_validate_inline_field_path_invalid_chars() -> None:
    """Test invalid characters in inline field path."""
    with pytest.raises(ValueError, match="Invalid inline field path"):
        validate_inline_field_path("vendor@name")


def test_validate_inline_field_path_valid() -> None:
    """Test valid inline field paths."""
    validate_inline_field_path("vendor")
    validate_inline_field_path("vendor_name")
    validate_inline_field_path("vendor123")


def test_validate_inline_filter_valid_operators() -> None:
    """Test validation of all supported inline filter operators."""
    for operator in ("equals", "contains", "exists", "in", "starts_with", "regex"):
        filter_obj = PropertyFilter(path="vendor", operator=operator, value="acme")
        validate_inline_filter(filter_obj)


def test_validate_inline_filter_invalid_operator() -> None:
    """Test validation with an invalid operator."""
    mock_filter = MagicMock()
    mock_filter.path = "vendor"
    mock_filter.operator = "invalid_operator"
    mock_filter.value = "acme"
    with pytest.raises(ValueError, match="Invalid operator"):
        validate_inline_filter(mock_filter)


def test_validate_inline_filters_max_count() -> None:
    """Test that more than MAX_INLINE_FILTERS raises ValueError."""
    filters = [PropertyFilter(path=f"field_{i}", operator="exists") for i in range(11)]
    with pytest.raises(ValueError, match="Maximum 10 inline filters allowed"):
        validate_inline_filters(filters)


def test_validate_inline_filters_none() -> None:
    """Test that None filters validate without error."""
    validate_inline_filters(None)


def test_build_inline_equals_condition() -> None:
    """Test equals condition with a non-None value."""
    condition = build_inline_equals_condition("inline_fields->>'vendor'", "acme")
    assert condition is not None


def test_build_inline_equals_condition_null_value() -> None:
    """Test equals condition with a None value produces IS NULL."""
    condition = build_inline_equals_condition("inline_fields->>'vendor'", None)
    assert "IS NULL" in condition.text


def test_build_inline_like_condition_contains() -> None:
    """Test LIKE condition for contains operator."""
    condition = build_inline_like_condition("inline_fields->>'vendor'", "%acme%")
    params = condition.compile().params
    assert params["pattern"] == "%acme%"


def test_build_inline_like_condition_starts_with() -> None:
    """Test LIKE condition for starts_with operator."""
    condition = build_inline_like_condition("inline_fields->>'vendor'", "acme%")
    params = condition.compile().params
    assert params["pattern"] == "acme%"


def test_build_inline_regex_condition_valid() -> None:
    """Test regex condition with a valid pattern."""
    condition = build_inline_regex_condition("inline_fields->>'vendor'", r"acme.*")
    assert condition is not None


def test_build_inline_regex_condition_invalid() -> None:
    """Test regex condition with an invalid pattern raises ValueError."""
    with pytest.raises(ValueError, match="Invalid regex pattern"):
        build_inline_regex_condition("inline_fields->>'vendor'", "[invalid")


def test_build_inline_in_condition_list() -> None:
    """Test in condition with a list of values."""
    condition = build_inline_in_condition(
        "inline_fields->>'vendor'", ["acme", "globex"]
    )
    assert condition is not None


def test_build_inline_in_condition_non_list() -> None:
    """Test in condition with a non-list value raises ValueError."""
    with pytest.raises(ValueError, match="'in' operator requires a list value"):
        build_inline_in_condition("inline_fields->>'vendor'", "acme")


def test_build_inline_exists_condition() -> None:
    """Test exists condition for inline_fields."""
    condition = build_inline_exists_condition("vendor")
    assert condition is not None


def test_apply_inline_field_filter_equals() -> None:
    """Test applying an equals inline field filter."""
    mock_query = MagicMock()
    mock_query.filter.return_value = mock_query
    filter_obj = PropertyFilter(path="vendor", operator="equals", value="acme")
    result = apply_inline_field_filter(mock_query, filter_obj)
    mock_query.filter.assert_called_once()
    assert result is mock_query


def test_apply_inline_field_filter_exists() -> None:
    """Test applying an exists inline field filter."""
    mock_query = MagicMock()
    mock_query.filter.return_value = mock_query
    filter_obj = PropertyFilter(path="vendor", operator="exists")
    result = apply_inline_field_filter(mock_query, filter_obj)
    mock_query.filter.assert_called_once()
    assert result is mock_query


def test_apply_inline_field_filter_contains() -> None:
    """Test applying a contains inline field filter."""
    mock_query = MagicMock()
    mock_query.filter.return_value = mock_query
    filter_obj = PropertyFilter(path="vendor", operator="contains", value="acme")
    result = apply_inline_field_filter(mock_query, filter_obj)
    mock_query.filter.assert_called_once()
    assert result is mock_query


def test_apply_inline_field_filter_regex() -> None:
    """Test applying a regex inline field filter."""
    mock_query = MagicMock()
    mock_query.filter.return_value = mock_query
    filter_obj = PropertyFilter(path="vendor", operator="regex", value=r"acme.*")
    result = apply_inline_field_filter(mock_query, filter_obj)
    mock_query.filter.assert_called_once()
    assert result is mock_query


def test_apply_inline_field_filter_in() -> None:
    """Test applying an in inline field filter."""
    mock_query = MagicMock()
    mock_query.filter.return_value = mock_query
    filter_obj = PropertyFilter(path="vendor", operator="in", value=["acme", "globex"])
    result = apply_inline_field_filter(mock_query, filter_obj)
    mock_query.filter.assert_called_once()
    assert result is mock_query


def test_apply_inline_field_filter_starts_with() -> None:
    """Test applying a starts_with inline field filter."""
    mock_query = MagicMock()
    mock_query.filter.return_value = mock_query
    filter_obj = PropertyFilter(path="vendor", operator="starts_with", value="acme")
    result = apply_inline_field_filter(mock_query, filter_obj)
    mock_query.filter.assert_called_once()
    assert result is mock_query


def test_apply_inline_field_filter_unsupported_operator_returns_query() -> None:
    """Test apply_inline_field_filter returns original query for unknown operator."""
    mock_query = MagicMock()
    mock_filter = MagicMock()
    mock_filter.path = "vendor"
    mock_filter.operator = "unsupported"
    mock_filter.value = "acme"
    result = apply_inline_field_filter(mock_query, mock_filter)
    mock_query.filter.assert_not_called()
    assert result is mock_query
