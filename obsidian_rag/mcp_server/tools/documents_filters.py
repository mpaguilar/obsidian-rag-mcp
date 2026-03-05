"""Property filtering utilities for document queries.

This module contains property filtering logic for both PostgreSQL and SQLite databases.
Supports dot-notation property paths and multiple filter operators.
"""

import logging
import re
from typing import TYPE_CHECKING

from sqlalchemy import text

from obsidian_rag.database.models import Document

if TYPE_CHECKING:
    from sqlalchemy.orm import Query

    from obsidian_rag.mcp_server.models import PropertyFilter

log = logging.getLogger(__name__)

# Maximum query complexity limits
MAX_PROPERTY_FILTERS = 10


def validate_property_path(path: str) -> None:
    """Validate property path format.

    Args:
        path: Property path to validate.

    Raises:
        ValueError: If path format is invalid.

    """
    if not path:
        msg = "Property path cannot be empty"
        raise ValueError(msg)
    parts = path.split(".")
    if len(parts) > 3:
        msg = "Property path cannot exceed 3 levels"
        raise ValueError(msg)
    for part in parts:
        if not part or not part.replace("_", "").isalnum():
            msg = f"Invalid property path segment: {part}"
            raise ValueError(msg)


def validate_property_filter(f: "PropertyFilter") -> None:
    """Validate a single property filter.

    Args:
        f: Property filter to validate.

    Raises:
        ValueError: If filter is invalid.

    """
    valid_operators = {"equals", "contains", "exists", "in", "starts_with", "regex"}
    if f.operator not in valid_operators:
        msg = f"Invalid operator '{f.operator}'. Valid operators: {', '.join(sorted(valid_operators))}"
        raise ValueError(msg)
    validate_property_path(f.path)


def validate_property_filters(filters: list["PropertyFilter"] | None) -> None:
    """Validate property filter parameters.

    Args:
        filters: List of property filters to validate.

    Raises:
        ValueError: If validation fails.

    """
    if filters is None:
        return
    if len(filters) > MAX_PROPERTY_FILTERS:
        msg = f"Maximum {MAX_PROPERTY_FILTERS} property filters allowed"
        raise ValueError(msg)
    for f in filters:
        validate_property_filter(f)


def get_jsonb_path_expression(path: str) -> str:
    """Build PostgreSQL JSONB path extraction expression.

    Args:
        path: Dot-notation path (e.g., "author.name").

    Returns:
        PostgreSQL JSONB expression string.

    """
    parts = path.split(".")
    if len(parts) == 1:
        return f"frontmatter_json->>'{parts[0]}'"
    # Multi-level path
    result = "frontmatter_json"
    for i, part in enumerate(parts[:-1]):
        result += f"->'{part}'"
    result += f"->>'{parts[-1]}'"
    return result


def build_exists_condition(path_expr: str, path: str) -> object:
    """Build SQL condition for 'exists' operator.

    Args:
        path_expr: JSONB path expression.
        path: Original property path.

    Returns:
        SQLAlchemy text condition.

    """
    parts = path.split(".")
    if len(parts) == 1:
        return text("frontmatter_json ? :key").bindparams(key=parts[0])
    return text(f"{path_expr} IS NOT NULL")


def build_equals_condition(path_expr: str, value: object) -> object:
    """Build SQL condition for 'equals' operator.

    Args:
        path_expr: JSONB path expression.
        value: Value to compare.

    Returns:
        SQLAlchemy text condition.

    """
    if value is None:
        return text(f"{path_expr} IS NULL")
    return text(f"{path_expr} = :value").bindparams(value=str(value))


def build_like_condition(path_expr: str, pattern: str) -> object:
    """Build SQL ILIKE condition.

    Args:
        path_expr: JSONB path expression.
        pattern: LIKE pattern.

    Returns:
        SQLAlchemy text condition.

    """
    return text(f"{path_expr} ILIKE :pattern").bindparams(pattern=pattern)


def build_regex_condition(path_expr: str, pattern: object) -> object:
    """Build SQL regex condition with validation.

    Args:
        path_expr: JSONB path expression.
        pattern: Regex pattern.

    Returns:
        SQLAlchemy text condition.

    Raises:
        ValueError: If regex pattern is invalid.

    """
    try:
        re.compile(str(pattern))
    except re.error as e:
        msg = f"Invalid regex pattern: {e}"
        raise ValueError(msg)
    return text(f"{path_expr} ~ :pattern").bindparams(pattern=str(pattern))


def build_in_condition(path_expr: str, values: object) -> object:
    """Build SQL 'in' condition.

    Args:
        path_expr: JSONB path expression.
        values: List of values.

    Returns:
        SQLAlchemy text condition.

    Raises:
        ValueError: If values is not a list.

    """
    if not isinstance(values, list):
        msg = "'in' operator requires a list value"
        raise ValueError(msg)
    str_values = [str(v) for v in values]
    return text(f"{path_expr} = ANY(:values)").bindparams(values=str_values)


def apply_postgresql_property_filter(
    query: "Query[Document]",
    prop_filter: "PropertyFilter",
) -> "Query[Document]":
    """Apply PostgreSQL JSONB property filter to query.

    Args:
        query: SQLAlchemy query object.
        prop_filter: Property filter to apply.

    Returns:
        Filtered query.

    """
    path_expr = get_jsonb_path_expression(prop_filter.path)
    operator = prop_filter.operator

    condition_builders = {
        "exists": lambda: build_exists_condition(path_expr, prop_filter.path),
        "equals": lambda: build_equals_condition(path_expr, prop_filter.value),
        "contains": lambda: build_like_condition(path_expr, f"%{prop_filter.value}%"),
        "starts_with": lambda: build_like_condition(path_expr, f"{prop_filter.value}%"),
        "regex": lambda: build_regex_condition(path_expr, prop_filter.value),
        "in": lambda: build_in_condition(path_expr, prop_filter.value),
    }

    if operator in condition_builders:
        condition = condition_builders[operator]()
        return query.filter(condition)

    return query


def get_nested_value(data: dict | None, path: str) -> object:
    """Get value from nested dictionary using dot notation.

    Args:
        data: Dictionary to navigate.
        path: Dot-notation path (e.g., "author.name").

    Returns:
        The value at the path, or None if path doesn't exist.

    """
    if data is None:
        return None
    parts = path.split(".")
    current = data
    for part in parts:
        if not isinstance(current, dict):
            return None
        current = current.get(part)
        if current is None:
            return None
    return current


def check_equals(value: object, filter_value: object) -> bool:
    """Check if value equals filter value (case-insensitive).

    Args:
        value: Document value.
        filter_value: Filter value to compare.

    Returns:
        True if values are equal.

    """
    if filter_value is None:
        return value is None
    return str(value).lower() == str(filter_value).lower()


def check_contains(value: object, filter_value: object) -> bool:
    """Check if value contains filter value (case-insensitive).

    Args:
        value: Document value.
        filter_value: Filter value to search for.

    Returns:
        True if value contains filter value.

    """
    if value is None:
        return False
    return str(filter_value).lower() in str(value).lower()


def check_starts_with(value: object, filter_value: object) -> bool:
    """Check if value starts with filter value (case-insensitive).

    Args:
        value: Document value.
        filter_value: Filter value to check.

    Returns:
        True if value starts with filter value.

    """
    if value is None:
        return False
    return str(value).lower().startswith(str(filter_value).lower())


def check_regex(value: object, pattern: object) -> bool:
    """Check if value matches regex pattern.

    Args:
        value: Document value.
        pattern: Regex pattern.

    Returns:
        True if value matches pattern.

    """
    if value is None:
        return False
    try:
        pattern_str = str(pattern)
        return re.search(pattern_str, str(value), re.IGNORECASE) is not None
    except re.error:
        return False


def check_in_list(value: object, filter_values: object) -> bool:
    """Check if value is in filter values list.

    Args:
        value: Document value.
        filter_values: List of values to check against.

    Returns:
        True if value is in the list.

    """
    if value is None or not isinstance(filter_values, list):
        return False
    search_values = [str(v).lower() for v in filter_values]
    return str(value).lower() in search_values


def matches_property_filter(doc: Document, prop_filter: "PropertyFilter") -> bool:
    """Check if document matches a single property filter.

    Args:
        doc: Document to check.
        prop_filter: Property filter to apply.

    Returns:
        True if document matches the filter.

    """
    frontmatter = doc.frontmatter_json or {}
    value = get_nested_value(frontmatter, prop_filter.path)
    filter_value = prop_filter.value

    operator_checks = {
        "exists": lambda: value is not None,
        "equals": lambda: check_equals(value, filter_value),
        "contains": lambda: check_contains(value, filter_value),
        "starts_with": lambda: check_starts_with(value, filter_value),
        "regex": lambda: check_regex(value, filter_value),
        "in": lambda: check_in_list(value, filter_value),
    }

    check_func = operator_checks.get(prop_filter.operator)
    return check_func() if check_func else False


def check_filters_match(
    doc: Document,
    filters: list["PropertyFilter"] | None,
    should_match: bool,
) -> bool:
    """Check if document matches (or doesn't match) filters.

    Args:
        doc: Document to check.
        filters: List of property filters.
        should_match: If True, all filters must match. If False, no filters should match.

    Returns:
        True if document passes the filter check.

    """
    if not filters:
        return True
    for f in filters:
        matches = matches_property_filter(doc, f)
        if should_match and not matches:
            return False
        if not should_match and matches:
            return False
    return True


def matches_property_filters(
    doc: Document,
    include_filters: list["PropertyFilter"] | None,
    exclude_filters: list["PropertyFilter"] | None,
) -> bool:
    """Check if document matches all include filters and no exclude filters.

    Args:
        doc: Document to check.
        include_filters: Filters document must match (AND logic).
        exclude_filters: Filters document must NOT match (OR logic).

    Returns:
        True if document passes all filters.

    """
    return check_filters_match(doc, include_filters, True) and check_filters_match(
        doc, exclude_filters, False
    )
