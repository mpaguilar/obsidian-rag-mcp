"""Inline field filtering utilities for task queries.

This module contains JSONB filtering logic for Task.inline_fields column.
Adapted from documents_filters.py for the inline_fields column.
Inline fields are flat key-value pairs (no nested paths).
"""

import logging
import re
from typing import TYPE_CHECKING

from sqlalchemy import text
from sqlalchemy.sql.elements import TextClause

from obsidian_rag.database.models import Task

if TYPE_CHECKING:
    from sqlalchemy.orm import Query

    from obsidian_rag.mcp_server.models import PropertyFilter

log = logging.getLogger(__name__)

MAX_INLINE_FILTERS = 10


def get_inline_field_path_expression(path: str) -> str:
    """Build PostgreSQL JSONB path extraction for inline_fields.

    Inline fields are flat (single-segment keys). No nested dot notation.

    Args:
        path: Inline field key name (e.g., "vendor").

    Returns:
        PostgreSQL JSONB expression: "inline_fields->>'vendor'"

    Raises:
        ValueError: If path contains dots (inline fields are flat).
    """
    if "." in path:
        msg = "Inline field paths must be flat keys (no dot notation)"
        raise ValueError(msg)
    return f"inline_fields->>'{path}'"


def build_inline_exists_condition(path: str) -> TextClause:
    """Build SQL condition for 'exists' operator on inline_fields.

    Args:
        path: Inline field key name to check for existence.

    Returns:
        SQLAlchemy TextClause condition using PostgreSQL JSONB `?` operator.

    """
    return text("inline_fields ? :key").bindparams(key=path)


def build_inline_equals_condition(path_expr: str, value: object) -> TextClause:
    """Build SQL condition for 'equals' operator.

    Args:
        path_expr: PostgreSQL JSONB path expression (e.g., "inline_fields->>'vendor'").
        value: Value to compare against. If None, generates IS NULL condition.

    Returns:
        SQLAlchemy TextClause condition using equality comparison.

    """
    if value is None:
        return text(f"{path_expr} IS NULL")
    return text(f"{path_expr} = :value").bindparams(value=str(value))


def build_inline_like_condition(path_expr: str, pattern: str) -> TextClause:
    """Build SQL ILIKE condition for contains/starts_with operators.

    Args:
        path_expr: PostgreSQL JSONB path expression (e.g., "inline_fields->>'vendor'").
        pattern: ILIKE pattern string (e.g., "%Amazon%" for contains, "Amaz%" for starts_with).

    Returns:
        SQLAlchemy TextClause condition using ILIKE comparison.

    """
    return text(f"{path_expr} ILIKE :pattern").bindparams(pattern=pattern)


def build_inline_regex_condition(path_expr: str, pattern: object) -> TextClause:
    """Build SQL regex condition with validation.

    Args:
        path_expr: PostgreSQL JSONB path expression (e.g., "inline_fields->>'vendor'").
        pattern: Regex pattern string. Validated via `re.compile()` before use.

    Returns:
        SQLAlchemy TextClause condition using PostgreSQL `~` regex operator.

    Raises:
        ValueError: If the regex pattern is invalid.

    """
    try:
        re.compile(str(pattern))
    except re.error as e:
        msg = f"Invalid regex pattern: {e}"
        raise ValueError(msg) from e
    return text(f"{path_expr} ~ :pattern").bindparams(pattern=str(pattern))


def build_inline_in_condition(path_expr: str, values: object) -> TextClause:
    """Build SQL 'in' condition.

    Args:
        path_expr: PostgreSQL JSONB path expression (e.g., "inline_fields->>'vendor'").
        values: List of values for the IN comparison.

    Returns:
        SQLAlchemy TextClause condition using PostgreSQL `= ANY()` operator.

    Raises:
        ValueError: If values is not a list.

    """
    if not isinstance(values, list):
        msg = "'in' operator requires a list value"
        raise ValueError(msg)
    str_values = [str(v) for v in values]
    return text(f"{path_expr} = ANY(:values)").bindparams(values=str_values)


def validate_inline_field_path(path: str) -> None:
    """Validate inline field path (must be flat key, no dots).

    Args:
        path: Inline field key name to validate.

    Raises:
        ValueError: If path is empty, contains dots, or has invalid characters.

    """
    if not path:
        msg = "Inline field path cannot be empty"
        raise ValueError(msg)
    if "." in path:
        msg = "Inline field paths must be flat keys (no dot notation)"
        raise ValueError(msg)
    if not path.replace("_", "").isalnum():
        msg = f"Invalid inline field path: {path}"
        raise ValueError(msg)


def validate_inline_filter(f: "PropertyFilter") -> None:
    """Validate a single inline field filter.

    Args:
        f: PropertyFilter object to validate.

    Raises:
        ValueError: If operator is invalid or path contains dots/invalid characters.

    """
    valid_operators = {"equals", "contains", "exists", "in", "starts_with", "regex"}
    if f.operator not in valid_operators:
        msg = f"Invalid operator '{f.operator}'. Valid operators: {', '.join(sorted(valid_operators))}"
        raise ValueError(msg)
    validate_inline_field_path(f.path)


def validate_inline_filters(filters: list["PropertyFilter"] | None) -> None:
    """Validate inline filter parameters.

    Args:
        filters: List of PropertyFilter objects to validate, or None.

    Raises:
        ValueError: If more than MAX_INLINE_FILTERS filters are provided,
            or if any individual filter is invalid.

    """
    _msg = "validate_inline_filters starting"
    log.debug(_msg)
    if filters is None:
        return
    if len(filters) > MAX_INLINE_FILTERS:
        msg = f"Maximum {MAX_INLINE_FILTERS} inline filters allowed"
        raise ValueError(msg)
    for f in filters:
        validate_inline_filter(f)
    _msg = "validate_inline_filters returning"
    log.debug(_msg)


def apply_inline_field_filter(
    query: "Query[Task]",
    prop_filter: "PropertyFilter",
) -> "Query[Task]":
    """Apply PostgreSQL JSONB inline field filter to task query.

    Args:
        query: SQLAlchemy Query object to filter.
        prop_filter: PropertyFilter specifying path, operator, and value.

    Returns:
        Filtered Query object with the inline field condition applied,
        or the original query if the operator is unrecognized.

    """
    _msg = "apply_inline_field_filter starting"
    log.debug(_msg)
    path_expr = get_inline_field_path_expression(prop_filter.path)
    operator = prop_filter.operator

    condition_builders = {
        "exists": lambda: build_inline_exists_condition(prop_filter.path),
        "equals": lambda: build_inline_equals_condition(path_expr, prop_filter.value),
        "contains": lambda: build_inline_like_condition(
            path_expr, f"%{prop_filter.value}%"
        ),
        "starts_with": lambda: build_inline_like_condition(
            path_expr, f"{prop_filter.value}%"
        ),
        "regex": lambda: build_inline_regex_condition(path_expr, prop_filter.value),
        "in": lambda: build_inline_in_condition(path_expr, prop_filter.value),
    }

    if operator in condition_builders:
        condition: TextClause = condition_builders[operator]()
        return query.filter(condition)

    _msg = "apply_inline_field_filter returning"
    log.debug(_msg)
    return query
