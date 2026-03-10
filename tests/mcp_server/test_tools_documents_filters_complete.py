"""Complete tests for documents_filters.py PostgreSQL functions.

Additional tests to cover remaining PostgreSQL-specific code paths.
"""

from unittest.mock import MagicMock, patch

import pytest

from obsidian_rag.mcp_server.models import PropertyFilter


class TestApplyPostgresqlPropertyFilterComplete:
    """Complete tests for apply_postgresql_property_filter (lines 197-230)."""

    def test_apply_postgresql_property_filter_unknown_operator(self):
        """Test apply_postgresql_property_filter with unknown operator (line 226)."""
        from obsidian_rag.mcp_server.tools.documents_filters import (
            apply_postgresql_property_filter,
        )

        mock_query = MagicMock()

        # Create a mock filter with unknown operator
        mock_filter = MagicMock()
        mock_filter.path = "status"
        mock_filter.operator = "unknown_operator"
        mock_filter.value = "draft"

        result = apply_postgresql_property_filter(mock_query, mock_filter)

        # Should return query unchanged when operator is unknown
        assert result is mock_query
        mock_query.filter.assert_not_called()

    def test_apply_postgresql_property_filter_all_operators(self):
        """Test apply_postgresql_property_filter with all valid operators."""
        from obsidian_rag.mcp_server.tools.documents_filters import (
            apply_postgresql_property_filter,
        )

        # Test "equals" operator
        mock_query = MagicMock()
        mock_query.filter.return_value = mock_query
        filter_obj = PropertyFilter(path="status", operator="equals", value="draft")
        result = apply_postgresql_property_filter(mock_query, filter_obj)
        mock_query.filter.assert_called_once()
        assert result is mock_query

        # Test "contains" operator
        mock_query = MagicMock()
        mock_query.filter.return_value = mock_query
        filter_obj = PropertyFilter(path="status", operator="contains", value="draft")
        result = apply_postgresql_property_filter(mock_query, filter_obj)
        mock_query.filter.assert_called_once()
        assert result is mock_query

        # Test "starts_with" operator
        mock_query = MagicMock()
        mock_query.filter.return_value = mock_query
        filter_obj = PropertyFilter(
            path="status", operator="starts_with", value="draft"
        )
        result = apply_postgresql_property_filter(mock_query, filter_obj)
        mock_query.filter.assert_called_once()
        assert result is mock_query

        # Test "regex" operator
        mock_query = MagicMock()
        mock_query.filter.return_value = mock_query
        filter_obj = PropertyFilter(path="status", operator="regex", value="draft")
        result = apply_postgresql_property_filter(mock_query, filter_obj)
        mock_query.filter.assert_called_once()
        assert result is mock_query

        # Test "exists" operator separately (no value needed)
        mock_query = MagicMock()
        mock_query.filter.return_value = mock_query
        filter_obj = PropertyFilter(path="status", operator="exists")
        result = apply_postgresql_property_filter(mock_query, filter_obj)
        mock_query.filter.assert_called_once()
        assert result is mock_query

        # Test "in" operator separately (list value)
        mock_query = MagicMock()
        mock_query.filter.return_value = mock_query
        filter_obj = PropertyFilter(
            path="status", operator="in", value=["draft", "published"]
        )
        result = apply_postgresql_property_filter(mock_query, filter_obj)
        mock_query.filter.assert_called_once()
        assert result is mock_query

        # Test "exists" operator separately (no value needed)
        mock_query = MagicMock()
        mock_query.filter.return_value = mock_query
        filter_obj = PropertyFilter(path="status", operator="exists")  # type: ignore[arg-type]
        result = apply_postgresql_property_filter(mock_query, filter_obj)
        mock_query.filter.assert_called_once()
        assert result is mock_query

        # Test "in" operator separately (list value)
        mock_query = MagicMock()
        mock_query.filter.return_value = mock_query
        filter_obj = PropertyFilter(
            path="status",
            operator="in",
            value=["draft", "published"],  # type: ignore[arg-type]
        )
        result = apply_postgresql_property_filter(mock_query, filter_obj)
        mock_query.filter.assert_called_once()
        assert result is mock_query


class TestGetJsonbPathExpressionEdgeCases:
    """Edge case tests for get_jsonb_path_expression."""

    def test_get_jsonb_path_expression_deep_nesting(self):
        """Test get_jsonb_path_expression with 3-level nesting."""
        from obsidian_rag.mcp_server.tools.documents_filters import (
            get_jsonb_path_expression,
        )

        result = get_jsonb_path_expression("a.b.c")
        assert result == "frontmatter_json->'a'->'b'->>'c'"

    def test_get_jsonb_path_expression_single_level(self):
        """Test get_jsonb_path_expression with single level."""
        from obsidian_rag.mcp_server.tools.documents_filters import (
            get_jsonb_path_expression,
        )

        result = get_jsonb_path_expression("status")
        assert result == "frontmatter_json->>'status'"

    def test_get_jsonb_path_expression_two_levels(self):
        """Test get_jsonb_path_expression with two levels."""
        from obsidian_rag.mcp_server.tools.documents_filters import (
            get_jsonb_path_expression,
        )

        result = get_jsonb_path_expression("author.name")
        assert result == "frontmatter_json->'author'->>'name'"


class TestBuildConditionsEdgeCases:
    """Edge case tests for condition builder functions."""

    def test_build_exists_condition_simple_path(self):
        """Test build_exists_condition with simple path."""
        from obsidian_rag.mcp_server.tools.documents_filters import (
            build_exists_condition,
        )

        result = build_exists_condition("frontmatter_json->>'status'", "status")

        assert result is not None

    def test_build_exists_condition_nested_path(self):
        """Test build_exists_condition with nested path."""
        from obsidian_rag.mcp_server.tools.documents_filters import (
            build_exists_condition,
        )

        result = build_exists_condition(
            "frontmatter_json->'author'->>'name'", "author.name"
        )

        assert result is not None

    def test_build_equals_condition_with_none(self):
        """Test build_equals_condition with None value."""
        from obsidian_rag.mcp_server.tools.documents_filters import (
            build_equals_condition,
        )

        result = build_equals_condition("frontmatter_json->>'status'", None)

        assert result is not None
        assert "IS NULL" in str(result)

    def test_build_equals_condition_with_value(self):
        """Test build_equals_condition with string value."""
        from obsidian_rag.mcp_server.tools.documents_filters import (
            build_equals_condition,
        )

        result = build_equals_condition("frontmatter_json->>'status'", "draft")

        assert result is not None

    def test_build_like_condition(self):
        """Test build_like_condition with pattern."""
        from obsidian_rag.mcp_server.tools.documents_filters import (
            build_like_condition,
        )

        result = build_like_condition("frontmatter_json->>'title'", "%test%")

        assert result is not None
        assert "ILIKE" in str(result)

    def test_build_regex_condition_valid(self):
        """Test build_regex_condition with valid pattern."""
        from obsidian_rag.mcp_server.tools.documents_filters import (
            build_regex_condition,
        )

        result = build_regex_condition("frontmatter_json->>'email'", r"@example\.com$")

        assert result is not None
        assert "~" in str(result)

    def test_build_regex_condition_invalid(self):
        """Test build_regex_condition with invalid pattern raises ValueError."""
        from obsidian_rag.mcp_server.tools.documents_filters import (
            build_regex_condition,
        )

        with pytest.raises(ValueError, match="Invalid regex pattern"):
            build_regex_condition("frontmatter_json->>'email'", "[invalid")

    def test_build_in_condition_with_list(self):
        """Test build_in_condition with list value."""
        from obsidian_rag.mcp_server.tools.documents_filters import (
            build_in_condition,
        )

        result = build_in_condition(
            "frontmatter_json->>'status'", ["draft", "published"]
        )

        assert result is not None
        assert "= ANY" in str(result)

    def test_build_in_condition_with_non_list(self):
        """Test build_in_condition with non-list raises ValueError."""
        from obsidian_rag.mcp_server.tools.documents_filters import (
            build_in_condition,
        )

        with pytest.raises(ValueError, match="'in' operator requires a list value"):
            build_in_condition("frontmatter_json->>'status'", "draft")


class TestValidatePropertyFiltersEdgeCases:
    """Edge case tests for property filter validation."""

    def test_validate_property_filters_none(self):
        """Test validate_property_filters with None."""
        from obsidian_rag.mcp_server.tools.documents_filters import (
            validate_property_filters,
        )

        # Should not raise
        validate_property_filters(None)

    def test_validate_property_filters_too_many(self):
        """Test validate_property_filters with too many filters."""
        from obsidian_rag.mcp_server.tools.documents_filters import (
            validate_property_filters,
        )

        filters = [
            PropertyFilter(path=f"field{i}", operator="equals", value="test")
            for i in range(15)
        ]

        with pytest.raises(ValueError, match="Maximum 10 property filters allowed"):
            validate_property_filters(filters)

    def test_validate_property_path_empty(self):
        """Test validate_property_path with empty string."""
        from obsidian_rag.mcp_server.tools.documents_filters import (
            validate_property_path,
        )

        with pytest.raises(ValueError, match="Property path cannot be empty"):
            validate_property_path("")

    def test_validate_property_path_too_deep(self):
        """Test validate_property_path exceeding max depth."""
        from obsidian_rag.mcp_server.tools.documents_filters import (
            validate_property_path,
        )

        with pytest.raises(ValueError, match="Property path cannot exceed 3 levels"):
            validate_property_path("a.b.c.d")

    def test_validate_property_path_invalid_chars(self):
        """Test validate_property_path with invalid characters."""
        from obsidian_rag.mcp_server.tools.documents_filters import (
            validate_property_path,
        )

        with pytest.raises(ValueError, match="Invalid property path segment"):
            validate_property_path("path@invalid")
