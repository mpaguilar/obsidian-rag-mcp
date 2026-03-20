"""Tests for task tag filtering functions."""

from unittest.mock import MagicMock


from obsidian_rag.mcp_server.tools.tasks import _apply_tag_filters
from obsidian_rag.mcp_server.tools.tasks_params import GetTasksFilterParams


class TestApplyTagFiltersLegacy:
    """Tests for legacy 'tags' parameter (backward compatibility)."""

    def test_legacy_tags_and_logic(self):
        """Test that legacy tags parameter uses AND logic."""
        mock_query = MagicMock()
        mock_query.filter.return_value = mock_query

        filters = GetTasksFilterParams(tags=["work", "urgent"])
        result = _apply_tag_filters(mock_query, filters)

        # Should call filter twice (once for each tag - AND logic)
        assert mock_query.filter.call_count == 2
        assert result == mock_query

    def test_legacy_tags_empty_list(self):
        """Test that empty legacy tags list applies no filtering."""
        mock_query = MagicMock()

        filters = GetTasksFilterParams(tags=[])
        result = _apply_tag_filters(mock_query, filters)

        # Should not call filter
        mock_query.filter.assert_not_called()
        assert result == mock_query

    def test_legacy_tags_none(self):
        """Test that None legacy tags applies no filtering."""
        mock_query = MagicMock()

        filters = GetTasksFilterParams(tags=None)
        result = _apply_tag_filters(mock_query, filters)

        # Should not call filter
        mock_query.filter.assert_not_called()
        assert result == mock_query


class TestApplyTagFiltersIncludeAll:
    """Tests for include_tags with tag_match_mode='all' (AND logic)."""

    def test_include_tags_all_mode_and_logic(self):
        """Test include_tags with 'all' mode uses AND logic."""
        mock_query = MagicMock()
        mock_query.filter.return_value = mock_query

        filters = GetTasksFilterParams(
            include_tags=["work", "urgent"],
            tag_match_mode="all",
        )
        result = _apply_tag_filters(mock_query, filters)

        # Should call filter twice (once for each tag - AND logic)
        assert mock_query.filter.call_count == 2
        assert result == mock_query

    def test_include_tags_single_tag_all_mode(self):
        """Test single include_tag with 'all' mode."""
        mock_query = MagicMock()
        mock_query.filter.return_value = mock_query

        filters = GetTasksFilterParams(
            include_tags=["work"],
            tag_match_mode="all",
        )
        result = _apply_tag_filters(mock_query, filters)

        # Should call filter once
        assert mock_query.filter.call_count == 1
        assert result == mock_query


class TestApplyTagFiltersIncludeAny:
    """Tests for include_tags with tag_match_mode='any' (OR logic)."""

    def test_include_tags_any_mode_or_logic(self):
        """Test include_tags with 'any' mode uses OR logic."""
        mock_query = MagicMock()
        mock_query.filter.return_value = mock_query

        filters = GetTasksFilterParams(
            include_tags=["work", "personal"],
            tag_match_mode="any",
        )
        result = _apply_tag_filters(mock_query, filters)

        # Should call filter once with OR conditions
        assert mock_query.filter.call_count == 1
        assert result == mock_query

    def test_include_tags_single_tag_any_mode(self):
        """Test single include_tag with 'any' mode."""
        mock_query = MagicMock()
        mock_query.filter.return_value = mock_query

        filters = GetTasksFilterParams(
            include_tags=["work"],
            tag_match_mode="any",
        )
        result = _apply_tag_filters(mock_query, filters)

        # Should call filter once
        assert mock_query.filter.call_count == 1
        assert result == mock_query


class TestApplyTagFiltersExclude:
    """Tests for exclude_tags parameter."""

    def test_exclude_tags_or_logic(self):
        """Test that exclude_tags uses OR logic (any match excludes)."""
        mock_query = MagicMock()
        mock_query.filter.return_value = mock_query

        filters = GetTasksFilterParams(exclude_tags=["blocked", "waiting"])
        result = _apply_tag_filters(mock_query, filters)

        # Should call filter once with NOT OR conditions
        assert mock_query.filter.call_count == 1
        assert result == mock_query

    def test_exclude_tags_empty_list(self):
        """Test that empty exclude_tags applies no filtering."""
        mock_query = MagicMock()

        filters = GetTasksFilterParams(exclude_tags=[])
        result = _apply_tag_filters(mock_query, filters)

        # Should not call filter
        mock_query.filter.assert_not_called()
        assert result == mock_query

    def test_exclude_tags_none(self):
        """Test that None exclude_tags applies no filtering."""
        mock_query = MagicMock()

        filters = GetTasksFilterParams(exclude_tags=None)
        result = _apply_tag_filters(mock_query, filters)

        # Should not call filter
        mock_query.filter.assert_not_called()
        assert result == mock_query


class TestApplyTagFiltersCombined:
    """Tests for combined include and exclude tag filtering."""

    def test_include_and_exclude_combined(self):
        """Test combining include_tags and exclude_tags."""
        mock_query = MagicMock()
        mock_query.filter.return_value = mock_query

        filters = GetTasksFilterParams(
            include_tags=["work"],
            exclude_tags=["blocked"],
            tag_match_mode="all",
        )
        result = _apply_tag_filters(mock_query, filters)

        # Should call filter twice (once for include, once for exclude)
        assert mock_query.filter.call_count == 2
        assert result == mock_query

    def test_legacy_and_new_tags_combined(self):
        """Test combining legacy tags with new include_tags."""
        mock_query = MagicMock()
        mock_query.filter.return_value = mock_query

        filters = GetTasksFilterParams(
            tags=["legacy"],
            include_tags=["work"],
            tag_match_mode="all",
        )
        result = _apply_tag_filters(mock_query, filters)

        # Should call filter twice (once for legacy, once for include)
        assert mock_query.filter.call_count == 2
        assert result == mock_query

    def test_all_tag_filter_types_combined(self):
        """Test combining legacy tags, include_tags, and exclude_tags."""
        mock_query = MagicMock()
        mock_query.filter.return_value = mock_query

        filters = GetTasksFilterParams(
            tags=["legacy"],
            include_tags=["work", "urgent"],
            exclude_tags=["blocked"],
            tag_match_mode="any",
        )
        result = _apply_tag_filters(mock_query, filters)

        # Should call filter 3 times:
        # - once for legacy tags (1 tag)
        # - once for include_tags with OR (2 tags in one filter)
        # - once for exclude_tags
        assert mock_query.filter.call_count == 3
        assert result == mock_query
