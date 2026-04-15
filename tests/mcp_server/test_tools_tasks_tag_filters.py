"""Tests for task tag filtering functions."""

from unittest.mock import MagicMock

import pytest

from obsidian_rag.mcp_server.tools.tasks import (
    _apply_exclude_tags,
    _apply_include_tags_all,
    _apply_include_tags_any,
    _apply_tag_filters,
    _strip_tag_list,
    _strip_tag_prefix,
    _validate_tag_filters,
)
from obsidian_rag.mcp_server.tools.tasks_params import GetTasksFilterParams


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


class TestStripTagPrefix:
    """Tests for _strip_tag_prefix function."""

    def test_strips_single_hash(self):
        """Test stripping a single # from tag."""
        assert _strip_tag_prefix("#work") == "work"

    def test_strips_multiple_hashes(self):
        """Test stripping multiple # characters from tag."""
        assert _strip_tag_prefix("##work") == "work"

    def test_no_hash(self):
        """Test that tag without # is unchanged."""
        assert _strip_tag_prefix("work") == "work"

    def test_only_hash(self):
        """Test that single # becomes empty string."""
        assert _strip_tag_prefix("#") == ""

    def test_only_hashes(self):
        """Test that multiple #s become empty string."""
        assert _strip_tag_prefix("###") == ""

    def test_empty_string(self):
        """Test that empty string returns empty string."""
        assert _strip_tag_prefix("") == ""

    def test_hierarchical_tag(self):
        """Test stripping # from hierarchical tag."""
        assert _strip_tag_prefix("#personal/expenses") == "personal/expenses"

    def test_hash_in_middle_not_stripped(self):
        """Test that # in middle of tag is not stripped."""
        assert _strip_tag_prefix("work#project") == "work#project"


class TestStripTagList:
    """Tests for _strip_tag_list function."""

    def test_strips_all_tags(self):
        """Test stripping # from all tags in list."""
        assert _strip_tag_list(["#work", "#urgent"]) == ["work", "urgent"]

    def test_removes_empty_after_strip(self):
        """Test that tags becoming empty after strip are removed."""
        assert _strip_tag_list(["#work", "#"]) == ["work"]

    def test_mixed_tags(self):
        """Test list with both # prefixed and non-prefixed tags."""
        assert _strip_tag_list(["#work", "personal"]) == ["work", "personal"]

    def test_all_become_empty(self):
        """Test that list with only # tags returns empty list."""
        assert _strip_tag_list(["#", "##"]) == []

    def test_empty_list(self):
        """Test that empty list returns empty list."""
        assert _strip_tag_list([]) == []


class TestTagPrefixStripping:
    """Tests that verify # prefix is stripped in tag filter functions."""

    def test_validate_tag_filters_strips_before_compare(self):
        """Verify conflict detection works after # stripping."""
        # Should raise ValueError because #work and work are the same after stripping
        with pytest.raises(ValueError, match="Conflicting tags"):
            _validate_tag_filters(["#work"], ["work"])

    def test_validate_tag_filters_multiple_hashes_stripped(self):
        """Verify tags with multiple # are properly stripped before conflict check."""
        # ##work and #work both become "work" after stripping
        with pytest.raises(ValueError, match="Conflicting tags"):
            _validate_tag_filters(["##work"], ["#work"])

    def test_validate_tag_filters_case_insensitive_after_strip(self):
        """Verify case-insensitive conflict detection after # stripping."""
        # #WORK and work should conflict (case-insensitive after strip)
        with pytest.raises(ValueError, match="Conflicting tags"):
            _validate_tag_filters(["#WORK"], ["work"])

    def test_validate_tag_filters_empty_tags_filtered(self):
        """Verify tags that become empty after strip are filtered out."""
        # Should not raise because # becomes empty and is filtered out
        _validate_tag_filters(["#work"], ["#"])  # No exception


class TestTagPrefixStrippingFilterFunctions:
    """Tests for # prefix stripping in filter application functions."""

    def test_include_tags_any_strips_hash(self):
        """Verify _apply_include_tags_any strips # from tag values."""
        mock_query = MagicMock()
        mock_query.filter.return_value = mock_query

        # Call with include_tags that have # prefix
        result = _apply_include_tags_any(mock_query, ["#work", "#personal"])

        # Should filter with OR conditions for stripped tags
        mock_query.filter.assert_called_once()
        assert result is mock_query

    def test_include_tags_any_single_hash_tag_ignored(self):
        """Verify tags that are just # are ignored in any mode."""
        mock_query = MagicMock()

        # Call with only # tags - should not filter
        result = _apply_include_tags_any(mock_query, ["#", "##"])

        mock_query.filter.assert_not_called()
        assert result is mock_query

    def test_include_tags_all_strips_hash(self):
        """Verify _apply_include_tags_all strips # from tag values."""
        mock_query = MagicMock()
        mock_query.filter.return_value = mock_query

        # Call with include_tags that have # prefix
        result = _apply_include_tags_all(mock_query, ["#work", "#urgent"])

        # Should filter twice (AND logic) with stripped tags
        assert mock_query.filter.call_count == 2
        assert result is mock_query

    def test_include_tags_all_single_hash_tag_ignored(self):
        """Verify tags that are just # are ignored in all mode."""
        mock_query = MagicMock()

        # Call with only # tags - should not filter
        result = _apply_include_tags_all(mock_query, ["#"])

        mock_query.filter.assert_not_called()
        assert result is mock_query

    def test_exclude_tags_strips_hash(self):
        """Verify _apply_exclude_tags strips # from tag values."""
        mock_query = MagicMock()
        mock_query.filter.return_value = mock_query

        # Call with exclude_tags that have # prefix
        result = _apply_exclude_tags(mock_query, ["#blocked", "#archived"])

        # Should filter with NOT OR conditions for stripped tags
        mock_query.filter.assert_called_once()
        assert result is mock_query

    def test_exclude_tags_single_hash_tag_ignored(self):
        """Verify tags that are just # are ignored in exclude."""
        mock_query = MagicMock()

        # Call with only # tags - should not filter
        result = _apply_exclude_tags(mock_query, ["###"])

        mock_query.filter.assert_not_called()
        assert result is mock_query

    def test_mixed_hash_and_no_hash_stripped_consistently(self):
        """Verify mixed # and non-# tags are handled consistently."""
        mock_query = MagicMock()
        mock_query.filter.return_value = mock_query

        # Mix of # prefixed and non-prefixed should work
        result = _apply_include_tags_all(mock_query, ["#work", "urgent", "##personal"])

        # Should filter 3 times (one for each unique tag after strip)
        assert mock_query.filter.call_count == 3
        assert result is mock_query
