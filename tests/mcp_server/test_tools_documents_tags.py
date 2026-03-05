"""Tests for documents_tags module.

Tests for tag filtering utilities in documents_tags.py.
"""

from unittest.mock import MagicMock, patch

import pytest

from obsidian_rag.mcp_server.models import TagFilter
from obsidian_rag.mcp_server.tools.documents_tags import (
    _check_tag_conflicts,
    _check_tag_count,
    _has_any_excluded_tags,
    _has_tags,
    _is_untagged,
    _matches_all_tags,
    _matches_any_tags,
    _tag_in_doc_tags,
    apply_postgresql_exclude_tags,
    apply_postgresql_include_tags,
    apply_postgresql_tag_filter,
    matches_tag_filter,
    validate_tag_filter,
)


class TestHasTags:
    """Tests for _has_tags function (TASK-089)."""

    def test_has_tags_with_none_tags(self):
        """Test _has_tags when doc.tags is None (TASK-089)."""
        doc = MagicMock()
        doc.tags = None

        result = _has_tags(doc, "work")
        assert result is False

    def test_has_tags_with_matching_tag(self):
        """Test _has_tags with matching tag."""
        doc = MagicMock()
        doc.tags = ["work", "urgent"]

        result = _has_tags(doc, "work")
        assert result is True

    def test_has_tags_with_substring_match(self):
        """Test _has_tags with substring match."""
        doc = MagicMock()
        doc.tags = ["workplace", "urgent"]

        result = _has_tags(doc, "work")
        assert result is True

    def test_has_tags_case_insensitive(self):
        """Test _has_tags is case-insensitive."""
        doc = MagicMock()
        doc.tags = ["WORK", "URGENT"]

        result = _has_tags(doc, "work")
        assert result is True


class TestIsUntagged:
    """Tests for _is_untagged function (TASK-090)."""

    def test_is_untagged_with_none(self):
        """Test _is_untagged with None tags (TASK-089)."""
        doc = MagicMock()
        doc.tags = None

        result = _is_untagged(doc)
        assert result is True

    def test_is_untagged_with_empty_list(self):
        """Test _is_untagged with empty tags list (TASK-090)."""
        doc = MagicMock()
        doc.tags = []

        result = _is_untagged(doc)
        assert result is True

    def test_is_untagged_with_tags(self):
        """Test _is_untagged with tags."""
        doc = MagicMock()
        doc.tags = ["work"]

        result = _is_untagged(doc)
        assert result is False


class TestCheckTagCount:
    """Tests for _check_tag_count function (TASK-091)."""

    def test_check_tag_count_with_exclude_tags_exceeding_max(self):
        """Test _check_tag_count with exclude_tags exceeding MAX_TAGS (TASK-091)."""
        tag_filter = TagFilter(
            include_tags=["work"],
            exclude_tags=[f"tag{i}" for i in range(60)],
        )

        with pytest.raises(ValueError, match="Maximum 50 exclude_tags allowed"):
            _check_tag_count(tag_filter)

    def test_check_tag_count_with_include_tags_exceeding_max(self):
        """Test _check_tag_count with include_tags exceeding MAX_TAGS."""
        tag_filter = TagFilter(
            include_tags=[f"tag{i}" for i in range(60)],
            exclude_tags=[],
        )

        with pytest.raises(ValueError, match="Maximum 50 include_tags allowed"):
            _check_tag_count(tag_filter)

    def test_check_tag_count_valid(self):
        """Test _check_tag_count with valid counts."""
        tag_filter = TagFilter(
            include_tags=["work", "urgent"],
            exclude_tags=["archived"],
        )

        # Should not raise
        _check_tag_count(tag_filter)


class TestTagInDocTags:
    """Tests for _tag_in_doc_tags function."""

    def test_tag_in_doc_tags_with_match(self):
        """Test _tag_in_doc_tags with matching tag."""
        doc_tags_lower = {"work", "urgent"}
        result = _tag_in_doc_tags("work", doc_tags_lower)
        assert result is True

    def test_tag_in_doc_tags_with_substring_match(self):
        """Test _tag_in_doc_tags with substring match."""
        doc_tags_lower = {"workplace", "urgent"}
        result = _tag_in_doc_tags("work", doc_tags_lower)
        assert result is True

    def test_tag_in_doc_tags_no_match(self):
        """Test _tag_in_doc_tags with no match."""
        doc_tags_lower = {"personal", "ideas"}
        result = _tag_in_doc_tags("work", doc_tags_lower)
        assert result is False


class TestMatchesAllTags:
    """Tests for _matches_all_tags function."""

    def test_matches_all_tags_all_match(self):
        """Test when all tags match."""
        doc = MagicMock()
        doc.tags = ["work", "urgent", "ideas"]

        result = _matches_all_tags(doc, ["work", "urgent"])
        assert result is True

    def test_matches_all_tags_some_missing(self):
        """Test when some tags are missing."""
        doc = MagicMock()
        doc.tags = ["work", "ideas"]

        result = _matches_all_tags(doc, ["work", "urgent"])
        assert result is False

    def test_matches_all_tags_empty_list(self):
        """Test with empty list."""
        doc = MagicMock()
        doc.tags = ["work"]

        result = _matches_all_tags(doc, [])
        assert result is True

    def test_matches_all_tags_no_doc_tags(self):
        """Test when document has no tags."""
        doc = MagicMock()
        doc.tags = None

        result = _matches_all_tags(doc, ["work"])
        assert result is False


class TestMatchesAnyTags:
    """Tests for _matches_any_tags function."""

    def test_matches_any_tags_one_matches(self):
        """Test when one tag matches."""
        doc = MagicMock()
        doc.tags = ["work", "ideas"]

        result = _matches_any_tags(doc, ["work", "urgent"])
        assert result is True

    def test_matches_any_tags_none_match(self):
        """Test when no tags match."""
        doc = MagicMock()
        doc.tags = ["personal", "ideas"]

        result = _matches_any_tags(doc, ["work", "urgent"])
        assert result is False

    def test_matches_any_tags_empty_list(self):
        """Test with empty list."""
        doc = MagicMock()
        doc.tags = ["work"]

        result = _matches_any_tags(doc, [])
        assert result is True


class TestHasAnyExcludedTags:
    """Tests for _has_any_excluded_tags function."""

    def test_has_any_excluded_tags_with_match(self):
        """Test when doc has excluded tag."""
        doc = MagicMock()
        doc.tags = ["work", "archived"]

        result = _has_any_excluded_tags(doc, ["archived"])
        assert result is True

    def test_has_any_excluded_tags_no_match(self):
        """Test when doc has no excluded tags."""
        doc = MagicMock()
        doc.tags = ["work", "urgent"]

        result = _has_any_excluded_tags(doc, ["archived"])
        assert result is False

    def test_has_any_excluded_tags_empty_excluded(self):
        """Test with empty excluded list."""
        doc = MagicMock()
        doc.tags = ["work"]

        result = _has_any_excluded_tags(doc, [])
        assert result is False

    def test_has_any_excluded_tags_none_doc_tags(self):
        """Test when doc.tags is None."""
        doc = MagicMock()
        doc.tags = None

        result = _has_any_excluded_tags(doc, ["archived"])
        assert result is False


class TestCheckTagConflicts:
    """Tests for _check_tag_conflicts function."""

    def test_check_tag_conflicts_with_conflicts(self):
        """Test with conflicting tags."""
        tag_filter = TagFilter(
            include_tags=["work", "urgent"],
            exclude_tags=["work", "archived"],
        )

        with pytest.raises(ValueError, match="Conflicting tags"):
            _check_tag_conflicts(tag_filter)

    def test_check_tag_conflicts_no_conflicts(self):
        """Test with no conflicting tags."""
        tag_filter = TagFilter(
            include_tags=["work", "urgent"],
            exclude_tags=["archived"],
        )

        # Should not raise
        _check_tag_conflicts(tag_filter)

    def test_check_tag_conflicts_empty_lists(self):
        """Test with empty lists."""
        tag_filter = TagFilter(
            include_tags=[],
            exclude_tags=[],
        )

        # Should not raise
        _check_tag_conflicts(tag_filter)


class TestValidateTagFilter:
    """Tests for validate_tag_filter function."""

    def test_validate_tag_filter_none(self):
        """Test with None filter."""
        # Should not raise
        validate_tag_filter(None)

    def test_validate_tag_filter_valid(self):
        """Test with valid filter."""
        tag_filter = TagFilter(include_tags=["work"], exclude_tags=["archived"])

        # Should not raise
        validate_tag_filter(tag_filter)

    def test_validate_tag_filter_conflicting(self):
        """Test with conflicting tags."""
        tag_filter = TagFilter(
            include_tags=["work", "urgent"],
            exclude_tags=["work", "archived"],
        )

        with pytest.raises(ValueError, match="Conflicting tags"):
            validate_tag_filter(tag_filter)

    def test_validate_tag_filter_too_many_tags(self):
        """Test with too many tags."""
        tag_filter = TagFilter(include_tags=[f"tag{i}" for i in range(60)])

        with pytest.raises(ValueError, match="Maximum 50 include_tags allowed"):
            validate_tag_filter(tag_filter)


class TestMatchesTagFilter:
    """Tests for matches_tag_filter function."""

    def test_matches_tag_filter_none(self):
        """Test with None filter."""
        doc = MagicMock()
        doc.tags = ["work"]

        result = matches_tag_filter(doc, None)
        assert result is True

    def test_matches_tag_filter_include_all(self):
        """Test include all tags."""
        doc = MagicMock()
        doc.tags = ["work", "urgent"]

        tag_filter = TagFilter(include_tags=["work", "urgent"], match_mode="all")
        result = matches_tag_filter(doc, tag_filter)
        assert result is True

    def test_matches_tag_filter_include_any(self):
        """Test include any tag."""
        doc = MagicMock()
        doc.tags = ["work", "ideas"]

        tag_filter = TagFilter(include_tags=["work", "urgent"], match_mode="any")
        result = matches_tag_filter(doc, tag_filter)
        assert result is True

    def test_matches_tag_filter_exclude(self):
        """Test exclude tags."""
        doc = MagicMock()
        doc.tags = ["work", "urgent"]

        tag_filter = TagFilter(exclude_tags=["archived"])
        result = matches_tag_filter(doc, tag_filter)
        assert result is True

    def test_matches_tag_filter_exclude_hit(self):
        """Test exclude tags when doc has excluded tag."""
        doc = MagicMock()
        doc.tags = ["work", "archived"]

        tag_filter = TagFilter(exclude_tags=["archived"])
        result = matches_tag_filter(doc, tag_filter)
        assert result is False

    def test_matches_tag_filter_include_and_exclude(self):
        """Test with both include and exclude."""
        doc = MagicMock()
        doc.tags = ["work", "ideas"]

        tag_filter = TagFilter(
            include_tags=["work"],
            exclude_tags=["archived"],
            match_mode="all",
        )
        result = matches_tag_filter(doc, tag_filter)
        assert result is True


class TestApplyPostgresqlIncludeTags:
    """Tests for apply_postgresql_include_tags function (TASK-092)."""

    def test_apply_postgresql_include_tags_match_mode_all(self):
        """Test apply_postgresql_include_tags with match_mode='all'."""
        mock_query = MagicMock()
        mock_query.filter.return_value = mock_query

        tag_filter = TagFilter(
            include_tags=["work", "urgent"],
            match_mode="all",
        )

        result = apply_postgresql_include_tags(mock_query, tag_filter)

        # Should filter twice (once per tag)
        assert mock_query.filter.call_count == 2
        assert result is mock_query

    def test_apply_postgresql_include_tags_match_mode_any(self):
        """Test apply_postgresql_include_tags with match_mode='any' (TASK-092)."""
        from sqlalchemy import or_

        mock_query = MagicMock()
        mock_query.filter.return_value = mock_query

        tag_filter = TagFilter(
            include_tags=["work", "urgent"],
            match_mode="any",
        )

        with patch("obsidian_rag.mcp_server.tools.documents_tags.or_") as mock_or:
            mock_or.return_value = "or_condition"

            result = apply_postgresql_include_tags(mock_query, tag_filter)

            # Should use or_ for 'any' mode
            mock_or.assert_called_once()
            mock_query.filter.assert_called_once_with("or_condition")
            assert result is mock_query

    def test_apply_postgresql_include_tags_empty_include(self):
        """Test apply_postgresql_include_tags with empty include_tags."""
        mock_query = MagicMock()

        tag_filter = TagFilter(include_tags=[], match_mode="all")

        result = apply_postgresql_include_tags(mock_query, tag_filter)

        # Should return query unchanged
        assert result is mock_query
        mock_query.filter.assert_not_called()


class TestApplyPostgresqlExcludeTags:
    """Tests for apply_postgresql_exclude_tags function (TASK-093)."""

    def test_apply_postgresql_exclude_tags(self):
        """Test apply_postgresql_exclude_tags (TASK-093)."""
        mock_query = MagicMock()
        mock_query.filter.return_value = mock_query

        tag_filter = TagFilter(exclude_tags=["archived", "deleted"])

        with patch("obsidian_rag.mcp_server.tools.documents_tags.or_") as mock_or:
            with patch(
                "obsidian_rag.mcp_server.tools.documents_tags.text"
            ) as mock_text:
                mock_or.return_value = "or_condition"
                mock_text.side_effect = lambda x: x

                result = apply_postgresql_exclude_tags(mock_query, tag_filter)

                # Should create exclude conditions
                mock_or.assert_called_once()
                mock_query.filter.assert_called_once()
                assert result is mock_query

    def test_apply_postgresql_exclude_tags_empty(self):
        """Test apply_postgresql_exclude_tags with empty exclude_tags."""
        mock_query = MagicMock()

        tag_filter = TagFilter(exclude_tags=[])

        result = apply_postgresql_exclude_tags(mock_query, tag_filter)

        # Should return query unchanged
        assert result is mock_query
        mock_query.filter.assert_not_called()


class TestApplyPostgresqlTagFilter:
    """Tests for apply_postgresql_tag_filter function (TASK-094)."""

    def test_apply_postgresql_tag_filter(self):
        """Test apply_postgresql_tag_filter (TASK-094)."""
        mock_query = MagicMock()

        tag_filter = TagFilter(
            include_tags=["work"],
            exclude_tags=["archived"],
        )

        with patch(
            "obsidian_rag.mcp_server.tools.documents_tags.apply_postgresql_include_tags"
        ) as mock_include:
            with patch(
                "obsidian_rag.mcp_server.tools.documents_tags.apply_postgresql_exclude_tags"
            ) as mock_exclude:
                mock_include.return_value = mock_query
                mock_exclude.return_value = mock_query

                result = apply_postgresql_tag_filter(mock_query, tag_filter)

                mock_include.assert_called_once_with(mock_query, tag_filter)
                mock_exclude.assert_called_once_with(mock_query, tag_filter)
                assert result is mock_query

    def test_apply_postgresql_tag_filter_none(self):
        """Test apply_postgresql_tag_filter with None filter."""
        mock_query = MagicMock()

        result = apply_postgresql_tag_filter(mock_query, None)

        # Should return query unchanged
        assert result is mock_query


class TestHasTagsAdditional:
    """Additional tests for _has_tags (line 42)."""

    def test_has_tags_second_branch(self):
        """Test _has_tags when doc.tags exists but tag not found - covers line 42."""
        from obsidian_rag.mcp_server.tools.documents_tags import _has_tags

        doc = MagicMock()
        doc.tags = ["personal", "ideas"]

        # Search for a tag that doesn't exist in the list
        result = _has_tags(doc, "work")
        assert result is False
