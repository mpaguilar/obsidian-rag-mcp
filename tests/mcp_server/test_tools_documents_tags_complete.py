"""Complete tests for documents_tags.py PostgreSQL functions.

Additional tests to cover remaining PostgreSQL-specific code paths.
"""

from unittest.mock import MagicMock, patch

import pytest

from obsidian_rag.mcp_server.models import TagFilter


class TestApplyPostgresqlIncludeTagsComplete:
    """Complete tests for apply_postgresql_include_tags."""

    def test_apply_postgresql_include_tags_all_mode(self):
        """Test apply_postgresql_include_tags with match_mode='all'."""
        from obsidian_rag.mcp_server.tools.documents_tags import (
            apply_postgresql_include_tags,
        )

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

    def test_apply_postgresql_include_tags_any_mode(self):
        """Test apply_postgresql_include_tags with match_mode='any'."""
        from obsidian_rag.mcp_server.tools.documents_tags import (
            apply_postgresql_include_tags,
        )

        mock_query = MagicMock()
        mock_query.filter.return_value = mock_query

        tag_filter = TagFilter(
            include_tags=["work", "urgent"],
            match_mode="any",
        )

        with patch("obsidian_rag.mcp_server.tools.documents_tags.or_") as mock_or:
            mock_or.return_value = "or_condition"

            result = apply_postgresql_include_tags(mock_query, tag_filter)

            mock_or.assert_called_once()
            mock_query.filter.assert_called_once_with("or_condition")
            assert result is mock_query

    def test_apply_postgresql_include_tags_empty(self):
        """Test apply_postgresql_include_tags with empty include_tags."""
        from obsidian_rag.mcp_server.tools.documents_tags import (
            apply_postgresql_include_tags,
        )

        mock_query = MagicMock()

        tag_filter = TagFilter(include_tags=[], match_mode="all")

        result = apply_postgresql_include_tags(mock_query, tag_filter)

        # Should return query unchanged
        assert result is mock_query
        mock_query.filter.assert_not_called()


class TestApplyPostgresqlExcludeTagsComplete:
    """Complete tests for apply_postgresql_exclude_tags."""

    def test_apply_postgresql_exclude_tags_with_tags(self):
        """Test apply_postgresql_exclude_tags with exclude_tags."""
        from obsidian_rag.mcp_server.tools.documents_tags import (
            apply_postgresql_exclude_tags,
        )

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

                mock_or.assert_called_once()
                mock_query.filter.assert_called_once()
                assert result is mock_query

    def test_apply_postgresql_exclude_tags_empty(self):
        """Test apply_postgresql_exclude_tags with empty exclude_tags."""
        from obsidian_rag.mcp_server.tools.documents_tags import (
            apply_postgresql_exclude_tags,
        )

        mock_query = MagicMock()

        tag_filter = TagFilter(exclude_tags=[])

        result = apply_postgresql_exclude_tags(mock_query, tag_filter)

        # Should return query unchanged
        assert result is mock_query
        mock_query.filter.assert_not_called()


class TestApplyPostgresqlTagFilterComplete:
    """Complete tests for apply_postgresql_tag_filter."""

    def test_apply_postgresql_tag_filter_both_include_and_exclude(self):
        """Test apply_postgresql_tag_filter with both include and exclude."""
        from obsidian_rag.mcp_server.tools.documents_tags import (
            apply_postgresql_tag_filter,
        )

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
        from obsidian_rag.mcp_server.tools.documents_tags import (
            apply_postgresql_tag_filter,
        )

        mock_query = MagicMock()

        result = apply_postgresql_tag_filter(mock_query, None)

        # Should return query unchanged
        assert result is mock_query


class TestTagFilterValidationEdgeCases:
    """Edge case tests for tag filter validation."""

    def test_check_tag_count_include_exceeds_max(self):
        """Test _check_tag_count with include_tags exceeding MAX_TAGS."""
        from obsidian_rag.mcp_server.tools.documents_tags import _check_tag_count

        tag_filter = TagFilter(
            include_tags=[f"tag{i}" for i in range(60)],
            exclude_tags=[],
        )

        with pytest.raises(ValueError, match="Maximum 50 include_tags allowed"):
            _check_tag_count(tag_filter)

    def test_check_tag_count_exclude_exceeds_max(self):
        """Test _check_tag_count with exclude_tags exceeding MAX_TAGS."""
        from obsidian_rag.mcp_server.tools.documents_tags import _check_tag_count

        tag_filter = TagFilter(
            include_tags=[],
            exclude_tags=[f"tag{i}" for i in range(60)],
        )

        with pytest.raises(ValueError, match="Maximum 50 exclude_tags allowed"):
            _check_tag_count(tag_filter)

    def test_check_tag_conflicts_with_conflicts(self):
        """Test _check_tag_conflicts with conflicting tags."""
        from obsidian_rag.mcp_server.tools.documents_tags import _check_tag_conflicts

        tag_filter = TagFilter(
            include_tags=["work", "urgent"],
            exclude_tags=["work", "archived"],
        )

        with pytest.raises(ValueError, match="Conflicting tags"):
            _check_tag_conflicts(tag_filter)

    def test_validate_tag_filter_with_conflict(self):
        """Test validate_tag_filter raises on conflicting tags."""
        from obsidian_rag.mcp_server.tools.documents_tags import validate_tag_filter

        tag_filter = TagFilter(
            include_tags=["work"],
            exclude_tags=["work"],
        )

        with pytest.raises(ValueError, match="Conflicting tags"):
            validate_tag_filter(tag_filter)

    def test_validate_tag_filter_with_too_many_tags(self):
        """Test validate_tag_filter raises on too many tags."""
        from obsidian_rag.mcp_server.tools.documents_tags import validate_tag_filter

        tag_filter = TagFilter(include_tags=[f"tag{i}" for i in range(60)])

        with pytest.raises(ValueError, match="Maximum 50 include_tags allowed"):
            validate_tag_filter(tag_filter)
