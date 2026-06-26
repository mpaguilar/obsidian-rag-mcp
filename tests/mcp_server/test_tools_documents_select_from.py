"""Tests for _extract_tags_postgresql subquery approach (REQ-004)."""

from unittest.mock import MagicMock


class TestExtractTagsPostgresqlSubquery:
    """Tests for _extract_tags_postgresql subquery approach."""

    def test_extract_tags_postgresql_returns_tags(self):
        """Verify _extract_tags_postgresql returns tags from mock results."""
        from obsidian_rag.mcp_server.tools.documents import _extract_tags_postgresql

        mock_session = MagicMock()
        mock_query = MagicMock()
        mock_query.filter.return_value = mock_query
        mock_query.order_by.return_value = mock_query

        mock_row = MagicMock()
        mock_row.tag = "work"
        mock_query.all.return_value = [mock_row]

        mock_session.query.return_value = mock_query

        result = _extract_tags_postgresql(mock_session, pattern=None)

        assert "work" in result

    def test_extract_tags_postgresql_filters_none_tags(self):
        """Verify _extract_tags_postgresql filters out None tags from results."""
        from obsidian_rag.mcp_server.tools.documents import _extract_tags_postgresql

        mock_session = MagicMock()
        mock_query = MagicMock()
        mock_query.filter.return_value = mock_query
        mock_query.order_by.return_value = mock_query

        mock_row_valid = MagicMock()
        mock_row_valid.tag = "work"
        mock_row_none = MagicMock()
        mock_row_none.tag = None
        mock_query.all.return_value = [mock_row_valid, mock_row_none]

        mock_session.query.return_value = mock_query

        result = _extract_tags_postgresql(mock_session, pattern=None)

        assert "work" in result
        assert None not in result

    def test_extract_tags_postgresql_with_pattern(self):
        """Verify pattern filtering works with subquery approach."""
        from obsidian_rag.mcp_server.tools.documents import _extract_tags_postgresql

        mock_session = MagicMock()
        mock_query = MagicMock()
        mock_query.filter.return_value = mock_query
        mock_query.order_by.return_value = mock_query

        mock_row = MagicMock()
        mock_row.tag = "work-item"
        mock_query.all.return_value = [mock_row]

        mock_session.query.return_value = mock_query

        result = _extract_tags_postgresql(mock_session, pattern="work*")

        assert "work-item" in result
        # filter is called twice: once for isnot(None), once for pattern
        assert mock_query.filter.call_count == 2

    def test_extract_tags_postgresql_no_select_from_on_outer_query(self):
        """Verify the outer query does NOT call select_from — it's in the subquery now."""
        from obsidian_rag.mcp_server.tools.documents import _extract_tags_postgresql

        mock_session = MagicMock()
        mock_query = MagicMock()
        mock_query.filter.return_value = mock_query
        mock_query.order_by.return_value = mock_query
        mock_query.all.return_value = []

        mock_session.query.return_value = mock_query

        _extract_tags_postgresql(mock_session, pattern=None)

        mock_query.select_from.assert_not_called()

    def test_extract_tags_postgresql_empty_results(self):
        """Verify empty result list when no tags exist."""
        from obsidian_rag.mcp_server.tools.documents import _extract_tags_postgresql

        mock_session = MagicMock()
        mock_query = MagicMock()
        mock_query.filter.return_value = mock_query
        mock_query.order_by.return_value = mock_query
        mock_query.all.return_value = []

        mock_session.query.return_value = mock_query

        result = _extract_tags_postgresql(mock_session, pattern=None)

        assert result == []
