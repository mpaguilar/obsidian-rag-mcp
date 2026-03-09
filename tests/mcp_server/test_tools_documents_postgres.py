"""Tests for MCP document tools PostgreSQL path."""

import uuid
from unittest.mock import MagicMock, patch

import pytest

from obsidian_rag.mcp_server.tools.documents_params import PaginationParams


class TestQueryDocumentsPostgres:
    """Tests for query_documents with PostgreSQL dialect."""

    def test_query_documents_postgresql(self):
        """Test query_documents with PostgreSQL dialect."""
        from obsidian_rag.mcp_server.tools.documents import query_documents

        # Create a mock session with PostgreSQL dialect
        mock_session = MagicMock()
        mock_bind = MagicMock()
        mock_bind.dialect.name = "postgresql"
        mock_session.bind = mock_bind

        # Mock the query results
        mock_doc = MagicMock()
        mock_doc.id = uuid.uuid4()
        mock_doc.file_path = "/test/path.md"
        mock_doc.file_name = "path.md"
        mock_doc.content = "Test content"
        mock_doc.content_vector = [0.1] * 1536
        mock_doc.frontmatter_json = {"kind": "note"}
        mock_doc.tags = ["test"]
        mock_doc.created_at_fs = MagicMock()
        mock_doc.modified_at_fs = MagicMock()
        mock_doc.vault = MagicMock()
        mock_doc.vault.name = "test_vault"

        mock_distance = 0.5

        # Setup query chain mocks
        mock_query = MagicMock()
        mock_query.filter.return_value = mock_query
        mock_query.order_by.return_value = mock_query
        mock_query.count.return_value = 1
        mock_query.offset.return_value.limit.return_value.all.return_value = [
            (mock_doc, mock_distance)
        ]

        mock_session.query.return_value = mock_query

        query_embedding = [0.1] * 1536
        pagination = PaginationParams(limit=20, offset=0)
        result = query_documents(mock_session, query_embedding, pagination=pagination)

        assert result.total_count == 1
        assert len(result.results) == 1
        assert result.results[0].file_name == "path.md"
        assert result.has_more is False
        assert result.next_offset is None

    def test_query_documents_postgresql_empty_results(self):
        """Test query_documents with PostgreSQL returning empty results."""
        from obsidian_rag.mcp_server.tools.documents import query_documents

        # Create a mock session with PostgreSQL dialect
        mock_session = MagicMock()
        mock_bind = MagicMock()
        mock_bind.dialect.name = "postgresql"
        mock_session.bind = mock_bind

        # Setup query chain mocks for empty results
        mock_query = MagicMock()
        mock_query.filter.return_value = mock_query
        mock_query.order_by.return_value = mock_query
        mock_query.count.return_value = 0
        mock_query.offset.return_value.limit.return_value.all.return_value = []

        mock_session.query.return_value = mock_query

        query_embedding = [0.1] * 1536
        pagination = PaginationParams(limit=20, offset=0)
        result = query_documents(mock_session, query_embedding, pagination=pagination)

        assert result.total_count == 0
        assert len(result.results) == 0
        assert result.has_more is False
        assert result.next_offset is None


class TestFilterResultsByExclude:
    """Tests for _filter_results_by_exclude function (TASK-086)."""

    def test_filter_results_by_exclude_with_property_filters(self):
        """Test _filter_results_by_exclude with property filters (TASK-086)."""
        from obsidian_rag.mcp_server.tools.documents_postgres import (
            _filter_results_by_exclude,
        )
        from obsidian_rag.mcp_server.models import PropertyFilter

        # Create mock documents
        mock_doc1 = MagicMock()
        mock_doc1.frontmatter_json = {"status": "draft"}

        mock_doc2 = MagicMock()
        mock_doc2.frontmatter_json = {"status": "published"}

        mock_doc3 = MagicMock()
        mock_doc3.frontmatter_json = {"status": "archived"}

        # Create results as tuples (doc, distance)
        results = [
            (mock_doc1, 0.1),
            (mock_doc2, 0.2),
            (mock_doc3, 0.3),
        ]

        # Create exclude filter for status=draft
        exclude_filters = [
            PropertyFilter(path="status", operator="equals", value="draft")
        ]

        filtered = _filter_results_by_exclude(results, exclude_filters)

        # Should exclude doc1 (status=draft), keep doc2 and doc3
        assert len(filtered) == 2
        assert mock_doc1 not in [r[0] for r in filtered]
        assert mock_doc2 in [r[0] for r in filtered]
        assert mock_doc3 in [r[0] for r in filtered]

    def test_filter_results_by_exclude_empty_filters(self):
        """Test _filter_results_by_exclude with empty filters."""
        from obsidian_rag.mcp_server.tools.documents_postgres import (
            _filter_results_by_exclude,
        )

        results = [(MagicMock(), 0.1), (MagicMock(), 0.2)]

        # Empty filters should return all results
        filtered = _filter_results_by_exclude(results, [])
        assert len(filtered) == 2

        # None filters should also return all results
        filtered = _filter_results_by_exclude(results, None)
        assert len(filtered) == 2

    def test_filter_results_by_exclude_with_row_objects(self):
        """Test _filter_results_by_exclude with row objects having Document attribute."""
        from obsidian_rag.mcp_server.tools.documents_postgres import (
            _filter_results_by_exclude,
        )
        from obsidian_rag.mcp_server.models import PropertyFilter

        # Create mock row objects with Document attribute
        mock_row1 = MagicMock()
        mock_row1.Document = MagicMock()
        mock_row1.Document.frontmatter_json = {"status": "draft"}

        mock_row2 = MagicMock()
        mock_row2.Document = MagicMock()
        mock_row2.Document.frontmatter_json = {"status": "published"}

        results = [mock_row1, mock_row2]

        exclude_filters = [
            PropertyFilter(path="status", operator="equals", value="draft")
        ]

        filtered = _filter_results_by_exclude(results, exclude_filters)

        # Should exclude row1 (status=draft), keep row2
        assert len(filtered) == 1
        assert filtered[0] == mock_row2


class TestApplyPostgresqlFilters:
    """Tests for _apply_postgresql_filters function (TASK-087)."""

    def test_apply_postgresql_filters_with_property_and_tag_filters(self):
        """Test _apply_postgresql_filters with property and tag filters (TASK-087)."""
        from obsidian_rag.mcp_server.tools.documents_postgres import (
            _apply_postgresql_filters,
        )
        from obsidian_rag.mcp_server.tools.documents_params import TagFilterParams
        from obsidian_rag.mcp_server.models import PropertyFilter, TagFilter

        mock_query = MagicMock()
        mock_query.filter.return_value = mock_query

        # Create property filters
        property_filters = [
            PropertyFilter(path="status", operator="equals", value="draft")
        ]

        # Create tag filter params
        tag_filter = TagFilter(include_tags=["work"])
        tag_params = TagFilterParams(tag_filter=tag_filter)

        with patch(
            "obsidian_rag.mcp_server.tools.documents_postgres.apply_postgresql_property_filter"
        ) as mock_apply_prop:
            with patch(
                "obsidian_rag.mcp_server.tools.documents_postgres.apply_postgresql_tag_filter"
            ) as mock_apply_tag:
                mock_apply_prop.return_value = mock_query
                mock_apply_tag.return_value = mock_query

                result = _apply_postgresql_filters(
                    mock_query, property_filters, tag_params
                )

                # Verify property filter was applied
                mock_apply_prop.assert_called_once_with(mock_query, property_filters[0])
                # Verify tag filter was applied
                mock_apply_tag.assert_called_once_with(mock_query, tag_filter)
                assert result is mock_query

    def test_apply_postgresql_filters_no_property_filters(self):
        """Test _apply_postgresql_filters with no property filters."""
        from obsidian_rag.mcp_server.tools.documents_postgres import (
            _apply_postgresql_filters,
        )
        from obsidian_rag.mcp_server.tools.documents_params import TagFilterParams
        from obsidian_rag.mcp_server.models import TagFilter

        mock_query = MagicMock()

        tag_filter = TagFilter(include_tags=["work"])
        tag_params = TagFilterParams(tag_filter=tag_filter)

        with patch(
            "obsidian_rag.mcp_server.tools.documents_postgres.apply_postgresql_tag_filter"
        ) as mock_apply_tag:
            mock_apply_tag.return_value = mock_query

            result = _apply_postgresql_filters(mock_query, None, tag_params)

            # Verify only tag filter was applied
            mock_apply_tag.assert_called_once_with(mock_query, tag_filter)
            assert result is mock_query


class TestGetDocumentsByPropertyPostgresql:
    """Tests for get_documents_by_property_postgresql function (TASK-088)."""

    def test_get_documents_by_property_postgresql_full_flow(self):
        """Test get_documents_by_property_postgresql full flow (TASK-088)."""
        from obsidian_rag.mcp_server.tools.documents_postgres import (
            get_documents_by_property_postgresql,
        )
        from obsidian_rag.mcp_server.tools.documents_params import (
            PaginationParams,
            PropertyFilterParams,
            PropertyQueryParams,
            TagFilterParams,
        )
        from obsidian_rag.mcp_server.models import PropertyFilter, TagFilter

        # Create mock session
        mock_session = MagicMock()

        # Create mock documents
        mock_doc1 = MagicMock()
        mock_doc1.file_name = "doc1.md"
        mock_doc1.frontmatter_json = {"status": "draft"}

        mock_doc2 = MagicMock()
        mock_doc2.file_name = "doc2.md"
        mock_doc2.frontmatter_json = {"status": "published"}

        # Setup query chain
        mock_query = MagicMock()
        mock_query.filter.return_value = mock_query
        mock_query.order_by.return_value = mock_query
        mock_query.count.return_value = 2
        mock_query.offset.return_value.limit.return_value.all.return_value = [
            mock_doc1,
            mock_doc2,
        ]

        mock_session.query.return_value = mock_query

        # Create params
        property_filters = PropertyFilterParams(
            include_filters=[
                PropertyFilter(path="status", operator="equals", value="draft")
            ],
            exclude_filters=None,
        )
        tag_params = TagFilterParams(tag_filter=None)
        pagination = PaginationParams(limit=20, offset=0)

        params = PropertyQueryParams(
            session=mock_session,
            property_filters=property_filters,
            tag_params=tag_params,
            vault_name=None,
            pagination=pagination,
        )

        with patch(
            "obsidian_rag.mcp_server.tools.documents_postgres.apply_postgresql_property_filter"
        ) as mock_apply_prop:
            with patch(
                "obsidian_rag.mcp_server.tools.documents_postgres.apply_postgresql_tag_filter"
            ) as mock_apply_tag:
                mock_apply_prop.return_value = mock_query
                mock_apply_tag.return_value = mock_query

                results, total_count = get_documents_by_property_postgresql(params)

                assert total_count == 2
                assert len(results) == 2
                assert results[0] == mock_doc1
                assert results[1] == mock_doc2

    def test_get_documents_by_property_postgresql_with_vault_root(self):
        """Test get_documents_by_property_postgresql with vault_root filter."""
        from obsidian_rag.mcp_server.tools.documents_postgres import (
            get_documents_by_property_postgresql,
        )
        from obsidian_rag.mcp_server.tools.documents_params import (
            PaginationParams,
            PropertyFilterParams,
            PropertyQueryParams,
            TagFilterParams,
        )

        mock_session = MagicMock()
        mock_query = MagicMock()
        mock_query.filter.return_value = mock_query
        mock_query.order_by.return_value = mock_query
        mock_query.count.return_value = 1
        mock_query.offset.return_value.limit.return_value.all.return_value = [
            MagicMock()
        ]

        mock_session.query.return_value = mock_query

        property_filters = PropertyFilterParams(
            include_filters=None, exclude_filters=None
        )
        tag_params = TagFilterParams(tag_filter=None)
        pagination = PaginationParams(limit=20, offset=0)

        params = PropertyQueryParams(
            session=mock_session,
            property_filters=property_filters,
            tag_params=tag_params,
            vault_name="test_vault",
            pagination=pagination,
        )

        with patch(
            "obsidian_rag.mcp_server.tools.documents_postgres.apply_postgresql_tag_filter"
        ) as mock_apply_tag:
            mock_apply_tag.return_value = mock_query

            results, total_count = get_documents_by_property_postgresql(params)

            # Verify vault_name filter was applied
            # When vault_name is provided, the function joins with Vault table
            # and applies a filter - verify query was called
            assert mock_session.query.call_count >= 1

    def test_get_documents_by_property_postgresql_with_exclude_filters(self):
        """Test get_documents_by_property_postgresql with exclude filters."""
        from obsidian_rag.mcp_server.tools.documents_postgres import (
            get_documents_by_property_postgresql,
        )
        from obsidian_rag.mcp_server.tools.documents_params import (
            PaginationParams,
            PropertyFilterParams,
            PropertyQueryParams,
            TagFilterParams,
        )
        from obsidian_rag.mcp_server.models import PropertyFilter

        mock_session = MagicMock()

        # Create mock documents
        mock_doc1 = MagicMock()
        mock_doc1.frontmatter_json = {"status": "draft"}

        mock_doc2 = MagicMock()
        mock_doc2.frontmatter_json = {"status": "published"}

        mock_query = MagicMock()
        mock_query.filter.return_value = mock_query
        mock_query.order_by.return_value = mock_query
        mock_query.count.return_value = 2
        mock_query.offset.return_value.limit.return_value.all.return_value = [
            mock_doc1,
            mock_doc2,
        ]

        mock_session.query.return_value = mock_query

        property_filters = PropertyFilterParams(
            include_filters=None,
            exclude_filters=[
                PropertyFilter(path="status", operator="equals", value="draft")
            ],
        )
        tag_params = TagFilterParams(tag_filter=None)
        pagination = PaginationParams(limit=20, offset=0)

        params = PropertyQueryParams(
            session=mock_session,
            property_filters=property_filters,
            tag_params=tag_params,
            vault_name=None,
            pagination=pagination,
        )

        with patch(
            "obsidian_rag.mcp_server.tools.documents_postgres.apply_postgresql_tag_filter"
        ) as mock_apply_tag:
            mock_apply_tag.return_value = mock_query

            results, total_count = get_documents_by_property_postgresql(params)

            # Should filter out doc1 (status=draft), keep only doc2
            assert total_count == 1
            assert len(results) == 1
            assert results[0] == mock_doc2


class TestGetDocumentsByTagPostgres:
    """Tests for get_documents_by_tag with PostgreSQL (lines 232-239)."""

    def test_get_documents_by_tag_postgresql_path(self):
        """Test get_documents_by_tag with PostgreSQL dialect."""
        from obsidian_rag.mcp_server.tools.documents import get_documents_by_tag
        from obsidian_rag.mcp_server.models import TagFilter

        # Create mock session with PostgreSQL dialect
        mock_session = MagicMock()
        mock_bind = MagicMock()
        mock_bind.dialect.name = "postgresql"
        mock_session.bind = mock_bind

        # Setup mock query chain
        mock_query = MagicMock()
        mock_query.filter.return_value = mock_query
        mock_query.order_by.return_value = mock_query
        mock_query.count.return_value = 2

        # Create mock document
        mock_doc = MagicMock()
        mock_doc.id = uuid.uuid4()
        mock_doc.file_path = "/test/doc.md"
        mock_doc.file_name = "doc.md"
        mock_doc.content = "Test content"
        mock_doc.tags = ["work"]
        mock_doc.frontmatter_json = {"kind": "note"}
        mock_doc.created_at_fs = MagicMock()
        mock_doc.modified_at_fs = MagicMock()
        mock_doc.vault = MagicMock()
        mock_doc.vault.name = "test_vault"

        mock_query.offset.return_value.limit.return_value.all.return_value = [
            mock_doc,
            mock_doc,
        ]
        mock_session.query.return_value = mock_query

        with patch(
            "obsidian_rag.mcp_server.tools.documents_tags.apply_postgresql_tag_filter"
        ) as mock_apply_filter:
            mock_apply_filter.return_value = mock_query

            result = get_documents_by_tag(
                mock_session,
                TagFilter(include_tags=["work"]),
                limit=20,
                offset=0,
            )

            assert result.total_count == 2
            assert len(result.results) == 2


class TestGetDocumentsByPropertyPostgresPath:
    """Tests for get_documents_by_property PostgreSQL path (line 331)."""

    def test_get_documents_by_property_postgresql_branch(self):
        """Test get_documents_by_property with PostgreSQL dialect (line 331)."""
        from obsidian_rag.mcp_server.tools.documents import get_documents_by_property
        from obsidian_rag.mcp_server.models import PropertyFilter
        from obsidian_rag.mcp_server.tools.documents_params import (
            PaginationParams,
            PropertyFilterParams,
        )

        # Create mock session with PostgreSQL dialect
        mock_session = MagicMock()
        mock_bind = MagicMock()
        mock_bind.dialect.name = "postgresql"
        mock_session.bind = mock_bind

        # Create mock document
        mock_doc = MagicMock()
        mock_doc.id = uuid.uuid4()
        mock_doc.file_path = "/test/doc.md"
        mock_doc.file_name = "doc.md"
        mock_doc.content = "Test content"
        mock_doc.tags = ["work"]
        mock_doc.frontmatter_json = {"status": "draft"}
        mock_doc.created_at_fs = MagicMock()
        mock_doc.modified_at_fs = MagicMock()
        mock_doc.vault = MagicMock()
        mock_doc.vault.name = "test_vault"

        with patch(
            "obsidian_rag.mcp_server.tools.documents.get_documents_by_property_postgresql"
        ) as mock_get:
            mock_get.return_value = ([mock_doc], 1)

            filter_params = PropertyFilterParams(
                include_filters=[
                    PropertyFilter(path="status", operator="equals", value="draft")
                ],
                exclude_filters=None,
            )
            pagination = PaginationParams(limit=20, offset=0)

            result = get_documents_by_property(
                mock_session,
                property_filters=filter_params,
                pagination=pagination,
            )

            assert result.total_count == 1
            assert len(result.results) == 1
            mock_get.assert_called_once()


class TestExtractTagsPostgresqlWithPattern:
    """Tests for _extract_tags_postgresql with pattern (lines 358-364)."""

    def test_extract_tags_postgresql_with_pattern(self):
        """Test _extract_tags_postgresql with pattern filtering (lines 358-364)."""
        from obsidian_rag.mcp_server.tools.documents import _extract_tags_postgresql

        # Create mock session
        mock_session = MagicMock()

        # Setup mock query chain for tags with pattern
        mock_query = MagicMock()
        mock_query.filter.return_value = mock_query
        mock_query.order_by.return_value = mock_query

        # Mock the distinct/unnest query
        mock_tag_row = MagicMock()
        mock_tag_row.tag = "work"
        mock_query.all.return_value = [mock_tag_row]

        mock_session.query.return_value = mock_query

        result = _extract_tags_postgresql(mock_session, pattern="wor*")

        assert "work" in result


class TestGetAllTagsPostgresPath:
    """Tests for get_all_tags PostgreSQL path (line 430)."""

    def test_get_all_tags_postgresql_branch(self):
        """Test get_all_tags with PostgreSQL dialect (line 430)."""
        from obsidian_rag.mcp_server.tools.documents import get_all_tags

        # Create mock session with PostgreSQL dialect
        mock_session = MagicMock()
        mock_bind = MagicMock()
        mock_bind.dialect.name = "postgresql"
        mock_session.bind = mock_bind

        with patch(
            "obsidian_rag.mcp_server.tools.documents._extract_tags_postgresql"
        ) as mock_extract:
            mock_extract.return_value = ["work", "personal", "ideas"]

            result = get_all_tags(mock_session, pattern=None, limit=20, offset=0)

            assert result.total_count == 3
            assert result.tags == ["work", "personal", "ideas"]
            mock_extract.assert_called_once_with(mock_session, None)


class TestExtractDocumentFromRow:
    """Tests for _extract_document_from_row function (lines 47, 50)."""

    def test_extract_document_from_row_with_document_attr(self):
        """Test _extract_document_from_row when row has Document attribute (line 47)."""
        from obsidian_rag.mcp_server.tools.documents_postgres import (
            _extract_document_from_row,
        )

        mock_doc = MagicMock()
        mock_row = MagicMock()
        mock_row.Document = mock_doc

        result = _extract_document_from_row(mock_row)
        assert result == mock_doc

    def test_extract_document_from_row_direct(self):
        """Test _extract_document_from_row when row is a Document directly (line 50)."""
        from obsidian_rag.mcp_server.tools.documents_postgres import (
            _extract_document_from_row,
        )

        # Use a plain object that doesn't have Document attribute and isn't a tuple
        class PlainDoc:
            pass

        doc = PlainDoc()
        result = _extract_document_from_row(doc)
        assert result == doc


class TestExtractDistanceFromRow:
    """Tests for _extract_distance_from_row function (lines 64, 67)."""

    def test_extract_distance_from_row_with_distance_attr(self):
        """Test _extract_distance_from_row when row has distance attribute (line 64)."""
        from obsidian_rag.mcp_server.tools.documents_postgres import (
            _extract_distance_from_row,
        )

        mock_row = MagicMock()
        mock_row.distance = 0.75

        result = _extract_distance_from_row(mock_row)
        assert result == 0.75

    def test_extract_distance_from_row_default(self):
        """Test _extract_distance_from_row when row has no distance (line 67)."""
        from obsidian_rag.mcp_server.tools.documents_postgres import (
            _extract_distance_from_row,
        )

        # Use a plain object that doesn't have distance attribute and isn't a tuple
        class PlainRow:
            pass

        row = PlainRow()
        result = _extract_distance_from_row(row)
        assert result == 0.0
