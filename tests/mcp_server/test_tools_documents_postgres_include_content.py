"""Tests for include_content threading in documents_postgres."""

import uuid
from unittest.mock import MagicMock, patch

from obsidian_rag.mcp_server.models import PropertyFilter
from obsidian_rag.mcp_server.tools.documents_params import (
    DocumentQueryParams,
    PaginationParams,
    PropertyFilterParams,
    PropertyQueryParams,
    QueryFilterParams,
    TagFilterParams,
)
from obsidian_rag.mcp_server.tools.documents_postgres import (
    _apply_postgresql_filters,
    _extract_distance_from_row,
    _extract_document_from_row,
    _filter_results_by_exclude,
    get_documents_by_property_postgresql,
    query_documents_postgresql,
)


def _create_mock_session_with_results(results, total_count=1):
    """Create a mock session with a query chain returning results."""
    mock_session = MagicMock()
    mock_query = MagicMock()
    mock_query.filter.return_value = mock_query
    mock_query.join.return_value = mock_query
    mock_query.order_by.return_value = mock_query
    mock_query.count.return_value = total_count
    mock_query.offset.return_value.limit.return_value.all.return_value = results
    mock_session.query.return_value = mock_query
    return mock_session, mock_query


def _create_mock_document(content="Test content"):
    """Create a mock Document with required attributes for create_document_response."""
    mock_doc = MagicMock()
    mock_doc.id = uuid.uuid4()
    mock_doc.file_path = "/test/path.md"
    mock_doc.file_name = "path.md"
    mock_doc.content = content
    mock_doc.content_vector = [0.1] * 1536
    mock_doc.frontmatter_json = {"kind": "note"}
    mock_doc.tags = ["test"]
    mock_doc.created_at_fs = MagicMock()
    mock_doc.modified_at_fs = MagicMock()
    mock_doc.vault = MagicMock()
    mock_doc.vault.name = "test_vault"
    return mock_doc


def test_query_documents_postgresql_include_content_true():
    """Test query_documents_postgresql threads include_content=True."""
    mock_doc = _create_mock_document(content="Included content")
    mock_session, _ = _create_mock_session_with_results(
        [(mock_doc, 0.5)], total_count=1
    )

    params = DocumentQueryParams(
        session=mock_session,
        query_embedding=[0.1] * 1536,
        filter_params=QueryFilterParams(
            property_filters=PropertyFilterParams(
                include_filters=None,
                exclude_filters=None,
            ),
            tag_params=TagFilterParams(tag_filter=None),
        ),
        pagination=PaginationParams(limit=20, offset=0, include_content=True),
    )

    result = query_documents_postgresql(params)

    assert result.total_count == 1
    assert len(result.results) == 1
    assert result.results[0].content == "Included content"


def test_query_documents_postgresql_include_content_false():
    """Test query_documents_postgresql threads include_content=False."""
    mock_doc = _create_mock_document(content="Should be hidden")
    mock_session, _ = _create_mock_session_with_results(
        [(mock_doc, 0.5)], total_count=1
    )

    params = DocumentQueryParams(
        session=mock_session,
        query_embedding=[0.1] * 1536,
        filter_params=QueryFilterParams(
            property_filters=PropertyFilterParams(
                include_filters=None,
                exclude_filters=None,
            ),
            tag_params=TagFilterParams(tag_filter=None),
        ),
        pagination=PaginationParams(limit=20, offset=0, include_content=False),
    )

    result = query_documents_postgresql(params)

    assert result.total_count == 1
    assert len(result.results) == 1
    assert result.results[0].content == ""


def test_query_documents_postgresql_empty_results():
    """Test query_documents_postgresql with empty results and include_content False."""
    mock_session, _ = _create_mock_session_with_results([], total_count=0)

    params = DocumentQueryParams(
        session=mock_session,
        query_embedding=[0.1] * 1536,
        filter_params=QueryFilterParams(
            property_filters=PropertyFilterParams(
                include_filters=None,
                exclude_filters=None,
            ),
            tag_params=TagFilterParams(tag_filter=None),
        ),
        pagination=PaginationParams(limit=20, offset=0, include_content=False),
    )

    result = query_documents_postgresql(params)

    assert result.total_count == 0
    assert len(result.results) == 0
    assert result.has_more is False
    assert result.next_offset is None


def test_query_documents_postgresql_with_exclude_filters():
    """Test query_documents_postgresql applies exclude filters and threads include_content."""
    mock_doc = _create_mock_document(content="Kept content")
    mock_doc.frontmatter_json = {"status": "published"}
    mock_session, _ = _create_mock_session_with_results(
        [(mock_doc, 0.5)], total_count=1
    )

    params = DocumentQueryParams(
        session=mock_session,
        query_embedding=[0.1] * 1536,
        filter_params=QueryFilterParams(
            property_filters=PropertyFilterParams(
                include_filters=None,
                exclude_filters=[
                    PropertyFilter(path="status", operator="equals", value="draft")
                ],
            ),
            tag_params=TagFilterParams(tag_filter=None),
        ),
        pagination=PaginationParams(limit=20, offset=0, include_content=False),
    )

    result = query_documents_postgresql(params)

    assert result.total_count == 1
    assert len(result.results) == 1
    assert result.results[0].content == ""


def test_get_documents_by_property_postgresql_no_include_content_change():
    """Test get_documents_by_property_postgresql returns raw documents unchanged."""
    mock_doc = _create_mock_document()
    mock_session, _ = _create_mock_session_with_results([mock_doc], total_count=1)

    params = PropertyQueryParams(
        session=mock_session,
        property_filters=PropertyFilterParams(
            include_filters=None,
            exclude_filters=None,
        ),
        tag_params=TagFilterParams(tag_filter=None),
        vault_name=None,
        pagination=PaginationParams(limit=20, offset=0, include_content=False),
    )

    results, total_count = get_documents_by_property_postgresql(params)

    assert total_count == 1
    assert len(results) == 1
    assert results[0] is mock_doc


def test_get_documents_by_property_postgresql_with_vault_name():
    """Test get_documents_by_property_postgresql with vault_name filter."""
    mock_doc = _create_mock_document()
    mock_session, _ = _create_mock_session_with_results([mock_doc], total_count=1)

    params = PropertyQueryParams(
        session=mock_session,
        property_filters=PropertyFilterParams(
            include_filters=None,
            exclude_filters=None,
        ),
        tag_params=TagFilterParams(tag_filter=None),
        vault_name="test_vault",
        pagination=PaginationParams(limit=20, offset=0, include_content=True),
    )

    results, total_count = get_documents_by_property_postgresql(params)

    assert total_count == 1
    assert results[0] is mock_doc


def test_get_documents_by_property_postgresql_with_include_filters():
    """Test get_documents_by_property_postgresql with include filters."""
    mock_doc = _create_mock_document()
    mock_doc.frontmatter_json = {"status": "published"}
    mock_session, _ = _create_mock_session_with_results([mock_doc], total_count=1)

    params = PropertyQueryParams(
        session=mock_session,
        property_filters=PropertyFilterParams(
            include_filters=[
                PropertyFilter(path="status", operator="equals", value="published")
            ],
            exclude_filters=None,
        ),
        tag_params=TagFilterParams(tag_filter=None),
        vault_name=None,
        pagination=PaginationParams(limit=20, offset=0, include_content=True),
    )

    results, total_count = get_documents_by_property_postgresql(params)

    assert total_count == 1
    assert results[0] is mock_doc


def test_get_documents_by_property_postgresql_with_exclude_filters():
    """Test get_documents_by_property_postgresql with exclude filters."""
    mock_doc1 = _create_mock_document()
    mock_doc1.frontmatter_json = {"status": "draft"}
    mock_doc2 = _create_mock_document()
    mock_doc2.frontmatter_json = {"status": "published"}

    mock_session, _ = _create_mock_session_with_results(
        [mock_doc1, mock_doc2], total_count=2
    )

    params = PropertyQueryParams(
        session=mock_session,
        property_filters=PropertyFilterParams(
            include_filters=None,
            exclude_filters=[
                PropertyFilter(path="status", operator="equals", value="draft")
            ],
        ),
        tag_params=TagFilterParams(tag_filter=None),
        vault_name=None,
        pagination=PaginationParams(limit=20, offset=0, include_content=True),
    )

    results, total_count = get_documents_by_property_postgresql(params)

    assert total_count == 1
    assert len(results) == 1
    assert results[0] is mock_doc2


def test_filter_results_by_exclude_no_filters():
    """Test _filter_results_by_exclude returns all results when no filters."""
    results = [(MagicMock(), 0.1), (MagicMock(), 0.2)]
    filtered = _filter_results_by_exclude(results, None)
    assert filtered == results

    filtered = _filter_results_by_exclude(results, [])
    assert filtered == results


def test_filter_results_by_exclude_with_document_attr():
    """Test _filter_results_by_exclude with row objects having Document attribute."""
    from obsidian_rag.mcp_server.models import PropertyFilter

    mock_row1 = MagicMock()
    mock_row1.Document = MagicMock()
    mock_row1.Document.frontmatter_json = {"status": "draft"}

    mock_row2 = MagicMock()
    mock_row2.Document = MagicMock()
    mock_row2.Document.frontmatter_json = {"status": "published"}

    results = [mock_row1, mock_row2]
    exclude_filters = [PropertyFilter(path="status", operator="equals", value="draft")]

    filtered = _filter_results_by_exclude(results, exclude_filters)

    assert len(filtered) == 1
    assert filtered[0] is mock_row2


def test_filter_results_by_exclude_with_tuple():
    """Test _filter_results_by_exclude with tuple rows."""
    from obsidian_rag.mcp_server.models import PropertyFilter

    mock_doc1 = MagicMock()
    mock_doc1.frontmatter_json = {"status": "draft"}
    mock_doc2 = MagicMock()
    mock_doc2.frontmatter_json = {"status": "published"}

    results = [(mock_doc1, 0.1), (mock_doc2, 0.2)]
    exclude_filters = [PropertyFilter(path="status", operator="equals", value="draft")]

    filtered = _filter_results_by_exclude(results, exclude_filters)

    assert len(filtered) == 1
    assert filtered[0][0] is mock_doc2


def test_apply_postgresql_filters_with_property_filters():
    """Test _apply_postgresql_filters applies property and tag filters."""
    from obsidian_rag.mcp_server.models import TagFilter

    mock_query = MagicMock()
    mock_query.filter.return_value = mock_query

    property_filters = [PropertyFilter(path="status", operator="equals", value="draft")]
    tag_params = TagFilterParams(tag_filter=TagFilter(include_tags=["work"]))

    with patch(
        "obsidian_rag.mcp_server.tools.documents_postgres.apply_postgresql_property_filter"
    ) as mock_apply_prop:
        with patch(
            "obsidian_rag.mcp_server.tools.documents_postgres.apply_postgresql_tag_filter"
        ) as mock_apply_tag:
            mock_apply_prop.return_value = mock_query
            mock_apply_tag.return_value = mock_query

            result = _apply_postgresql_filters(mock_query, property_filters, tag_params)

            mock_apply_prop.assert_called_once_with(mock_query, property_filters[0])
            mock_apply_tag.assert_called_once()
            assert result is mock_query


def test_apply_postgresql_filters_no_property_filters():
    """Test _apply_postgresql_filters with no property filters."""
    from obsidian_rag.mcp_server.models import TagFilter

    mock_query = MagicMock()
    tag_params = TagFilterParams(tag_filter=TagFilter(include_tags=["work"]))

    with patch(
        "obsidian_rag.mcp_server.tools.documents_postgres.apply_postgresql_tag_filter"
    ) as mock_apply_tag:
        mock_apply_tag.return_value = mock_query

        result = _apply_postgresql_filters(mock_query, None, tag_params)

        mock_apply_tag.assert_called_once_with(mock_query, tag_params.tag_filter)
        assert result is mock_query


def test_extract_document_from_row_with_document_attr():
    """Test _extract_document_from_row when row has Document attribute."""
    mock_doc = MagicMock()
    mock_row = MagicMock()
    mock_row.Document = mock_doc
    assert _extract_document_from_row(mock_row) is mock_doc


def test_extract_document_from_row_with_tuple():
    """Test _extract_document_from_row when row is a tuple."""
    mock_doc = MagicMock()
    assert _extract_document_from_row((mock_doc, 0.5)) is mock_doc


def test_extract_document_from_row_direct():
    """Test _extract_document_from_row when row is a Document directly."""
    mock_doc = object()
    assert _extract_document_from_row(mock_doc) is mock_doc


def test_extract_distance_from_row_with_distance_attr():
    """Test _extract_distance_from_row when row has distance attribute."""
    mock_row = MagicMock()
    mock_row.distance = 0.75
    assert _extract_distance_from_row(mock_row) == 0.75


def test_extract_distance_from_row_with_tuple():
    """Test _extract_distance_from_row when row is a tuple with distance."""
    assert _extract_distance_from_row((MagicMock(), 0.75)) == 0.75


def test_extract_distance_from_row_default():
    """Test _extract_distance_from_row default value."""

    class PlainRow:
        pass

    assert _extract_distance_from_row(PlainRow()) == 0.0
