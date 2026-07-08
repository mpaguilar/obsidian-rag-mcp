"""Tests for document query parameter dataclasses."""

from obsidian_rag.mcp_server.tools.documents_params import (
    DocumentQueryParams,
    PaginationParams,
    PropertyFilterParams,
    QueryFilterParams,
    TagFilterParams,
)


def test_document_query_params_has_vault_name_field():
    """DocumentQueryParams must have vault_name as the last field."""
    from dataclasses import fields

    field_names = [f.name for f in fields(DocumentQueryParams)]
    assert field_names == [
        "session",
        "query_embedding",
        "filter_params",
        "pagination",
        "vault_name",
    ]


def test_document_query_params_vault_name_default():
    """vault_name defaults to None."""
    from dataclasses import fields

    vault_name_field = [
        f for f in fields(DocumentQueryParams) if f.name == "vault_name"
    ][0]
    assert vault_name_field.default is None


def test_document_query_params_can_be_constructed_with_vault_name():
    """DocumentQueryParams accepts vault_name keyword argument."""
    mock_session = object()
    params = DocumentQueryParams(
        session=mock_session,  # type: ignore[arg-type]
        query_embedding=[0.1, 0.2],
        filter_params=QueryFilterParams(
            property_filters=PropertyFilterParams(
                include_filters=None,
                exclude_filters=None,
            ),
            tag_params=TagFilterParams(tag_filter=None),
        ),
        pagination=PaginationParams(limit=10, offset=0),
        vault_name="Personal",
    )
    assert params.vault_name == "Personal"


def test_document_query_params_vault_name_defaults_to_none():
    """When vault_name is omitted, it defaults to None."""
    mock_session = object()
    params = DocumentQueryParams(
        session=mock_session,  # type: ignore[arg-type]
        query_embedding=[0.1, 0.2],
        filter_params=QueryFilterParams(
            property_filters=PropertyFilterParams(
                include_filters=None,
                exclude_filters=None,
            ),
            tag_params=TagFilterParams(tag_filter=None),
        ),
        pagination=PaginationParams(limit=10, offset=0),
    )
    assert params.vault_name is None


def test_document_query_params_vault_name_can_be_none_explicitly():
    """vault_name can be set to None explicitly."""
    mock_session = object()
    params = DocumentQueryParams(
        session=mock_session,  # type: ignore[arg-type]
        query_embedding=[0.1, 0.2],
        filter_params=QueryFilterParams(
            property_filters=PropertyFilterParams(
                include_filters=None,
                exclude_filters=None,
            ),
            tag_params=TagFilterParams(tag_filter=None),
        ),
        pagination=PaginationParams(limit=10, offset=0),
        vault_name=None,
    )
    assert params.vault_name is None
