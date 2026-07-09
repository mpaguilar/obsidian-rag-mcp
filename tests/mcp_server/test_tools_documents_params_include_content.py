"""Tests for include_content parameter in document params dataclasses."""

import pytest

from obsidian_rag.mcp_server.tools.documents_params import (
    GetDocumentParams,
    ListDocumentsParams,
    PaginationParams,
)


def test_pagination_params_default_include_content_true() -> None:
    """PaginationParams defaults include_content to True."""
    params = PaginationParams(limit=10, offset=0)
    assert params.include_content is True


def test_pagination_params_include_content_false() -> None:
    """PaginationParams accepts include_content=False."""
    params = PaginationParams(limit=10, offset=0, include_content=False)
    assert params.include_content is False


def test_get_document_params_default_include_content_true() -> None:
    """GetDocumentParams defaults include_content to True."""
    params = GetDocumentParams()
    assert params.include_content is True


def test_get_document_params_include_content_false() -> None:
    """GetDocumentParams accepts include_content=False."""
    params = GetDocumentParams(include_content=False)
    assert params.include_content is False


def test_list_documents_params_no_include_content_field() -> None:
    """ListDocumentsParams no longer has an include_content field."""
    params = ListDocumentsParams(file_name="x")
    assert not hasattr(params, "include_content")


def test_list_documents_params_rejects_include_content() -> None:
    """ListDocumentsParams rejects include_content construction."""
    with pytest.raises(TypeError):
        ListDocumentsParams(file_name="x", include_content=True)
