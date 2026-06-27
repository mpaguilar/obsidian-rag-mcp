"""Tests for include_content parameter in document params dataclasses."""

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


def test_list_documents_params_default_include_content_true() -> None:
    """ListDocumentsParams defaults include_content to True."""
    params = ListDocumentsParams()
    assert params.include_content is True


def test_list_documents_params_include_content_false() -> None:
    """ListDocumentsParams accepts include_content=False."""
    params = ListDocumentsParams(include_content=False)
    assert params.include_content is False
