"""Tests for GetDocumentParams and ListDocumentsParams dataclasses."""
from obsidian_rag.mcp_server.tools.documents_params import (
    GetDocumentParams,
    ListDocumentsParams,
)


def test_get_document_params_defaults():
    """All fields default to None."""
    params = GetDocumentParams()
    assert params.vault_name is None
    assert params.file_path is None
    assert params.document_id is None


def test_get_document_params_with_values():
    """All fields populated."""
    params = GetDocumentParams(
        vault_name="Personal",
        file_path="notes/test.md",
        document_id="abc-123",
    )
    assert params.vault_name == "Personal"
    assert params.file_path == "notes/test.md"
    assert params.document_id == "abc-123"


def test_list_documents_params_defaults():
    """file_name/vault_name default None, limit=20, offset=0."""
    params = ListDocumentsParams()
    assert params.file_name is None
    assert params.vault_name is None
    assert params.limit == 20
    assert params.offset == 0


def test_list_documents_params_with_values():
    """All fields populated."""
    params = ListDocumentsParams(
        file_name="test.md",
        vault_name="Personal",
        limit=10,
        offset=5,
    )
    assert params.file_name == "test.md"
    assert params.vault_name == "Personal"
    assert params.limit == 10
    assert params.offset == 5
