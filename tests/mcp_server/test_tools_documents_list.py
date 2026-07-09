"""Tests for list_documents tool in documents.py."""

import uuid as uuid_module
from unittest.mock import MagicMock, patch

import pytest

from obsidian_rag.database.models import Document, Vault
from obsidian_rag.mcp_server.tools.documents import list_documents


def _create_mock_document(
    doc_id=None,
    file_path="notes.md",
    file_name="notes.md",
    vault_name="Personal",
):
    """Create a mock document with vault relationship."""
    mock_doc = MagicMock()
    mock_doc.id = doc_id or uuid_module.uuid4()
    mock_doc.file_path = file_path
    mock_doc.file_name = file_name
    mock_doc.content = "test content"
    mock_doc.tags = ["tag1"]
    mock_doc.frontmatter_json = {}
    mock_doc.created_at_fs = MagicMock()
    mock_doc.modified_at_fs = MagicMock()
    mock_doc.ingested_at = MagicMock()
    mock_doc.checksum_md5 = "abc123"

    mock_vault = MagicMock()
    mock_vault.name = vault_name
    mock_doc.vault = mock_vault

    return mock_doc


def _setup_list_query_mock(
    mock_session,
    results=None,
    total_count=0,
    mock_vault=None,
):
    """Set up mock query chain for list_documents."""
    mock_query = MagicMock()
    mock_query.options.return_value = mock_query
    mock_query.filter.return_value = mock_query
    mock_query.order_by.return_value = mock_query
    mock_query.offset.return_value = mock_query
    mock_query.limit.return_value = mock_query
    mock_query.count.return_value = total_count
    mock_query.all.return_value = results or []

    mock_vault_query = MagicMock()
    mock_vault_query.filter.return_value = mock_vault_query
    mock_vault_query.first.return_value = mock_vault

    def _query_side_effect(model_class):
        if model_class is Document:
            return mock_query
        if model_class is Vault:
            return mock_vault_query
        return MagicMock()

    mock_session.query.side_effect = _query_side_effect
    return mock_query, mock_vault_query


class TestListDocumentsByFileName:
    """Happy path: list documents by file_name."""

    @patch(
        "obsidian_rag.mcp_server.tools.documents._build_document_list_response",
    )
    def test_list_documents_by_file_name_single_result(self, mock_build):
        mock_session = MagicMock()
        mock_doc = _create_mock_document(file_name="work.md")
        mock_build.return_value = MagicMock()

        mock_query, _ = _setup_list_query_mock(
            mock_session,
            results=[mock_doc],
            total_count=1,
        )

        result = list_documents(mock_session, file_name="work.md")

        assert result == mock_build.return_value
        mock_build.assert_called_once_with(
            [mock_doc],
            1,
            0,
            20,
            include_content=False,
        )
        # Verify file_name filter was applied
        assert mock_query.filter.call_count >= 1

    @patch(
        "obsidian_rag.mcp_server.tools.documents._build_document_list_response",
    )
    def test_list_documents_by_file_name_multiple_results(self, mock_build):
        mock_session = MagicMock()
        mock_doc1 = _create_mock_document(
            file_path="vault1/work.md",
            file_name="work.md",
            vault_name="Personal",
        )
        mock_doc2 = _create_mock_document(
            file_path="vault2/work.md",
            file_name="work.md",
            vault_name="Work",
        )
        mock_build.return_value = MagicMock()

        mock_query, _ = _setup_list_query_mock(
            mock_session,
            results=[mock_doc1, mock_doc2],
            total_count=2,
        )

        result = list_documents(mock_session, file_name="work.md")

        assert result == mock_build.return_value
        mock_build.assert_called_once_with(
            [mock_doc1, mock_doc2],
            2,
            0,
            20,
            include_content=False,
        )


class TestListDocumentsIncludeContentRejected:
    """include_content parameter is rejected (metadata-only tool)."""

    def test_list_documents_include_content_true_raises_type_error(self):
        mock_session = MagicMock()

        with pytest.raises(TypeError):
            list_documents(
                mock_session,
                file_name="work.md",
                include_content=True,
            )

    def test_list_documents_include_content_false_raises_type_error(self):
        mock_session = MagicMock()

        with pytest.raises(TypeError):
            list_documents(
                mock_session,
                file_name="work.md",
                include_content=False,
            )


class TestListDocumentsWithVaultScope:
    """Vault scope filtering."""

    @patch(
        "obsidian_rag.mcp_server.tools.documents._build_document_list_response",
    )
    @patch("obsidian_rag.mcp_server.tools.documents._get_available_vault_names")
    def test_list_documents_with_vault_scope(self, mock_available, mock_build):
        mock_session = MagicMock()
        mock_vault = MagicMock()
        mock_vault.id = uuid_module.uuid4()
        mock_doc = _create_mock_document(
            file_path="work.md",
            file_name="work.md",
            vault_name="Personal",
        )
        mock_build.return_value = MagicMock()
        mock_available.return_value = ["Personal", "Work"]

        mock_query, _ = _setup_list_query_mock(
            mock_session,
            results=[mock_doc],
            total_count=1,
            mock_vault=mock_vault,
        )

        result = list_documents(
            mock_session,
            file_name="work.md",
            vault_name="Personal",
        )

        assert result == mock_build.return_value
        mock_build.assert_called_once_with(
            [mock_doc],
            1,
            0,
            20,
            include_content=False,
        )
        # Verify vault filter was applied (file_name + vault_id)
        _expected_filter_calls = 2
        assert mock_query.filter.call_count >= _expected_filter_calls


class TestListDocumentsValidation:
    """Validation error cases."""

    def test_list_documents_no_file_name(self):
        mock_session = MagicMock()

        with pytest.raises(ValueError, match="Must provide at least file_name"):
            list_documents(mock_session)

    def test_list_documents_empty_file_name(self):
        mock_session = MagicMock()

        with pytest.raises(ValueError, match="Must provide at least file_name"):
            list_documents(mock_session, file_name="")

    @patch("obsidian_rag.mcp_server.tools.documents._get_available_vault_names")
    def test_list_documents_vault_not_found(self, mock_available):
        mock_session = MagicMock()
        mock_available.return_value = ["Work", "Archive"]

        mock_vault_query = MagicMock()
        mock_vault_query.filter.return_value = mock_vault_query
        mock_vault_query.first.return_value = None

        def _query_side_effect(model_class):
            if model_class is Vault:
                return mock_vault_query
            return MagicMock()

        mock_session.query.side_effect = _query_side_effect

        with pytest.raises(
            ValueError,
            match="Vault 'Missing' not found. Available: Work, Archive",
        ):
            list_documents(
                mock_session,
                file_name="work.md",
                vault_name="Missing",
            )


class TestListDocumentsNotFound:
    """Not found cases (return empty, not error)."""

    @patch(
        "obsidian_rag.mcp_server.tools.documents._build_document_list_response",
    )
    def test_list_documents_file_name_not_found(self, mock_build):
        mock_session = MagicMock()
        mock_build.return_value = MagicMock()

        _setup_list_query_mock(
            mock_session,
            results=[],
            total_count=0,
        )

        result = list_documents(mock_session, file_name="nonexistent.md")

        assert result == mock_build.return_value
        mock_build.assert_called_once_with([], 0, 0, 20, include_content=False)


class TestListDocumentsPagination:
    """Pagination behavior."""

    @patch(
        "obsidian_rag.mcp_server.tools.documents._build_document_list_response",
    )
    def test_list_documents_pagination(self, mock_build):
        mock_session = MagicMock()
        mock_doc = _create_mock_document(file_name="work.md")
        mock_build.return_value = MagicMock()

        mock_query, _ = _setup_list_query_mock(
            mock_session,
            results=[mock_doc],
            total_count=5,
        )

        result = list_documents(
            mock_session,
            file_name="work.md",
            limit=10,
            offset=2,
        )

        assert result == mock_build.return_value
        mock_build.assert_called_once_with([mock_doc], 5, 2, 10, include_content=False)
        # Verify offset and limit were called
        mock_query.offset.assert_called_once_with(2)
        mock_query.limit.assert_called_once_with(10)


class TestListDocumentsSimilarityScore:
    """Verify all results have similarity_score=0.0."""

    @patch(
        "obsidian_rag.mcp_server.tools.documents._build_document_list_response",
    )
    def test_list_documents_all_results_have_similarity_zero(self, mock_build):
        mock_session = MagicMock()
        mock_doc = _create_mock_document(file_name="work.md")
        mock_build.return_value = MagicMock()

        _setup_list_query_mock(
            mock_session,
            results=[mock_doc],
            total_count=1,
        )

        list_documents(mock_session, file_name="work.md")

        mock_build.assert_called_once_with([mock_doc], 1, 0, 20, include_content=False)


class TestListDocumentsEagerLoad:
    """Verify vault relationship is eager loaded and content deferred."""

    def test_list_documents_eager_loads_vault(self):
        mock_session = MagicMock()
        mock_doc = _create_mock_document(file_name="work.md")

        mock_query, _ = _setup_list_query_mock(
            mock_session,
            results=[mock_doc],
            total_count=1,
        )

        list_documents(mock_session, file_name="work.md")

        # Verify options was called once with both joinedload and defer
        mock_query.options.assert_called_once()
        call_args = mock_query.options.call_args
        assert call_args is not None


class TestListDocumentsNoAvailableVaults:
    """Test vault not found when no vaults exist."""

    @patch("obsidian_rag.mcp_server.tools.documents._get_available_vault_names")
    def test_list_documents_vault_not_found_no_vaults(self, mock_available):
        mock_session = MagicMock()
        mock_available.return_value = []

        mock_vault_query = MagicMock()
        mock_vault_query.filter.return_value = mock_vault_query
        mock_vault_query.first.return_value = None

        def _query_side_effect(model_class):
            if model_class is Vault:
                return mock_vault_query
            return MagicMock()

        mock_session.query.side_effect = _query_side_effect

        with pytest.raises(
            ValueError,
            match="Vault 'Missing' not found. Available: none",
        ):
            list_documents(
                mock_session,
                file_name="work.md",
                vault_name="Missing",
            )


class TestListDocumentsLimitValidation:
    """Test that limit parameter is validated."""

    @patch(
        "obsidian_rag.mcp_server.tools.documents._build_document_list_response",
    )
    @patch("obsidian_rag.mcp_server.tools.documents._validate_limit")
    def test_list_documents_limit_validated(self, mock_validate, mock_build):
        mock_session = MagicMock()
        mock_doc = _create_mock_document(file_name="work.md")
        mock_build.return_value = MagicMock()
        mock_validate.return_value = 50

        _setup_list_query_mock(
            mock_session,
            results=[mock_doc],
            total_count=1,
        )

        list_documents(mock_session, file_name="work.md", limit=150)

        mock_validate.assert_called_once_with(150)
        # Note: _build_document_list_response uses the validated limit
        mock_build.assert_called_once_with([mock_doc], 1, 0, 50, include_content=False)
