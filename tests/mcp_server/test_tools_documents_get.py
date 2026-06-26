"""Tests for get_document tool in documents.py."""

import uuid as uuid_module
from unittest.mock import MagicMock, patch

import pytest

from obsidian_rag.database.models import Document, Vault
from obsidian_rag.mcp_server.tools.documents import get_document


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


def _setup_query_mocks(mock_session, mock_doc=None, mock_vault=None):
    """Set up mock query chains for Document and Vault lookups."""
    mock_doc_query = MagicMock()
    mock_doc_query.options.return_value = mock_doc_query
    mock_doc_query.filter.return_value = mock_doc_query
    mock_doc_query.first.return_value = mock_doc

    mock_vault_query = MagicMock()
    mock_vault_query.filter.return_value = mock_vault_query
    mock_vault_query.first.return_value = mock_vault

    def _query_side_effect(model_class):
        if model_class is Document:
            return mock_doc_query
        if model_class is Vault:
            return mock_vault_query
        return MagicMock()

    mock_session.query.side_effect = _query_side_effect
    return mock_doc_query, mock_vault_query


class TestGetDocumentByVaultPath:
    """Happy path: lookup by vault_name + file_path."""

    @patch("obsidian_rag.mcp_server.tools.documents.create_document_response")
    @patch("obsidian_rag.mcp_server.tools.documents._get_available_vault_names")
    def test_get_document_by_vault_and_path(self, mock_available, mock_response):
        mock_session = MagicMock()
        mock_vault = MagicMock()
        mock_vault.id = uuid_module.uuid4()
        mock_doc = _create_mock_document()

        _setup_query_mocks(mock_session, mock_doc=mock_doc, mock_vault=mock_vault)
        mock_available.return_value = ["Personal", "Work"]
        mock_response.return_value = MagicMock()

        result = get_document(
            mock_session,
            vault_name="Personal",
            file_path="notes.md",
        )

        assert result == mock_response.return_value
        mock_response.assert_called_once_with(mock_doc, similarity_score=0.0)


class TestGetDocumentByDocumentId:
    """Happy path: lookup by document_id."""

    @patch("obsidian_rag.mcp_server.tools.documents.create_document_response")
    def test_get_document_by_document_id(self, mock_response):
        mock_session = MagicMock()
        doc_id = str(uuid_module.uuid4())
        mock_doc = _create_mock_document(doc_id=uuid_module.UUID(doc_id))

        _setup_query_mocks(mock_session, mock_doc=mock_doc)
        mock_response.return_value = MagicMock()

        result = get_document(mock_session, document_id=doc_id)

        assert result == mock_response.return_value
        mock_response.assert_called_once_with(mock_doc, similarity_score=0.0)


class TestGetDocumentValidationErrors:
    """Validation error cases."""

    def test_get_document_file_path_without_vault_name(self):
        mock_session = MagicMock()

        with pytest.raises(
            ValueError,
            match="Must provide either document_id, or vault_name and file_path",
        ):
            get_document(mock_session, file_path="notes.md")

    def test_get_document_no_parameters(self):
        mock_session = MagicMock()

        with pytest.raises(ValueError, match="Must provide either document_id"):
            get_document(mock_session)

    def test_get_document_only_vault_name(self):
        mock_session = MagicMock()

        with pytest.raises(ValueError, match="Must provide either document_id"):
            get_document(mock_session, vault_name="Personal")

    def test_get_document_id_with_file_path_no_vault(self):
        mock_session = MagicMock()
        doc_id = str(uuid_module.uuid4())

        with pytest.raises(ValueError, match="vault_name is required"):
            get_document(
                mock_session,
                document_id=doc_id,
                file_path="notes.md",
            )


class TestGetDocumentNotFound:
    """Not found error cases."""

    @patch("obsidian_rag.mcp_server.tools.documents._get_available_vault_names")
    def test_get_document_document_id_not_found(self, mock_available):
        mock_session = MagicMock()
        doc_id = str(uuid_module.uuid4())
        mock_available.return_value = ["Personal", "Work"]

        mock_doc_query = MagicMock()
        mock_doc_query.options.return_value = mock_doc_query
        mock_doc_query.filter.return_value = mock_doc_query
        mock_doc_query.first.return_value = None

        def _query_side_effect(model_class):
            if model_class is Document:
                return mock_doc_query
            return MagicMock()

        mock_session.query.side_effect = _query_side_effect

        with pytest.raises(ValueError, match=f"Document with id '{doc_id}' not found"):
            get_document(mock_session, document_id=doc_id)

    def test_get_document_invalid_uuid_format(self):
        mock_session = MagicMock()

        with pytest.raises(ValueError, match="Invalid document_id UUID format"):
            get_document(mock_session, document_id="not-a-uuid")

    @patch("obsidian_rag.mcp_server.tools.documents._get_available_vault_names")
    def test_get_document_vault_not_found(self, mock_available):
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
            match="Vault 'Personal' not found. Available: Work, Archive",
        ):
            get_document(
                mock_session,
                vault_name="Personal",
                file_path="notes.md",
            )

    @patch("obsidian_rag.mcp_server.tools.documents._get_available_vault_names")
    def test_get_document_path_not_found_in_vault(self, mock_available):
        mock_session = MagicMock()
        mock_vault = MagicMock()
        mock_vault.id = uuid_module.uuid4()
        mock_available.return_value = ["Personal"]

        mock_doc_query = MagicMock()
        mock_doc_query.options.return_value = mock_doc_query
        mock_doc_query.filter.return_value = mock_doc_query
        mock_doc_query.first.return_value = None

        mock_vault_query = MagicMock()
        mock_vault_query.filter.return_value = mock_vault_query
        mock_vault_query.first.return_value = mock_vault

        def _query_side_effect(model_class):
            if model_class is Document:
                return mock_doc_query
            if model_class is Vault:
                return mock_vault_query
            return MagicMock()

        mock_session.query.side_effect = _query_side_effect

        with pytest.raises(
            ValueError,
            match="Document 'missing.md' not found in vault 'Personal'",
        ):
            get_document(
                mock_session,
                vault_name="Personal",
                file_path="missing.md",
            )


class TestGetDocumentEdgeCases:
    """Edge case behaviors."""

    @patch("obsidian_rag.mcp_server.tools.documents.create_document_response")
    @patch("obsidian_rag.mcp_server.tools.documents._get_available_vault_names")
    def test_get_document_file_path_case_sensitive(
        self,
        mock_available,
        mock_response,
    ):
        mock_session = MagicMock()
        mock_vault = MagicMock()
        mock_vault.id = uuid_module.uuid4()
        mock_doc = _create_mock_document(file_path="Notes.md")

        _setup_query_mocks(mock_session, mock_doc=mock_doc, mock_vault=mock_vault)
        mock_available.return_value = ["Personal"]
        mock_response.return_value = MagicMock()

        result = get_document(
            mock_session,
            vault_name="Personal",
            file_path="Notes.md",
        )

        assert result == mock_response.return_value
        # Verify that query was called and filter chain executed
        assert mock_session.query.call_count >= 1

    @patch("obsidian_rag.mcp_server.tools.documents.create_document_response")
    @patch("obsidian_rag.mcp_server.tools.documents._get_available_vault_names")
    def test_get_document_file_path_leading_slash(
        self,
        mock_available,
        mock_response,
    ):
        mock_session = MagicMock()
        mock_vault = MagicMock()
        mock_vault.id = uuid_module.uuid4()
        mock_doc = _create_mock_document(file_path="/notes.md")

        _setup_query_mocks(mock_session, mock_doc=mock_doc, mock_vault=mock_vault)
        mock_available.return_value = ["Personal"]
        mock_response.return_value = MagicMock()

        result = get_document(
            mock_session,
            vault_name="Personal",
            file_path="/notes.md",
        )

        assert result == mock_response.return_value
        assert mock_session.query.call_count >= 1

    @patch("obsidian_rag.mcp_server.tools.documents.create_document_response")
    def test_get_document_returns_similarity_zero(self, mock_response):
        mock_session = MagicMock()
        doc_id = str(uuid_module.uuid4())
        mock_doc = _create_mock_document(doc_id=uuid_module.UUID(doc_id))

        _setup_query_mocks(mock_session, mock_doc=mock_doc)
        mock_response.return_value = MagicMock()

        get_document(mock_session, document_id=doc_id)

        mock_response.assert_called_once_with(mock_doc, similarity_score=0.0)

    @patch("obsidian_rag.mcp_server.tools.documents.create_document_response")
    @patch("obsidian_rag.mcp_server.tools.documents._get_available_vault_names")
    def test_get_document_returns_full_content(
        self,
        mock_available,
        mock_response,
    ):
        mock_session = MagicMock()
        mock_vault = MagicMock()
        mock_vault.id = uuid_module.uuid4()
        mock_doc = _create_mock_document()
        mock_doc.content = "# Full Content\n\nThis is the full document."

        _setup_query_mocks(mock_session, mock_doc=mock_doc, mock_vault=mock_vault)
        mock_available.return_value = ["Personal"]
        mock_response.return_value = MagicMock()

        get_document(
            mock_session,
            vault_name="Personal",
            file_path="notes.md",
        )

        mock_response.assert_called_once_with(mock_doc, similarity_score=0.0)


class TestGetDocumentPriority:
    """Test that document_id takes priority over vault_name+file_path."""

    @patch("obsidian_rag.mcp_server.tools.documents.create_document_response")
    def test_document_id_takes_priority(self, mock_response):
        mock_session = MagicMock()
        doc_id = str(uuid_module.uuid4())
        mock_doc = _create_mock_document(doc_id=uuid_module.UUID(doc_id))

        _setup_query_mocks(mock_session, mock_doc=mock_doc)
        mock_response.return_value = MagicMock()

        result = get_document(
            mock_session,
            document_id=doc_id,
            vault_name="Personal",
            file_path="notes.md",
        )

        assert result == mock_response.return_value
        # Verify that Vault query was NOT called (document_id path used)
        vault_query_calls = [
            call for call in mock_session.query.call_args_list if call[0][0] is Vault
        ]
        assert len(vault_query_calls) == 0


class TestGetDocumentNoAvailableVaults:
    """Test vault not found when no vaults exist."""

    @patch("obsidian_rag.mcp_server.tools.documents._get_available_vault_names")
    def test_get_document_vault_not_found_no_vaults(self, mock_available):
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
            get_document(
                mock_session,
                vault_name="Missing",
                file_path="notes.md",
            )
