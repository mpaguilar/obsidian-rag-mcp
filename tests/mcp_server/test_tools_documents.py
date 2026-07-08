"""Tests for documents.py vault_name parameter wiring (Task 3.1)."""

from unittest.mock import MagicMock, patch

import pytest

from obsidian_rag.mcp_server.models import TagFilter
from obsidian_rag.mcp_server.tools.documents import (
    _extract_tags_postgresql,
    get_all_tags,
    get_document,
    get_documents_by_tag,
    query_documents,
)


class TestQueryDocumentsVaultName:
    """Tests for query_documents vault_name parameter."""

    @patch("obsidian_rag.mcp_server.tools.documents.query_chunks")
    def test_query_documents_chunks_passes_vault_name(
        self,
        mock_query_chunks: MagicMock,
        db_session: MagicMock,
    ) -> None:
        """query_chunks receives vault_name when use_chunks=True."""
        mock_query_chunks.return_value = []

        query_documents(
            db_session,
            [0.1] * 1536,
            use_chunks=True,
            vault_name="Work",
        )

        mock_query_chunks.assert_called_once()
        call_kwargs = mock_query_chunks.call_args.kwargs
        assert call_kwargs.get("vault_name") == "Work"

    @patch("obsidian_rag.mcp_server.tools.documents.query_documents_postgresql")
    def test_query_documents_document_path_passes_vault_name(
        self,
        mock_postgresql: MagicMock,
        db_session: MagicMock,
    ) -> None:
        """DocumentQueryParams receives vault_name in document path."""
        mock_postgresql.return_value = MagicMock()
        mock_postgresql.return_value.results = []
        mock_postgresql.return_value.total_count = 0
        mock_postgresql.return_value.has_more = False
        mock_postgresql.return_value.next_offset = None

        query_documents(
            db_session,
            [0.1] * 1536,
            vault_name="Personal",
        )

        mock_postgresql.assert_called_once()
        call_args = mock_postgresql.call_args[0][0]
        assert call_args.vault_name == "Personal"

    @patch("obsidian_rag.mcp_server.tools.documents.query_chunks")
    def test_query_documents_empty_vault_name_normalized(
        self,
        mock_query_chunks: MagicMock,
        db_session: MagicMock,
    ) -> None:
        """Empty string vault_name is normalized to None."""
        mock_query_chunks.return_value = []

        query_documents(
            db_session,
            [0.1] * 1536,
            use_chunks=True,
            vault_name="",
        )

        call_kwargs = mock_query_chunks.call_args.kwargs
        assert call_kwargs.get("vault_name") is None


class TestExtractTagsPostgresqlVaultName:
    """Tests for _extract_tags_postgresql vault_name parameter."""

    def test_extract_tags_with_vault_name_calls_validate(self):
        """_extract_tags_postgresql calls _validate_vault_exists when vault_name given."""
        mock_session = MagicMock()
        mock_query = MagicMock()
        mock_query.filter.return_value = mock_query
        mock_query.order_by.return_value = mock_query
        mock_query.all.return_value = []
        mock_session.query.return_value = mock_query

        with patch(
            "obsidian_rag.mcp_server.tools.documents._validate_vault_exists"
        ) as mock_validate:
            _extract_tags_postgresql(mock_session, pattern=None, vault_name="Work")

            mock_validate.assert_called_once_with(mock_session, "Work")

    def test_extract_tags_without_vault_name_no_validate(self):
        """_extract_tags_postgresql does not call _validate_vault_exists when vault_name is None."""
        mock_session = MagicMock()
        mock_query = MagicMock()
        mock_query.filter.return_value = mock_query
        mock_query.order_by.return_value = mock_query
        mock_query.all.return_value = []
        mock_session.query.return_value = mock_query

        with patch(
            "obsidian_rag.mcp_server.tools.documents._validate_vault_exists"
        ) as mock_validate:
            _extract_tags_postgresql(mock_session, pattern=None, vault_name=None)

            mock_validate.assert_not_called()

    def test_extract_tags_empty_vault_name_normalized(self):
        """Empty string vault_name is normalized to None."""
        mock_session = MagicMock()
        mock_query = MagicMock()
        mock_query.filter.return_value = mock_query
        mock_query.order_by.return_value = mock_query
        mock_query.all.return_value = []
        mock_session.query.return_value = mock_query

        with patch(
            "obsidian_rag.mcp_server.tools.documents._validate_vault_exists"
        ) as mock_validate:
            _extract_tags_postgresql(mock_session, pattern=None, vault_name="")

            mock_validate.assert_not_called()

    def test_extract_tags_invalid_vault_raises(self):
        """_extract_tags_postgresql raises ValueError for invalid vault."""
        mock_session = MagicMock()

        with patch(
            "obsidian_rag.mcp_server.tools.documents._validate_vault_exists",
            side_effect=ValueError("Vault 'Missing' not found. Available: none"),
        ):
            with pytest.raises(ValueError, match="Vault 'Missing' not found"):
                _extract_tags_postgresql(
                    mock_session, pattern=None, vault_name="Missing"
                )


class TestGetAllTagsVaultName:
    """Tests for get_all_tags vault_name parameter."""

    def test_get_all_tags_with_vault_name(self, db_session: MagicMock):
        """get_all_tags passes vault_name to _extract_tags_postgresql."""
        with patch(
            "obsidian_rag.mcp_server.tools.documents._extract_tags_postgresql"
        ) as mock_extract:
            mock_extract.return_value = []

            get_all_tags(
                db_session, pattern=None, limit=20, offset=0, vault_name="Work"
            )

            mock_extract.assert_called_once_with(db_session, None, "Work")

    def test_get_all_tags_without_vault_name(self, db_session: MagicMock):
        """get_all_tags passes None vault_name by default."""
        with patch(
            "obsidian_rag.mcp_server.tools.documents._extract_tags_postgresql"
        ) as mock_extract:
            mock_extract.return_value = []

            get_all_tags(db_session, pattern=None, limit=20, offset=0)

            mock_extract.assert_called_once_with(db_session, None, None)

    def test_get_all_tags_empty_vault_name_normalized(self, db_session: MagicMock):
        """Empty string vault_name is normalized to None."""
        with patch(
            "obsidian_rag.mcp_server.tools.documents._extract_tags_postgresql"
        ) as mock_extract:
            mock_extract.return_value = []

            get_all_tags(db_session, pattern=None, limit=20, offset=0, vault_name="")

            mock_extract.assert_called_once_with(db_session, None, None)


class TestGetDocumentsByTagVaultName:
    """Tests for get_documents_by_tag vault_name validation."""

    def test_get_documents_by_tag_valid_vault(self, db_session: MagicMock):
        """get_documents_by_tag calls _validate_vault_exists for valid vault."""
        tag_filter = TagFilter(include_tags=["work"])

        with patch(
            "obsidian_rag.mcp_server.tools.documents._validate_vault_exists"
        ) as mock_validate:
            with patch(
                "obsidian_rag.mcp_server.tools.documents_tags.apply_postgresql_tag_filter"
            ) as mock_apply:
                mock_query = MagicMock()
                mock_query.order_by.return_value = mock_query
                mock_query.count.return_value = 0
                mock_query.offset.return_value.limit.return_value.all.return_value = []
                mock_apply.return_value = mock_query
                db_session.query.return_value = mock_query

                get_documents_by_tag(
                    db_session, tag_filter, vault_name="Work", limit=20, offset=0
                )

                mock_validate.assert_called_once_with(db_session, "Work")

    def test_get_documents_by_tag_invalid_vault_raises(self, db_session: MagicMock):
        """get_documents_by_tag raises ValueError for invalid vault."""
        tag_filter = TagFilter(include_tags=["work"])

        with patch(
            "obsidian_rag.mcp_server.tools.documents._validate_vault_exists",
            side_effect=ValueError("Vault 'Missing' not found. Available: none"),
        ):
            with pytest.raises(ValueError, match="Vault 'Missing' not found"):
                get_documents_by_tag(
                    db_session, tag_filter, vault_name="Missing", limit=20, offset=0
                )

    def test_get_documents_by_tag_empty_vault_name_normalized(
        self, db_session: MagicMock
    ):
        """Empty string vault_name is normalized to None, skipping validation."""
        tag_filter = TagFilter(include_tags=["work"])

        with patch(
            "obsidian_rag.mcp_server.tools.documents._validate_vault_exists"
        ) as mock_validate:
            with patch(
                "obsidian_rag.mcp_server.tools.documents_tags.apply_postgresql_tag_filter"
            ) as mock_apply:
                mock_query = MagicMock()
                mock_query.order_by.return_value = mock_query
                mock_query.count.return_value = 0
                mock_query.offset.return_value.limit.return_value.all.return_value = []
                mock_apply.return_value = mock_query
                db_session.query.return_value = mock_query

                get_documents_by_tag(
                    db_session, tag_filter, vault_name="", limit=20, offset=0
                )

                mock_validate.assert_not_called()

    def test_get_documents_by_tag_none_vault_no_validation(self, db_session: MagicMock):
        """None vault_name skips validation and uses base query."""
        tag_filter = TagFilter(include_tags=["work"])

        with patch(
            "obsidian_rag.mcp_server.tools.documents._validate_vault_exists"
        ) as mock_validate:
            with patch(
                "obsidian_rag.mcp_server.tools.documents_tags.apply_postgresql_tag_filter"
            ) as mock_apply:
                mock_query = MagicMock()
                mock_query.order_by.return_value = mock_query
                mock_query.count.return_value = 0
                mock_query.offset.return_value.limit.return_value.all.return_value = []
                mock_apply.return_value = mock_query
                db_session.query.return_value = mock_query

                get_documents_by_tag(
                    db_session, tag_filter, vault_name=None, limit=20, offset=0
                )

                mock_validate.assert_not_called()


class TestQueryDocumentsDocumentPathVaultName:
    """Tests for query_documents document-level path with vault_name."""

    @patch("obsidian_rag.mcp_server.tools.documents.query_documents_postgresql")
    def test_document_path_vault_name_empty_normalized(
        self,
        mock_postgresql: MagicMock,
        db_session: MagicMock,
    ) -> None:
        """Empty string vault_name is normalized to None in document path."""
        mock_postgresql.return_value = MagicMock()
        mock_postgresql.return_value.results = []
        mock_postgresql.return_value.total_count = 0
        mock_postgresql.return_value.has_more = False
        mock_postgresql.return_value.next_offset = None

        query_documents(
            db_session,
            [0.1] * 1536,
            vault_name="",
        )

        call_args = mock_postgresql.call_args[0][0]
        assert call_args.vault_name is None


class TestGetDocumentDefensiveBranches:
    """Tests for get_document defensive RuntimeError branches."""

    def test_get_document_vault_name_none_after_validation(self, db_session: MagicMock):
        """Defensive RuntimeError when vault_name is None after validation."""
        with patch(
            "obsidian_rag.mcp_server.tools.documents._validate_get_document_params"
        ):
            with pytest.raises(RuntimeError, match="vault_name is None"):
                get_document(
                    db_session,
                    document_id=None,
                    vault_name=None,
                    file_path="test.md",
                )

    def test_get_document_file_path_none_after_validation(self, db_session: MagicMock):
        """Defensive RuntimeError when file_path is None after validation."""
        with patch(
            "obsidian_rag.mcp_server.tools.documents._validate_get_document_params"
        ):
            with pytest.raises(RuntimeError, match="file_path is None"):
                get_document(
                    db_session,
                    document_id=None,
                    vault_name="Work",
                    file_path=None,
                )
