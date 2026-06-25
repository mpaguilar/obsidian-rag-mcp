"""Unit tests for MCP server handlers and additional tests."""

from pathlib import Path
from typing import cast
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner


runner = CliRunner()


class TestDocumentHandlers:
    """Tests for document tool handlers."""

    def test_get_documents_by_tag_handler_full_flow(self):
        """Test _get_documents_by_tag_handler full flow."""
        from obsidian_rag.mcp_server.handlers import _get_documents_by_tag_handler

        mock_db_manager = MagicMock()
        mock_session = MagicMock()
        mock_db_manager.get_session.return_value.__enter__ = MagicMock(
            return_value=mock_session
        )
        mock_db_manager.get_session.return_value.__exit__ = MagicMock(
            return_value=False
        )

        with patch(
            "obsidian_rag.mcp_server.handlers.get_documents_by_tag_tool"
        ) as mock_tool:
            mock_result = MagicMock()
            mock_result.model_dump.return_value = {
                "results": [],
                "total_count": 0,
                "has_more": False,
            }
            mock_tool.return_value = mock_result

            # Create params as a dict-like object
            params = {
                "include_tags": ["work"],
                "exclude_tags": [],
                "match_mode": "all",
                "vault_root": None,
                "limit": 20,
                "offset": 0,
            }
            result = _get_documents_by_tag_handler(mock_db_manager, params)  # type: ignore[arg-type]

            assert result == {"results": [], "total_count": 0, "has_more": False}
            mock_tool.assert_called_once()

    def test_get_all_tags_handler_full_flow(self):
        """Test _get_all_tags_handler full flow."""
        from obsidian_rag.mcp_server.handlers import _get_all_tags_handler

        mock_db_manager = MagicMock()
        mock_session = MagicMock()
        mock_db_manager.get_session.return_value.__enter__ = MagicMock(
            return_value=mock_session
        )
        mock_db_manager.get_session.return_value.__exit__ = MagicMock(
            return_value=False
        )

        with patch("obsidian_rag.mcp_server.handlers.get_all_tags_tool") as mock_tool:
            mock_result = MagicMock()
            mock_result.model_dump.return_value = {
                "tags": [],
                "total_count": 0,
                "has_more": False,
            }
            mock_tool.return_value = mock_result

            result = _get_all_tags_handler(mock_db_manager, "work*", 20, 0)

            assert result == {"tags": [], "total_count": 0, "has_more": False}
            mock_tool.assert_called_once_with(
                session=mock_session, pattern="work*", limit=20, offset=0
            )

    def test_convert_property_filters_with_valid_filters(self):
        """Test _convert_property_filters with valid filters."""
        from obsidian_rag.mcp_server.handlers import _convert_property_filters

        properties = [
            {"path": "status", "operator": "equals", "value": "draft"},
            {"path": "priority", "operator": "equals", "value": "high"},
        ]

        result = _convert_property_filters(properties)

        assert result is not None
        assert len(result) == len(properties)
        assert result[0].path == "status"
        assert result[0].operator == "equals"
        assert result[0].value == "draft"

    def test_create_tag_filter_with_none_filters(self):
        """Test _create_tag_filter with None filters."""
        from obsidian_rag.mcp_server.handlers import _create_tag_filter

        result = _create_tag_filter(None)

        assert result is None

    def test_create_tag_filter_with_empty_tags(self):
        """Test _create_tag_filter with empty tags."""
        from obsidian_rag.mcp_server.handlers import (
            _create_tag_filter,
            QueryFilterParams,
        )

        filters = QueryFilterParams(
            include_properties=None,
            exclude_properties=None,
            include_tags=None,
            exclude_tags=None,
            match_mode="all",
        )
        result = _create_tag_filter(filters)

        assert result is None


class TestValidateIngestPath:
    """Tests for _validate_ingest_path."""

    def test_validate_ingest_path_with_parent_directory_references(self):
        """Test _validate_ingest_path with parent directory references."""
        from obsidian_rag.mcp_server.handlers import _validate_ingest_path

        with pytest.raises(ValueError, match="parent directory references"):
            _validate_ingest_path("/vault/../etc/passwd")

    def test_validate_ingest_path_with_nonexistent_path(self):
        """Test _validate_ingest_path with non-existent path."""
        from obsidian_rag.mcp_server.handlers import _validate_ingest_path

        with pytest.raises(ValueError, match="does not exist"):
            _validate_ingest_path("/nonexistent/path")

    def test_validate_ingest_path_with_non_directory_path(self):
        """Test _validate_ingest_path with non-directory path."""
        from obsidian_rag.mcp_server.handlers import _validate_ingest_path

        with runner.isolated_filesystem():
            test_file = "test.txt"
            Path(test_file).write_text("test")
            with pytest.raises(ValueError, match="not a directory"):
                _validate_ingest_path(test_file)


class TestIngestHandler:
    """Tests for _ingest_handler."""

    @patch("obsidian_rag.mcp_server.handlers._validate_ingest_path")
    def test_ingest_handler_with_path_override(self, mock_validate):
        """Test _ingest_handler with path_override."""
        from obsidian_rag.mcp_server.handlers import _ingest_handler

        mock_db_manager = MagicMock()
        mock_settings = MagicMock()
        mock_settings.vault_root = "/default/vault"

        vault_config = MagicMock()
        vault_config.container_path = "/test/vault"
        mock_settings.get_vault.return_value = vault_config

        mock_validate.return_value = Path("/custom/path")

        with runner.isolated_filesystem():
            # Create a temporary directory for testing
            vault_path = Path("vault")
            vault_path.mkdir()

            with patch(
                "obsidian_rag.mcp_server.handlers.IngestionService"
            ) as mock_service:
                mock_instance = MagicMock()
                mock_result = MagicMock()
                mock_result.total = 1
                mock_result.new = 1
                mock_result.updated = 0
                mock_result.unchanged = 0
                mock_result.errors = 0
                mock_result.processing_time_seconds = 1.0
                mock_result.to_dict.return_value = {
                    "total": 1,
                    "new": 1,
                    "updated": 0,
                    "unchanged": 0,
                    "errors": 0,
                    "processing_time_seconds": 1.0,
                    "message": "Ingested 1 documents",
                }
                mock_instance.ingest_vault.return_value = mock_result
                mock_service.return_value = mock_instance

                mock_embedding_provider = MagicMock()
                from obsidian_rag.mcp_server.handlers import IngestHandlerParams
                from obsidian_rag.services.ingestion import IngestVaultOptions

                params = IngestHandlerParams(
                    settings=mock_settings,
                    db_manager=mock_db_manager,
                    embedding_provider=mock_embedding_provider,
                    vault_name="Obsidian Vault",
                    path_override="/custom/path",
                )
                result = _ingest_handler(params)

                assert result["total"] == 1
                mock_validate.assert_called_once_with("/custom/path")
                call_args = mock_instance.ingest_vault.call_args
                ingest_options = call_args.args[1]
                assert isinstance(ingest_options, IngestVaultOptions)
                assert ingest_options.no_delete is False


class TestListVaultsHandler:
    """Tests for _list_vaults_handler."""

    def test_list_vaults_handler(self):
        """Test _list_vaults_handler returns vault list."""
        from obsidian_rag.mcp_server.handlers import _list_vaults_handler

        mock_db_manager = MagicMock()
        mock_session = MagicMock()
        mock_db_manager.get_session.return_value.__enter__ = MagicMock(
            return_value=mock_session
        )
        mock_db_manager.get_session.return_value.__exit__ = MagicMock(return_value=None)

        # Mock the list_vaults_tool response
        mock_result = MagicMock()
        mock_result.model_dump.return_value = {
            "total_count": 1,
            "results": [
                {
                    "id": "vault-1",
                    "name": "Personal",
                    "description": "Personal vault",
                    "host_path": "/data/personal",
                    "document_count": 5,
                }
            ],
            "has_more": False,
            "next_offset": None,
        }

        with patch(
            "obsidian_rag.mcp_server.handlers.list_vaults_tool"
        ) as mock_list_vaults:
            mock_list_vaults.return_value = mock_result
            result = _list_vaults_handler(
                db_manager=mock_db_manager, limit=20, offset=0
            )

        assert result["total_count"] == 1
        results = cast("list[dict[str, object]]", result["results"])
        assert len(results) == 1
        assert results[0]["name"] == "Personal"


class TestIngestHandlerVaultNotFound:
    """Tests for _ingest_handler vault not found error."""

    def test_ingest_handler_vault_not_found(self):
        """Test _ingest_handler raises error when vault not found."""
        from obsidian_rag.mcp_server.handlers import (
            _ingest_handler,
            IngestHandlerParams,
        )

        mock_db_manager = MagicMock()
        mock_settings = MagicMock()
        mock_settings.get_vault.return_value = None
        mock_settings.get_vault_names.return_value = ["Personal", "Work"]

        mock_embedding_provider = MagicMock()

        params = IngestHandlerParams(
            settings=mock_settings,
            db_manager=mock_db_manager,
            embedding_provider=mock_embedding_provider,
            vault_name="NonExistent",
            path_override=None,
        )

        with pytest.raises(ValueError) as exc_info:
            _ingest_handler(params)

        assert "Vault 'NonExistent' not found" in str(exc_info.value)
        assert "Personal" in str(exc_info.value)
        assert "Work" in str(exc_info.value)


class TestConvertPropertyFilters:
    """Tests for _convert_property_filters."""

    def test_convert_property_filters_empty(self):
        """Test _convert_property_filters with empty properties."""
        from obsidian_rag.mcp_server.handlers import _convert_property_filters

        result = _convert_property_filters([])
        assert result is None

    def test_convert_property_filters_none(self):
        """Test _convert_property_filters with None."""
        from obsidian_rag.mcp_server.handlers import _convert_property_filters

        result = _convert_property_filters(None)
        assert result is None


class TestCreateTagFilter:
    """Tests for _create_tag_filter."""

    def test_create_tag_filter_invalid_match_mode(self):
        """Test _create_tag_filter with invalid match_mode defaults to 'all'."""
        from obsidian_rag.mcp_server.handlers import (
            _create_tag_filter,
            QueryFilterParams,
        )

        filters = QueryFilterParams(
            include_properties=None,
            exclude_properties=None,
            include_tags=["tag1"],
            exclude_tags=None,
            match_mode="invalid",  # type: ignore[arg-type]
        )
        result = _create_tag_filter(filters)

        assert result is not None
        assert result.match_mode == "all"
        assert result.include_tags == ["tag1"]


class TestIngestHandlerForce:
    """Tests for _ingest_handler force parameter."""

    def test_ingest_handler_params_has_force_field(self):
        """Test IngestHandlerParams accepts force field."""
        from obsidian_rag.mcp_server.handlers import IngestHandlerParams
        from unittest.mock import MagicMock

        params = IngestHandlerParams(
            settings=MagicMock(),
            db_manager=MagicMock(),
            embedding_provider=None,
            vault_name="test",
            path_override=None,
            no_delete=False,
            force=True,
        )
        assert params.force is True

    def test_ingest_handler_passes_force_to_ingest_vault_options(self):
        """Test _ingest_handler passes force to IngestVaultOptions."""
        from obsidian_rag.mcp_server.handlers import _ingest_handler, IngestHandlerParams
        from obsidian_rag.services.ingestion import IngestVaultOptions
        from unittest.mock import MagicMock, patch
        from pathlib import Path

        params = IngestHandlerParams(
            settings=MagicMock(),
            db_manager=MagicMock(),
            embedding_provider=None,
            vault_name="test",
            path_override=None,
            no_delete=False,
            force=True,
        )

        mock_settings = MagicMock()
        vault_config = MagicMock()
        vault_config.container_path = "/test/vault"
        mock_settings.get_vault.return_value = vault_config
        params.settings = mock_settings

        with patch("obsidian_rag.mcp_server.handlers._validate_ingest_path", return_value=Path("/test/vault")):
            with patch("obsidian_rag.mcp_server.handlers.IngestionService") as mock_service:
                mock_instance = MagicMock()
                mock_result = MagicMock()
                mock_result.to_dict.return_value = {"total": 0}
                mock_instance.ingest_vault.return_value = mock_result
                mock_service.return_value = mock_instance

                _ingest_handler(params)

        mock_instance.ingest_vault.assert_called_once()
        call_args = mock_instance.ingest_vault.call_args
        ingest_options = call_args.args[1]
        assert isinstance(ingest_options, IngestVaultOptions)
        assert ingest_options.force is True


class TestValidateIngestPathSuccess:
    """Tests for _validate_ingest_path success path."""

    def test_validate_ingest_path_success(self):
        """Test _validate_ingest_path returns Path on success."""
        from obsidian_rag.mcp_server.handlers import _validate_ingest_path

        with runner.isolated_filesystem():
            # Create a valid directory
            test_dir = Path("test_vault")
            test_dir.mkdir()

            result = _validate_ingest_path(str(test_dir))

            assert isinstance(result, Path)
            assert result.exists()
            assert result.is_dir()


class TestGetDocumentHandlerParams:
    """Tests for GetDocumentHandlerParams dataclass."""

    def test_get_document_handler_params_dataclass(self):
        """Fields populated correctly."""
        from obsidian_rag.mcp_server.handlers import GetDocumentHandlerParams

        db_manager = MagicMock()
        params = GetDocumentHandlerParams(
            db_manager=db_manager,
            vault_name="Personal",
            file_path="notes.md",
            document_id="abc-123",
        )
        assert params.db_manager is db_manager
        assert params.vault_name == "Personal"
        assert params.file_path == "notes.md"
        assert params.document_id == "abc-123"

    def test_get_document_handler_params_defaults(self):
        """Default values are correct."""
        from obsidian_rag.mcp_server.handlers import GetDocumentHandlerParams

        params = GetDocumentHandlerParams(db_manager=MagicMock())
        assert params.vault_name is None
        assert params.file_path is None
        assert params.document_id is None


class TestListDocumentsHandlerParams:
    """Tests for ListDocumentsHandlerParams dataclass."""

    def test_list_documents_handler_params_dataclass(self):
        """Fields populated correctly."""
        from obsidian_rag.mcp_server.handlers import ListDocumentsHandlerParams

        db_manager = MagicMock()
        limit = 10
        offset = 5
        params = ListDocumentsHandlerParams(
            db_manager=db_manager,
            file_name="notes.md",
            vault_name="Personal",
            limit=limit,
            offset=offset,
        )
        assert params.db_manager is db_manager
        assert params.file_name == "notes.md"
        assert params.vault_name == "Personal"
        assert params.limit == limit
        assert params.offset == offset

    def test_list_documents_handler_params_defaults(self):
        """Default values are correct."""
        from obsidian_rag.mcp_server.handlers import ListDocumentsHandlerParams

        default_limit = 20
        params = ListDocumentsHandlerParams(db_manager=MagicMock())
        assert params.file_name is None
        assert params.vault_name is None
        assert params.limit == default_limit
        assert params.offset == 0


class TestGetDocumentHandler:
    """Tests for _get_document_handler."""

    @patch(
        "obsidian_rag.mcp_server.tools.documents.get_document",
        create=True,
    )
    def test_get_document_handler_success(self, mock_get_document):
        """Test successful document retrieval."""
        from obsidian_rag.mcp_server.handlers import (
            _get_document_handler,
            GetDocumentHandlerParams,
        )

        mock_db_manager = MagicMock()
        mock_session = MagicMock()
        mock_db_manager.get_session.return_value.__enter__ = MagicMock(
            return_value=mock_session
        )
        mock_db_manager.get_session.return_value.__exit__ = MagicMock(
            return_value=False
        )

        mock_result = MagicMock()
        mock_result.model_dump.return_value = {
            "id": "doc-1",
            "file_path": "notes.md",
        }
        mock_get_document.return_value = mock_result

        params = GetDocumentHandlerParams(
            db_manager=mock_db_manager,
            vault_name="Personal",
            file_path="notes.md",
        )
        result = _get_document_handler(params)

        assert result == {"id": "doc-1", "file_path": "notes.md"}
        mock_get_document.assert_called_once_with(
            session=mock_session,
            vault_name="Personal",
            file_path="notes.md",
            document_id=None,
        )

    @patch(
        "obsidian_rag.mcp_server.tools.documents.get_document",
        create=True,
    )
    def test_get_document_handler_by_id_success(self, mock_get_document):
        """Test successful document retrieval by document_id."""
        from obsidian_rag.mcp_server.handlers import (
            _get_document_handler,
            GetDocumentHandlerParams,
        )

        mock_db_manager = MagicMock()
        mock_session = MagicMock()
        mock_db_manager.get_session.return_value.__enter__ = MagicMock(
            return_value=mock_session
        )
        mock_db_manager.get_session.return_value.__exit__ = MagicMock(
            return_value=False
        )

        mock_result = MagicMock()
        mock_result.model_dump.return_value = {
            "id": "doc-1",
            "file_path": "notes.md",
        }
        mock_get_document.return_value = mock_result

        params = GetDocumentHandlerParams(
            db_manager=mock_db_manager,
            document_id="abc-123",
        )
        result = _get_document_handler(params)

        assert result == {"id": "doc-1", "file_path": "notes.md"}
        mock_get_document.assert_called_once_with(
            session=mock_session,
            vault_name=None,
            file_path=None,
            document_id="abc-123",
        )

    @patch(
        "obsidian_rag.mcp_server.tools.documents.get_document",
        create=True,
    )
    def test_get_document_handler_by_vault_path_success(self, mock_get_document):
        """Test successful document retrieval by vault_name and file_path."""
        from obsidian_rag.mcp_server.handlers import (
            _get_document_handler,
            GetDocumentHandlerParams,
        )

        mock_db_manager = MagicMock()
        mock_session = MagicMock()
        mock_db_manager.get_session.return_value.__enter__ = MagicMock(
            return_value=mock_session
        )
        mock_db_manager.get_session.return_value.__exit__ = MagicMock(
            return_value=False
        )

        mock_result = MagicMock()
        mock_result.model_dump.return_value = {
            "id": "doc-1",
            "file_path": "notes.md",
            "vault_name": "Personal",
        }
        mock_get_document.return_value = mock_result

        params = GetDocumentHandlerParams(
            db_manager=mock_db_manager,
            vault_name="Personal",
            file_path="notes.md",
        )
        result = _get_document_handler(params)

        assert result == {
            "id": "doc-1",
            "file_path": "notes.md",
            "vault_name": "Personal",
        }
        mock_get_document.assert_called_once_with(
            session=mock_session,
            vault_name="Personal",
            file_path="notes.md",
            document_id=None,
        )

    @patch(
        "obsidian_rag.mcp_server.tools.documents.get_document",
        create=True,
    )
    def test_get_document_handler_no_params(self, mock_get_document):
        """Test error when no lookup parameters provided."""
        from obsidian_rag.mcp_server.handlers import (
            _get_document_handler,
            GetDocumentHandlerParams,
        )

        mock_db_manager = MagicMock()
        mock_session = MagicMock()
        mock_db_manager.get_session.return_value.__enter__ = MagicMock(
            return_value=mock_session
        )
        mock_db_manager.get_session.return_value.__exit__ = MagicMock(
            return_value=False
        )

        mock_get_document.side_effect = ValueError(
            "Must provide either document_id, or vault_name and file_path"
        )

        params = GetDocumentHandlerParams(
            db_manager=mock_db_manager,
        )
        result = _get_document_handler(params)

        assert result == {
            "success": False,
            "error": "Must provide either document_id, or vault_name and file_path",
        }

    @patch(
        "obsidian_rag.mcp_server.tools.documents.get_document",
        create=True,
    )
    def test_get_document_handler_path_without_vault(self, mock_get_document):
        """Test error when file_path provided without vault_name."""
        from obsidian_rag.mcp_server.handlers import (
            _get_document_handler,
            GetDocumentHandlerParams,
        )

        mock_db_manager = MagicMock()
        mock_session = MagicMock()
        mock_db_manager.get_session.return_value.__enter__ = MagicMock(
            return_value=mock_session
        )
        mock_db_manager.get_session.return_value.__exit__ = MagicMock(
            return_value=False
        )

        mock_get_document.side_effect = ValueError(
            "vault_name is required when using file_path "
            "(file_path is only unique per vault)"
        )

        params = GetDocumentHandlerParams(
            db_manager=mock_db_manager,
            file_path="notes.md",
        )
        result = _get_document_handler(params)

        assert result == {
            "success": False,
            "error": "vault_name is required when using file_path "
            "(file_path is only unique per vault)",
        }

    @patch(
        "obsidian_rag.mcp_server.tools.documents.get_document",
        create=True,
    )
    def test_get_document_handler_not_found(self, mock_get_document):
        """Test error when document not found."""
        from obsidian_rag.mcp_server.handlers import (
            _get_document_handler,
            GetDocumentHandlerParams,
        )

        mock_db_manager = MagicMock()
        mock_session = MagicMock()
        mock_db_manager.get_session.return_value.__enter__ = MagicMock(
            return_value=mock_session
        )
        mock_db_manager.get_session.return_value.__exit__ = MagicMock(
            return_value=False
        )

        mock_get_document.side_effect = ValueError("Document not found")

        params = GetDocumentHandlerParams(
            db_manager=mock_db_manager,
            vault_name="Personal",
            file_path="missing.md",
        )
        result = _get_document_handler(params)

        assert result == {"success": False, "error": "Document not found"}

    @patch(
        "obsidian_rag.mcp_server.tools.documents.get_document",
        create=True,
    )
    def test_get_document_handler_invalid_uuid(self, mock_get_document):
        """Test error when document_id is invalid UUID."""
        from obsidian_rag.mcp_server.handlers import (
            _get_document_handler,
            GetDocumentHandlerParams,
        )

        mock_db_manager = MagicMock()
        mock_session = MagicMock()
        mock_db_manager.get_session.return_value.__enter__ = MagicMock(
            return_value=mock_session
        )
        mock_db_manager.get_session.return_value.__exit__ = MagicMock(
            return_value=False
        )

        mock_get_document.side_effect = ValueError(
            "Invalid document_id UUID format: 'invalid-uuid'"
        )

        params = GetDocumentHandlerParams(
            db_manager=mock_db_manager,
            document_id="invalid-uuid",
        )
        result = _get_document_handler(params)

        assert result == {
            "success": False,
            "error": "Invalid document_id UUID format: 'invalid-uuid'",
        }


class TestListDocumentsHandler:
    """Tests for _list_documents_handler."""

    @patch(
        "obsidian_rag.mcp_server.tools.documents.list_documents",
        create=True,
    )
    def test_list_documents_handler_success(self, mock_list_documents):
        """Test successful document list retrieval."""
        from obsidian_rag.mcp_server.handlers import (
            _list_documents_handler,
            ListDocumentsHandlerParams,
        )

        mock_db_manager = MagicMock()
        mock_session = MagicMock()
        mock_db_manager.get_session.return_value.__enter__ = MagicMock(
            return_value=mock_session
        )
        mock_db_manager.get_session.return_value.__exit__ = MagicMock(
            return_value=False
        )

        mock_result = MagicMock()
        mock_result.model_dump.return_value = {
            "results": [],
            "total_count": 0,
            "has_more": False,
        }
        mock_list_documents.return_value = mock_result

        params = ListDocumentsHandlerParams(
            db_manager=mock_db_manager,
            file_name="notes.md",
            vault_name="Personal",
            limit=10,
            offset=5,
        )
        result = _list_documents_handler(params)

        assert result == {"results": [], "total_count": 0, "has_more": False}
        mock_list_documents.assert_called_once_with(
            session=mock_session,
            file_name="notes.md",
            vault_name="Personal",
            limit=10,
            offset=5,
        )

    @patch(
        "obsidian_rag.mcp_server.tools.documents.list_documents",
        create=True,
    )
    def test_list_documents_handler_value_error(self, mock_list_documents):
        """Test error handling for invalid parameters."""
        from obsidian_rag.mcp_server.handlers import (
            _list_documents_handler,
            ListDocumentsHandlerParams,
        )

        mock_db_manager = MagicMock()
        mock_session = MagicMock()
        mock_db_manager.get_session.return_value.__enter__ = MagicMock(
            return_value=mock_session
        )
        mock_db_manager.get_session.return_value.__exit__ = MagicMock(
            return_value=False
        )

        mock_list_documents.side_effect = ValueError("Invalid vault")

        params = ListDocumentsHandlerParams(db_manager=mock_db_manager)
        result = _list_documents_handler(params)

        assert result == {"success": False, "error": "Invalid vault"}

    @patch(
        "obsidian_rag.mcp_server.tools.documents.list_documents",
        create=True,
    )
    def test_list_documents_handler_empty_results(self, mock_list_documents):
        """Test empty results return empty list dict, not error."""
        from obsidian_rag.mcp_server.handlers import (
            _list_documents_handler,
            ListDocumentsHandlerParams,
        )

        mock_db_manager = MagicMock()
        mock_session = MagicMock()
        mock_db_manager.get_session.return_value.__enter__ = MagicMock(
            return_value=mock_session
        )
        mock_db_manager.get_session.return_value.__exit__ = MagicMock(
            return_value=False
        )

        mock_result = MagicMock()
        mock_result.model_dump.return_value = {
            "results": [],
            "total_count": 0,
            "has_more": False,
        }
        mock_list_documents.return_value = mock_result

        params = ListDocumentsHandlerParams(
            db_manager=mock_db_manager,
            file_name="nonexistent.md",
            limit=10,
            offset=0,
        )
        result = _list_documents_handler(params)

        assert result == {"results": [], "total_count": 0, "has_more": False}
        mock_list_documents.assert_called_once_with(
            session=mock_session,
            file_name="nonexistent.md",
            vault_name=None,
            limit=10,
            offset=0,
        )

    @patch(
        "obsidian_rag.mcp_server.tools.documents.list_documents",
        create=True,
    )
    def test_list_documents_handler_no_file_name(self, mock_list_documents):
        """Test error when file_name is not provided."""
        from obsidian_rag.mcp_server.handlers import (
            _list_documents_handler,
            ListDocumentsHandlerParams,
        )

        mock_db_manager = MagicMock()
        mock_session = MagicMock()
        mock_db_manager.get_session.return_value.__enter__ = MagicMock(
            return_value=mock_session
        )
        mock_db_manager.get_session.return_value.__exit__ = MagicMock(
            return_value=False
        )

        mock_list_documents.side_effect = ValueError(
            "Must provide at least file_name"
        )

        params = ListDocumentsHandlerParams(db_manager=mock_db_manager)
        result = _list_documents_handler(params)

        assert result == {
            "success": False,
            "error": "Must provide at least file_name",
        }

    @patch(
        "obsidian_rag.mcp_server.tools.documents.list_documents",
        create=True,
    )
    def test_list_documents_handler_vault_not_found(self, mock_list_documents):
        """Test error when vault_name is not found."""
        from obsidian_rag.mcp_server.handlers import (
            _list_documents_handler,
            ListDocumentsHandlerParams,
        )

        mock_db_manager = MagicMock()
        mock_session = MagicMock()
        mock_db_manager.get_session.return_value.__enter__ = MagicMock(
            return_value=mock_session
        )
        mock_db_manager.get_session.return_value.__exit__ = MagicMock(
            return_value=False
        )

        mock_list_documents.side_effect = ValueError(
            "Vault 'Missing' not found. Available: Personal, Work"
        )

        params = ListDocumentsHandlerParams(
            db_manager=mock_db_manager,
            file_name="notes.md",
            vault_name="Missing",
        )
        result = _list_documents_handler(params)

        assert result == {
            "success": False,
            "error": "Vault 'Missing' not found. Available: Personal, Work",
        }


class TestIsIncrementalPath:
    """Tests for _is_incremental_path."""

    def test_is_incremental_path_none_returns_false(self):
        """path=None returns False."""
        from obsidian_rag.mcp_server.handlers import _is_incremental_path

        result = _is_incremental_path(None, "/vault")
        assert result is False

    def test_is_incremental_path_equals_container_returns_false(self):
        """path == container_path (after resolve) returns False."""
        from obsidian_rag.mcp_server.handlers import _is_incremental_path

        with runner.isolated_filesystem():
            vault_path = Path("vault")
            vault_path.mkdir()
            result = _is_incremental_path(str(vault_path), str(vault_path))
            assert result is False

    def test_is_incremental_path_subdirectory_returns_true(self):
        """path = container_path + '/sub' returns True."""
        from obsidian_rag.mcp_server.handlers import _is_incremental_path

        with runner.isolated_filesystem():
            vault_path = Path("vault")
            vault_path.mkdir()
            sub_path = vault_path / "sub"
            sub_path.mkdir()
            result = _is_incremental_path(str(sub_path), str(vault_path))
            assert result is True

    def test_is_incremental_path_outside_container_returns_false(self):
        """path = '/other/location' returns False."""
        from obsidian_rag.mcp_server.handlers import _is_incremental_path

        with runner.isolated_filesystem():
            vault_path = Path("vault")
            vault_path.mkdir()
            other_path = Path("other")
            other_path.mkdir()
            result = _is_incremental_path(str(other_path), str(vault_path))
            assert result is False

    def test_is_incremental_path_trailing_slash_normalizes(self):
        """path = container_path + '/' returns False (equals after resolve)."""
        from obsidian_rag.mcp_server.handlers import _is_incremental_path

        with runner.isolated_filesystem():
            vault_path = Path("vault")
            vault_path.mkdir()
            result = _is_incremental_path(str(vault_path) + "/", str(vault_path))
            assert result is False

    def test_is_incremental_path_nested_subdirectory_returns_true(self):
        """path = container_path + '/a/b/c' returns True."""
        from obsidian_rag.mcp_server.handlers import _is_incremental_path

        with runner.isolated_filesystem():
            vault_path = Path("vault")
            vault_path.mkdir()
            nested = vault_path / "a" / "b" / "c"
            nested.mkdir(parents=True)
            result = _is_incremental_path(str(nested), str(vault_path))
            assert result is True

    def test_is_incremental_path_symlink_resolution(self):
        """Symlink into container resolves and returns True."""
        import os
        from obsidian_rag.mcp_server.handlers import _is_incremental_path

        with runner.isolated_filesystem():
            vault_path = Path("vault")
            vault_path.mkdir()
            sub_path = vault_path / "sub"
            sub_path.mkdir()
            link_path = Path("link")
            os.symlink(str(sub_path), str(link_path))
            result = _is_incremental_path(str(link_path), str(vault_path))
            assert result is True


class TestResolveNoDelete:
    """Tests for _resolve_no_delete."""

    def test_resolve_no_delete_none_incremental_returns_true(self):
        """None + incremental path returns True."""
        from obsidian_rag.mcp_server.handlers import _resolve_no_delete

        with runner.isolated_filesystem():
            vault_path = Path("vault")
            vault_path.mkdir()
            sub_path = vault_path / "sub"
            sub_path.mkdir()
            result = _resolve_no_delete(
                str(sub_path), str(vault_path), no_delete=None
            )
            assert result is True

    def test_resolve_no_delete_none_full_vault_returns_false(self):
        """None + not incremental returns False."""
        from obsidian_rag.mcp_server.handlers import _resolve_no_delete

        with runner.isolated_filesystem():
            vault_path = Path("vault")
            vault_path.mkdir()
            result = _resolve_no_delete(None, str(vault_path), no_delete=None)
            assert result is False

    def test_resolve_no_delete_true_honored(self):
        """explicit True returns True regardless of path."""
        from obsidian_rag.mcp_server.handlers import _resolve_no_delete

        with runner.isolated_filesystem():
            vault_path = Path("vault")
            vault_path.mkdir()
            sub_path = vault_path / "sub"
            sub_path.mkdir()
            result = _resolve_no_delete(
                str(sub_path), str(vault_path), no_delete=True
            )
            assert result is True

    def test_resolve_no_delete_false_honored(self):
        """explicit False + incremental returns False (client accepts risk)."""
        from obsidian_rag.mcp_server.handlers import _resolve_no_delete

        with runner.isolated_filesystem():
            vault_path = Path("vault")
            vault_path.mkdir()
            sub_path = vault_path / "sub"
            sub_path.mkdir()
            result = _resolve_no_delete(
                str(sub_path), str(vault_path), no_delete=False
            )
            assert result is False

    def test_resolve_no_delete_none_path_none_returns_false(self):
        """None + path_override=None returns False."""
        from obsidian_rag.mcp_server.handlers import _resolve_no_delete

        with runner.isolated_filesystem():
            vault_path = Path("vault")
            vault_path.mkdir()
            result = _resolve_no_delete(None, str(vault_path), no_delete=None)
            assert result is False

    def test_resolve_no_delete_auto_force_logs_info(self, caplog):
        """None + incremental logs INFO auto-force message."""
        import logging
        from obsidian_rag.mcp_server.handlers import _resolve_no_delete

        with runner.isolated_filesystem():
            vault_path = Path("vault")
            vault_path.mkdir()
            sub_path = vault_path / "sub"
            sub_path.mkdir()
            with caplog.at_level(logging.INFO):
                result = _resolve_no_delete(
                    str(sub_path), str(vault_path), no_delete=None
                )
            assert result is True
            assert "no_delete auto-enabled" in caplog.text


class TestIngestHandlerNoDeleteResolution:
    """Tests for _ingest_handler no_delete resolution integration."""

    @patch("obsidian_rag.mcp_server.handlers._validate_ingest_path")
    def test_ingest_handler_no_delete_none_incremental_resolves_true(self, mock_validate):
        """no_delete=None + subdir path resolves to True."""
        from obsidian_rag.mcp_server.handlers import _ingest_handler, IngestHandlerParams
        from obsidian_rag.services.ingestion import IngestVaultOptions

        with runner.isolated_filesystem():
            vault_path = Path("vault")
            vault_path.mkdir()
            sub_path = vault_path / "sub"
            sub_path.mkdir()

            mock_settings = MagicMock()
            vault_config = MagicMock()
            vault_config.container_path = str(vault_path)
            mock_settings.get_vault.return_value = vault_config

            mock_validate.return_value = sub_path

            with patch("obsidian_rag.mcp_server.handlers.IngestionService") as mock_service:
                mock_instance = MagicMock()
                mock_result = MagicMock()
                mock_result.to_dict.return_value = {"total": 0}
                mock_instance.ingest_vault.return_value = mock_result
                mock_service.return_value = mock_instance

                params = IngestHandlerParams(
                    settings=mock_settings,
                    db_manager=MagicMock(),
                    embedding_provider=None,
                    vault_name="test",
                    path_override=str(sub_path),
                    no_delete=None,
                )
                _ingest_handler(params)

                call_args = mock_instance.ingest_vault.call_args
                ingest_options = call_args.args[1]
                assert isinstance(ingest_options, IngestVaultOptions)
                assert ingest_options.no_delete is True

    @patch("obsidian_rag.mcp_server.handlers._validate_ingest_path")
    def test_ingest_handler_no_delete_none_full_vault_resolves_false(self, mock_validate):
        """no_delete=None + path_override=None resolves to False."""
        from obsidian_rag.mcp_server.handlers import _ingest_handler, IngestHandlerParams
        from obsidian_rag.services.ingestion import IngestVaultOptions

        with runner.isolated_filesystem():
            vault_path = Path("vault")
            vault_path.mkdir()

            mock_settings = MagicMock()
            vault_config = MagicMock()
            vault_config.container_path = str(vault_path)
            mock_settings.get_vault.return_value = vault_config

            mock_validate.return_value = vault_path

            with patch("obsidian_rag.mcp_server.handlers.IngestionService") as mock_service:
                mock_instance = MagicMock()
                mock_result = MagicMock()
                mock_result.to_dict.return_value = {"total": 0}
                mock_instance.ingest_vault.return_value = mock_result
                mock_service.return_value = mock_instance

                params = IngestHandlerParams(
                    settings=mock_settings,
                    db_manager=MagicMock(),
                    embedding_provider=None,
                    vault_name="test",
                    path_override=None,
                    no_delete=None,
                )
                _ingest_handler(params)

                call_args = mock_instance.ingest_vault.call_args
                ingest_options = call_args.args[1]
                assert isinstance(ingest_options, IngestVaultOptions)
                assert ingest_options.no_delete is False

    @patch("obsidian_rag.mcp_server.handlers._validate_ingest_path")
    def test_ingest_handler_no_delete_true_honored(self, mock_validate):
        """no_delete=True + subdir path remains True."""
        from obsidian_rag.mcp_server.handlers import _ingest_handler, IngestHandlerParams
        from obsidian_rag.services.ingestion import IngestVaultOptions

        with runner.isolated_filesystem():
            vault_path = Path("vault")
            vault_path.mkdir()
            sub_path = vault_path / "sub"
            sub_path.mkdir()

            mock_settings = MagicMock()
            vault_config = MagicMock()
            vault_config.container_path = str(vault_path)
            mock_settings.get_vault.return_value = vault_config

            mock_validate.return_value = sub_path

            with patch("obsidian_rag.mcp_server.handlers.IngestionService") as mock_service:
                mock_instance = MagicMock()
                mock_result = MagicMock()
                mock_result.to_dict.return_value = {"total": 0}
                mock_instance.ingest_vault.return_value = mock_result
                mock_service.return_value = mock_instance

                params = IngestHandlerParams(
                    settings=mock_settings,
                    db_manager=MagicMock(),
                    embedding_provider=None,
                    vault_name="test",
                    path_override=str(sub_path),
                    no_delete=True,
                )
                _ingest_handler(params)

                call_args = mock_instance.ingest_vault.call_args
                ingest_options = call_args.args[1]
                assert isinstance(ingest_options, IngestVaultOptions)
                assert ingest_options.no_delete is True

    @patch("obsidian_rag.mcp_server.handlers._validate_ingest_path")
    def test_ingest_handler_no_delete_false_honored_with_incremental(self, mock_validate):
        """no_delete=False + subdir path remains False."""
        from obsidian_rag.mcp_server.handlers import _ingest_handler, IngestHandlerParams
        from obsidian_rag.services.ingestion import IngestVaultOptions

        with runner.isolated_filesystem():
            vault_path = Path("vault")
            vault_path.mkdir()
            sub_path = vault_path / "sub"
            sub_path.mkdir()

            mock_settings = MagicMock()
            vault_config = MagicMock()
            vault_config.container_path = str(vault_path)
            mock_settings.get_vault.return_value = vault_config

            mock_validate.return_value = sub_path

            with patch("obsidian_rag.mcp_server.handlers.IngestionService") as mock_service:
                mock_instance = MagicMock()
                mock_result = MagicMock()
                mock_result.to_dict.return_value = {"total": 0}
                mock_instance.ingest_vault.return_value = mock_result
                mock_service.return_value = mock_instance

                params = IngestHandlerParams(
                    settings=mock_settings,
                    db_manager=MagicMock(),
                    embedding_provider=None,
                    vault_name="test",
                    path_override=str(sub_path),
                    no_delete=False,
                )
                _ingest_handler(params)

                call_args = mock_instance.ingest_vault.call_args
                ingest_options = call_args.args[1]
                assert isinstance(ingest_options, IngestVaultOptions)
                assert ingest_options.no_delete is False

    @patch("obsidian_rag.mcp_server.handlers._validate_ingest_path")
    def test_ingest_handler_auto_force_emits_info_log(self, mock_validate, caplog):
        """no_delete=None + subdir emits INFO log."""
        import logging
        from obsidian_rag.mcp_server.handlers import _ingest_handler, IngestHandlerParams

        with runner.isolated_filesystem():
            vault_path = Path("vault")
            vault_path.mkdir()
            sub_path = vault_path / "sub"
            sub_path.mkdir()

            mock_settings = MagicMock()
            vault_config = MagicMock()
            vault_config.container_path = str(vault_path)
            mock_settings.get_vault.return_value = vault_config

            mock_validate.return_value = sub_path

            with patch("obsidian_rag.mcp_server.handlers.IngestionService") as mock_service:
                mock_instance = MagicMock()
                mock_result = MagicMock()
                mock_result.to_dict.return_value = {"total": 0}
                mock_instance.ingest_vault.return_value = mock_result
                mock_service.return_value = mock_instance

                params = IngestHandlerParams(
                    settings=mock_settings,
                    db_manager=MagicMock(),
                    embedding_provider=None,
                    vault_name="test",
                    path_override=str(sub_path),
                    no_delete=None,
                )
                with caplog.at_level(logging.INFO):
                    _ingest_handler(params)

                assert "no_delete auto-enabled" in caplog.text
