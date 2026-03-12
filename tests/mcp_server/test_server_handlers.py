"""Unit tests for MCP server handlers and additional tests."""

import json
import os
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
        assert len(result) == 2
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
        results = cast(list[dict[str, object]], result["results"])
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
