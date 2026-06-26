"""Unit tests for MCP server ingest handlers."""

from pathlib import Path
from typing import cast
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner


runner = CliRunner()


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
        from obsidian_rag.mcp_server.handlers import (
            _ingest_handler,
            IngestHandlerParams,
        )
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

        with patch(
            "obsidian_rag.mcp_server.handlers._validate_ingest_path",
            return_value=Path("/test/vault"),
        ):
            with patch(
                "obsidian_rag.mcp_server.handlers.IngestionService"
            ) as mock_service:
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
            result = _resolve_no_delete(str(sub_path), str(vault_path), no_delete=None)
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
            result = _resolve_no_delete(str(sub_path), str(vault_path), no_delete=True)
            assert result is True

    def test_resolve_no_delete_false_honored(self):
        """explicit False + incremental returns False (client accepts risk)."""
        from obsidian_rag.mcp_server.handlers import _resolve_no_delete

        with runner.isolated_filesystem():
            vault_path = Path("vault")
            vault_path.mkdir()
            sub_path = vault_path / "sub"
            sub_path.mkdir()
            result = _resolve_no_delete(str(sub_path), str(vault_path), no_delete=False)
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
    def test_ingest_handler_no_delete_none_incremental_resolves_true(
        self, mock_validate
    ):
        """no_delete=None + subdir path resolves to True."""
        from obsidian_rag.mcp_server.handlers import (
            _ingest_handler,
            IngestHandlerParams,
        )
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

            with patch(
                "obsidian_rag.mcp_server.handlers.IngestionService"
            ) as mock_service:
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
    def test_ingest_handler_no_delete_none_full_vault_resolves_false(
        self, mock_validate
    ):
        """no_delete=None + path_override=None resolves to False."""
        from obsidian_rag.mcp_server.handlers import (
            _ingest_handler,
            IngestHandlerParams,
        )
        from obsidian_rag.services.ingestion import IngestVaultOptions

        with runner.isolated_filesystem():
            vault_path = Path("vault")
            vault_path.mkdir()

            mock_settings = MagicMock()
            vault_config = MagicMock()
            vault_config.container_path = str(vault_path)
            mock_settings.get_vault.return_value = vault_config

            mock_validate.return_value = vault_path

            with patch(
                "obsidian_rag.mcp_server.handlers.IngestionService"
            ) as mock_service:
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
        from obsidian_rag.mcp_server.handlers import (
            _ingest_handler,
            IngestHandlerParams,
        )
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

            with patch(
                "obsidian_rag.mcp_server.handlers.IngestionService"
            ) as mock_service:
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
    def test_ingest_handler_no_delete_false_honored_with_incremental(
        self, mock_validate
    ):
        """no_delete=False + subdir path remains False."""
        from obsidian_rag.mcp_server.handlers import (
            _ingest_handler,
            IngestHandlerParams,
        )
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

            with patch(
                "obsidian_rag.mcp_server.handlers.IngestionService"
            ) as mock_service:
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
        from obsidian_rag.mcp_server.handlers import (
            _ingest_handler,
            IngestHandlerParams,
        )

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

            with patch(
                "obsidian_rag.mcp_server.handlers.IngestionService"
            ) as mock_service:
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
