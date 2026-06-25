"""End-to-end MCP incremental ingestion safety tests (TASK-014)."""

import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

from click.testing import CliRunner

runner = CliRunner()


class TestIngestIncrementalIntegration:
    """End-to-end MCP incremental ingestion safety (TASK-014)."""

    @patch("obsidian_rag.mcp_server.server._get_registry")
    @patch("obsidian_rag.mcp_server.server._ingest_handler")
    def test_incremental_path_no_delete_none_passes_none_to_handler(
        self, mock_handler, mock_get_registry, caplog
    ):
        """ingest(subdir, no_delete=None) -> handler receives no_delete=None."""
        from obsidian_rag.mcp_server.server import ingest, _clear_ingest_tracker

        mock_registry = MagicMock()
        mock_get_registry.return_value = mock_registry
        mock_result = {"total": 5, "new": 5, "deleted": 0, "message": "ok"}
        mock_handler.return_value = mock_result

        with caplog.at_level(logging.INFO):
            result = ingest("vault", "/vault/subdir", no_delete=None)

        # Server wrapper passed None through to IngestHandlerParams
        call_params = mock_handler.call_args.args[0]
        assert call_params.no_delete is None
        assert call_params.path_override == "/vault/subdir"
        assert result == mock_result
        _clear_ingest_tracker()

    @patch("obsidian_rag.mcp_server.server._get_registry")
    @patch("obsidian_rag.mcp_server.server._ingest_handler")
    def test_full_vault_no_delete_none_passes_none_to_handler(
        self, mock_handler, mock_get_registry
    ):
        """ingest(vault, no_delete=None) with no path -> handler receives None."""
        from obsidian_rag.mcp_server.server import ingest, _clear_ingest_tracker

        mock_registry = MagicMock()
        mock_get_registry.return_value = mock_registry
        mock_handler.return_value = {"total": 0}

        ingest("vault", no_delete=None)

        call_params = mock_handler.call_args.args[0]
        assert call_params.no_delete is None
        assert call_params.path_override is None
        _clear_ingest_tracker()

    @patch("obsidian_rag.mcp_server.server._get_registry")
    @patch("obsidian_rag.mcp_server.server._ingest_handler")
    def test_incremental_path_explicit_false_honored(
        self, mock_handler, mock_get_registry
    ):
        """ingest(subdir, no_delete=False) -> explicit False preserved."""
        from obsidian_rag.mcp_server.server import ingest, _clear_ingest_tracker

        mock_registry = MagicMock()
        mock_get_registry.return_value = mock_registry
        mock_handler.return_value = {"total": 0}

        ingest("vault", "/vault/subdir", no_delete=False)

        call_params = mock_handler.call_args.args[0]
        assert call_params.no_delete is False
        assert call_params.path_override == "/vault/subdir"
        _clear_ingest_tracker()

    @patch("obsidian_rag.mcp_server.server._get_registry")
    def test_incremental_path_auto_force_log_emitted_end_to_end(
        self, mock_get_registry, caplog
    ):
        """True end-to-end: auto-force INFO log emitted across server->handler boundary."""
        import logging
        from obsidian_rag.mcp_server.server import ingest, _clear_ingest_tracker

        mock_registry = MagicMock()
        mock_get_registry.return_value = mock_registry

        with runner.isolated_filesystem():
            # Create real temp dirs so Path.resolve() works
            vault_dir = Path("vault")
            vault_dir.mkdir()
            subdir = vault_dir / "subdir"
            subdir.mkdir()

            mock_settings = MagicMock()
            vault_config = MagicMock()
            vault_config.container_path = str(vault_dir.resolve())
            mock_settings.get_vault.return_value = vault_config
            mock_registry.settings = mock_settings

            with patch(
                "obsidian_rag.mcp_server.handlers.IngestionService"
            ) as mock_service:
                mock_instance = MagicMock()
                mock_result = MagicMock()
                mock_result.to_dict.return_value = {"total": 0}
                mock_instance.ingest_vault.return_value = mock_result
                mock_service.return_value = mock_instance

                with caplog.at_level(logging.INFO):
                    ingest("vault", str(subdir.resolve()), no_delete=None)

            # Verify the auto-force INFO log was emitted
            assert any(
                "no_delete auto-enabled" in record.message for record in caplog.records
            )

        _clear_ingest_tracker()
