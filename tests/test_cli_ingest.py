"""Tests for CLI commands."""

from unittest.mock import MagicMock, patch

import click
import pytest
from click.testing import CliRunner

from obsidian_rag.cli import cli


class TestCreateProgressCallback:
    """Test progress_callback function via partial application."""

    def test_create_progress_callback_with_verbose_true(self):
        """Test progress_callback with verbose=True via partial (TASK-042)."""
        from functools import partial

        import click
        from click.testing import CliRunner

        from obsidian_rag.cli_commands import progress_callback

        runner = CliRunner()

        callback = partial(progress_callback, verbose=True)

        @click.command()
        def test_cmd():
            callback(10, 100, 8, 2)

        result = runner.invoke(test_cmd)
        assert result.exit_code == 0
        assert (
            "Progress: 10/100 files processed (8 successful, 2 errors)" in result.output
        )


class TestReportIngestResults:
    """Test _report_ingest_results function."""

    def test_report_ingest_results_with_errors(self):
        """Test _report_ingest_results with errors (TASK-043)."""
        import click
        from click.testing import CliRunner

        from obsidian_rag.cli_commands import _report_ingest_results

        runner = CliRunner()

        @click.command()
        def test_cmd():
            stats = {"new": 5, "updated": 3, "unchanged": 2, "errors": 2}
            _report_ingest_results(10, stats, 5.5, deleted=1, no_delete=False)

        result = runner.invoke(test_cmd)
        assert result.exit_code == 0
        assert "Successfully ingested 10 documents" in result.output
        assert "(5 new, 3 updated, 2 unchanged, 2 errors, 1 deleted)" in result.output
        assert "Completed in 5.5 seconds" in result.output


class TestIngestCommand:
    """Test ingest command with various options."""

    @patch("obsidian_rag.cli_commands._scan_vault")
    @patch("obsidian_rag.cli_commands.process_files_in_batches")
    @patch("obsidian_rag.cli_commands.DatabaseManager")
    @patch("obsidian_rag.cli_commands._get_embedding_provider")
    @patch("obsidian_rag.cli_commands.IngestionService")
    def test_ingest_command_with_dry_run(
        self,
        mock_ingestion_service,
        mock_get_provider,
        mock_db_manager,
        mock_process,
        mock_scan,
    ):
        """Test ingest command with dry_run flag (TASK-044)."""
        from pathlib import Path

        from obsidian_rag.config import VaultConfig
        from obsidian_rag.services.ingestion import IngestionResult

        runner = CliRunner()

        mock_scan.return_value = [Path("test.md")]
        mock_process.return_value = []

        mock_settings = MagicMock()
        mock_settings.database.url = "postgresql+psycopg://localhost/test"
        mock_settings.ingestion.batch_size = 100
        mock_settings.ingestion.progress_interval = 10
        mock_settings.logging.level = "INFO"
        mock_settings.logging.format = "text"

        # Set up vault configuration
        vault_config = VaultConfig(
            container_path="/vault",
            host_path="/vault",
        )
        mock_settings.get_vault.return_value = vault_config
        mock_settings.get_vault_names.return_value = ["test-vault"]

        mock_embedding_provider = MagicMock()
        mock_get_provider.return_value = mock_embedding_provider

        mock_db_instance = MagicMock()
        mock_db_manager.return_value = mock_db_instance

        mock_service = MagicMock()
        mock_result = IngestionResult(
            total=1,
            new=0,
            updated=0,
            unchanged=1,
            errors=0,
            deleted=0,
            chunks_created=0,
            empty_documents=0,
            processing_time_seconds=1.0,
            message="Dry run completed",
        )
        mock_service.ingest_vault.return_value = mock_result
        mock_ingestion_service.return_value = mock_service

        with patch("obsidian_rag.cli.get_settings", return_value=mock_settings):
            with runner.isolated_filesystem() as fs:
                vault_path = Path(fs) / "vault"
                vault_path.mkdir()
                test_file = vault_path / "test.md"
                test_file.write_text("# Test")
                # Update vault config to match actual path
                vault_config.container_path = str(vault_path)
                vault_config.host_path = str(vault_path)
                result = runner.invoke(
                    cli,
                    ["ingest", str(vault_path), "--vault", "test-vault", "--dry-run"],
                )

        assert result.exit_code == 0
        assert "DRY RUN: No changes will be written to the database" in result.output

    @patch("obsidian_rag.cli_commands._scan_vault")
    @patch("obsidian_rag.cli_commands.process_files_in_batches")
    @patch("obsidian_rag.cli_commands.DatabaseManager")
    @patch("obsidian_rag.cli_commands._get_embedding_provider")
    @patch("obsidian_rag.cli_commands.IngestionService")
    def test_ingest_command_full_flow(
        self,
        mock_ingestion_service,
        mock_get_provider,
        mock_db_manager,
        mock_process,
        mock_scan,
    ):
        """Test ingest command full flow including embedding provider and IngestionService (TASK-045)."""
        from pathlib import Path

        from obsidian_rag.config import VaultConfig
        from obsidian_rag.services.ingestion import IngestionResult

        runner = CliRunner()

        mock_scan.return_value = [Path("test.md")]

        # Create mock file info
        mock_file_info = MagicMock()
        mock_file_info.path = Path("test.md")
        mock_file_info.name = "test.md"
        mock_process.return_value = [mock_file_info]

        mock_settings = MagicMock()
        mock_settings.database.url = "postgresql+psycopg://localhost/test"
        mock_settings.ingestion.batch_size = 100
        mock_settings.ingestion.progress_interval = 10
        mock_settings.logging.level = "INFO"
        mock_settings.logging.format = "text"

        # Set up vault configuration
        vault_config = VaultConfig(
            container_path="/vault",
            host_path="/vault",
        )
        mock_settings.get_vault.return_value = vault_config
        mock_settings.get_vault_names.return_value = ["test-vault"]

        mock_embedding_provider = MagicMock()
        mock_get_provider.return_value = mock_embedding_provider

        mock_db_instance = MagicMock()
        mock_db_manager.return_value = mock_db_instance

        mock_service = MagicMock()
        mock_result = IngestionResult(
            total=1,
            new=1,
            updated=0,
            unchanged=0,
            errors=0,
            deleted=0,
            chunks_created=0,
            empty_documents=0,
            processing_time_seconds=2.5,
            message="Ingestion completed successfully",
        )
        mock_service.ingest_vault.return_value = mock_result
        mock_ingestion_service.return_value = mock_service

        with patch("obsidian_rag.cli.get_settings", return_value=mock_settings):
            with runner.isolated_filesystem() as fs:
                vault_path = Path(fs) / "vault"
                vault_path.mkdir()
                test_file = vault_path / "test.md"
                test_file.write_text("# Test content")
                # Update vault config to match actual path
                vault_config.container_path = str(vault_path)
                vault_config.host_path = str(vault_path)
                result = runner.invoke(
                    cli, ["ingest", str(vault_path), "--vault", "test-vault"]
                )

        assert result.exit_code == 0
        assert "Found 1 markdown files" in result.output
        assert "Successfully ingested 1 documents" in result.output
        assert "(1 new, 0 updated, 0 unchanged, 0 errors, 0 deleted)" in result.output
        mock_ingestion_service.assert_called_once()


class TestCreateProgressCallbackCoverage:
    """Test progress_callback with verbose=False via partial (line 173->exit)."""

    def test_create_progress_callback_with_verbose_false(self):
        """Test progress callback does not print when verbose=False."""
        from functools import partial

        from obsidian_rag.cli_commands import progress_callback

        callback = partial(progress_callback, verbose=False)

        # When verbose=False, callback should not output anything
        # This covers the branch where if verbose: is False
        callback(5, 10, 3, 0)  # Should not raise or print


class TestProgressCallback:
    """Test progress_callback module-level function (refactored for testability)."""

    def test_progress_callback_verbose_true(self):
        """Test progress_callback with verbose=True prints message."""
        from obsidian_rag.cli_commands import progress_callback

        with patch("obsidian_rag.cli_commands.click.echo") as mock_echo:
            progress_callback(5, 10, 4, 1, verbose=True)

            mock_echo.assert_called_once()
            assert "Progress: 5/10" in mock_echo.call_args[0][0]
            assert "4 successful" in mock_echo.call_args[0][0]
            assert "1 errors" in mock_echo.call_args[0][0]

    def test_progress_callback_verbose_false(self):
        """Test progress_callback with verbose=False does not print."""
        from obsidian_rag.cli_commands import progress_callback

        with patch("obsidian_rag.cli_commands.click.echo") as mock_echo:
            progress_callback(5, 10, 4, 1, verbose=False)

            mock_echo.assert_not_called()


class TestReportIngestResultsCoverage:
    """Test _report_ingest_results with no_delete=True (line 205)."""

    def test_report_ingest_results_with_no_delete(self):
        """Test report shows 'deletion skipped' when no_delete=True."""
        import sys
        from io import StringIO

        from obsidian_rag.cli_commands import _report_ingest_results

        stats = {"new": 5, "updated": 3, "unchanged": 2, "errors": 0}

        # Capture stdout
        old_stdout = sys.stdout
        sys.stdout = StringIO()

        try:
            _report_ingest_results(10, stats, 1.5, 0, no_delete=True)
            output = sys.stdout.getvalue()
        finally:
            sys.stdout = old_stdout

        assert "deletion skipped" in output


class TestResolveIngestPath:
    """Test _resolve_ingest_path path resolution logic."""

    def test_resolve_ingest_path_explicit_path_provided(self):
        """When explicit path is provided, it is returned as-is (REQ-009)."""
        import tempfile

        from obsidian_rag.cli_ingest import _resolve_ingest_path
        from obsidian_rag.config import Settings, VaultConfig

        with tempfile.TemporaryDirectory() as tmpdir:
            vault_config = VaultConfig(
                container_path=tmpdir,
                host_path=tmpdir,
            )
            settings = MagicMock(spec=Settings)
            settings.get_vault.return_value = vault_config
            # Explicit path should be returned even if vault config exists
            result = _resolve_ingest_path(settings, tmpdir, "any-vault")
            assert result == tmpdir

    def test_resolve_ingest_path_uses_vault_container_path(self):
        """When no path provided, vault's container_path is used (REQ-008)."""
        import tempfile

        from obsidian_rag.cli_ingest import _resolve_ingest_path
        from obsidian_rag.config import Settings, VaultConfig

        with tempfile.TemporaryDirectory() as tmpdir:
            vault_config = VaultConfig(
                container_path=tmpdir,
                host_path=tmpdir,
            )
            settings = MagicMock(spec=Settings)
            settings.get_vault.return_value = vault_config

            result = _resolve_ingest_path(settings, None, "Personal")
            assert result == tmpdir

    def test_resolve_ingest_path_vault_not_found_no_path(self):
        """Vault not found in config and no PATH provided → click.BadParameter."""
        from obsidian_rag.cli_ingest import _resolve_ingest_path
        from obsidian_rag.config import Settings

        settings = MagicMock(spec=Settings)
        settings.get_vault.return_value = None
        settings.get_vault_names.return_value = ["vault1", "vault2"]

        with pytest.raises(click.BadParameter, match="not found in configuration"):
            _resolve_ingest_path(settings, None, "Nonexistent")

    def test_resolve_ingest_path_configured_path_does_not_exist(self):
        """Vault's container_path doesn't exist → click.BadParameter."""
        from obsidian_rag.cli_ingest import _resolve_ingest_path
        from obsidian_rag.config import Settings

        vault_config = MagicMock()
        vault_config.container_path = "/nonexistent/path"
        settings = MagicMock(spec=Settings)
        settings.get_vault.return_value = vault_config

        with pytest.raises(click.BadParameter, match="does not exist"):
            _resolve_ingest_path(settings, None, "Personal")

    def test_resolve_ingest_path_explicit_path_does_not_exist(self):
        """Explicit PATH that doesn't exist → click.BadParameter."""
        from obsidian_rag.cli_ingest import _resolve_ingest_path
        from obsidian_rag.config import Settings

        vault_config = MagicMock()
        vault_config.container_path = "/nonexistent/path"
        settings = MagicMock(spec=Settings)
        settings.get_vault.return_value = vault_config

        with pytest.raises(click.BadParameter, match="does not exist"):
            _resolve_ingest_path(settings, "/nonexistent/path", "Personal")

    def test_resolve_ingest_path_is_not_directory(self):
        """Resolved path is a file, not a directory → click.BadParameter."""
        import tempfile
        from pathlib import Path

        from obsidian_rag.cli_ingest import _resolve_ingest_path
        from obsidian_rag.config import Settings

        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "file.txt"
            file_path.write_text("not a directory")

            vault_config = MagicMock()
            vault_config.container_path = str(file_path)
            settings = MagicMock(spec=Settings)
            settings.get_vault.return_value = vault_config

            with pytest.raises(click.BadParameter, match="not a directory"):
                _resolve_ingest_path(settings, None, "Personal")

    def test_resolve_ingest_path_no_vaults_configured(self):
        """No vaults configured at all → error message shows 'none configured'."""
        from obsidian_rag.cli_ingest import _resolve_ingest_path
        from obsidian_rag.config import Settings

        settings = MagicMock(spec=Settings)
        settings.get_vault.return_value = None
        settings.get_vault_names.return_value = []

        with pytest.raises(click.BadParameter, match="none configured"):
            _resolve_ingest_path(settings, None, "Nonexistent")

    def test_resolve_ingest_path_vault_not_found_error_includes_available(self):
        """Error message includes list of available vaults."""
        from obsidian_rag.cli_ingest import _resolve_ingest_path
        from obsidian_rag.config import Settings

        settings = MagicMock(spec=Settings)
        settings.get_vault.return_value = None
        settings.get_vault_names.return_value = ["Personal", "Work"]

        with pytest.raises(click.BadParameter, match="Personal, Work"):
            _resolve_ingest_path(settings, None, "Nonexistent")

    def test_resolve_ingest_path_path_mismatch(self):
        """Explicit path that doesn't match vault's container_path → error."""
        import tempfile
        from pathlib import Path

        from obsidian_rag.cli_ingest import _resolve_ingest_path
        from obsidian_rag.config import Settings, VaultConfig

        with tempfile.TemporaryDirectory() as tmpdir:
            vault_a = Path(tmpdir) / "vault_a"
            vault_b = Path(tmpdir) / "vault_b"
            vault_a.mkdir()
            vault_b.mkdir()

            vault_config = VaultConfig(
                container_path=str(vault_a),
                host_path=str(vault_a),
            )
            mock_settings = MagicMock(spec=Settings)
            mock_settings.get_vault.return_value = vault_config

            with pytest.raises(click.BadParameter, match="does not match"):
                _resolve_ingest_path(mock_settings, str(vault_b), "test-vault")


class TestIngestCommandOptionalPath:
    """Test ingest command with optional PATH behavior."""

    def test_ingest_no_path_vault_not_configured(self):
        """CLI shows error when vault not found and no PATH provided."""
        runner = CliRunner()
        mock_settings = MagicMock()
        mock_settings.get_vault.return_value = None
        mock_settings.get_vault_names.return_value = ["Personal"]
        mock_settings.logging.level = "INFO"
        mock_settings.logging.format = "text"

        with patch("obsidian_rag.cli.get_settings", return_value=mock_settings):
            result = runner.invoke(cli, ["ingest", "--vault", "Nonexistent"])

        assert result.exit_code != 0
        assert "not found in configuration" in result.output

    def test_ingest_no_path_container_path_not_found(self):
        """CLI shows error when vault's container_path doesn't exist."""
        runner = CliRunner()
        vault_config = MagicMock()
        vault_config.container_path = "/nonexistent/path"
        mock_settings = MagicMock()
        mock_settings.get_vault.return_value = vault_config
        mock_settings.get_vault_names.return_value = ["Personal"]
        mock_settings.logging.level = "INFO"
        mock_settings.logging.format = "text"

        with patch("obsidian_rag.cli.get_settings", return_value=mock_settings):
            result = runner.invoke(cli, ["ingest", "--vault", "Personal"])

        assert result.exit_code != 0
        assert "does not exist" in result.output

    @patch("obsidian_rag.cli_commands.scan_markdown_files")
    @patch("obsidian_rag.cli_commands.process_files_in_batches")
    @patch("obsidian_rag.cli_commands.DatabaseManager")
    @patch("obsidian_rag.cli_commands._get_embedding_provider")
    def test_ingest_explicit_path_backward_compatible(
        self, _mock_get_provider, _mock_db_manager, _mock_process, mock_scan
    ):
        """Existing 'ingest <path> --vault <name>' usage still works."""
        from pathlib import Path

        from obsidian_rag.config import VaultConfig

        runner = CliRunner()
        mock_scan.return_value = []
        mock_settings = MagicMock()
        mock_settings.database.url = "postgresql+psycopg://localhost/test"
        mock_settings.ingestion.batch_size = 100
        mock_settings.ingestion.progress_interval = 10
        mock_settings.logging.level = "INFO"
        mock_settings.logging.format = "text"

        vault_config = VaultConfig(container_path="/vault", host_path="/vault")
        mock_settings.get_vault.return_value = vault_config
        mock_settings.get_vault_names.return_value = ["test-vault"]

        with patch("obsidian_rag.cli.get_settings", return_value=mock_settings):
            with runner.isolated_filesystem() as fs:
                vault_path = Path(fs) / "vault"
                vault_path.mkdir()
                vault_config.container_path = str(vault_path)
                vault_config.host_path = str(vault_path)
                result = runner.invoke(
                    cli, ["ingest", str(vault_path), "--vault", "test-vault"]
                )

        assert result.exit_code == 0
        assert "No markdown files found" in result.output

    @patch("obsidian_rag.cli_commands.scan_markdown_files")
    @patch("obsidian_rag.cli_commands.process_files_in_batches")
    @patch("obsidian_rag.cli_commands.DatabaseManager")
    @patch("obsidian_rag.cli_commands._get_embedding_provider")
    def test_ingest_no_path_uses_container_path(
        self, _mock_get_provider, _mock_db_manager, _mock_process, mock_scan
    ):
        """Ingest without PATH uses vault's container_path from config."""
        from pathlib import Path

        from obsidian_rag.config import VaultConfig

        runner = CliRunner()
        mock_scan.return_value = []
        mock_settings = MagicMock()
        mock_settings.database.url = "postgresql+psycopg://localhost/test"
        mock_settings.ingestion.batch_size = 100
        mock_settings.ingestion.progress_interval = 10
        mock_settings.logging.level = "INFO"
        mock_settings.logging.format = "text"

        with runner.isolated_filesystem() as fs:
            vault_path = Path(fs) / "vault"
            vault_path.mkdir()
            vault_config = VaultConfig(
                container_path=str(vault_path),
                host_path=str(vault_path),
            )
            mock_settings.get_vault.return_value = vault_config
            mock_settings.get_vault_names.return_value = ["test-vault"]

            with patch("obsidian_rag.cli.get_settings", return_value=mock_settings):
                result = runner.invoke(cli, ["ingest", "--vault", "test-vault"])

        assert result.exit_code == 0
        assert "No markdown files found" in result.output

    @patch("obsidian_rag.cli_commands.scan_markdown_files")
    @patch("obsidian_rag.cli_commands.process_files_in_batches")
    @patch("obsidian_rag.cli_commands.DatabaseManager")
    @patch("obsidian_rag.cli_commands._get_embedding_provider")
    def test_ingest_path_overrides_container_path(
        self, _mock_get_provider, _mock_db_manager, _mock_process, mock_scan
    ):
        """Explicit PATH takes precedence over vault's container_path."""
        from pathlib import Path

        from obsidian_rag.config import VaultConfig

        runner = CliRunner()
        mock_scan.return_value = []
        mock_settings = MagicMock()
        mock_settings.database.url = "postgresql+psycopg://localhost/test"
        mock_settings.ingestion.batch_size = 100
        mock_settings.ingestion.progress_interval = 10
        mock_settings.logging.level = "INFO"
        mock_settings.logging.format = "text"

        with runner.isolated_filesystem() as fs:
            vault_path = Path(fs) / "vault"
            vault_path.mkdir()
            # Config points to /other/path, but explicit path is used
            vault_config = VaultConfig(
                container_path=str(vault_path),
                host_path=str(vault_path),
            )
            mock_settings.get_vault.return_value = vault_config
            mock_settings.get_vault_names.return_value = ["test-vault"]

            with patch("obsidian_rag.cli.get_settings", return_value=mock_settings):
                result = runner.invoke(
                    cli, ["ingest", str(vault_path), "--vault", "test-vault"]
                )

        assert result.exit_code == 0


class TestRunIngestionCoverage:
    """Test _run_ingestion with no_delete option (line 282)."""

    @patch("obsidian_rag.cli_commands._scan_vault")
    @patch("obsidian_rag.cli_commands.process_files_in_batches")
    @patch("obsidian_rag.cli_commands.DatabaseManager")
    @patch("obsidian_rag.cli_commands._get_embedding_provider")
    def test_run_ingestion_with_no_delete(
        self, mock_get_provider, mock_db_manager, mock_process, mock_scan
    ):
        """Test _run_ingestion outputs message when no_delete=True (line 282)."""
        import sys
        from io import StringIO
        from pathlib import Path

        from obsidian_rag.cli_commands import _run_ingestion, IngestOptions

        mock_settings = MagicMock()
        mock_settings.database = MagicMock()
        mock_settings.database.url = "postgresql+psycopg://localhost/test"
        mock_settings.ingestion = MagicMock()
        mock_settings.ingestion.batch_size = 100
        mock_settings.ingestion.progress_interval = 10
        mock_settings.get_vault.return_value.container_path = "/vault"

        mock_embedding_provider = MagicMock()
        mock_get_provider.return_value = mock_embedding_provider

        mock_scan.return_value = [Path("test.md")]
        mock_process.return_value = []

        mock_session = MagicMock()
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)

        mock_db_instance = MagicMock()
        mock_db_instance.get_session.return_value = mock_session
        mock_db_manager.return_value = mock_db_instance

        options = IngestOptions(
            vault="test-vault", dry_run=False, verbose=False, no_delete=True
        )

        # Capture stdout
        old_stdout = sys.stdout
        sys.stdout = StringIO()

        try:
            _run_ingestion(mock_settings, Path("/vault"), "test-vault", options)
            output = sys.stdout.getvalue()
        finally:
            sys.stdout = old_stdout

        assert "Deletion phase skipped" in output
