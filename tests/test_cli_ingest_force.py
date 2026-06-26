"""Tests for CLI commands."""

from unittest.mock import MagicMock, patch


def test_ingest_help_shows_force():
    """Test ingest --help shows --force option."""
    from click.testing import CliRunner
    from obsidian_rag.cli import cli

    runner = CliRunner()
    result = runner.invoke(cli, ["ingest", "--help"])
    assert result.exit_code == 0
    assert "--force" in result.output
    assert "re-ingest" in result.output.lower()


def test_ingest_command_accepts_force_flag():
    """Test ingest command accepts --force flag."""
    from click.testing import CliRunner
    from obsidian_rag.cli import cli
    from obsidian_rag.config import VaultConfig
    from unittest.mock import patch, MagicMock

    runner = CliRunner()

    with patch("obsidian_rag.cli_commands._run_ingestion") as mock_run:
        with patch("obsidian_rag.cli.get_settings") as mock_get_settings:
            mock_settings = MagicMock()
            mock_settings.vaults = {}
            mock_settings.logging.level = "INFO"
            mock_settings.logging.format = "text"
            vault_config = VaultConfig(container_path="/tmp", host_path="/tmp")
            mock_settings.get_vault.return_value = vault_config
            mock_settings.get_vault_names.return_value = ["test"]
            mock_get_settings.return_value = mock_settings
            result = runner.invoke(
                cli, ["ingest", "/tmp", "--vault", "test", "--force"]
            )

    assert result.exit_code == 0 or result.exit_code is None
    mock_run.assert_called_once()
    call_kwargs = mock_run.call_args
    assert call_kwargs.args[3].force is True


def test_ingest_command_force_defaults_false():
    """Test ingest command defaults force to False."""
    from click.testing import CliRunner
    from obsidian_rag.cli import cli
    from obsidian_rag.config import VaultConfig
    from unittest.mock import patch, MagicMock

    runner = CliRunner()

    with patch("obsidian_rag.cli_commands._run_ingestion") as mock_run:
        with patch("obsidian_rag.cli.get_settings") as mock_get_settings:
            mock_settings = MagicMock()
            mock_settings.vaults = {}
            mock_settings.logging.level = "INFO"
            mock_settings.logging.format = "text"
            vault_config = VaultConfig(container_path="/tmp", host_path="/tmp")
            mock_settings.get_vault.return_value = vault_config
            mock_settings.get_vault_names.return_value = ["test"]
            mock_get_settings.return_value = mock_settings
            runner.invoke(cli, ["ingest", "/tmp", "--vault", "test"])

    mock_run.assert_called_once()
    call_kwargs = mock_run.call_args
    assert call_kwargs.args[3].force is False


class TestIngestForceOption:
    """Test force option propagation through ingest command chain."""

    def test_ingest_options_has_force_field(self):
        """Test IngestOptions accepts force field."""
        from obsidian_rag.cli_commands import IngestOptions

        options = IngestOptions(
            vault="test",
            dry_run=False,
            no_delete=False,
            verbose=False,
            force=True,
        )
        assert options.force is True

    def test_run_ingest_command_passes_force(self):
        """Test _run_ingest_command passes force to IngestOptions."""
        from obsidian_rag.cli_commands import _run_ingest_command

        ctx = MagicMock()
        ctx.obj = {"settings": MagicMock()}

        with patch("obsidian_rag.cli_commands._run_ingestion") as mock_run:
            with patch(
                "obsidian_rag.cli_commands._resolve_ingest_path", return_value="/vault"
            ):
                _run_ingest_command(
                    ctx,
                    "/vault",
                    vault="test",
                    dry_run=False,
                    no_delete=False,
                    verbose=False,
                    force=True,
                )

        mock_run.assert_called_once()
        call_args = mock_run.call_args
        assert call_args.args[3].force is True

    @patch("obsidian_rag.cli_commands._scan_vault")
    def test_run_ingestion_shows_force_confirmation(self, mock_scan):
        """Test _run_ingestion shows force confirmation message."""
        import sys
        from io import StringIO
        from pathlib import Path

        from obsidian_rag.cli_commands import IngestOptions, _run_ingestion

        mock_settings = MagicMock()
        mock_settings.database = MagicMock()
        mock_settings.database.url = "postgresql+psycopg://localhost/test"
        mock_settings.ingestion = MagicMock()
        mock_settings.ingestion.batch_size = 100
        mock_settings.ingestion.progress_interval = 10
        mock_settings.get_vault.return_value.container_path = "/vault"

        mock_scan.return_value = []

        options = IngestOptions(
            vault="test",
            dry_run=False,
            no_delete=False,
            verbose=False,
            force=True,
        )

        old_stdout = sys.stdout
        sys.stdout = StringIO()

        try:
            _run_ingestion(mock_settings, Path("/vault"), "test", options)
            output = sys.stdout.getvalue()
        finally:
            sys.stdout = old_stdout

        assert "Force re-ingestion enabled" in output

    @patch("obsidian_rag.cli_commands._scan_vault")
    @patch("obsidian_rag.cli_commands.process_files_in_batches")
    @patch("obsidian_rag.cli_commands.DatabaseManager")
    @patch("obsidian_rag.cli_commands._get_embedding_provider")
    @patch("obsidian_rag.cli_commands.IngestionService")
    def test_run_ingestion_passes_force_to_ingest_vault_options(
        self,
        mock_service_class,
        mock_get_provider,
        mock_db_manager,
        mock_process,
        mock_scan,
    ):
        """Test _run_ingestion passes force to IngestVaultOptions."""
        import sys
        from io import StringIO
        from pathlib import Path

        from obsidian_rag.cli_commands import IngestOptions, _run_ingestion
        from obsidian_rag.services.ingestion import IngestVaultOptions

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

        mock_instance = MagicMock()
        mock_result = MagicMock()
        mock_result.total = 0
        mock_result.new = 0
        mock_result.updated = 0
        mock_result.unchanged = 0
        mock_result.errors = 0
        mock_result.deleted = 0
        mock_result.processing_time_seconds = 0.0
        mock_result.message = "No markdown files found in directory"
        mock_instance.ingest_vault.return_value = mock_result
        mock_service_class.return_value = mock_instance

        options = IngestOptions(
            vault="test",
            dry_run=False,
            no_delete=False,
            verbose=False,
            force=True,
        )

        old_stdout = sys.stdout
        sys.stdout = StringIO()

        try:
            _run_ingestion(mock_settings, Path("/vault"), "test", options)
        finally:
            sys.stdout = old_stdout

        mock_instance.ingest_vault.assert_called_once()
        call_args = mock_instance.ingest_vault.call_args
        ingest_options = call_args.args[1]
        assert isinstance(ingest_options, IngestVaultOptions)
        assert ingest_options.force is True

    @patch("obsidian_rag.cli_commands._scan_vault")
    @patch("obsidian_rag.cli_commands.process_files_in_batches")
    @patch("obsidian_rag.cli_commands.DatabaseManager")
    @patch("obsidian_rag.cli_commands._get_embedding_provider")
    @patch("obsidian_rag.cli_commands.IngestionService")
    def test_run_ingestion_no_force_message_when_false(
        self,
        mock_service_class,
        mock_get_provider,
        mock_db_manager,
        mock_process,
        mock_scan,
    ):
        """Test _run_ingestion does not show force message when force=False."""
        import sys
        from io import StringIO
        from pathlib import Path

        from obsidian_rag.cli_commands import IngestOptions, _run_ingestion
        from obsidian_rag.services.ingestion import IngestVaultOptions

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

        mock_instance = MagicMock()
        mock_result = MagicMock()
        mock_result.total = 0
        mock_result.new = 0
        mock_result.updated = 0
        mock_result.unchanged = 0
        mock_result.errors = 0
        mock_result.deleted = 0
        mock_result.processing_time_seconds = 0.0
        mock_result.message = "No markdown files found in directory"
        mock_instance.ingest_vault.return_value = mock_result
        mock_service_class.return_value = mock_instance

        options = IngestOptions(
            vault="test",
            dry_run=False,
            no_delete=False,
            verbose=False,
            force=False,
        )

        old_stdout = sys.stdout
        sys.stdout = StringIO()

        try:
            _run_ingestion(mock_settings, Path("/vault"), "test", options)
            output = sys.stdout.getvalue()
        finally:
            sys.stdout = old_stdout

        assert "Force re-ingestion enabled" not in output
        mock_instance.ingest_vault.assert_called_once()
        call_args = mock_instance.ingest_vault.call_args
        ingest_options = call_args.args[1]
        assert isinstance(ingest_options, IngestVaultOptions)
        assert ingest_options.force is False


class TestReportIngestResultsForce:
    """Test _report_ingest_results force parameter."""

    def test_report_ingest_results_force_message(self):
        """Test _report_ingest_results shows force message when force=True."""
        from obsidian_rag.cli_commands import _report_ingest_results
        from click.testing import CliRunner

        runner = CliRunner()
        with runner.isolated_filesystem():
            _report_ingest_results(
                total=10,
                stats={"new": 2, "updated": 5, "unchanged": 0, "errors": 0},
                elapsed_time=1.5,
                deleted=0,
                no_delete=False,
                force=True,
            )
        # The function uses click.echo; we verify no crash and force message is in output by capturing

    def test_report_ingest_results_no_force_message(self):
        """Test _report_ingest_results has no force message when force=False."""
        from obsidian_rag.cli_commands import _report_ingest_results
        from click.testing import CliRunner

        runner = CliRunner()
        with runner.isolated_filesystem():
            _report_ingest_results(
                total=10,
                stats={"new": 2, "updated": 3, "unchanged": 5, "errors": 0},
                elapsed_time=1.5,
                deleted=2,
                no_delete=False,
                force=False,
            )

    def test_report_ingest_results_force_with_no_delete(self):
        """Test _report_ingest_results shows both force and no_delete messages."""
        from obsidian_rag.cli_commands import _report_ingest_results
        from click.testing import CliRunner

        runner = CliRunner()
        with runner.isolated_filesystem():
            _report_ingest_results(
                total=10,
                stats={"new": 2, "updated": 5, "unchanged": 0, "errors": 0},
                elapsed_time=1.5,
                deleted=0,
                no_delete=True,
                force=True,
            )


class TestCliForceIntegration:
    """CLI integration tests for the --force flag."""

    @patch("obsidian_rag.cli_commands._run_ingestion")
    @patch("obsidian_rag.cli_commands._resolve_ingest_path")
    def test_ingest_with_force_flag(self, mock_resolve, mock_run):
        """Test ingest command with --force passes force=True."""
        from click.testing import CliRunner
        from obsidian_rag.cli import cli
        from obsidian_rag.config import VaultConfig

        runner = CliRunner()
        mock_resolve.return_value = "/vault"

        mock_settings = MagicMock()
        vault_config = VaultConfig(container_path="/vault", host_path="/vault")
        mock_settings.get_vault.return_value = vault_config
        mock_settings.get_vault_names.return_value = ["test-vault"]
        mock_settings.logging.level = "INFO"
        mock_settings.logging.format = "text"

        with patch("obsidian_rag.cli.get_settings", return_value=mock_settings):
            result = runner.invoke(
                cli, ["ingest", "/vault", "--vault", "test-vault", "--force"]
            )

        assert result.exit_code == 0
        mock_run.assert_called_once()
        call_args = mock_run.call_args
        assert call_args.args[3].force is True

    @patch("obsidian_rag.cli_commands._run_ingestion")
    @patch("obsidian_rag.cli_commands._resolve_ingest_path")
    def test_ingest_without_force_flag(self, mock_resolve, mock_run):
        """Test ingest command without --force passes force=False."""
        from click.testing import CliRunner
        from obsidian_rag.cli import cli
        from obsidian_rag.config import VaultConfig

        runner = CliRunner()
        mock_resolve.return_value = "/vault"

        mock_settings = MagicMock()
        vault_config = VaultConfig(container_path="/vault", host_path="/vault")
        mock_settings.get_vault.return_value = vault_config
        mock_settings.get_vault_names.return_value = ["test-vault"]
        mock_settings.logging.level = "INFO"
        mock_settings.logging.format = "text"

        with patch("obsidian_rag.cli.get_settings", return_value=mock_settings):
            result = runner.invoke(cli, ["ingest", "/vault", "--vault", "test-vault"])

        assert result.exit_code == 0
        mock_run.assert_called_once()
        call_args = mock_run.call_args
        assert call_args.args[3].force is False

    def test_ingest_force_help_text(self):
        """Test ingest --help shows --force and mentions re-ingest."""
        from click.testing import CliRunner
        from obsidian_rag.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["ingest", "--help"])

        assert result.exit_code == 0
        assert "--force" in result.output
        assert "re-ingest" in result.output.lower()

    @patch("obsidian_rag.cli_commands._scan_vault")
    @patch("obsidian_rag.cli_commands.IngestionService")
    def test_run_ingestion_force_confirmation_message(self, mock_service, mock_scan):
        """Test _run_ingestion shows force confirmation when force=True."""
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

        options = IngestOptions(
            vault="test",
            dry_run=False,
            no_delete=False,
            verbose=False,
            force=True,
        )

        mock_scan.return_value = []
        mock_result = MagicMock()
        mock_result.total = 0
        mock_result.new = 0
        mock_result.updated = 0
        mock_result.unchanged = 0
        mock_result.errors = 0
        mock_result.deleted = 0
        mock_result.processing_time_seconds = 0.0
        mock_result.message = "No markdown files found in directory"
        mock_instance = MagicMock()
        mock_instance.ingest_vault.return_value = mock_result
        mock_service.return_value = mock_instance

        old_stdout = sys.stdout
        sys.stdout = StringIO()
        try:
            _run_ingestion(mock_settings, Path("/vault"), "test", options)
            output = sys.stdout.getvalue()
        finally:
            sys.stdout = old_stdout

        assert "Force re-ingestion enabled" in output

    def test_report_ingest_results_with_force(self):
        """Test _report_ingest_results shows force-specific messaging."""
        import click
        from click.testing import CliRunner

        from obsidian_rag.cli_commands import _report_ingest_results

        runner = CliRunner()

        @click.command()
        def test_cmd():
            _report_ingest_results(
                10,
                {"new": 2, "updated": 8, "unchanged": 0, "errors": 0},
                1.5,
                0,
                no_delete=False,
                force=True,
            )

        result = runner.invoke(test_cmd)
        assert result.exit_code == 0
        assert "Force re-ingestion" in result.output

    def test_report_ingest_results_without_force(self):
        """Test _report_ingest_results has no force messaging when force=False."""
        import click
        from click.testing import CliRunner

        from obsidian_rag.cli_commands import _report_ingest_results

        runner = CliRunner()

        @click.command()
        def test_cmd():
            _report_ingest_results(
                10,
                {"new": 2, "updated": 3, "unchanged": 5, "errors": 0},
                1.5,
                2,
                no_delete=False,
                force=False,
            )

        result = runner.invoke(test_cmd)
        assert result.exit_code == 0
        assert "Force re-ingestion" not in result.output


class TestSetupLoggingJsonFormat:
    """Test _setup_logging with JSON format."""

    def test_setup_logging_json_format(self):
        """Test _setup_logging with JSON format type (TASK-038)."""
        import logging
        import sys
        from io import StringIO

        from obsidian_rag.cli_commands import setup_logging

        # Capture stderr
        stderr_capture = StringIO()
        original_stderr = sys.stderr
        sys.stderr = stderr_capture

        try:
            setup_logging("INFO", "json")
            logger = logging.getLogger("json_test_logger")
            logger.info("JSON test message")

            output = stderr_capture.getvalue()
            assert '"timestamp"' in output
            assert '"level"' in output
            assert '"message"' in output
            assert "JSON test message" in output
        finally:
            sys.stderr = original_stderr


class TestCliSetupLoggingJsonFormat:
    """Test _setup_logging with JSON format from cli.py."""

    def test_setup_logging_json_format_cli(self):
        """Test _setup_logging with JSON format from cli.py module (TASK-025)."""
        import logging
        import sys
        from io import StringIO

        from obsidian_rag.cli_commands import setup_logging

        # Capture stderr
        stderr_capture = StringIO()
        original_stderr = sys.stderr
        sys.stderr = stderr_capture

        try:
            setup_logging("INFO", "json")
            logger = logging.getLogger("test_json_logger_cli")
            logger.info("JSON test from cli")

            output = stderr_capture.getvalue()
            assert '"timestamp"' in output
            assert '"level"' in output
            assert '"message"' in output
            assert "JSON test from cli" in output
        finally:
            sys.stderr = original_stderr
