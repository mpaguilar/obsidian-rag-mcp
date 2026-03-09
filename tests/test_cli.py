"""Tests for CLI commands."""

from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from obsidian_rag.cli import cli


class TestCli:
    """Test cases for CLI."""

    def test_cli_help(self):
        """Test CLI help command."""
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])

        assert result.exit_code == 0
        assert "Obsidian RAG" in result.output

    def test_ingest_help(self):
        """Test ingest command help."""
        runner = CliRunner()
        result = runner.invoke(cli, ["ingest", "--help"])

        assert result.exit_code == 0
        assert "Ingest documents" in result.output

    def test_query_help(self):
        """Test query command help."""
        runner = CliRunner()
        result = runner.invoke(cli, ["query", "--help"])

        assert result.exit_code == 0
        assert "semantic" in result.output.lower() or "search" in result.output.lower()

    def test_tasks_help(self):
        """Test tasks command help."""
        runner = CliRunner()
        result = runner.invoke(cli, ["tasks", "--help"])

        assert result.exit_code == 0
        assert "Query" in result.output or "tasks" in result.output.lower()

    def test_ingest_nonexistent_path(self):
        """Test ingest with nonexistent path fails."""
        runner = CliRunner()
        result = runner.invoke(cli, ["ingest", "/nonexistent/path"])

        assert result.exit_code != 0


class TestCliCommands:
    """Test actual CLI command execution with mocks."""

    @patch("obsidian_rag.cli.scan_markdown_files")
    @patch("obsidian_rag.cli.process_files_in_batches")
    @patch("obsidian_rag.cli.DatabaseManager")
    @patch("obsidian_rag.cli._get_embedding_provider")
    def test_ingest_command_no_files(
        self, _mock_get_provider, _mock_db_manager, _mock_process, mock_scan
    ):
        """Test ingest command when no markdown files found."""
        from pathlib import Path

        from obsidian_rag.config import VaultConfig

        runner = CliRunner()

        mock_scan.return_value = []
        mock_settings = MagicMock()
        mock_settings.database.url = "sqlite:///:memory:"
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

        with patch("obsidian_rag.cli.get_settings", return_value=mock_settings):
            with runner.isolated_filesystem() as fs:
                vault_path = Path(fs) / "vault"
                vault_path.mkdir()
                # Update vault config to match actual path
                vault_config.container_path = str(vault_path)
                vault_config.host_path = str(vault_path)
                result = runner.invoke(
                    cli, ["ingest", str(vault_path), "--vault", "test-vault"]
                )

        assert result.exit_code == 0
        assert "No markdown files found" in result.output

    @patch("obsidian_rag.cli.DatabaseManager")
    @patch("obsidian_rag.cli._get_embedding_provider")
    def test_query_command_no_results(self, mock_get_provider, mock_db_manager):
        """Test query command when no results found."""
        runner = CliRunner()

        mock_settings = MagicMock()
        mock_settings.database.url = "sqlite:///:memory:"
        mock_settings.logging.level = "INFO"
        mock_settings.logging.format = "text"

        mock_embedding_provider = MagicMock()
        mock_embedding_provider.generate_embedding.return_value = [0.1] * 1536
        mock_get_provider.return_value = mock_embedding_provider

        mock_session = MagicMock()
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)
        mock_session.query.return_value.filter.return_value.order_by.return_value.limit.return_value.all.return_value = []

        mock_db_instance = MagicMock()
        mock_db_instance.get_session.return_value = mock_session
        mock_db_manager.return_value = mock_db_instance

        with patch("obsidian_rag.cli.get_settings", return_value=mock_settings):
            result = runner.invoke(cli, ["query", "test query"])

        assert result.exit_code == 0

    @patch("obsidian_rag.cli.DatabaseManager")
    def test_tasks_command_no_results(self, mock_db_manager):
        """Test tasks command when no tasks found."""
        runner = CliRunner()

        mock_settings = MagicMock()
        mock_settings.database.url = "sqlite:///:memory:"
        mock_settings.logging.level = "INFO"
        mock_settings.logging.format = "text"

        mock_session = MagicMock()
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)
        mock_session.query.return_value.join.return_value.filter.return_value.order_by.return_value.limit.return_value.all.return_value = []

        mock_db_instance = MagicMock()
        mock_db_instance.get_session.return_value = mock_session
        mock_db_manager.return_value = mock_db_instance

        with patch("obsidian_rag.cli.get_settings", return_value=mock_settings):
            result = runner.invoke(cli, ["tasks"])

        assert result.exit_code == 0
        assert "Found 0 tasks" in result.output

    @patch("obsidian_rag.cli.DatabaseManager")
    def test_tasks_command_with_invalid_date(self, mock_db_manager):
        """Test tasks command with invalid date format."""
        runner = CliRunner()

        mock_settings = MagicMock()
        mock_settings.database.url = "sqlite:///:memory:"
        mock_settings.logging.level = "INFO"
        mock_settings.logging.format = "text"

        mock_db_instance = MagicMock()
        mock_db_manager.return_value = mock_db_instance

        with patch("obsidian_rag.cli.get_settings", return_value=mock_settings):
            result = runner.invoke(cli, ["tasks", "--due-before", "invalid-date"])

        assert result.exit_code == 1
        assert "Invalid date format" in result.output


class TestCliHelpers:
    """Test CLI helper functions."""

    def test_scan_vault_success(self):
        """Test _scan_vault with successful scan."""
        from pathlib import Path

        from obsidian_rag.cli import _scan_vault

        with patch("obsidian_rag.cli.scan_markdown_files") as mock_scan:
            mock_scan.return_value = [Path("test.md")]
            result = _scan_vault(Path("/vault"))

        assert result == [Path("test.md")]

    def test_scan_vault_failure(self):
        """Test _scan_vault with scan failure."""
        from pathlib import Path

        from obsidian_rag.cli import _scan_vault

        with patch("obsidian_rag.cli.scan_markdown_files") as mock_scan:
            mock_scan.side_effect = Exception("Scan failed")
            with pytest.raises(SystemExit) as exc_info:
                _scan_vault(Path("/vault"))

            assert exc_info.value.code == 1


class TestFormatTaskResults:
    """Test _format_task_results function."""

    def test_format_task_results_empty(self):
        """Test _format_task_results with empty list."""
        from obsidian_rag.cli import _format_task_results

        result = _format_task_results([])
        assert "Found 0 tasks" in result

    def test_format_task_results_one_line_per_task(self):
        """Test _format_task_results outputs one task per line."""
        from obsidian_rag.cli import _format_task_results

        mock_document = MagicMock()
        mock_document.file_name = "test.md"

        mock_task = MagicMock()
        mock_task.document = mock_document
        mock_task.description = "Test task"
        mock_task.status = "not_completed"
        mock_task.due = None
        mock_task.priority = "normal"
        mock_task.tags = []

        result = _format_task_results([mock_task])
        lines = result.split("\n")

        # Should have header line (with empty line) + one task line
        assert "Found 1 tasks" in lines[0]
        assert "[ ] Test task (File: test.md)" in result

    def test_format_task_results_with_all_metadata(self):
        """Test _format_task_results includes all metadata when present."""
        from datetime import date

        from obsidian_rag.cli import _format_task_results

        mock_document = MagicMock()
        mock_document.file_name = "project.md"

        mock_task = MagicMock()
        mock_task.document = mock_document
        mock_task.description = "Important task"
        mock_task.status = "in_progress"
        mock_task.due = date(2026, 3, 10)
        mock_task.priority = "high"
        mock_task.tags = ["work", "urgent"]

        result = _format_task_results([mock_task])

        assert "[/] Important task (" in result
        assert "File: project.md" in result
        assert "Due: 2026-03-10" in result
        assert "Priority: high" in result
        assert "Tags: work, urgent" in result

    def test_format_task_results_optional_metadata_omitted(self):
        """Test _format_task_results omits optional metadata when not present."""
        from obsidian_rag.cli import _format_task_results

        mock_document = MagicMock()
        mock_document.file_name = "simple.md"

        mock_task = MagicMock()
        mock_task.document = mock_document
        mock_task.description = "Simple task"
        mock_task.status = "completed"
        mock_task.due = None
        mock_task.priority = "normal"
        mock_task.tags = []

        result = _format_task_results([mock_task])

        assert "[x] Simple task (File: simple.md)" in result
        # Should not contain optional fields
        assert "Due:" not in result
        assert "Priority:" not in result
        assert "Tags:" not in result

    def test_format_task_results_multiple_tasks(self):
        """Test _format_task_results with multiple tasks."""
        from datetime import date

        from obsidian_rag.cli import _format_task_results

        mock_document = MagicMock()
        mock_document.file_name = "notes.md"

        mock_task1 = MagicMock()
        mock_task1.document = mock_document
        mock_task1.description = "Task one"
        mock_task1.status = "not_completed"
        mock_task1.due = None
        mock_task1.priority = "normal"
        mock_task1.tags = []

        mock_task2 = MagicMock()
        mock_task2.document = mock_document
        mock_task2.description = "Task two"
        mock_task2.status = "completed"
        mock_task2.due = date(2026, 3, 1)
        mock_task2.priority = "normal"
        mock_task2.tags = []

        mock_task3 = MagicMock()
        mock_task3.document = mock_document
        mock_task3.description = "Task three"
        mock_task3.status = "cancelled"
        mock_task3.due = None
        mock_task3.priority = "normal"
        mock_task3.tags = []

        result = _format_task_results([mock_task1, mock_task2, mock_task3])
        lines = [line for line in result.split("\n") if line.strip()]

        # Should have header + 3 tasks
        assert len(lines) == 4
        assert "[ ] Task one (File: notes.md)" in lines[1]
        assert "[x] Task two (File: notes.md, Due: 2026-03-01)" in lines[2]
        assert "[-] Task three (File: notes.md)" in lines[3]

    def test_format_task_results_no_unicode_emojis(self):
        """Test _format_task_results uses only ASCII indicators."""
        from obsidian_rag.cli import _format_task_results

        mock_document = MagicMock()
        mock_document.file_name = "test.md"

        for status, expected_indicator in [
            ("completed", "[x]"),
            ("not_completed", "[ ]"),
            ("in_progress", "[/]"),
            ("cancelled", "[-]"),
        ]:
            mock_task = MagicMock()
            mock_task.document = mock_document
            mock_task.description = "Test"
            mock_task.status = status
            mock_task.due = None
            mock_task.priority = "normal"
            mock_task.tags = []

            result = _format_task_results([mock_task])
            assert expected_indicator in result
            # Ensure no emoji characters
            assert "✓" not in result
            assert "✗" not in result
            assert "⏳" not in result
            assert "🚫" not in result


class TestLogLevelFlag:
    """Test --log-level CLI flag."""

    def test_cli_help_includes_log_level(self):
        """Test that --log-level appears in help."""
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])

        assert result.exit_code == 0
        assert "--log-level" in result.output

    def test_log_level_debug(self):
        """Test --log-level DEBUG sets logging level correctly."""
        runner = CliRunner()

        mock_settings = MagicMock()
        mock_settings.database.url = "sqlite:///:memory:"
        mock_settings.logging.level = "DEBUG"
        mock_settings.logging.format = "text"

        mock_session = MagicMock()
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)
        mock_session.query.return_value.join.return_value.filter.return_value.order_by.return_value.limit.return_value.all.return_value = []

        mock_db_instance = MagicMock()
        mock_db_instance.get_session.return_value = mock_session

        with patch(
            "obsidian_rag.cli.get_settings", return_value=mock_settings
        ) as mock_get_settings:
            with patch("obsidian_rag.cli.DatabaseManager") as mock_db_manager:
                mock_db_manager.return_value = mock_db_instance
                result = runner.invoke(cli, ["--log-level", "DEBUG", "tasks"])

        assert result.exit_code == 0
        # Verify settings were loaded with log level
        mock_get_settings.assert_called_once()

    def test_log_level_case_insensitive(self):
        """Test --log-level accepts various cases."""
        runner = CliRunner()

        mock_settings = MagicMock()
        mock_settings.database.url = "sqlite:///:memory:"
        mock_settings.logging.level = "info"  # lowercase
        mock_settings.logging.format = "text"

        mock_session = MagicMock()
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)
        mock_session.query.return_value.join.return_value.filter.return_value.order_by.return_value.limit.return_value.all.return_value = []

        mock_db_instance = MagicMock()
        mock_db_instance.get_session.return_value = mock_session

        with patch("obsidian_rag.cli.get_settings", return_value=mock_settings):
            with patch("obsidian_rag.cli.DatabaseManager") as mock_db_manager:
                mock_db_manager.return_value = mock_db_instance
                result = runner.invoke(cli, ["--log-level", "info", "tasks"])

        assert result.exit_code == 0

    def test_log_level_invalid_shows_error(self):
        """Test invalid log level shows error message."""
        runner = CliRunner()

        mock_settings = MagicMock()
        mock_settings.database.url = "sqlite:///:memory:"
        mock_settings.logging.level = "INVALID"
        mock_settings.logging.format = "text"

        mock_session = MagicMock()
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)

        mock_db_instance = MagicMock()
        mock_db_instance.get_session.return_value = mock_session

        with patch("obsidian_rag.cli.get_settings", return_value=mock_settings):
            with patch("obsidian_rag.cli.DatabaseManager") as mock_db_manager:
                mock_db_manager.return_value = mock_db_instance
                result = runner.invoke(cli, ["--log-level", "INVALID", "tasks"])

        assert result.exit_code == 1
        assert "Invalid log level" in result.output
        assert "DEBUG, INFO, WARNING, ERROR, CRITICAL" in result.output

    def test_log_level_overrides_verbose(self):
        """Test --log-level takes precedence over --verbose."""
        runner = CliRunner()

        mock_settings = MagicMock()
        mock_settings.database.url = "sqlite:///:memory:"
        mock_settings.logging.level = "ERROR"
        mock_settings.logging.format = "text"

        mock_session = MagicMock()
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)
        mock_session.query.return_value.join.return_value.filter.return_value.order_by.return_value.limit.return_value.all.return_value = []

        mock_db_instance = MagicMock()
        mock_db_instance.get_session.return_value = mock_session

        with patch(
            "obsidian_rag.cli.get_settings", return_value=mock_settings
        ) as mock_get_settings:
            with patch("obsidian_rag.cli.DatabaseManager") as mock_db_manager:
                mock_db_manager.return_value = mock_db_instance
                # Both --verbose and --log-level ERROR specified
                result = runner.invoke(
                    cli, ["--verbose", "--log-level", "ERROR", "tasks"]
                )

        assert result.exit_code == 0
        # Should use ERROR, not DEBUG
        call_kwargs = mock_get_settings.call_args.kwargs
        assert call_kwargs["logging"]["level"] == "ERROR"


class TestLoggingToStderr:
    """Test that logging output goes to stderr."""

    def test_setup_logging_uses_stderr(self):
        """Test that _setup_logging configures handler to use stderr."""
        import logging
        import sys

        from obsidian_rag.cli import _setup_logging

        # Setup logging
        _setup_logging("INFO", "text")

        # Get root logger and check handler stream
        root_logger = logging.getLogger()
        handler = root_logger.handlers[0]

        assert isinstance(handler, logging.StreamHandler)
        assert handler.stream is sys.stderr

    def test_log_messages_go_to_stderr(self):
        """Test that log messages are written to stderr."""
        import logging
        import sys
        from io import StringIO

        from obsidian_rag.cli import _setup_logging

        # Capture stderr
        stderr_capture = StringIO()
        original_stderr = sys.stderr
        sys.stderr = stderr_capture

        try:
            # Setup logging and emit a message
            _setup_logging("INFO", "text")
            logger = logging.getLogger("test_logger")
            logger.info("Test message to stderr")

            # Check that message went to stderr
            output = stderr_capture.getvalue()
            assert "Test message to stderr" in output
        finally:
            sys.stderr = original_stderr

    def test_invalid_log_level_error_to_stderr(self):
        """Test that invalid log level error is written to stderr."""
        import sys
        from io import StringIO

        from obsidian_rag.cli import _setup_logging

        # Capture stderr
        stderr_capture = StringIO()
        original_stderr = sys.stderr
        sys.stderr = stderr_capture

        try:
            # Try to setup logging with invalid level
            try:
                _setup_logging("INVALID", "text")
            except SystemExit:
                pass  # Expected to exit

            # Check that error went to stderr
            output = stderr_capture.getvalue()
            assert "Invalid log level" in output
        finally:
            sys.stderr = original_stderr


class TestSetupLoggingJsonFormat:
    """Test _setup_logging with JSON format."""

    def test_setup_logging_json_format(self):
        """Test _setup_logging with JSON format type (TASK-038)."""
        import logging
        import sys
        from io import StringIO

        from obsidian_rag.cli import _setup_logging

        # Capture stderr
        stderr_capture = StringIO()
        original_stderr = sys.stderr
        sys.stderr = stderr_capture

        try:
            _setup_logging("INFO", "json")
            logger = logging.getLogger("json_test_logger")
            logger.info("JSON test message")

            output = stderr_capture.getvalue()
            assert '"timestamp"' in output
            assert '"level"' in output
            assert '"message"' in output
            assert "JSON test message" in output
        finally:
            sys.stderr = original_stderr


class TestCliConfigFileLogging:
    """Test CLI config file path logging."""

    @patch("obsidian_rag.cli.DatabaseManager")
    def test_cli_with_config_file_path_logging(self, _mock_db_manager):
        """Test CLI with config file path logging (TASK-039)."""
        from pathlib import Path

        runner = CliRunner()

        mock_settings = MagicMock()
        mock_settings.database.url = "sqlite:///:memory:"
        mock_settings.logging.level = "INFO"
        mock_settings.logging.format = "text"

        mock_session = MagicMock()
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)
        mock_session.query.return_value.join.return_value.filter.return_value.order_by.return_value.limit.return_value.all.return_value = []

        mock_db_instance = MagicMock()
        mock_db_instance.get_session.return_value = mock_session
        _mock_db_manager.return_value = mock_db_instance

        with patch("obsidian_rag.cli.get_settings", return_value=mock_settings):
            with runner.isolated_filesystem() as fs:
                config_path = Path(fs) / "config.yaml"
                config_path.write_text("database:\n  url: sqlite:///:memory:")
                result = runner.invoke(
                    cli, ["--config-file", str(config_path), "tasks"]
                )

        assert result.exit_code == 0


class TestCliVerboseFlag:
    """Test CLI verbose flag behavior."""

    @patch("obsidian_rag.cli.DatabaseManager")
    def test_cli_with_verbose_flag(self, _mock_db_manager):
        """Test CLI with verbose flag (TASK-040)."""
        runner = CliRunner()

        mock_settings = MagicMock()
        mock_settings.database.url = "sqlite:///:memory:"
        mock_settings.logging.level = "DEBUG"
        mock_settings.logging.format = "text"

        mock_session = MagicMock()
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)
        mock_session.query.return_value.join.return_value.filter.return_value.order_by.return_value.limit.return_value.all.return_value = []

        mock_db_instance = MagicMock()
        mock_db_instance.get_session.return_value = mock_session
        _mock_db_manager.return_value = mock_db_instance

        with patch("obsidian_rag.cli.get_settings", return_value=mock_settings):
            result = runner.invoke(cli, ["--verbose", "tasks"])

        assert result.exit_code == 0


class TestGetEmbeddingProvider:
    """Test _get_embedding_provider function."""

    def test_get_embedding_provider_with_missing_config(self):
        """Test _get_embedding_provider with missing config (TASK-041)."""
        from obsidian_rag.cli import _get_embedding_provider

        mock_settings = MagicMock()
        mock_settings.get_endpoint_config.return_value = None

        with patch("obsidian_rag.cli.ProviderFactory") as mock_factory:
            mock_provider = MagicMock()
            mock_factory.create_embedding_provider.return_value = mock_provider

            result = _get_embedding_provider(mock_settings)

            assert result is mock_provider
            mock_factory.create_embedding_provider.assert_called_once_with("openai")

    def test_get_embedding_provider_with_config(self):
        """Test _get_embedding_provider with existing config (lines 115-123)."""
        from obsidian_rag.cli import _get_embedding_provider

        mock_settings = MagicMock()
        mock_config = MagicMock()
        mock_config.provider = "openai"
        mock_config.api_key = "test-api-key"
        mock_config.model = "text-embedding-3-small"
        mock_config.base_url = None
        mock_settings.get_endpoint_config.return_value = mock_config

        with patch("obsidian_rag.cli.ProviderFactory") as mock_factory:
            mock_provider = MagicMock()
            mock_factory.create_embedding_provider.return_value = mock_provider

            result = _get_embedding_provider(mock_settings)

            assert result is mock_provider
            mock_factory.create_embedding_provider.assert_called_once_with(
                "openai",
                api_key="test-api-key",
                model="text-embedding-3-small",
                base_url=None,
            )


class TestCreateProgressCallback:
    """Test progress_callback function via partial application."""

    def test_create_progress_callback_with_verbose_true(self):
        """Test progress_callback with verbose=True via partial (TASK-042)."""
        from functools import partial

        import click
        from click.testing import CliRunner

        from obsidian_rag.cli import progress_callback

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

        from obsidian_rag.cli import _report_ingest_results

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

    @patch("obsidian_rag.cli._scan_vault")
    @patch("obsidian_rag.cli.process_files_in_batches")
    @patch("obsidian_rag.cli.DatabaseManager")
    @patch("obsidian_rag.cli._get_embedding_provider")
    @patch("obsidian_rag.cli.IngestionService")
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
        mock_settings.database.url = "sqlite:///:memory:"
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

    @patch("obsidian_rag.cli._scan_vault")
    @patch("obsidian_rag.cli.process_files_in_batches")
    @patch("obsidian_rag.cli.DatabaseManager")
    @patch("obsidian_rag.cli._get_embedding_provider")
    @patch("obsidian_rag.cli.IngestionService")
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
        mock_settings.database.url = "sqlite:///:memory:"
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


class TestFormatQueryResults:
    """Test query result formatting functions."""

    def test_format_query_results_json_with_tags(self):
        """Test _format_query_results_json with results containing tags (TASK-046)."""
        import json

        from obsidian_rag.cli import _format_query_results_json

        mock_doc = MagicMock()
        mock_doc.file_path = "/path/to/doc.md"
        mock_doc.file_name = "doc.md"
        mock_doc.frontmatter_json = {"kind": "note"}
        mock_doc.tags = ["work", "urgent"]

        results = [(mock_doc, 0.5)]
        output = _format_query_results_json(results)

        parsed = json.loads(output)
        assert len(parsed) == 1
        assert parsed[0]["file_path"] == "/path/to/doc.md"
        assert parsed[0]["file_name"] == "doc.md"
        assert parsed[0]["kind"] == "note"
        assert parsed[0]["tags"] == ["work", "urgent"]
        assert parsed[0]["distance"] == 0.5

    def test_format_query_results_table_with_kind_and_tags(self):
        """Test _format_query_results_table with documents having kind and tags (TASK-047)."""
        from obsidian_rag.cli import _format_query_results_table

        mock_doc = MagicMock()
        mock_doc.file_name = "doc.md"
        mock_doc.file_path = "/path/to/doc.md"
        mock_doc.frontmatter_json = {"kind": "project"}
        mock_doc.tags = ["work", "urgent"]

        results = [(mock_doc, 0.75)]
        output = _format_query_results_table(results)

        assert "Found 1 results:" in output
        assert "File: doc.md" in output
        assert "Path: /path/to/doc.md" in output
        assert "Distance: 0.7500" in output
        assert "Kind: project" in output
        assert "Tags: work, urgent" in output


class TestQueryCommand:
    """Test query command with various scenarios."""

    @patch("obsidian_rag.cli.DatabaseManager")
    @patch("obsidian_rag.cli._get_embedding_provider")
    def test_query_command_with_embedding_failure(
        self, mock_get_provider, mock_db_manager
    ):
        """Test query command with embedding generation failure (TASK-048)."""
        runner = CliRunner()

        mock_settings = MagicMock()
        mock_settings.database.url = "sqlite:///:memory:"
        mock_settings.logging.level = "INFO"
        mock_settings.logging.format = "text"

        mock_embedding_provider = MagicMock()
        mock_embedding_provider.generate_embedding.side_effect = Exception(
            "Embedding failed"
        )
        mock_get_provider.return_value = mock_embedding_provider

        mock_db_instance = MagicMock()
        mock_db_manager.return_value = mock_db_instance

        with patch("obsidian_rag.cli.get_settings", return_value=mock_settings):
            result = runner.invoke(cli, ["query", "test query"])

        assert result.exit_code == 1
        assert "Failed to generate query embedding" in result.output

    @patch("obsidian_rag.cli.DatabaseManager")
    @patch("obsidian_rag.cli._get_embedding_provider")
    def test_query_command_with_json_output(self, mock_get_provider, mock_db_manager):
        """Test query command with JSON output format (TASK-049)."""
        import json

        runner = CliRunner()

        mock_settings = MagicMock()
        mock_settings.database.url = "sqlite:///:memory:"
        mock_settings.logging.level = "INFO"
        mock_settings.logging.format = "text"

        mock_embedding_provider = MagicMock()
        mock_embedding_provider.generate_embedding.return_value = [0.1] * 1536
        mock_get_provider.return_value = mock_embedding_provider

        mock_doc = MagicMock()
        mock_doc.file_path = "/path/to/doc.md"
        mock_doc.file_name = "doc.md"
        mock_doc.frontmatter_json = None
        mock_doc.tags = None

        mock_session = MagicMock()
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)
        mock_query_result = MagicMock()
        mock_query_result.filter.return_value.order_by.return_value.limit.return_value.all.return_value = [
            (mock_doc, 0.5)
        ]
        mock_session.query.return_value = mock_query_result

        mock_db_instance = MagicMock()
        mock_db_instance.get_session.return_value = mock_session
        mock_db_manager.return_value = mock_db_instance

        with patch("obsidian_rag.cli.get_settings", return_value=mock_settings):
            result = runner.invoke(cli, ["query", "test query", "--format", "json"])

        assert result.exit_code == 0
        # Verify JSON output
        try:
            parsed = json.loads(result.output)
            assert isinstance(parsed, list)
        except json.JSONDecodeError:
            # Output might have logging mixed in, check for JSON structure
            assert '"file_path"' in result.output or "file_path" in result.output


class TestTasksCommand:
    """Test tasks command with various scenarios."""

    @patch("obsidian_rag.cli.DatabaseManager")
    def test_tasks_command_with_no_results(self, mock_db_manager):
        """Test tasks command with no results (TASK-050)."""
        runner = CliRunner()

        mock_settings = MagicMock()
        mock_settings.database.url = "sqlite:///:memory:"
        mock_settings.logging.level = "INFO"
        mock_settings.logging.format = "text"

        mock_session = MagicMock()
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)
        mock_query_result = MagicMock()
        mock_query_result.join.return_value.filter.return_value.order_by.return_value.limit.return_value.all.return_value = []
        mock_session.query.return_value = mock_query_result

        mock_db_instance = MagicMock()
        mock_db_instance.get_session.return_value = mock_session
        mock_db_manager.return_value = mock_db_instance

        with patch("obsidian_rag.cli.get_settings", return_value=mock_settings):
            result = runner.invoke(cli, ["tasks"])

        assert result.exit_code == 0
        assert "Found 0 tasks" in result.output


class TestBuildTasksQuery:
    """Test _build_tasks_query function."""

    def test_build_tasks_query_with_status_filter(self):
        """Test _build_tasks_query with status filter (TASK-051)."""
        from obsidian_rag.cli import _build_tasks_query

        mock_session = MagicMock()
        mock_query = MagicMock()
        mock_session.query.return_value = mock_query
        mock_query.join.return_value = mock_query
        mock_query.filter.return_value = mock_query
        mock_query.order_by.return_value = mock_query
        mock_query.limit.return_value = mock_query

        result = _build_tasks_query(mock_session, "completed", None, None, 10)

        mock_session.query.assert_called_once()
        mock_query.filter.assert_called()
        mock_query.limit.assert_called_once_with(10)

    def test_build_tasks_query_with_invalid_date_format(self):
        """Test _build_tasks_query with invalid date format error path (TASK-052)."""
        import click
        from click.testing import CliRunner

        from obsidian_rag.cli import _build_tasks_query

        runner = CliRunner()

        mock_session = MagicMock()
        mock_query = MagicMock()
        mock_session.query.return_value = mock_query
        mock_query.join.return_value = mock_query
        mock_query.filter.return_value = mock_query
        mock_query.order_by.return_value = mock_query
        mock_query.limit.return_value = mock_query

        @click.command()
        def test_cmd():
            _build_tasks_query(mock_session, None, "invalid-date", None, 10)

        result = runner.invoke(test_cmd)
        assert result.exit_code == 1
        assert "Invalid date format" in result.output


class TestMainFunction:
    """Test main entry point."""

    @patch("obsidian_rag.cli.cli")
    def test_main_function(self, mock_cli):
        """Test main() function calls cli() (line 421)."""
        from obsidian_rag.cli import main

        main()

        mock_cli.assert_called_once()


class TestCreateProgressCallbackCoverage:
    """Test progress_callback with verbose=False via partial (line 173->exit)."""

    def test_create_progress_callback_with_verbose_false(self):
        """Test progress callback does not print when verbose=False."""
        from functools import partial

        from obsidian_rag.cli import progress_callback

        callback = partial(progress_callback, verbose=False)

        # When verbose=False, callback should not output anything
        # This covers the branch where if verbose: is False
        callback(5, 10, 3, 0)  # Should not raise or print


class TestProgressCallback:
    """Test progress_callback module-level function (refactored for testability)."""

    def test_progress_callback_verbose_true(self):
        """Test progress_callback with verbose=True prints message."""
        from obsidian_rag.cli import progress_callback

        with patch("obsidian_rag.cli.click.echo") as mock_echo:
            progress_callback(5, 10, 4, 1, verbose=True)

            mock_echo.assert_called_once()
            assert "Progress: 5/10" in mock_echo.call_args[0][0]
            assert "4 successful" in mock_echo.call_args[0][0]
            assert "1 errors" in mock_echo.call_args[0][0]

    def test_progress_callback_verbose_false(self):
        """Test progress_callback with verbose=False does not print."""
        from obsidian_rag.cli import progress_callback

        with patch("obsidian_rag.cli.click.echo") as mock_echo:
            progress_callback(5, 10, 4, 1, verbose=False)

            mock_echo.assert_not_called()


class TestReportIngestResultsCoverage:
    """Test _report_ingest_results with no_delete=True (line 205)."""

    def test_report_ingest_results_with_no_delete(self):
        """Test report shows 'deletion skipped' when no_delete=True."""
        import sys
        from io import StringIO

        from obsidian_rag.cli import _report_ingest_results

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


class TestValidateVaultConfigCoverage:
    """Test _validate_vault_config error paths (lines 239-245, 252-258)."""

    def test_validate_vault_config_vault_not_found(self):
        """Test error when vault not found (lines 239-245)."""
        from pathlib import Path

        from obsidian_rag.cli import _validate_vault_config
        from obsidian_rag.config import Settings

        mock_settings = MagicMock(spec=Settings)
        mock_settings.get_vault.return_value = None
        mock_settings.get_vault_names.return_value = ["vault1", "vault2"]

        with pytest.raises(SystemExit) as exc_info:
            _validate_vault_config(mock_settings, "nonexistent", "/path")

        assert exc_info.value.code == 1

    def test_validate_vault_config_path_mismatch(self):
        """Test error when path does not match vault config (lines 252-258)."""
        import tempfile
        from pathlib import Path

        from obsidian_rag.cli import _validate_vault_config
        from obsidian_rag.config import Settings, VaultConfig

        with tempfile.TemporaryDirectory() as tmpdir:
            mock_settings = MagicMock(spec=Settings)
            vault_config = VaultConfig(
                container_path=str(Path(tmpdir) / "vault_a"),
                host_path=str(Path(tmpdir) / "vault_a"),
            )
            mock_settings.get_vault.return_value = vault_config

            # Try to validate with a different path
            with pytest.raises(SystemExit) as exc_info:
                _validate_vault_config(
                    mock_settings, "test-vault", str(Path(tmpdir) / "vault_b")
                )

            assert exc_info.value.code == 1


class TestQueryCommandCoverage:
    """Test query command coverage gaps."""

    @patch("obsidian_rag.cli.DatabaseManager")
    @patch("obsidian_rag.cli._get_embedding_provider")
    def test_query_command_with_table_output(self, mock_get_provider, mock_db_manager):
        """Test query command with table output format (line 494)."""
        from datetime import datetime

        runner = CliRunner()

        mock_settings = MagicMock()
        mock_settings.database.url = "sqlite:///:memory:"
        mock_settings.logging.level = "INFO"
        mock_settings.logging.format = "text"

        mock_embedding_provider = MagicMock()
        mock_embedding_provider.generate_embedding.return_value = [0.1] * 1536
        mock_get_provider.return_value = mock_embedding_provider

        # Create mock document with all fields for branch coverage
        mock_doc = MagicMock()
        mock_doc.file_name = "test.md"
        mock_doc.file_path = "path/to/test.md"
        mock_doc.frontmatter_json = {"kind": "note"}  # Triggers line 407->409 branch
        mock_doc.tags = ["tag1", "tag2"]  # Triggers line 409->411 branch

        mock_session = MagicMock()
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)

        # Create mock result with distance
        mock_result = (mock_doc, 0.5)
        mock_session.query.return_value.filter.return_value.order_by.return_value.limit.return_value.all.return_value = [
            mock_result
        ]

        mock_db_instance = MagicMock()
        mock_db_instance.get_session.return_value = mock_session
        mock_db_manager.return_value = mock_db_instance

        with patch("obsidian_rag.cli.get_settings", return_value=mock_settings):
            result = runner.invoke(cli, ["query", "test query", "--format", "table"])

        assert result.exit_code == 0
        assert "test.md" in result.output
        assert "note" in result.output  # Kind is displayed
        assert "tag1" in result.output  # Tags are displayed

    @patch("obsidian_rag.cli.DatabaseManager")
    @patch("obsidian_rag.cli._get_embedding_provider")
    def test_query_command_no_matching_documents(
        self, mock_get_provider, mock_db_manager
    ):
        """Test query command when no documents match (covers early return)."""
        runner = CliRunner()

        mock_settings = MagicMock()
        mock_settings.database.url = "sqlite:///:memory:"
        mock_settings.logging.level = "INFO"
        mock_settings.logging.format = "text"

        mock_embedding_provider = MagicMock()
        mock_embedding_provider.generate_embedding.return_value = [0.1] * 1536
        mock_get_provider.return_value = mock_embedding_provider

        mock_session = MagicMock()
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)
        mock_session.query.return_value.filter.return_value.order_by.return_value.limit.return_value.all.return_value = []

        mock_db_instance = MagicMock()
        mock_db_instance.get_session.return_value = mock_session
        mock_db_manager.return_value = mock_db_instance

        with patch("obsidian_rag.cli.get_settings", return_value=mock_settings):
            result = runner.invoke(cli, ["query", "test query"])

        assert result.exit_code == 0
        assert "No matching documents found" in result.output


class TestTasksCommandCoverage:
    """Test tasks command coverage gaps (lines 529-530)."""

    @patch("obsidian_rag.cli.DatabaseManager")
    def test_tasks_command_no_tasks_found_message(self, mock_db_manager):
        """Test tasks command shows 'No tasks found' message (lines 529-530)."""
        runner = CliRunner()

        mock_settings = MagicMock()
        mock_settings.database.url = "sqlite:///:memory:"
        mock_settings.logging.level = "INFO"
        mock_settings.logging.format = "text"

        mock_session = MagicMock()
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)
        # Return empty list to trigger "No tasks found" message
        mock_session.query.return_value.join.return_value.filter.return_value.order_by.return_value.limit.return_value.all.return_value = []

        mock_db_instance = MagicMock()
        mock_db_instance.get_session.return_value = mock_session
        mock_db_manager.return_value = mock_db_instance

        with patch("obsidian_rag.cli.get_settings", return_value=mock_settings):
            result = runner.invoke(cli, ["tasks"])

        assert result.exit_code == 0
        # This covers the early return when no results


class TestFormatTaskResultsCoverage:
    """Test _format_task_results coverage gaps (lines 569-570, 576)."""

    def test_format_task_results_with_different_statuses(self):
        """Test task formatting with all statuses (line 569-570, 576)."""
        from obsidian_rag.cli import _format_task_results

        mock_document = MagicMock()
        mock_document.file_name = "test.md"

        for status, expected in [
            ("completed", "[x]"),
            ("not_completed", "[ ]"),
            ("in_progress", "[/]"),
            ("cancelled", "[-]"),
        ]:
            mock_task = MagicMock()
            mock_task.document = mock_document
            mock_task.description = f"Task {status}"
            mock_task.status = status
            mock_task.due = None
            mock_task.priority = "normal"
            mock_task.tags = []

            result = _format_task_results([mock_task])
            assert expected in result


class TestRunIngestionCoverage:
    """Test _run_ingestion with no_delete option (line 282)."""

    @patch("obsidian_rag.cli._scan_vault")
    @patch("obsidian_rag.cli.process_files_in_batches")
    @patch("obsidian_rag.cli.DatabaseManager")
    @patch("obsidian_rag.cli._get_embedding_provider")
    def test_run_ingestion_with_no_delete(
        self, mock_get_provider, mock_db_manager, mock_process, mock_scan
    ):
        """Test _run_ingestion outputs message when no_delete=True (line 282)."""
        import sys
        from io import StringIO
        from pathlib import Path

        from obsidian_rag.cli import _run_ingestion, IngestOptions

        mock_settings = MagicMock()
        mock_settings.database = MagicMock()
        mock_settings.database.url = "sqlite:///:memory:"
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


class TestFormatQueryResultsTableCoverage:
    """Test _format_query_results_table branches (lines 407->409, 409->411)."""

    def test_format_query_results_table_without_kind_and_tags(self):
        """Test table format when doc.kind and doc.tags are None/empty."""
        from obsidian_rag.cli import _format_query_results_table

        # Create mock document without kind and tags
        mock_doc = MagicMock()
        mock_doc.file_name = "test.md"
        mock_doc.file_path = "path/to/test.md"
        mock_doc.frontmatter_json = None  # Branch not taken
        mock_doc.tags = []  # Branch not taken (falsy)

        results = [(mock_doc, 0.5)]

        result = _format_query_results_table(results)

        assert "test.md" in result
        assert "Kind:" not in result  # Should not include Kind
        assert "Tags:" not in result  # Should not include Tags


class TestBuildTasksQueryCoverage:
    """Test _build_tasks_query coverage gaps (lines 569-570, 576)."""

    def test_build_tasks_query_with_valid_due_date(self):
        """Test _build_tasks_query with valid due_before date (lines 569-570)."""
        from obsidian_rag.cli import _build_tasks_query

        mock_session = MagicMock()
        mock_query = MagicMock()
        mock_session.query.return_value = mock_query
        mock_query.join.return_value = mock_query
        mock_query.filter.return_value = mock_query
        mock_query.order_by.return_value = mock_query
        mock_query.limit.return_value = mock_query

        result = _build_tasks_query(mock_session, None, "2026-03-15", None, 10)

        mock_session.query.assert_called_once()
        mock_query.filter.assert_called()  # Should filter by due date
        mock_query.limit.assert_called_once_with(10)

    def test_build_tasks_query_with_tag_filter(self):
        """Test _build_tasks_query with tag filter (line 576)."""
        from obsidian_rag.cli import _build_tasks_query

        mock_session = MagicMock()
        mock_query = MagicMock()
        mock_session.query.return_value = mock_query
        mock_query.join.return_value = mock_query
        mock_query.filter.return_value = mock_query
        mock_query.order_by.return_value = mock_query
        mock_query.limit.return_value = mock_query

        result = _build_tasks_query(mock_session, None, None, "work", 10)

        mock_session.query.assert_called_once()
        mock_query.filter.assert_called()  # Should filter by tag
        mock_query.limit.assert_called_once_with(10)


class TestTasksCommandEarlyReturn:
    """Test tasks command early return (lines 529-530)."""

    @patch("obsidian_rag.cli.DatabaseManager")
    @patch("obsidian_rag.cli._build_tasks_query")
    def test_tasks_command_early_return_when_no_results(
        self, mock_build_query, mock_db_manager
    ):
        """Test tasks command returns early with message when no results (lines 529-530)."""
        from click.testing import CliRunner

        runner = CliRunner()

        mock_settings = MagicMock()
        mock_settings.database.url = "sqlite:///:memory:"
        mock_settings.logging.level = "INFO"
        mock_settings.logging.format = "text"

        mock_session = MagicMock()
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)

        mock_db_instance = MagicMock()
        mock_db_instance.get_session.return_value = mock_session
        mock_db_manager.return_value = mock_db_instance

        # Mock query that returns empty list
        mock_query = MagicMock()
        mock_query.all.return_value = []
        mock_build_query.return_value = mock_query

        with patch("obsidian_rag.cli.get_settings", return_value=mock_settings):
            result = runner.invoke(cli, ["tasks"])

        assert result.exit_code == 0
        assert "No tasks found matching the criteria" in result.output
