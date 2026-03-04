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

        runner = CliRunner()

        mock_scan.return_value = []
        mock_settings = MagicMock()
        mock_settings.database.url = "sqlite:///:memory:"
        mock_settings.ingestion.batch_size = 100
        mock_settings.ingestion.progress_interval = 10
        mock_settings.logging.level = "INFO"
        mock_settings.logging.format = "text"

        with patch("obsidian_rag.cli.get_settings", return_value=mock_settings):
            with runner.isolated_filesystem() as fs:
                vault_path = Path(fs) / "vault"
                vault_path.mkdir()
                result = runner.invoke(cli, ["ingest", str(vault_path)])

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

    def test_update_stats_new(self):
        """Test _update_stats with new result."""
        from obsidian_rag.cli import _update_stats

        stats = {"new": 0, "updated": 0, "unchanged": 0, "errors": 0}
        _update_stats(stats, "new")

        assert stats["new"] == 1

    def test_update_stats_unknown(self):
        """Test _update_stats with unknown result."""
        from obsidian_rag.cli import _update_stats

        stats = {"new": 0, "updated": 0, "unchanged": 0, "errors": 0}
        _update_stats(stats, "unknown")

        # Stats should remain unchanged
        assert stats == {"new": 0, "updated": 0, "unchanged": 0, "errors": 0}


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


class TestProcessingContext:
    """Test ProcessingContext class."""

    def test_processing_context_init(self):
        """Test ProcessingContext initialization."""
        from obsidian_rag.cli import ProcessingContext

        mock_db_manager = MagicMock()
        mock_embedding_provider = MagicMock()
        stats = {"new": 0, "updated": 0, "unchanged": 0, "errors": 0}

        ctx = ProcessingContext(
            db_manager=mock_db_manager,
            embedding_provider=mock_embedding_provider,
            dry_run=True,
            verbose=True,
            stats=stats,
        )

        assert ctx.db_manager is mock_db_manager
        assert ctx.embedding_provider is mock_embedding_provider
        assert ctx.dry_run is True
        assert ctx.verbose is True
        assert ctx.stats is stats


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
