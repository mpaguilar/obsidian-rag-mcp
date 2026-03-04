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
        self, mock_get_provider, mock_db_manager, mock_process, mock_scan
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
