"""Tests for CLI commands."""

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
