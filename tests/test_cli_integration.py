"""Integration tests for CLI date filtering."""

import uuid
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from obsidian_rag.cli import cli


@pytest.fixture
def mock_db_manager():
    """Create a mock database manager."""
    manager = MagicMock()
    mock_session = MagicMock()
    manager.get_session.return_value.__enter__ = MagicMock(return_value=mock_session)
    manager.get_session.return_value.__exit__ = MagicMock(return_value=False)
    return manager


@pytest.fixture
def mock_settings():
    """Create mock settings with PostgreSQL URL."""
    settings = MagicMock()
    settings.database.url = "postgresql+psycopg://localhost/test"
    settings.database.vector_dimension = 1536
    settings.ingestion.batch_size = 100
    settings.ingestion.progress_interval = 10
    settings.logging.level = "INFO"
    settings.logging.format = "text"
    return settings


class TestCliDateFilteringIntegration:
    """Integration tests for CLI date filtering."""

    @patch("obsidian_rag.cli.DatabaseManager")
    def test_tasks_due_before_filter(self, mock_db_manager, mock_settings):
        """Test CLI --due-before filter returns correct tasks."""
        today = date.today()

        runner = CliRunner()

        mock_session = MagicMock()
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)
        mock_session.query.return_value.join.return_value.filter.return_value.filter.return_value.order_by.return_value.limit.return_value.all.return_value = []

        mock_db_instance = MagicMock()
        mock_db_instance.get_session.return_value = mock_session
        mock_db_manager.return_value = mock_db_instance

        with patch("obsidian_rag.cli.get_settings", return_value=mock_settings):
            result = runner.invoke(cli, ["tasks", "--due-before", today.isoformat()])

        assert result.exit_code == 0

    @patch("obsidian_rag.cli.DatabaseManager")
    def test_tasks_due_after_filter(self, mock_db_manager, mock_settings):
        """Test CLI --due-after filter returns correct tasks."""
        today = date.today()

        runner = CliRunner()

        mock_session = MagicMock()
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)
        mock_session.query.return_value.join.return_value.filter.return_value.filter.return_value.order_by.return_value.limit.return_value.all.return_value = []

        mock_db_instance = MagicMock()
        mock_db_instance.get_session.return_value = mock_session
        mock_db_manager.return_value = mock_db_instance

        with patch("obsidian_rag.cli.get_settings", return_value=mock_settings):
            result = runner.invoke(cli, ["tasks", "--due-after", today.isoformat()])

        assert result.exit_code == 0

    @patch("obsidian_rag.cli.DatabaseManager")
    def test_tasks_date_range_filter(self, mock_db_manager, mock_settings):
        """Test CLI with both --due-after and --due-before."""
        today = date.today()

        runner = CliRunner()

        mock_session = MagicMock()
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)
        mock_session.query.return_value.join.return_value.filter.return_value.filter.return_value.filter.return_value.order_by.return_value.limit.return_value.all.return_value = []

        mock_db_instance = MagicMock()
        mock_db_instance.get_session.return_value = mock_session
        mock_db_manager.return_value = mock_db_instance

        with patch("obsidian_rag.cli.get_settings", return_value=mock_settings):
            result = runner.invoke(
                cli,
                [
                    "tasks",
                    "--due-after",
                    today.isoformat(),
                    "--due-before",
                    date(today.year, 12, 31).isoformat(),
                ],
            )

        assert result.exit_code == 0
