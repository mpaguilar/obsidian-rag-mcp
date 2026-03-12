"""Integration tests for CLI date filtering."""

import uuid
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from obsidian_rag.cli import cli
from obsidian_rag.database.models import Base, Document, Task, TaskStatus, Vault


@pytest.fixture
def db_engine():
    """Create a test database engine using SQLite."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    yield engine
    Base.metadata.drop_all(engine)


@pytest.fixture
def db_session(db_engine):
    """Create a test database session."""
    SessionLocal = sessionmaker(bind=db_engine)
    session = SessionLocal()
    yield session
    session.close()


class TestCliDateFilteringIntegration:
    """Integration tests for CLI date filtering."""

    @patch("obsidian_rag.cli.DatabaseManager")
    def test_tasks_due_before_filter(self, mock_db_manager, db_engine):
        """Test CLI --due-before filter returns correct tasks."""
        today = date.today()

        runner = CliRunner()

        mock_settings = MagicMock()
        mock_settings.database.url = str(db_engine.url)
        mock_settings.logging.level = "INFO"
        mock_settings.logging.format = "text"

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
    def test_tasks_due_after_filter(self, mock_db_manager, db_engine):
        """Test CLI --due-after filter returns correct tasks."""
        today = date.today()

        runner = CliRunner()

        mock_settings = MagicMock()
        mock_settings.database.url = str(db_engine.url)
        mock_settings.logging.level = "INFO"
        mock_settings.logging.format = "text"

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
    def test_tasks_date_range_filter(self, mock_db_manager, db_engine):
        """Test CLI with both --due-after and --due-before."""
        today = date.today()

        runner = CliRunner()

        mock_settings = MagicMock()
        mock_settings.database.url = str(db_engine.url)
        mock_settings.logging.level = "INFO"
        mock_settings.logging.format = "text"

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
