"""Tests for CLI vault list/get ingest status display."""

import json
import uuid
from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from obsidian_rag.cli import cli


def _make_mock_vault(
    ingest_status="idle",
    ingest_started_at=None,
    ingest_pid=None,
    ingest_force=False,
):
    """Create a mock vault with configurable ingest fields."""
    vault_id = uuid.uuid4()
    mock_vault = MagicMock()
    mock_vault.id = vault_id
    mock_vault.name = "Test Vault"
    mock_vault.description = "A test vault"
    mock_vault.container_path = "/data/test"
    mock_vault.host_path = "/home/user/test"
    mock_vault.created_at = datetime(2024, 1, 15, 10, 30, 0, tzinfo=UTC)
    mock_vault.ingest_status = ingest_status
    mock_vault.ingest_started_at = ingest_started_at
    mock_vault.ingest_pid = ingest_pid
    mock_vault.ingest_force = ingest_force
    return mock_vault


def _make_mock_session(mock_result):
    """Create a mock session that returns the given vault result."""
    mock_session = MagicMock()
    mock_subquery = MagicMock()
    mock_subquery.c = MagicMock()
    mock_subquery.c.vault_id = "vault_id"
    mock_subquery.c.doc_count = "doc_count"

    mock_session.query.return_value.group_by.return_value.subquery.return_value = (
        mock_subquery
    )

    mock_query = MagicMock()
    mock_query.outerjoin.return_value.order_by.return_value = mock_query
    mock_query.count.return_value = len(mock_result)
    mock_query.offset.return_value.limit.return_value.all.return_value = mock_result

    mock_session.query.return_value = mock_query
    return mock_session


def _make_mock_db_manager(mock_session):
    """Create a mock DatabaseManager that yields the given session."""
    mock_db_manager = MagicMock()
    mock_db_manager.get_session.return_value.__enter__ = MagicMock(
        return_value=mock_session
    )
    mock_db_manager.get_session.return_value.__exit__ = MagicMock(return_value=None)
    return mock_db_manager


def _make_mock_settings():
    """Create mock settings with database config."""
    mock_settings = MagicMock()
    mock_settings.database.url = "postgresql+psycopg://localhost/test"
    mock_settings.database.pool_size = 5
    mock_settings.database.max_overflow = 10
    mock_settings.database.pool_timeout = 30
    mock_settings.database.pool_recycle = 3600
    mock_settings.logging.level = "INFO"
    mock_settings.logging.format = "text"
    return mock_settings


def _patch_db_manager(mock_db_manager_class, mock_session):
    """Patch DatabaseManager class with the mock manager."""
    mock_db_manager = _make_mock_db_manager(mock_session)
    mock_db_manager_class.return_value = mock_db_manager
    return mock_db_manager


@patch("obsidian_rag.cli_vault_commands.DatabaseManager")
def test_vault_list_table_shows_ingest_column(mock_db_manager_class) -> None:
    """Table output includes an Ingest column with the status values."""
    runner = CliRunner()

    mock_vault_idle = _make_mock_vault(ingest_status="idle")
    mock_vault_progress = _make_mock_vault(ingest_status="in_progress")
    mock_vault_failed = _make_mock_vault(ingest_status="failed")

    mock_result = [
        (mock_vault_idle, 10),
        (mock_vault_progress, 20),
        (mock_vault_failed, 30),
    ]
    mock_session = _make_mock_session(mock_result)
    _patch_db_manager(mock_db_manager_class, mock_session)

    mock_settings = _make_mock_settings()

    with patch("obsidian_rag.cli.get_settings", return_value=mock_settings):
        result = runner.invoke(cli, ["vault", "list"])

    assert result.exit_code == 0
    assert "idle" in result.output
    assert "in_progress" in result.output
    assert "failed" in result.output


@patch("obsidian_rag.cli_vault_commands.DatabaseManager")
def test_vault_list_json_includes_lock_fields(mock_db_manager_class) -> None:
    """JSON format output contains ingest_status, ingest_started_at, ingest_pid, ingest_force keys."""
    runner = CliRunner()

    started_at = datetime(2024, 1, 15, 10, 30, 0, tzinfo=UTC)
    mock_vault = _make_mock_vault(
        ingest_status="in_progress",
        ingest_started_at=started_at,
        ingest_pid=1234,
        ingest_force=True,
    )
    mock_result = [(mock_vault, 42)]
    mock_session = _make_mock_session(mock_result)
    _patch_db_manager(mock_db_manager_class, mock_session)

    mock_settings = _make_mock_settings()

    with patch("obsidian_rag.cli.get_settings", return_value=mock_settings):
        result = runner.invoke(cli, ["vault", "list", "--format", "json"])

    assert result.exit_code == 0
    output_data = json.loads(result.output)
    assert len(output_data) == 1
    assert output_data[0]["ingest_status"] == "in_progress"
    assert output_data[0]["ingest_started_at"] == started_at.isoformat()
    assert output_data[0]["ingest_pid"] == 1234
    assert output_data[0]["ingest_force"] is True


@patch("obsidian_rag.cli_vault_commands.DatabaseManager")
def test_vault_get_shows_ingest_status(mock_db_manager_class) -> None:
    """vault get --name X output includes the ingest status and other lock fields."""
    runner = CliRunner()

    started_at = datetime(2024, 1, 15, 10, 30, 0, tzinfo=UTC)
    mock_vault = _make_mock_vault(
        ingest_status="in_progress",
        ingest_started_at=started_at,
        ingest_pid=1234,
        ingest_force=True,
    )

    mock_session = MagicMock()
    mock_session.query.return_value.filter.return_value.first.return_value = mock_vault
    mock_session.query.return_value.filter.return_value.scalar.return_value = 42

    _patch_db_manager(mock_db_manager_class, mock_session)

    mock_settings = _make_mock_settings()

    with patch("obsidian_rag.cli.get_settings", return_value=mock_settings):
        result = runner.invoke(cli, ["vault", "get", "--name", "Test Vault"])

    assert result.exit_code == 0
    assert "in_progress" in result.output
    assert "1234" in result.output
    assert "True" in result.output


@patch("obsidian_rag.cli_vault_commands.DatabaseManager")
def test_vault_list_ingest_in_progress_shown(mock_db_manager_class) -> None:
    """A vault with ingest_status='in_progress' and ingest_pid=1234 is shown in table."""
    runner = CliRunner()

    mock_vault = _make_mock_vault(
        ingest_status="in_progress",
        ingest_pid=1234,
    )
    mock_result = [(mock_vault, 5)]
    mock_session = _make_mock_session(mock_result)
    _patch_db_manager(mock_db_manager_class, mock_session)

    mock_settings = _make_mock_settings()

    with patch("obsidian_rag.cli.get_settings", return_value=mock_settings):
        result = runner.invoke(cli, ["vault", "list"])

    assert result.exit_code == 0
    assert "in_progress" in result.output
    assert "1234" in result.output


@patch("obsidian_rag.cli_vault_commands.DatabaseManager")
def test_vault_list_table_shows_ingest_force(mock_db_manager_class) -> None:
    """Table output shows ingest_force=True when set."""
    runner = CliRunner()

    mock_vault = _make_mock_vault(ingest_status="in_progress", ingest_force=True)
    mock_result = [(mock_vault, 7)]
    mock_session = _make_mock_session(mock_result)
    _patch_db_manager(mock_db_manager_class, mock_session)

    mock_settings = _make_mock_settings()

    with patch("obsidian_rag.cli.get_settings", return_value=mock_settings):
        result = runner.invoke(cli, ["vault", "list"])

    assert result.exit_code == 0
    assert "True" in result.output


@patch("obsidian_rag.cli_vault_commands.DatabaseManager")
def test_vault_get_idle_status(mock_db_manager_class) -> None:
    """vault get shows idle status with None pid and False force."""
    runner = CliRunner()

    mock_vault = _make_mock_vault(
        ingest_status="idle",
        ingest_started_at=None,
        ingest_pid=None,
        ingest_force=False,
    )

    mock_session = MagicMock()
    mock_session.query.return_value.filter.return_value.first.return_value = mock_vault
    mock_session.query.return_value.filter.return_value.scalar.return_value = 0

    _patch_db_manager(mock_db_manager_class, mock_session)

    mock_settings = _make_mock_settings()

    with patch("obsidian_rag.cli.get_settings", return_value=mock_settings):
        result = runner.invoke(cli, ["vault", "get", "--name", "Test Vault"])

    assert result.exit_code == 0
    assert "idle" in result.output


@patch("obsidian_rag.cli_vault_commands.DatabaseManager")
def test_vault_list_json_idle_defaults(mock_db_manager_class) -> None:
    """JSON output for idle vault shows default ingest field values."""
    runner = CliRunner()

    mock_vault = _make_mock_vault(ingest_status="idle")
    mock_result = [(mock_vault, 3)]
    mock_session = _make_mock_session(mock_result)
    _patch_db_manager(mock_db_manager_class, mock_session)

    mock_settings = _make_mock_settings()

    with patch("obsidian_rag.cli.get_settings", return_value=mock_settings):
        result = runner.invoke(cli, ["vault", "list", "--format", "json"])

    assert result.exit_code == 0
    output_data = json.loads(result.output)
    assert output_data[0]["ingest_status"] == "idle"
    assert output_data[0]["ingest_started_at"] is None
    assert output_data[0]["ingest_pid"] is None
    assert output_data[0]["ingest_force"] is False


@patch("obsidian_rag.cli_vault_commands.DatabaseManager")
def test_vault_get_failed_status(mock_db_manager_class) -> None:
    """vault get shows failed status correctly."""
    runner = CliRunner()

    mock_vault = _make_mock_vault(ingest_status="failed")

    mock_session = MagicMock()
    mock_session.query.return_value.filter.return_value.first.return_value = mock_vault
    mock_session.query.return_value.filter.return_value.scalar.return_value = 0

    _patch_db_manager(mock_db_manager_class, mock_session)

    mock_settings = _make_mock_settings()

    with patch("obsidian_rag.cli.get_settings", return_value=mock_settings):
        result = runner.invoke(cli, ["vault", "get", "--name", "Test Vault"])

    assert result.exit_code == 0
    assert "failed" in result.output
