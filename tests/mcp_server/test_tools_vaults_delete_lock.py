"""Unit tests for vault delete/update ingest-in-progress blocking."""

import logging
import uuid
from datetime import UTC, datetime
from unittest.mock import MagicMock

from obsidian_rag.database.models import Vault
from obsidian_rag.mcp_server.tools.vaults import (
    _is_ingest_in_progress,
    delete_vault,
    update_vault,
)
from obsidian_rag.mcp_server.tools.vaults_params import VaultUpdateParams


def _make_mock_vault(
    ingest_status: str = "idle",
    ingest_started_at: datetime | None = None,
    ingest_pid: int | None = None,
) -> MagicMock:
    """Create a MagicMock vault with ingest lock attributes."""
    vault = MagicMock(spec=Vault)
    vault.id = uuid.uuid4()
    vault.name = "TestVault"
    vault.container_path = "/data/test"
    vault.host_path = "/home/test"
    vault.description = "Test vault"
    vault.created_at = datetime.now(UTC)
    vault.ingest_status = ingest_status
    vault.ingest_started_at = ingest_started_at
    vault.ingest_pid = ingest_pid
    return vault


def _setup_delete_query_chain(mock_session: MagicMock, vault: MagicMock) -> None:
    """Set up mock_session.query side-effect for delete_vault path."""
    mock_vault_query = MagicMock()
    mock_vault_query.filter.return_value = mock_vault_query
    mock_vault_query.first.return_value = vault

    mock_doc_count_query = MagicMock()
    mock_doc_count_query.filter.return_value = mock_doc_count_query
    mock_doc_count_query.scalar.return_value = 1

    mock_task_count_query = MagicMock()
    mock_task_count_query.join.return_value = mock_task_count_query
    mock_task_count_query.filter.return_value = mock_task_count_query
    mock_task_count_query.scalar.return_value = 2

    mock_chunk_count_query = MagicMock()
    mock_chunk_count_query.join.return_value = mock_chunk_count_query
    mock_chunk_count_query.filter.return_value = mock_chunk_count_query
    mock_chunk_count_query.scalar.return_value = 3

    query_returns = [
        mock_vault_query,
        mock_doc_count_query,
        mock_task_count_query,
        mock_chunk_count_query,
    ]
    query_index = 0

    def query_side_effect(model):
        nonlocal query_index
        if query_index < len(query_returns):
            result = query_returns[query_index]
            query_index += 1
            return result
        return MagicMock()

    mock_session.query.side_effect = query_side_effect


def test_delete_vault_blocked_when_in_progress(caplog):
    """Delete blocked when ingest_status is in_progress; returns error dict."""
    caplog.set_level(logging.WARNING)

    started_at = datetime(2026, 7, 9, 10, 0, 0, tzinfo=UTC)
    vault = _make_mock_vault(
        ingest_status="in_progress",
        ingest_started_at=started_at,
        ingest_pid=12345,
    )

    mock_session = MagicMock()
    mock_session.bind.dialect.name = "postgresql"

    _setup_delete_query_chain(mock_session, vault)

    result = delete_vault(mock_session, vault_name="TestVault", confirm=True)

    assert isinstance(result, dict)
    assert result["success"] is False
    assert "in progress" in result["error"].lower()
    assert "TestVault" in result["error"]
    assert "2026-07-09" in result["error"]
    assert "12345" in result["error"]

    # Verify warning log
    warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert any("in progress" in r.message for r in warning_records)

    # Verify cascade counting and deletion were NOT reached
    mock_session.delete.assert_not_called()


def test_delete_vault_proceeds_when_idle():
    """Delete proceeds normally when ingest_status is idle."""
    vault = _make_mock_vault(ingest_status="idle")

    mock_session = MagicMock()
    mock_session.bind.dialect.name = "postgresql"

    _setup_delete_query_chain(mock_session, vault)

    result = delete_vault(mock_session, vault_name="TestVault", confirm=True)

    assert result["success"] is True
    assert result["name"] == "TestVault"
    mock_session.delete.assert_called_once_with(vault)
    mock_session.flush.assert_called_once()


def test_delete_vault_proceeds_when_failed():
    """Delete proceeds when ingest_status is failed (auto-reclaimable)."""
    vault = _make_mock_vault(ingest_status="failed")

    mock_session = MagicMock()
    mock_session.bind.dialect.name = "postgresql"

    _setup_delete_query_chain(mock_session, vault)

    result = delete_vault(mock_session, vault_name="TestVault", confirm=True)

    assert result["success"] is True
    assert result["name"] == "TestVault"
    mock_session.delete.assert_called_once_with(vault)


def test_delete_vault_confirm_false_still_blocks_first():
    """confirm=False returns confirm error before ingest check."""
    mock_session = MagicMock()

    result = delete_vault(mock_session, vault_name="TestVault", confirm=False)

    assert result["success"] is False
    assert "confirm=True" in result["error"]
    # No query should have been issued because confirm check is first
    mock_session.query.assert_not_called()


def test_update_vault_container_path_blocked_when_in_progress(caplog):
    """container_path change blocked when ingest is in_progress."""
    caplog.set_level(logging.WARNING)

    started_at = datetime(2026, 7, 9, 10, 0, 0, tzinfo=UTC)
    vault = _make_mock_vault(
        ingest_status="in_progress",
        ingest_started_at=started_at,
        ingest_pid=12345,
    )

    mock_session = MagicMock()
    mock_session.bind.dialect.name = "postgresql"

    mock_vault_query = MagicMock()
    mock_vault_query.filter.return_value = mock_vault_query
    mock_vault_query.first.return_value = vault

    mock_session.query.return_value = mock_vault_query

    params = VaultUpdateParams(
        vault_name="TestVault",
        container_path="/new/path",
        force=True,
    )

    result = update_vault(mock_session, params)

    assert isinstance(result, dict)
    assert result["success"] is False
    assert "in progress" in result["error"].lower()
    assert "container_path" in result["error"].lower()
    assert "TestVault" in result["error"]
    assert "2026-07-09" in result["error"]
    assert "12345" in result["error"]

    # Verify no DB write occurred
    mock_session.flush.assert_not_called()

    # Verify warning log
    warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert any("in progress" in r.message for r in warning_records)


def test_update_vault_container_path_proceeds_when_idle_with_force():
    """container_path change proceeds when idle and force=True."""
    vault = _make_mock_vault(ingest_status="idle")

    mock_session = MagicMock()
    mock_session.bind.dialect.name = "postgresql"

    mock_vault_query = MagicMock()
    mock_vault_query.filter.return_value = mock_vault_query
    mock_vault_query.first.return_value = vault

    mock_doc_delete_query = MagicMock()
    mock_doc_delete_query.filter.return_value = mock_doc_delete_query
    mock_doc_delete_query.delete.return_value = 3

    mock_doc_count_query = MagicMock()
    mock_doc_count_query.filter.return_value = mock_doc_count_query
    mock_doc_count_query.scalar.return_value = 0

    call_count = 0

    def query_side_effect(model):
        nonlocal call_count
        call_count += 1
        if model is Vault:
            return mock_vault_query
        if call_count == 2:
            return mock_doc_delete_query
        return mock_doc_count_query

    mock_session.query.side_effect = query_side_effect

    params = VaultUpdateParams(
        vault_name="TestVault",
        container_path="/new/path",
        force=True,
    )

    result = update_vault(mock_session, params)

    # Should be VaultResponse, not error dict
    assert hasattr(result, "container_path")
    assert result.container_path == "/new/path"
    mock_doc_delete_query.delete.assert_called_once_with(synchronize_session=False)
    mock_session.flush.assert_called_once()


def test_update_vault_non_container_path_change_not_blocked():
    """Updating description while in_progress is allowed."""
    started_at = datetime(2026, 7, 9, 10, 0, 0, tzinfo=UTC)
    vault = _make_mock_vault(
        ingest_status="in_progress",
        ingest_started_at=started_at,
        ingest_pid=12345,
    )

    mock_session = MagicMock()
    mock_session.bind.dialect.name = "postgresql"

    mock_vault_query = MagicMock()
    mock_vault_query.filter.return_value = mock_vault_query
    mock_vault_query.first.return_value = vault

    mock_doc_query = MagicMock()
    mock_doc_query.filter.return_value = mock_doc_query
    mock_doc_query.scalar.return_value = 5

    def query_side_effect(model):
        if model is Vault:
            return mock_vault_query
        return mock_doc_query

    mock_session.query.side_effect = query_side_effect

    params = VaultUpdateParams(
        vault_name="TestVault",
        description="Updated description",
    )

    result = update_vault(mock_session, params)

    # Should proceed because container_path is not changing
    assert hasattr(result, "description")
    assert result.description == "Updated description"
    mock_session.flush.assert_called_once()


def test_is_ingest_in_progress_helper():
    """Test _is_ingest_in_progress for all status values."""
    in_progress_vault = _make_mock_vault(ingest_status="in_progress")
    idle_vault = _make_mock_vault(ingest_status="idle")
    failed_vault = _make_mock_vault(ingest_status="failed")
    other_vault = _make_mock_vault(ingest_status="some_other_value")

    assert _is_ingest_in_progress(in_progress_vault) is True
    assert _is_ingest_in_progress(idle_vault) is False
    assert _is_ingest_in_progress(failed_vault) is False
    assert _is_ingest_in_progress(other_vault) is False
