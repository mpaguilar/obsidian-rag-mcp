"""Tests for obsidian_rag.services.ingestion_lock."""

import uuid
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from obsidian_rag.services.ingestion_lock import (
    IngestLockAcquisition,
    IngestLockError,
    acquire_ingest_lock,
    apply_lock_policy,
    heartbeat_ingest_lock,
    reclaim_stale_lock,
    release_ingest_lock,
    try_acquire_ingest_lock,
)


# ─── apply_lock_policy ───


def test_apply_lock_policy_matrix() -> None:
    """Cover all four cells of the policy matrix."""
    assert apply_lock_policy(running_force=True, new_force=True) == "fail_fast"
    assert apply_lock_policy(running_force=True, new_force=False) == "no_op_skip"
    assert apply_lock_policy(running_force=False, new_force=True) == "fail_fast"
    assert apply_lock_policy(running_force=False, new_force=False) == "fail_fast"


# ─── acquire_ingest_lock happy / edge paths ───


def _make_db_manager(rowcount: int, vault=None) -> MagicMock:
    """Build a mock DatabaseManager whose session yields the given rowcount and vault."""
    mock_result = MagicMock()
    mock_result.rowcount = rowcount

    mock_query = MagicMock()
    mock_query.filter_by.return_value.first.return_value = vault

    mock_session = MagicMock()
    mock_session.execute.return_value = mock_result
    mock_session.query.return_value = mock_query

    mock_manager = MagicMock()
    mock_manager.get_session.return_value.__enter__ = MagicMock(
        return_value=mock_session
    )
    mock_manager.get_session.return_value.__exit__ = MagicMock(return_value=False)
    return mock_manager


def test_acquire_lock_happy_path_rowcount_one() -> None:
    """execute().rowcount=1 returns IngestLockAcquisition with pid=os.getpid()."""
    vault_id = uuid.uuid4()
    db_manager = _make_db_manager(rowcount=1)

    with patch("obsidian_rag.services.ingestion_lock.os.getpid", return_value=12345):
        result = acquire_ingest_lock(db_manager, vault_id, force=False, ttl_seconds=300)

    assert result is not None
    assert isinstance(result, IngestLockAcquisition)
    assert result.vault_id == vault_id
    assert result.pid == 12345
    assert result.force is False
    assert result.acquired_at.tzinfo is UTC


def test_acquire_lock_vault_not_found() -> None:
    """rowcount=0 AND query().first()=None raises IngestLockError."""
    vault_id = uuid.uuid4()
    db_manager = _make_db_manager(rowcount=0, vault=None)

    with patch("obsidian_rag.services.ingestion_lock.os.getpid", return_value=12345):
        with pytest.raises(IngestLockError, match="not found"):
            acquire_ingest_lock(db_manager, vault_id, force=False, ttl_seconds=300)


def test_acquire_lock_in_progress_not_stale_fail_fast() -> None:
    """rowcount=0, vault.ingest_status='in_progress', started_at recent → raises."""
    vault_id = uuid.uuid4()
    mock_vault = MagicMock()
    mock_vault.ingest_status = "in_progress"
    mock_vault.ingest_started_at = datetime.now(UTC) - timedelta(seconds=10)
    mock_vault.ingest_pid = 9999
    mock_vault.ingest_force = False
    mock_vault.name = "test-vault"

    db_manager = _make_db_manager(rowcount=0, vault=mock_vault)

    with patch("obsidian_rag.services.ingestion_lock.os.getpid", return_value=12345):
        with pytest.raises(
            IngestLockError, match="Ingest already in progress"
        ) as exc_info:
            acquire_ingest_lock(db_manager, vault_id, force=False, ttl_seconds=300)

    msg = str(exc_info.value)
    assert "test-vault" in msg
    assert "PID 9999" in msg
    assert "force=False" in msg


def test_acquire_lock_in_progress_not_stale_running_force_new_regular_no_op_skip() -> (
    None
):
    """running force=True, new force=False → returns None (sentinel)."""
    vault_id = uuid.uuid4()
    mock_vault = MagicMock()
    mock_vault.ingest_status = "in_progress"
    mock_vault.ingest_started_at = datetime.now(UTC) - timedelta(seconds=10)
    mock_vault.ingest_pid = 9999
    mock_vault.ingest_force = True
    mock_vault.name = "test-vault"

    db_manager = _make_db_manager(rowcount=0, vault=mock_vault)

    with patch("obsidian_rag.services.ingestion_lock.os.getpid", return_value=12345):
        result = acquire_ingest_lock(db_manager, vault_id, force=False, ttl_seconds=300)

    assert result is None


def test_acquire_lock_in_progress_not_stale_running_force_new_force_fail_fast() -> None:
    """both force=True → raises."""
    vault_id = uuid.uuid4()
    mock_vault = MagicMock()
    mock_vault.ingest_status = "in_progress"
    mock_vault.ingest_started_at = datetime.now(UTC) - timedelta(seconds=10)
    mock_vault.ingest_pid = 9999
    mock_vault.ingest_force = True
    mock_vault.name = "test-vault"

    db_manager = _make_db_manager(rowcount=0, vault=mock_vault)

    with patch("obsidian_rag.services.ingestion_lock.os.getpid", return_value=12345):
        with pytest.raises(IngestLockError, match="Ingest already in progress"):
            acquire_ingest_lock(db_manager, vault_id, force=True, ttl_seconds=300)


def test_acquire_lock_in_progress_not_stale_running_regular_new_force_fail_fast() -> (
    None
):
    """running force=False, new force=True → raises."""
    vault_id = uuid.uuid4()
    mock_vault = MagicMock()
    mock_vault.ingest_status = "in_progress"
    mock_vault.ingest_started_at = datetime.now(UTC) - timedelta(seconds=10)
    mock_vault.ingest_pid = 9999
    mock_vault.ingest_force = False
    mock_vault.name = "test-vault"

    db_manager = _make_db_manager(rowcount=0, vault=mock_vault)

    with patch("obsidian_rag.services.ingestion_lock.os.getpid", return_value=12345):
        with pytest.raises(IngestLockError, match="Ingest already in progress"):
            acquire_ingest_lock(db_manager, vault_id, force=True, ttl_seconds=300)


def test_acquire_lock_in_progress_stale_reclaim_wins() -> None:
    """started_at old, reclaim_stale_lock patched to True → returns acquisition."""
    vault_id = uuid.uuid4()
    mock_vault = MagicMock()
    mock_vault.ingest_status = "in_progress"
    mock_vault.ingest_started_at = datetime.now(UTC) - timedelta(seconds=400)
    mock_vault.ingest_pid = 9999
    mock_vault.ingest_force = False
    mock_vault.name = "test-vault"

    db_manager = _make_db_manager(rowcount=0, vault=mock_vault)

    with patch(
        "obsidian_rag.services.ingestion_lock.reclaim_stale_lock",
        return_value=True,
    ) as mock_reclaim:
        with patch(
            "obsidian_rag.services.ingestion_lock.os.getpid", return_value=12345
        ):
            result = acquire_ingest_lock(
                db_manager, vault_id, force=False, ttl_seconds=300
            )

    assert result is not None
    assert isinstance(result, IngestLockAcquisition)
    assert result.vault_id == vault_id
    assert result.pid == 12345
    mock_reclaim.assert_called_once()


def test_acquire_lock_in_progress_stale_reclaim_loses_race() -> None:
    """started_at old, reclaim returns False → raises IngestLockError."""
    vault_id = uuid.uuid4()
    mock_vault = MagicMock()
    mock_vault.ingest_status = "in_progress"
    mock_vault.ingest_started_at = datetime.now(UTC) - timedelta(seconds=400)
    mock_vault.ingest_pid = 9999
    mock_vault.ingest_force = False
    mock_vault.name = "test-vault"

    db_manager = _make_db_manager(rowcount=0, vault=mock_vault)

    with patch(
        "obsidian_rag.services.ingestion_lock.reclaim_stale_lock",
        return_value=False,
    ) as mock_reclaim:
        with patch(
            "obsidian_rag.services.ingestion_lock.os.getpid", return_value=12345
        ):
            with pytest.raises(IngestLockError, match="stale reclaim lost") as exc_info:
                acquire_ingest_lock(db_manager, vault_id, force=False, ttl_seconds=300)

    msg = str(exc_info.value)
    assert "stale reclaim lost" in msg
    mock_reclaim.assert_called_once()


def test_acquire_lock_failed_status_treated_as_idle() -> None:
    """vault.ingest_status='failed' → initial UPDATE includes 'failed', rowcount=1."""
    vault_id = uuid.uuid4()
    db_manager = _make_db_manager(rowcount=1)

    with patch("obsidian_rag.services.ingestion_lock.os.getpid", return_value=12345):
        result = acquire_ingest_lock(db_manager, vault_id, force=False, ttl_seconds=300)

    assert result is not None
    assert isinstance(result, IngestLockAcquisition)
    assert result.vault_id == vault_id


def test_acquire_lock_unexpected_status_defensive_fail_fast() -> None:
    """rowcount=0, vault found with unexpected status → raises IngestLockError."""
    vault_id = uuid.uuid4()
    mock_vault = MagicMock()
    mock_vault.ingest_status = "unknown_status"
    mock_vault.name = "test-vault"

    db_manager = _make_db_manager(rowcount=0, vault=mock_vault)

    with patch("obsidian_rag.services.ingestion_lock.os.getpid", return_value=12345):
        with pytest.raises(
            IngestLockError, match="Unexpected ingest_status"
        ) as exc_info:
            acquire_ingest_lock(db_manager, vault_id, force=False, ttl_seconds=300)

    msg = str(exc_info.value)
    assert "unknown_status" in msg
    assert "test-vault" in msg


# ─── heartbeat ───


def test_heartbeat_rowcount_one_returns_true() -> None:
    """execute().rowcount=1 → True."""
    vault_id = uuid.uuid4()
    db_manager = _make_db_manager(rowcount=1)

    result = heartbeat_ingest_lock(db_manager, vault_id)
    assert result is True


def test_heartbeat_rowcount_zero_returns_false() -> None:
    """execute().rowcount=0 → False and logs WARNING."""
    vault_id = uuid.uuid4()
    db_manager = _make_db_manager(rowcount=0)

    result = heartbeat_ingest_lock(db_manager, vault_id)
    assert result is False


# ─── release ───


def test_release_failed_false_sets_idle() -> None:
    """failed=False sets ingest_status='idle' and nulls timestamps/pid/force."""
    vault_id = uuid.uuid4()
    db_manager = _make_db_manager(rowcount=1)

    release_ingest_lock(db_manager, vault_id, failed=False)

    mock_session = db_manager.get_session.return_value.__enter__.return_value
    call_args = mock_session.execute.call_args[0][0]
    compiled = str(call_args.compile(compile_kwargs={"literal_binds": True}))
    assert "idle" in compiled


def test_release_failed_true_sets_failed() -> None:
    """failed=True sets ingest_status='failed' and nulls timestamps/pid/force."""
    vault_id = uuid.uuid4()
    db_manager = _make_db_manager(rowcount=1)

    release_ingest_lock(db_manager, vault_id, failed=True)

    mock_session = db_manager.get_session.return_value.__enter__.return_value
    call_args = mock_session.execute.call_args[0][0]
    compiled = str(call_args.compile(compile_kwargs={"literal_binds": True}))
    assert "failed" in compiled


def test_release_idempotent_rowcount_zero_no_error() -> None:
    """execute().rowcount=0, function returns None without raising."""
    vault_id = uuid.uuid4()
    db_manager = _make_db_manager(rowcount=0)

    result = release_ingest_lock(db_manager, vault_id, failed=False)
    assert result is None


# ─── reclaim_stale_lock ───


def test_reclaim_stale_rowcount_one_returns_true() -> None:
    """execute().rowcount=1 → True and logs INFO with 'reclaimed stale ingest lock'."""
    vault_id = uuid.uuid4()
    db_manager = _make_db_manager(rowcount=1)

    with patch("obsidian_rag.services.ingestion_lock.os.getpid", return_value=12345):
        result = reclaim_stale_lock(db_manager, vault_id, force=False, ttl_seconds=300)

    assert result is True


def test_reclaim_stale_rowcount_zero_returns_false() -> None:
    """execute().rowcount=0 → False."""
    vault_id = uuid.uuid4()
    db_manager = _make_db_manager(rowcount=0)

    with patch("obsidian_rag.services.ingestion_lock.os.getpid", return_value=12345):
        result = reclaim_stale_lock(db_manager, vault_id, force=False, ttl_seconds=300)

    assert result is False


# ─── try_acquire_ingest_lock ───


def test_try_acquire_ingest_lock_dry_run_no_acquire() -> None:
    """dry_run=True returns (False, None) without calling acquire_ingest_lock."""
    vault_id = uuid.uuid4()
    db_manager = MagicMock()
    options = MagicMock()
    options.dry_run = True
    settings = MagicMock()

    with patch(
        "obsidian_rag.services.ingestion_lock.acquire_ingest_lock"
    ) as mock_acquire:
        result = try_acquire_ingest_lock(db_manager, vault_id, options, settings, 0.0)

    assert result == (False, None)
    mock_acquire.assert_not_called()


def test_try_acquire_ingest_lock_vault_id_none_no_acquire() -> None:
    """vault_id=None returns (False, None) without calling acquire_ingest_lock."""
    db_manager = MagicMock()
    options = MagicMock()
    options.dry_run = False
    settings = MagicMock()

    with patch(
        "obsidian_rag.services.ingestion_lock.acquire_ingest_lock"
    ) as mock_acquire:
        result = try_acquire_ingest_lock(db_manager, None, options, settings, 0.0)

    assert result == (False, None)
    mock_acquire.assert_not_called()


def test_try_acquire_ingest_lock_acquired_returns_true() -> None:
    """Successful acquire returns (True, None)."""
    vault_id = uuid.uuid4()
    db_manager = MagicMock()
    options = MagicMock()
    options.dry_run = False
    options.force = False
    settings = MagicMock()
    settings.ingestion.ingest_lock_ttl_seconds = 300
    mock_acquisition = IngestLockAcquisition(
        vault_id=vault_id, pid=123, force=False, acquired_at=datetime.now(UTC)
    )

    with patch(
        "obsidian_rag.services.ingestion_lock.acquire_ingest_lock",
        return_value=mock_acquisition,
    ) as mock_acquire:
        result = try_acquire_ingest_lock(db_manager, vault_id, options, settings, 0.0)

    assert result == (True, None)
    mock_acquire.assert_called_once_with(
        db_manager, vault_id, force=False, ttl_seconds=300
    )


def test_try_acquire_ingest_lock_skip_with_str_vault_name() -> None:
    """No-op skip with str vault name uses the name in the skip message."""
    vault_id = uuid.uuid4()
    db_manager = MagicMock()
    options = MagicMock()
    options.dry_run = False
    options.force = False
    options.vault = "MyVault"
    settings = MagicMock()
    settings.ingestion.ingest_lock_ttl_seconds = 300

    with patch(
        "obsidian_rag.services.ingestion_lock.acquire_ingest_lock",
        return_value=None,
    ):
        lock_acquired, skip_result = try_acquire_ingest_lock(
            db_manager, vault_id, options, settings, 100.0
        )

    assert lock_acquired is False
    assert skip_result is not None
    assert skip_result.total == 0
    assert "MyVault" in skip_result.message


def test_try_acquire_ingest_lock_skip_with_vault_config() -> None:
    """No-op skip with VaultConfig uses container_path in the skip message."""
    vault_id = uuid.uuid4()
    db_manager = MagicMock()
    options = MagicMock()
    options.dry_run = False
    options.force = False
    options.vault = MagicMock()
    options.vault.container_path = "/data/myvault"
    settings = MagicMock()
    settings.ingestion.ingest_lock_ttl_seconds = 300

    with patch(
        "obsidian_rag.services.ingestion_lock.acquire_ingest_lock",
        return_value=None,
    ):
        lock_acquired, skip_result = try_acquire_ingest_lock(
            db_manager, vault_id, options, settings, 100.0
        )

    assert lock_acquired is False
    assert skip_result is not None
    assert "/data/myvault" in skip_result.message
