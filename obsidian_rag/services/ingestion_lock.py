"""Cross-process vault ingest lock primitives."""

import logging
import os
import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Literal, cast

from sqlalchemy import update

from obsidian_rag.database.models import IngestStatus, Vault

if TYPE_CHECKING:
    from sqlalchemy.engine import CursorResult

    from obsidian_rag.config import Settings
    from obsidian_rag.database.engine import DatabaseManager
    from obsidian_rag.services.ingestion_models import (
        IngestionResult,
        IngestVaultOptions,
    )

log = logging.getLogger(__name__)


class IngestLockError(RuntimeError):
    """Raised when a vault ingest lock cannot be acquired (fail-fast policy)."""


@dataclass
class IngestLockAcquisition:
    """Record of an acquired ingest lock.

    Attributes:
        vault_id: UUID of the locked vault.
        pid: OS PID holding the lock.
        force: Whether this ingest is a force re-ingest.
        acquired_at: When the lock was acquired.
    """

    vault_id: uuid.UUID
    pid: int
    force: bool
    acquired_at: datetime


def apply_lock_policy(
    *,
    running_force: bool,
    new_force: bool,
) -> Literal["fail_fast", "no_op_skip"]:
    """Decide fail-fast vs no-op-skip per the policy matrix.

    Matrix: (running=True,new=True)->fail_fast; (running=True,new=False)->no_op_skip;
            (running=False,new=True)->fail_fast; (running=False,new=False)->fail_fast.

    Args:
        running_force: Whether the currently running ingest is a force re-ingest.
        new_force: Whether the new caller requested a force re-ingest.

    Returns:
        "fail_fast" if the new caller should raise IngestLockError;
        "no_op_skip" if the new caller should return a synthetic skip result.

    """
    if running_force and not new_force:
        return "no_op_skip"
    return "fail_fast"


def _resolve_in_progress_lock(
    db_manager: "DatabaseManager",
    vault: Vault,
    vault_id: uuid.UUID,
    pid: int,
    *,
    force: bool,
    ttl_seconds: int,
) -> IngestLockAcquisition | None:
    """Handle acquire when vault is already in_progress. Check stale, reclaim, or policy.

    Args:
        db_manager: Database manager for session access.
        vault: Currently locked vault row read from the database.
        vault_id: UUID of the vault to lock.
        pid: OS PID of the calling process.
        force: Whether this is a force re-ingest.
        ttl_seconds: Seconds before a lock is considered stale.

    Returns:
        IngestLockAcquisition if reclaimed, or None on no-op-skip policy.

    Raises:
        IngestLockError: On fail-fast policy or when stale reclaim loses the race.

    """
    now = datetime.now(UTC)
    if (
        vault.ingest_started_at is not None
        and vault.ingest_started_at < now - timedelta(seconds=ttl_seconds)
    ):
        if reclaim_stale_lock(
            db_manager, vault_id, force=force, ttl_seconds=ttl_seconds
        ):
            return IngestLockAcquisition(
                vault_id=vault_id, pid=pid, force=force, acquired_at=datetime.now(UTC)
            )
        _msg = f"Ingest already in progress for vault '{vault.name}' (stale reclaim lost). Wait for completion."
        log.error(_msg)
        raise IngestLockError(_msg)
    policy = apply_lock_policy(running_force=vault.ingest_force, new_force=force)
    if policy == "no_op_skip":
        _msg = f"Skipped: force re-ingest already in progress for vault '{vault.name}'"
        log.info(_msg)
        return None
    _msg = (
        f"Ingest already in progress for vault '{vault.name}' "
        f"(started at {vault.ingest_started_at}, PID {vault.ingest_pid}, "
        f"force={vault.ingest_force}). Wait for completion or reclaim the stale lock."
    )
    log.error(_msg)
    raise IngestLockError(_msg)


def acquire_ingest_lock(
    db_manager: "DatabaseManager",
    vault_id: uuid.UUID,
    *,
    force: bool,
    ttl_seconds: int,
) -> IngestLockAcquisition | None:
    """Atomically acquire the ingest lock for a vault.

    Args:
        db_manager: Database manager for session access.
        vault_id: UUID of the vault to lock.
        force: Whether this is a force re-ingest.
        ttl_seconds: Seconds before a lock is considered stale and reclaimable.

    Returns:
        IngestLockAcquisition if the lock was acquired, or None on no-op-skip
        policy (a force re-ingest is already running).

    Raises:
        IngestLockError: On fail-fast policy, when the vault is not found,
            or when an unexpected ingest_status is encountered.

    Notes:
        Performs an atomic UPDATE on the vaults row as the entry gate. Each
        acquire runs in its own short database session.

    """
    _msg = f"acquire_ingest_lock starting for vault {vault_id}, force={force}"
    log.debug(_msg)
    pid = os.getpid()
    now = datetime.now(UTC)
    with db_manager.get_session() as session:
        result = cast(
            "CursorResult",
            session.execute(
                update(Vault)
                .where(Vault.id == vault_id)
                .where(
                    Vault.ingest_status.in_(
                        [IngestStatus.IDLE.value, IngestStatus.FAILED.value]
                    )
                )
                .values(
                    ingest_status=IngestStatus.IN_PROGRESS.value,
                    ingest_started_at=now,
                    ingest_pid=pid,
                    ingest_force=force,
                )
            ),
        )
        if result.rowcount == 1:
            _msg = (
                f"acquired ingest lock for vault {vault_id} (pid={pid}, force={force})"
            )
            log.info(_msg)
            return IngestLockAcquisition(
                vault_id=vault_id, pid=pid, force=force, acquired_at=now
            )
        vault = session.query(Vault).filter_by(id=vault_id).first()
        if vault is None:
            _msg = f"vault {vault_id} not found during lock acquire"
            log.error(_msg)
            raise IngestLockError(_msg)
        current_status = vault.ingest_status
        if current_status == IngestStatus.IN_PROGRESS.value:
            return _resolve_in_progress_lock(
                db_manager, vault, vault_id, pid, force=force, ttl_seconds=ttl_seconds
            )
        _msg = f"Unexpected ingest_status '{current_status}' for vault '{vault.name}'"
        log.error(_msg)
        raise IngestLockError(_msg)


def heartbeat_ingest_lock(db_manager: "DatabaseManager", vault_id: uuid.UUID) -> bool:
    """Refresh ingest_started_at. Return True if lock still held, False if lost.

    Args:
        db_manager: Database manager for session access.
        vault_id: UUID of the vault whose lock heartbeat is being refreshed.

    Returns:
        True if the lock is still held (rowcount==1); False if the lock was
        lost (rowcount==0 — e.g. another process reclaimed it).

    Notes:
        Performs an atomic UPDATE on the vaults row in its own short session.

    """
    _msg = f"heartbeat_ingest_lock starting for vault {vault_id}"
    log.debug(_msg)
    with db_manager.get_session() as session:
        result = cast(
            "CursorResult",
            session.execute(
                update(Vault)
                .where(Vault.id == vault_id)
                .where(Vault.ingest_status == IngestStatus.IN_PROGRESS.value)
                .values(ingest_started_at=datetime.now(UTC))
            ),
        )
    if result.rowcount == 1:
        _msg = f"heartbeat refreshed for vault {vault_id}"
        log.debug(_msg)
        return True
    _msg = f"heartbeat lost for vault {vault_id} (lock no longer held)"
    log.warning(_msg)
    return False


def release_ingest_lock(
    db_manager: "DatabaseManager",
    vault_id: uuid.UUID,
    *,
    failed: bool = False,
) -> None:
    """Release the lock: idle on success, failed on error. Idempotent.

    Args:
        db_manager: Database manager for session access.
        vault_id: UUID of the vault whose lock is being released.
        failed: If True, set ingest_status to 'failed' (visible last-error
            marker, auto-reclaimable by the next acquire); if False, set to
            'idle'.

    Notes:
        Idempotent — rowcount==0 (lock was reclaimed by another process) is
        silently accepted without raising. Runs in its own short session.

    """
    _msg = f"release_ingest_lock starting for vault {vault_id}, failed={failed}"
    log.debug(_msg)
    status = IngestStatus.FAILED.value if failed else IngestStatus.IDLE.value
    with db_manager.get_session() as session:
        session.execute(
            update(Vault)
            .where(Vault.id == vault_id)
            .values(
                ingest_status=status,
                ingest_started_at=None,
                ingest_pid=None,
                ingest_force=False,
            )
        )
    _msg = f"released ingest lock for vault {vault_id} (status={status})"
    log.info(_msg)
    # Idempotent: rowcount==0 (lock was reclaimed) is silently fine — no error.


def reclaim_stale_lock(
    db_manager: "DatabaseManager",
    vault_id: uuid.UUID,
    *,
    force: bool,
    ttl_seconds: int,
) -> bool:
    """Optimistically reclaim an in_progress lock older than TTL. Return True on win.

    Args:
        db_manager: Database manager for session access.
        vault_id: UUID of the vault whose stale lock is being reclaimed.
        force: Whether the new holder is a force re-ingest.
        ttl_seconds: Seconds before a lock is considered stale.

    Returns:
        True if this caller won the reclaim race (rowcount==1); False if
        another process reclaimed it concurrently (rowcount==0).

    Notes:
        Performs an optimistic atomic UPDATE in its own short session. Only
        one concurrent reclaimer wins; losers fall through to fail-fast.

    """
    _msg = f"reclaim_stale_lock starting for vault {vault_id}"
    log.debug(_msg)
    now = datetime.now(UTC)
    pid = os.getpid()
    with db_manager.get_session() as session:
        result = cast(
            "CursorResult",
            session.execute(
                update(Vault)
                .where(Vault.id == vault_id)
                .where(Vault.ingest_status == IngestStatus.IN_PROGRESS.value)
                .where(Vault.ingest_started_at < now - timedelta(seconds=ttl_seconds))
                .values(
                    ingest_status=IngestStatus.IN_PROGRESS.value,
                    ingest_started_at=now,
                    ingest_pid=pid,
                    ingest_force=force,
                )
            ),
        )
    if result.rowcount == 1:
        _msg = f"reclaimed stale ingest lock for vault {vault_id} (pid={pid})"
        log.info(_msg)
        return True
    _msg = f"stale reclaim lost for vault {vault_id}"
    log.debug(_msg)
    return False


def check_ingest_heartbeat(
    db_manager: "DatabaseManager",
    vault_id: uuid.UUID | None,
    idx: int,
    heartbeat_interval: int,
    lock_lost_flag: list[bool] | None,
) -> bool:
    """Send heartbeat and return True if lock is still held.

    Args:
        db_manager: Database manager for session access.
        vault_id: UUID of the vault, or None when no lock is held (no-op).
        idx: Current file index in the ingest loop (0-based).
        heartbeat_interval: Files processed between heartbeat UPDATEs; 0 disables.
        lock_lost_flag: Optional mutable single-element container used to flag
            lock loss back to the caller so it can skip releasing someone
            else's lock.

    Returns:
        True if the lock is still held (or no heartbeat was due); False if the
        heartbeat returned False (lock lost) and the loop should break.

    """
    if (
        vault_id is not None
        and idx > 0
        and heartbeat_interval > 0
        and idx % heartbeat_interval == 0
    ):
        if not heartbeat_ingest_lock(db_manager, vault_id):
            _msg = f"Lock lost during heartbeat at file idx {idx}; breaking ingest loop"
            log.warning(_msg)
            if lock_lost_flag is not None:
                lock_lost_flag[0] = True
            return False
    return True


def try_acquire_ingest_lock(
    db_manager: "DatabaseManager",
    vault_id: uuid.UUID | None,
    options: "IngestVaultOptions",
    settings: "Settings",
    start_time: float,
) -> tuple[bool, "IngestionResult | None"]:
    """Attempt to acquire the ingest lock.

    Args:
        db_manager: Database manager for session access.
        vault_id: UUID of the vault, or None for dry_run (no lock acquired).
        options: Ingest options (dry_run, force, vault name/config).
        settings: Application settings (used for TTL config).
        start_time: Monotonic start time of the ingest, used to compute the
            synthetic skip result's processing_time_seconds.

    Returns:
        Tuple of (lock_acquired, skip_result). lock_acquired is True when the
        lock was acquired (caller must release it in finally). skip_result is
        set when another force ingest already holds the lock (caller returns
        it directly without releasing). Both are False/None for dry_run.

    Raises:
        IngestLockError: Propagated from acquire_ingest_lock on fail-fast policy.

    Notes:
        Performs a database UPDATE via acquire_ingest_lock. No lock is
        acquired for dry_run or when vault_id is None.

    """
    if options.dry_run or vault_id is None:
        return (False, None)
    acquisition = acquire_ingest_lock(
        db_manager,
        vault_id,
        force=options.force,
        ttl_seconds=settings.ingestion.ingest_lock_ttl_seconds,
    )
    if acquisition is None:
        _vault_label = (
            options.vault
            if isinstance(options.vault, str)
            else options.vault.container_path
        )
        _msg = (
            f"Skipped: force re-ingest already in progress for vault '{_vault_label}'"
        )
        log.info(_msg)
        from obsidian_rag.services.ingestion_models import IngestionResult

        return (
            False,
            IngestionResult(
                total=0,
                new=0,
                updated=0,
                unchanged=0,
                errors=0,
                deleted=0,
                chunks_created=0,
                empty_documents=0,
                processing_time_seconds=time.time() - start_time,
                message=_msg,
                skipped=True,  # NEW — mark no-op-skip (REQ-002)
            ),
        )
    return (True, None)
