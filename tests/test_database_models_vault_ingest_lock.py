"""Tests for Vault ingest lock columns.

Tests the IngestStatus enum and the four new Vault columns
(ingest_status, ingest_started_at, ingest_pid, ingest_force).
Uses SQLAlchemy introspection to verify defaults without a DB connection.
"""

from obsidian_rag.database.models import IngestStatus, Vault


def test_vault_ingest_status_default_idle():
    """Vault.ingest_status column has default 'idle' configured."""
    col = Vault.__table__.c.ingest_status
    assert col.default is not None
    assert col.default.arg == "idle"
    assert col.server_default is not None
    assert col.server_default.arg == "idle"


def test_vault_ingest_force_default_false():
    """Vault.ingest_force column has default False configured."""
    col = Vault.__table__.c.ingest_force
    assert col.default is not None
    assert col.default.arg is False
    assert col.server_default is not None
    assert col.server_default.arg == "false"


def test_vault_ingest_started_at_default_none():
    """Vault.ingest_started_at column is nullable (default None)."""
    col = Vault.__table__.c.ingest_started_at
    assert col.nullable is True
    assert col.default is None


def test_vault_ingest_pid_default_none():
    """Vault.ingest_pid column is nullable (default None)."""
    col = Vault.__table__.c.ingest_pid
    assert col.nullable is True
    assert col.default is None


def test_vault_ingest_status_assignable():
    """ingest_status can be set to IngestStatus.IN_PROGRESS.value."""
    vault = Vault(
        name="TestVault",
        container_path="/data/test",
        host_path="/home/user/test",
    )
    vault.ingest_status = IngestStatus.IN_PROGRESS.value
    assert vault.ingest_status == "in_progress"


def test_ingest_status_enum_values():
    """IngestStatus enum values match expected strings."""
    assert IngestStatus.IDLE.value == "idle"
    assert IngestStatus.IN_PROGRESS.value == "in_progress"
    assert IngestStatus.FAILED.value == "failed"


def test_vault_repr_still_works():
    """repr(Vault) does not raise with the new columns present."""
    vault = Vault(
        name="TestVault",
        container_path="/data/test",
        host_path="/home/user/test",
    )
    result = repr(vault)
    assert "Vault" in result
    assert "TestVault" in result
