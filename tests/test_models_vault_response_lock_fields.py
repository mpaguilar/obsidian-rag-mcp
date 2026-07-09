"""Tests for VaultResponse lock fields (ingest status tracking).

Tests that VaultResponse accepts and propagates the four new ingest-lock
fields: ingest_status, ingest_started_at, ingest_pid, and ingest_force.
"""

import uuid
from datetime import datetime
from unittest.mock import MagicMock

from obsidian_rag.mcp_server.models import VaultResponse, create_vault_response


def test_vault_response_accepts_lock_fields():
    """Construct VaultResponse with all required fields including the 4 new ones."""
    vault_id = uuid.uuid4()
    now = datetime.now()
    response = VaultResponse(
        id=vault_id,
        name="Test Vault",
        description="A test vault",
        container_path="/data/test",
        host_path="/host/test",
        document_count=3,
        created_at=now,
        ingest_status="in_progress",
        ingest_started_at=now,
        ingest_pid=12345,
        ingest_force=True,
    )
    assert response.id == vault_id
    assert response.name == "Test Vault"
    assert response.ingest_status == "in_progress"
    assert response.ingest_started_at == now
    assert response.ingest_pid == 12345
    assert response.ingest_force is True


def test_create_vault_response_populates_lock_fields():
    """Mock Vault with active ingest locks; create_vault_response copies them."""
    vault = MagicMock()
    vault.id = uuid.uuid4()
    vault.name = "Work"
    vault.description = "Work vault"
    vault.container_path = "/data/work"
    vault.host_path = "/host/work"
    vault.created_at = datetime.now()
    vault.ingest_status = "in_progress"
    vault.ingest_started_at = datetime(2026, 7, 9, 10, 30, 0)
    vault.ingest_pid = 12345
    vault.ingest_force = True

    response = create_vault_response(vault, document_count=5)

    assert response.ingest_status == "in_progress"
    assert response.ingest_started_at == datetime(2026, 7, 9, 10, 30, 0)
    assert response.ingest_pid == 12345
    assert response.ingest_force is True


def test_create_vault_response_idle_defaults():
    """Mock Vault with idle defaults; create_vault_response reflects them."""
    vault = MagicMock()
    vault.id = uuid.uuid4()
    vault.name = "Personal"
    vault.description = "Personal vault"
    vault.container_path = "/data/personal"
    vault.host_path = "/host/personal"
    vault.created_at = datetime.now()
    vault.ingest_status = "idle"
    vault.ingest_started_at = None
    vault.ingest_pid = None
    vault.ingest_force = False

    response = create_vault_response(vault, document_count=0)

    assert response.ingest_status == "idle"
    assert response.ingest_started_at is None
    assert response.ingest_pid is None
    assert response.ingest_force is False


def test_create_vault_response_failed_state():
    """Failed ingest status propagates; other fields are None/False."""
    vault = MagicMock()
    vault.id = uuid.uuid4()
    vault.name = "Failed"
    vault.description = None
    vault.container_path = "/data/failed"
    vault.host_path = "/host/failed"
    vault.created_at = datetime.now()
    vault.ingest_status = "failed"
    vault.ingest_started_at = None
    vault.ingest_pid = None
    vault.ingest_force = False

    response = create_vault_response(vault, document_count=1)

    assert response.ingest_status == "failed"
    assert response.ingest_started_at is None
    assert response.ingest_pid is None
    assert response.ingest_force is False


def test_create_vault_response_missing_lock_fields():
    """Vault without lock fields gets safe defaults via getattr."""
    vault = MagicMock(
        spec=["id", "name", "description", "container_path", "host_path", "created_at"]
    )
    vault.id = uuid.uuid4()
    vault.name = "Legacy"
    vault.description = None
    vault.container_path = "/data/legacy"
    vault.host_path = "/host/legacy"
    vault.created_at = datetime.now()
    # ingest_* attributes are not in spec, so getattr will use defaults

    response = create_vault_response(vault, document_count=2)

    assert response.ingest_status == "idle"
    assert response.ingest_started_at is None
    assert response.ingest_pid is None
    assert response.ingest_force is False
