"""Unit tests for MCP vault update and delete tools."""

import logging
import uuid
from datetime import datetime, UTC
from unittest.mock import MagicMock, patch

import pytest

from obsidian_rag.database.models import Vault
from obsidian_rag.mcp_server.tools.vaults import (
    _apply_vault_updates,
    _handle_flush_with_integrity_check,
    delete_vault,
    update_vault,
)
from obsidian_rag.mcp_server.tools.vaults_params import VaultUpdateParams


class TestDeleteVault:
    """Test suite for delete_vault function."""

    def test_delete_vault_confirmed(self):
        """Test deleting vault with confirm=True returns success dict."""

        vault_id = uuid.uuid4()
        vault = MagicMock(spec=Vault)
        vault.id = vault_id
        vault.name = "TestVault"

        mock_session = MagicMock()
        mock_session.bind.dialect.name = "postgresql"

        # Setup mock for vault lookup
        mock_vault_query = MagicMock()
        mock_vault_query.filter.return_value = mock_vault_query
        mock_vault_query.first.return_value = vault

        # Setup mocks for cascade counting - use distinct scalar return values
        # to identify which query is being executed
        mock_doc_count_query = MagicMock()
        mock_doc_count_query.filter.return_value = mock_doc_count_query
        mock_doc_count_query.scalar.return_value = 5

        mock_task_count_query = MagicMock()
        mock_task_count_query.join.return_value = mock_task_count_query
        mock_task_count_query.filter.return_value = mock_task_count_query
        mock_task_count_query.scalar.return_value = 10

        mock_chunk_count_query = MagicMock()
        mock_chunk_count_query.join.return_value = mock_chunk_count_query
        mock_chunk_count_query.filter.return_value = mock_chunk_count_query
        mock_chunk_count_query.scalar.return_value = 15

        # Query order: Vault -> Document count -> Task count -> Chunk count
        query_returns = [
            mock_vault_query,  # 1st: Vault lookup
            mock_doc_count_query,  # 2nd: Document count
            mock_task_count_query,  # 3rd: Task count
            mock_chunk_count_query,  # 4th: Chunk count
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

        result = delete_vault(mock_session, vault_name="TestVault", confirm=True)

        assert result["success"] is True
        assert result["name"] == "TestVault"
        assert result["id"] == str(vault_id)
        assert result["documents_deleted"] == 5
        assert result["tasks_deleted"] == 10
        assert result["chunks_deleted"] == 15
        assert "warning" in result
        assert "config entry" in result["warning"].lower()

    def test_delete_vault_without_confirm(self):
        """Test deleting vault without confirm returns error dict."""

        mock_session = MagicMock()

        result = delete_vault(mock_session, vault_name="TestVault", confirm=False)

        assert result["success"] is False
        assert "error" in result
        assert "confirm=True" in result["error"]
        assert "irreversible" in result["error"]
        assert "cascade-delete" in result["error"]

    def test_delete_vault_not_found(self):
        """Test ValueError when vault not found."""

        mock_session = MagicMock()
        mock_session.bind.dialect.name = "postgresql"

        # Setup mock for vault lookup (not found)
        mock_vault_query = MagicMock()
        mock_vault_query.filter.return_value = mock_vault_query
        mock_vault_query.first.return_value = None

        # Setup mock for available vaults
        available_vault = MagicMock(spec=Vault)
        available_vault.name = "ExistingVault"

        mock_vault_list_query = MagicMock()
        mock_vault_list_query.all.return_value = [available_vault]

        call_count = 0

        def query_side_effect(model):
            nonlocal call_count
            call_count += 1
            if model is Vault:
                if call_count == 1:
                    return mock_vault_query
                return mock_vault_list_query
            return MagicMock()

        mock_session.query.side_effect = query_side_effect

        with pytest.raises(ValueError) as exc_info:
            delete_vault(mock_session, vault_name="NonExistent", confirm=True)

        error_msg = str(exc_info.value)
        assert "Vault 'NonExistent' not found" in error_msg
        assert "ExistingVault" in error_msg

    def test_delete_vault_with_documents(self):
        """Test cascade counts are correct when vault has documents."""

        vault_id = uuid.uuid4()
        vault = MagicMock(spec=Vault)
        vault.id = vault_id
        vault.name = "WorkVault"

        mock_session = MagicMock()
        mock_session.bind.dialect.name = "postgresql"

        # Setup mock for vault lookup
        mock_vault_query = MagicMock()
        mock_vault_query.filter.return_value = mock_vault_query
        mock_vault_query.first.return_value = vault

        # Setup mocks for cascade counting
        mock_doc_count_query = MagicMock()
        mock_doc_count_query.filter.return_value = mock_doc_count_query
        mock_doc_count_query.scalar.return_value = 3

        mock_task_count_query = MagicMock()
        mock_task_count_query.join.return_value = mock_task_count_query
        mock_task_count_query.filter.return_value = mock_task_count_query
        mock_task_count_query.scalar.return_value = 7

        mock_chunk_count_query = MagicMock()
        mock_chunk_count_query.join.return_value = mock_chunk_count_query
        mock_chunk_count_query.filter.return_value = mock_chunk_count_query
        mock_chunk_count_query.scalar.return_value = 12

        # Query order: Vault -> Document count -> Task count -> Chunk count
        query_returns = [
            mock_vault_query,  # 1st: Vault lookup
            mock_doc_count_query,  # 2nd: Document count
            mock_task_count_query,  # 3rd: Task count
            mock_chunk_count_query,  # 4th: Chunk count
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

        result = delete_vault(mock_session, vault_name="WorkVault", confirm=True)

        assert result["documents_deleted"] == 3
        assert result["tasks_deleted"] == 7
        assert result["chunks_deleted"] == 12

    def test_delete_vault_zero_documents(self):
        """Test deleting vault with zero documents succeeds."""

        vault_id = uuid.uuid4()
        vault = MagicMock(spec=Vault)
        vault.id = vault_id
        vault.name = "EmptyVault"

        mock_session = MagicMock()
        mock_session.bind.dialect.name = "postgresql"

        # Setup mock for vault lookup
        mock_vault_query = MagicMock()
        mock_vault_query.filter.return_value = mock_vault_query
        mock_vault_query.first.return_value = vault

        # Setup mocks for cascade counting (all zero)
        mock_doc_count_query = MagicMock()
        mock_doc_count_query.filter.return_value = mock_doc_count_query
        mock_doc_count_query.scalar.return_value = 0

        mock_task_count_query = MagicMock()
        mock_task_count_query.join.return_value = mock_task_count_query
        mock_task_count_query.filter.return_value = mock_task_count_query
        mock_task_count_query.scalar.return_value = 0

        mock_chunk_count_query = MagicMock()
        mock_chunk_count_query.join.return_value = mock_chunk_count_query
        mock_chunk_count_query.filter.return_value = mock_chunk_count_query
        mock_chunk_count_query.scalar.return_value = 0

        # Query order: Vault -> Document count -> Task count -> Chunk count
        query_returns = [
            mock_vault_query,  # 1st: Vault lookup
            mock_doc_count_query,  # 2nd: Document count
            mock_task_count_query,  # 3rd: Task count
            mock_chunk_count_query,  # 4th: Chunk count
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

        result = delete_vault(mock_session, vault_name="EmptyVault", confirm=True)

        assert result["success"] is True
        assert result["documents_deleted"] == 0
        assert result["tasks_deleted"] == 0
        assert result["chunks_deleted"] == 0

    def test_delete_vault_logs_warning(self, caplog):
        """Test that WARNING log is emitted before deletion."""

        vault_id = uuid.uuid4()
        vault = MagicMock(spec=Vault)
        vault.id = vault_id
        vault.name = "LogTestVault"

        mock_session = MagicMock()
        mock_session.bind.dialect.name = "postgresql"

        # Setup mock for vault lookup
        mock_vault_query = MagicMock()
        mock_vault_query.filter.return_value = mock_vault_query
        mock_vault_query.first.return_value = vault

        # Setup mocks for cascade counting
        mock_doc_count_query = MagicMock()
        mock_doc_count_query.filter.return_value = mock_doc_count_query
        mock_doc_count_query.scalar.return_value = 2

        mock_task_count_query = MagicMock()
        mock_task_count_query.join.return_value = mock_task_count_query
        mock_task_count_query.filter.return_value = mock_task_count_query
        mock_task_count_query.scalar.return_value = 4

        mock_chunk_count_query = MagicMock()
        mock_chunk_count_query.join.return_value = mock_chunk_count_query
        mock_chunk_count_query.filter.return_value = mock_chunk_count_query
        mock_chunk_count_query.scalar.return_value = 8

        # Query order: Vault -> Document count -> Task count -> Chunk count
        query_returns = [
            mock_vault_query,  # 1st: Vault lookup
            mock_doc_count_query,  # 2nd: Document count
            mock_task_count_query,  # 3rd: Task count
            mock_chunk_count_query,  # 4th: Chunk count
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

        with caplog.at_level(logging.WARNING):
            result = delete_vault(mock_session, vault_name="LogTestVault", confirm=True)

        assert result["success"] is True
        # Verify warning log message
        assert "Deleting vault 'LogTestVault'" in caplog.text
        assert "2 documents" in caplog.text
        assert "4 tasks" in caplog.text
        assert "8 chunks" in caplog.text

    def test_delete_vault_response_includes_config_warning(self):
        """Test that response includes warning about config entry."""

        vault_id = uuid.uuid4()
        vault = MagicMock(spec=Vault)
        vault.id = vault_id
        vault.name = "ConfigWarningVault"

        mock_session = MagicMock()
        mock_session.bind.dialect.name = "postgresql"

        # Setup mock for vault lookup
        mock_vault_query = MagicMock()
        mock_vault_query.filter.return_value = mock_vault_query
        mock_vault_query.first.return_value = vault

        # Setup mocks for cascade counting
        mock_doc_count_query = MagicMock()
        mock_doc_count_query.filter.return_value = mock_doc_count_query
        mock_doc_count_query.scalar.return_value = 1

        mock_task_count_query = MagicMock()
        mock_task_count_query.join.return_value = mock_task_count_query
        mock_task_count_query.filter.return_value = mock_task_count_query
        mock_task_count_query.scalar.return_value = 0

        mock_chunk_count_query = MagicMock()
        mock_chunk_count_query.join.return_value = mock_chunk_count_query
        mock_chunk_count_query.filter.return_value = mock_chunk_count_query
        mock_chunk_count_query.scalar.return_value = 2

        call_count = 0

        def query_side_effect(model):
            nonlocal call_count
            call_count += 1
            if model is Vault:
                return mock_vault_query
            if hasattr(model, "element") or hasattr(model, "clauses"):
                if call_count <= 2:
                    return mock_doc_count_query
                elif call_count <= 4:
                    return mock_task_count_query
                else:
                    return mock_chunk_count_query
            return MagicMock()

        mock_session.query.side_effect = query_side_effect

        result = delete_vault(
            mock_session, vault_name="ConfigWarningVault", confirm=True
        )

        assert "warning" in result
        assert "config entry" in result["warning"].lower()
        assert "next ingestion" in result["warning"].lower()
        assert "recreate" in result["warning"].lower()


class TestUpdateVault:
    """Test suite for update_vault function."""

    def test_update_vault_description_only(self):
        """Updates description, other fields unchanged."""

        vault_id = uuid.uuid4()
        vault = MagicMock(spec=Vault)
        vault.id = vault_id
        vault.name = "Personal"
        vault.description = "Old description"
        vault.container_path = "/data/personal"
        vault.host_path = "/home/user/personal"
        vault.created_at = datetime.now(UTC)

        mock_session = MagicMock()
        mock_session.bind.dialect.name = "postgresql"

        # Setup mock for vault lookup
        mock_vault_query = MagicMock()
        mock_vault_query.filter.return_value = mock_vault_query
        mock_vault_query.first.return_value = vault

        # Setup mock for document count
        mock_doc_query = MagicMock()
        mock_doc_query.filter.return_value = mock_doc_query
        mock_doc_query.scalar.return_value = 5

        def query_side_effect(model):
            if model is Vault:
                return mock_vault_query
            return mock_doc_query

        mock_session.query.side_effect = query_side_effect

        params = VaultUpdateParams(
            vault_name="Personal",
            description="New description",
        )

        result = update_vault(mock_session, params)

        # Verify description was updated
        assert vault.description == "New description"
        # Verify other fields unchanged
        assert vault.container_path == "/data/personal"
        assert vault.host_path == "/home/user/personal"
        # Verify flush was called
        mock_session.flush.assert_called_once()
        # Verify result is VaultResponse
        assert result.name == "Personal"
        assert result.description == "New description"
        assert result.document_count == 5

    def test_update_vault_host_path_only(self):
        """Updates host_path only."""

        vault_id = uuid.uuid4()
        vault = MagicMock(spec=Vault)
        vault.id = vault_id
        vault.name = "Personal"
        vault.description = "Personal knowledge base"
        vault.container_path = "/data/personal"
        vault.host_path = "/home/user/personal"
        vault.created_at = datetime.now(UTC)

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
            vault_name="Personal",
            host_path="/new/host/path",
        )

        result = update_vault(mock_session, params)

        # Verify host_path was updated
        assert vault.host_path == "/new/host/path"
        # Verify other fields unchanged
        assert vault.container_path == "/data/personal"
        assert vault.description == "Personal knowledge base"
        mock_session.flush.assert_called_once()
        assert result.host_path == "/new/host/path"

    def test_update_vault_container_path_without_force(self):
        """Returns error dict, no DB write."""

        vault_id = uuid.uuid4()
        vault = MagicMock(spec=Vault)
        vault.id = vault_id
        vault.name = "Personal"
        vault.description = "Personal knowledge base"
        vault.container_path = "/data/personal"
        vault.host_path = "/home/user/personal"
        vault.created_at = datetime.now(UTC)

        mock_session = MagicMock()
        mock_session.bind.dialect.name = "postgresql"

        mock_vault_query = MagicMock()
        mock_vault_query.filter.return_value = mock_vault_query
        mock_vault_query.first.return_value = vault

        mock_session.query.return_value = mock_vault_query

        params = VaultUpdateParams(
            vault_name="Personal",
            container_path="/new/container/path",
        )

        result = update_vault(mock_session, params)

        # Verify error dict returned
        assert isinstance(result, dict)
        assert result["success"] is False
        assert "force=True" in result["error"]
        # Verify no DB write occurred
        mock_session.flush.assert_not_called()
        # Verify container_path not changed
        assert vault.container_path == "/data/personal"

    def test_update_vault_container_path_with_force(self):
        """Deletes docs, updates path."""

        vault_id = uuid.uuid4()
        vault = MagicMock(spec=Vault)
        vault.id = vault_id
        vault.name = "Personal"
        vault.description = "Personal knowledge base"
        vault.container_path = "/data/personal"
        vault.host_path = "/home/user/personal"
        vault.created_at = datetime.now(UTC)

        mock_session = MagicMock()
        mock_session.bind.dialect.name = "postgresql"

        mock_vault_query = MagicMock()
        mock_vault_query.filter.return_value = mock_vault_query
        mock_vault_query.first.return_value = vault

        # Setup mock for document deletion and count
        mock_doc_delete_query = MagicMock()
        mock_doc_delete_query.filter.return_value = mock_doc_delete_query
        mock_doc_delete_query.delete.return_value = 3  # 3 documents deleted

        mock_doc_count_query = MagicMock()
        mock_doc_count_query.filter.return_value = mock_doc_count_query
        mock_doc_count_query.scalar.return_value = 0  # 0 docs after deletion

        call_count = 0

        def query_side_effect(model):
            nonlocal call_count
            call_count += 1
            if model is Vault:
                return mock_vault_query
            # First Document query is for deletion, second for count
            if call_count == 2:
                return mock_doc_delete_query
            return mock_doc_count_query

        mock_session.query.side_effect = query_side_effect

        params = VaultUpdateParams(
            vault_name="Personal",
            container_path="/new/container/path",
            force=True,
        )

        result = update_vault(mock_session, params)

        # Verify documents were deleted
        mock_doc_delete_query.delete.assert_called_once_with(synchronize_session=False)
        # Verify container_path was updated
        assert vault.container_path == "/new/container/path"
        mock_session.flush.assert_called_once()
        # Verify result shows 0 documents
        assert result.container_path == "/new/container/path"
        assert result.document_count == 0

    def test_update_vault_no_fields_changed(self):
        """Returns current VaultResponse without DB write."""

        vault_id = uuid.uuid4()
        vault = MagicMock(spec=Vault)
        vault.id = vault_id
        vault.name = "Personal"
        vault.description = "Personal knowledge base"
        vault.container_path = "/data/personal"
        vault.host_path = "/home/user/personal"
        vault.created_at = datetime.now(UTC)

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

        # Pass same values - no change
        params = VaultUpdateParams(
            vault_name="Personal",
            description="Personal knowledge base",  # Same as current
        )

        result = update_vault(mock_session, params)

        # Verify no DB write occurred
        mock_session.flush.assert_not_called()
        # Verify VaultResponse returned with current values
        assert result.name == "Personal"
        assert result.description == "Personal knowledge base"
        assert result.document_count == 5

    def test_update_vault_duplicate_container_path(self):
        """IntegrityError caught, returns error dict."""

        vault_id = uuid.uuid4()
        vault = MagicMock(spec=Vault)
        vault.id = vault_id
        vault.name = "Personal"
        vault.description = "Personal knowledge base"
        vault.container_path = "/data/personal"
        vault.host_path = "/home/user/personal"
        vault.created_at = datetime.now(UTC)

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

        # Simulate IntegrityError on flush
        mock_session.flush.side_effect = Exception(
            "(psycopg2.errors.UniqueViolation) duplicate key value violates unique constraint"
        )

        params = VaultUpdateParams(
            vault_name="Personal",
            container_path="/data/other",  # Different from current
            force=True,
        )

        result = update_vault(mock_session, params)

        # Verify error dict returned
        assert isinstance(result, dict)
        assert result["success"] is False
        assert "Duplicate" in result["error"]

    def test_update_vault_not_found(self):
        """Raises ValueError."""

        mock_session = MagicMock()
        mock_session.bind.dialect.name = "postgresql"

        mock_vault_query = MagicMock()
        mock_vault_query.filter.return_value = mock_vault_query
        mock_vault_query.first.return_value = None

        mock_session.query.return_value = mock_vault_query

        params = VaultUpdateParams(
            vault_name="NonExistent",
            description="New description",
        )

        with pytest.raises(ValueError) as exc_info:
            update_vault(mock_session, params)

        assert "Vault 'NonExistent' not found" in str(exc_info.value)

    def test_update_vault_multiple_fields(self):
        """Updates description + host_path together."""

        vault_id = uuid.uuid4()
        vault = MagicMock(spec=Vault)
        vault.id = vault_id
        vault.name = "Personal"
        vault.description = "Old description"
        vault.container_path = "/data/personal"
        vault.host_path = "/home/user/personal"
        vault.created_at = datetime.now(UTC)

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
            vault_name="Personal",
            description="New description",
            host_path="/new/host/path",
        )

        result = update_vault(mock_session, params)

        # Verify both fields updated
        assert vault.description == "New description"
        assert vault.host_path == "/new/host/path"
        # Verify container_path unchanged
        assert vault.container_path == "/data/personal"
        mock_session.flush.assert_called_once()
        assert result.description == "New description"
        assert result.host_path == "/new/host/path"

    def test_update_vault_container_path_force_logs_warning(self, caplog):
        """Verify WARNING log before deletion."""

        caplog.set_level(logging.WARNING)

        vault_id = uuid.uuid4()
        vault = MagicMock(spec=Vault)
        vault.id = vault_id
        vault.name = "Personal"
        vault.description = "Personal knowledge base"
        vault.container_path = "/data/personal"
        vault.host_path = "/home/user/personal"
        vault.created_at = datetime.now(UTC)

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
            vault_name="Personal",
            container_path="/new/container/path",
            force=True,
        )

        update_vault(mock_session, params)

        # Verify warning was logged
        warning_messages = [
            record.message
            for record in caplog.records
            if record.levelno == logging.WARNING
        ]
        assert any("Deleting all documents" in msg for msg in warning_messages)
        assert any("Personal" in msg for msg in warning_messages)

    def test_update_vault_non_integrity_error_reraised(self):
        """Non-integrity errors should be re-raised (TASK-025)."""

        mock_session = MagicMock()
        # Simulate a non-integrity error (e.g., database connection error)
        mock_session.flush.side_effect = Exception(
            "(psycopg2.OperationalError) could not connect to server"
        )

        # The error should be re-raised since it's not an integrity error
        with pytest.raises(Exception) as exc_info:
            _handle_flush_with_integrity_check(mock_session, "/data/path")

        assert "could not connect to server" in str(exc_info.value)

    def test_apply_vault_updates_runtime_error_when_container_path_none(self):
        """Guard raises RuntimeError when container_path is unexpectedly None."""

        mock_session = MagicMock()
        mock_vault = MagicMock()
        mock_vault.id = "vault-uuid"

        params = VaultUpdateParams(
            vault_name="Test",
            container_path=None,
            force=True,
        )

        with (
            patch(
                "obsidian_rag.mcp_server.tools.vaults._is_container_path_changing",
                return_value=True,
            ),
            patch("obsidian_rag.mcp_server.tools.vaults._delete_vault_documents"),
            patch("obsidian_rag.mcp_server.tools.vaults.log") as mock_log,
        ):
            with pytest.raises(RuntimeError) as exc_info:
                _apply_vault_updates(mock_vault, params, mock_session)

        expected_msg = "params.container_path is None despite validation guarantee"
        assert str(exc_info.value) == expected_msg
        mock_log.error.assert_called_once_with(expected_msg)
