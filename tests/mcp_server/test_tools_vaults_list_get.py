"""Unit tests for MCP vault list and get tools."""

import uuid
from datetime import datetime, UTC
from unittest.mock import MagicMock, patch

import pytest

from obsidian_rag.database.models import Document as DocumentModel, Vault
from obsidian_rag.mcp_server.tools.vaults import (
    _validate_vault_exists,
    get_vault,
    list_vaults,
)


class TestListVaults:
    """Test suite for list_vaults function."""

    def test_list_vaults_basic(self):
        """Test listing all vaults with document counts."""

        # Create mock vaults
        vault1 = MagicMock(spec=Vault)
        vault1.id = uuid.uuid4()
        vault1.name = "Empty Vault"
        vault1.description = "Vault with no documents"
        vault1.container_path = "/data/empty"
        vault1.host_path = "/home/user/empty"
        vault1.created_at = datetime.now(UTC)

        vault2 = MagicMock(spec=Vault)
        vault2.id = uuid.uuid4()
        vault2.name = "Personal"
        vault2.description = "Personal knowledge base"
        vault2.container_path = "/data/personal"
        vault2.host_path = "/home/user/personal"
        vault2.created_at = datetime.now(UTC)

        vault3 = MagicMock(spec=Vault)
        vault3.id = uuid.uuid4()
        vault3.name = "Work"
        vault3.description = "Work notes"
        vault3.container_path = "/data/work"
        vault3.host_path = "/home/user/work"
        vault3.created_at = datetime.now(UTC)

        # Create mock session
        mock_session = MagicMock()
        mock_session.bind.dialect.name = "postgresql"

        # Setup mock query chain
        mock_query = MagicMock()
        mock_query.group_by.return_value = mock_query
        mock_query.subquery.return_value = MagicMock()
        mock_query.outerjoin.return_value = mock_query
        mock_query.order_by.return_value = mock_query
        mock_query.limit.return_value = mock_query
        mock_query.offset.return_value = mock_query
        mock_query.count.return_value = 3
        mock_query.all.return_value = [
            (vault1, 0),
            (vault2, 2),
            (vault3, 1),
        ]

        mock_session.query.return_value = mock_query

        result = list_vaults(mock_session)

        assert result.total_count == 3
        assert len(result.results) == 3
        assert result.has_more is False
        assert result.next_offset is None

        # Verify vaults are ordered by name
        assert result.results[0].name == "Empty Vault"
        assert result.results[1].name == "Personal"
        assert result.results[2].name == "Work"

    def test_list_vaults_document_counts(self):
        """Test that document counts are correct."""

        vault1 = MagicMock(spec=Vault)
        vault1.id = uuid.uuid4()
        vault1.name = "Personal"
        vault1.description = "Personal knowledge base"
        vault1.container_path = "/data/personal"
        vault1.host_path = "/home/user/personal"
        vault1.created_at = datetime.now(UTC)

        vault2 = MagicMock(spec=Vault)
        vault2.id = uuid.uuid4()
        vault2.name = "Work"
        vault2.description = "Work notes"
        vault2.container_path = "/data/work"
        vault2.host_path = "/home/user/work"
        vault2.created_at = datetime.now(UTC)

        vault3 = MagicMock(spec=Vault)
        vault3.id = uuid.uuid4()
        vault3.name = "Empty Vault"
        vault3.description = "Vault with no documents"
        vault3.container_path = "/data/empty"
        vault3.host_path = "/home/user/empty"
        vault3.created_at = datetime.now(UTC)

        mock_session = MagicMock()
        mock_session.bind.dialect.name = "postgresql"

        mock_query = MagicMock()
        mock_query.group_by.return_value = mock_query
        mock_query.subquery.return_value = MagicMock()
        mock_query.outerjoin.return_value = mock_query
        mock_query.order_by.return_value = mock_query
        mock_query.limit.return_value = mock_query
        mock_query.offset.return_value = mock_query
        mock_query.count.return_value = 3
        mock_query.all.return_value = [
            (vault1, 2),
            (vault2, 1),
            (vault3, 0),
        ]

        mock_session.query.return_value = mock_query

        result = list_vaults(mock_session)

        # Find vaults by name
        vault_map = {v.name: v for v in result.results}

        assert vault_map["Personal"].document_count == 2
        assert vault_map["Work"].document_count == 1
        assert vault_map["Empty Vault"].document_count == 0

    def test_list_vaults_pagination(self):
        """Test pagination with limit and offset."""

        vault1 = MagicMock(spec=Vault)
        vault1.id = uuid.uuid4()
        vault1.name = "Empty Vault"
        vault1.description = "Vault with no documents"
        vault1.container_path = "/data/empty"
        vault1.host_path = "/home/user/empty"
        vault1.created_at = datetime.now(UTC)

        vault2 = MagicMock(spec=Vault)
        vault2.id = uuid.uuid4()
        vault2.name = "Personal"
        vault2.description = "Personal knowledge base"
        vault2.container_path = "/data/personal"
        vault2.host_path = "/home/user/personal"
        vault2.created_at = datetime.now(UTC)

        vault3 = MagicMock(spec=Vault)
        vault3.id = uuid.uuid4()
        vault3.name = "Work"
        vault3.description = "Work notes"
        vault3.container_path = "/data/work"
        vault3.host_path = "/home/user/work"
        vault3.created_at = datetime.now(UTC)

        mock_session = MagicMock()
        mock_session.bind.dialect.name = "postgresql"

        # First page with limit of 2
        mock_query = MagicMock()
        mock_query.group_by.return_value = mock_query
        mock_query.subquery.return_value = MagicMock()
        mock_query.outerjoin.return_value = mock_query
        mock_query.order_by.return_value = mock_query
        mock_query.limit.return_value = mock_query
        mock_query.offset.return_value = mock_query
        mock_query.count.return_value = 3
        mock_query.all.return_value = [
            (vault1, 0),
            (vault2, 2),
        ]

        mock_session.query.return_value = mock_query

        result = list_vaults(mock_session, limit=2, offset=0)

        assert result.total_count == 3
        assert len(result.results) == 2
        assert result.has_more is True
        assert result.next_offset == 2

    def test_list_vaults_empty_database(self):
        """Test listing vaults when database is empty."""

        mock_session = MagicMock()
        mock_session.bind.dialect.name = "postgresql"

        mock_query = MagicMock()
        mock_query.group_by.return_value = mock_query
        mock_query.subquery.return_value = MagicMock()
        mock_query.outerjoin.return_value = mock_query
        mock_query.order_by.return_value = mock_query
        mock_query.limit.return_value = mock_query
        mock_query.offset.return_value = mock_query
        mock_query.count.return_value = 0
        mock_query.all.return_value = []

        mock_session.query.return_value = mock_query

        result = list_vaults(mock_session)

        assert result.total_count == 0
        assert len(result.results) == 0
        assert result.has_more is False
        assert result.next_offset is None

    def test_list_vaults_limit_validation(self):
        """Test that limit is validated (clamped to max 10000)."""

        vault1 = MagicMock(spec=Vault)
        vault1.id = uuid.uuid4()
        vault1.name = "Personal"
        vault1.description = "Personal knowledge base"
        vault1.container_path = "/data/personal"
        vault1.host_path = "/home/user/personal"
        vault1.created_at = datetime.now(UTC)

        vault2 = MagicMock(spec=Vault)
        vault2.id = uuid.uuid4()
        vault2.name = "Work"
        vault2.description = "Work notes"
        vault2.container_path = "/data/work"
        vault2.host_path = "/home/user/work"
        vault2.created_at = datetime.now(UTC)

        vault3 = MagicMock(spec=Vault)
        vault3.id = uuid.uuid4()
        vault3.name = "Empty Vault"
        vault3.description = "Vault with no documents"
        vault3.container_path = "/data/empty"
        vault3.host_path = "/home/user/empty"
        vault3.created_at = datetime.now(UTC)

        mock_session = MagicMock()
        mock_session.bind.dialect.name = "postgresql"

        mock_query = MagicMock()
        mock_query.group_by.return_value = mock_query
        mock_query.subquery.return_value = MagicMock()
        mock_query.outerjoin.return_value = mock_query
        mock_query.order_by.return_value = mock_query
        mock_query.limit.return_value = mock_query
        mock_query.offset.return_value = mock_query
        mock_query.count.return_value = 3
        mock_query.all.return_value = [
            (vault1, 2),
            (vault2, 1),
            (vault3, 0),
        ]

        mock_session.query.return_value = mock_query

        result = list_vaults(mock_session, limit=20000, offset=0)

        # Should be clamped to 10000
        assert len(result.results) == 3  # All vaults returned
        mock_query.limit.assert_called_with(10000)

    def test_list_vaults_negative_offset(self):
        """Test that negative offset is clamped to 0."""

        vault1 = MagicMock(spec=Vault)
        vault1.id = uuid.uuid4()
        vault1.name = "Empty Vault"
        vault1.description = "Vault with no documents"
        vault1.container_path = "/data/empty"
        vault1.host_path = "/home/user/empty"
        vault1.created_at = datetime.now(UTC)

        vault2 = MagicMock(spec=Vault)
        vault2.id = uuid.uuid4()
        vault2.name = "Personal"
        vault2.description = "Personal knowledge base"
        vault2.container_path = "/data/personal"
        vault2.host_path = "/home/user/personal"
        vault2.created_at = datetime.now(UTC)

        vault3 = MagicMock(spec=Vault)
        vault3.id = uuid.uuid4()
        vault3.name = "Work"
        vault3.description = "Work notes"
        vault3.container_path = "/data/work"
        vault3.host_path = "/home/user/work"
        vault3.created_at = datetime.now(UTC)

        mock_session = MagicMock()
        mock_session.bind.dialect.name = "postgresql"

        mock_query = MagicMock()
        mock_query.group_by.return_value = mock_query
        mock_query.subquery.return_value = MagicMock()
        mock_query.outerjoin.return_value = mock_query
        mock_query.order_by.return_value = mock_query
        mock_query.limit.return_value = mock_query
        mock_query.offset.return_value = mock_query
        mock_query.count.return_value = 3
        mock_query.all.return_value = [
            (vault1, 0),
            (vault2, 2),
            (vault3, 1),
        ]

        mock_session.query.return_value = mock_query

        result = list_vaults(mock_session, limit=10, offset=-5)

        # Should start from beginning
        assert len(result.results) == 3
        assert result.results[0].name == "Empty Vault"

    def test_list_vaults_response_fields(self):
        """Test that all expected fields are in response."""

        vault1 = MagicMock(spec=Vault)
        vault1.id = uuid.uuid4()
        vault1.name = "Personal"
        vault1.description = "Personal knowledge base"
        vault1.container_path = "/data/personal"
        vault1.host_path = "/home/user/personal"
        vault1.created_at = datetime.now(UTC)

        mock_session = MagicMock()
        mock_session.bind.dialect.name = "postgresql"

        mock_query = MagicMock()
        mock_query.group_by.return_value = mock_query
        mock_query.subquery.return_value = MagicMock()
        mock_query.outerjoin.return_value = mock_query
        mock_query.order_by.return_value = mock_query
        mock_query.limit.return_value = mock_query
        mock_query.offset.return_value = mock_query
        mock_query.count.return_value = 1
        mock_query.all.return_value = [(vault1, 2)]

        mock_session.query.return_value = mock_query

        result = list_vaults(mock_session)

        for vault_response in result.results:
            assert vault_response.id is not None
            assert vault_response.name is not None
            assert vault_response.description is not None
            assert vault_response.container_path is not None
            assert vault_response.host_path is not None
            assert isinstance(vault_response.document_count, int)
            assert vault_response.created_at is not None

    def test_list_vaults_specific_vault_fields(self):
        """Test specific vault field values."""

        vault1 = MagicMock(spec=Vault)
        vault1.id = uuid.uuid4()
        vault1.name = "Personal"
        vault1.description = "Personal knowledge base"
        vault1.container_path = "/data/personal"
        vault1.host_path = "/home/user/personal"
        vault1.created_at = datetime.now(UTC)

        mock_session = MagicMock()
        mock_session.bind.dialect.name = "postgresql"

        mock_query = MagicMock()
        mock_query.group_by.return_value = mock_query
        mock_query.subquery.return_value = MagicMock()
        mock_query.outerjoin.return_value = mock_query
        mock_query.order_by.return_value = mock_query
        mock_query.limit.return_value = mock_query
        mock_query.offset.return_value = mock_query
        mock_query.count.return_value = 1
        mock_query.all.return_value = [(vault1, 2)]

        mock_session.query.return_value = mock_query

        result = list_vaults(mock_session)

        personal_vault = result.results[0]

        assert personal_vault.description == "Personal knowledge base"
        assert personal_vault.container_path == "/data/personal"
        assert personal_vault.host_path == "/home/user/personal"
        assert personal_vault.document_count == 2
        assert personal_vault.created_at is not None

    def test_list_vaults_no_documents_table(self):
        """Test listing vaults when no documents exist yet."""

        vault1 = MagicMock(spec=Vault)
        vault1.id = uuid.uuid4()
        vault1.name = "Personal"
        vault1.description = "Personal knowledge base"
        vault1.container_path = "/data/personal"
        vault1.host_path = "/home/user/personal"
        vault1.created_at = datetime.now(UTC)

        vault2 = MagicMock(spec=Vault)
        vault2.id = uuid.uuid4()
        vault2.name = "Work"
        vault2.description = "Work notes"
        vault2.container_path = "/data/work"
        vault2.host_path = "/home/user/work"
        vault2.created_at = datetime.now(UTC)

        vault3 = MagicMock(spec=Vault)
        vault3.id = uuid.uuid4()
        vault3.name = "Empty Vault"
        vault3.description = "Vault with no documents"
        vault3.container_path = "/data/empty"
        vault3.host_path = "/home/user/empty"
        vault3.created_at = datetime.now(UTC)

        mock_session = MagicMock()
        mock_session.bind.dialect.name = "postgresql"

        mock_query = MagicMock()
        mock_query.group_by.return_value = mock_query
        mock_query.subquery.return_value = MagicMock()
        mock_query.outerjoin.return_value = mock_query
        mock_query.order_by.return_value = mock_query
        mock_query.limit.return_value = mock_query
        mock_query.offset.return_value = mock_query
        mock_query.count.return_value = 3
        mock_query.all.return_value = [
            (vault1, 0),
            (vault2, 0),
            (vault3, 0),
        ]

        mock_session.query.return_value = mock_query

        result = list_vaults(mock_session)

        assert result.total_count == 3
        for vault in result.results:
            assert vault.document_count == 0

    def test_list_vaults_offset_beyond_total(self):
        """Test pagination with offset beyond total count."""

        mock_session = MagicMock()
        mock_session.bind.dialect.name = "postgresql"

        mock_query = MagicMock()
        mock_query.group_by.return_value = mock_query
        mock_query.subquery.return_value = MagicMock()
        mock_query.outerjoin.return_value = mock_query
        mock_query.order_by.return_value = mock_query
        mock_query.limit.return_value = mock_query
        mock_query.offset.return_value = mock_query
        mock_query.count.return_value = 3
        mock_query.all.return_value = []

        mock_session.query.return_value = mock_query

        result = list_vaults(mock_session, limit=10, offset=100)

        assert result.total_count == 3
        assert len(result.results) == 0
        assert result.has_more is False
        assert result.next_offset is None

    def test_list_vaults_single_vault(self):
        """Test listing with single vault."""

        vault = MagicMock(spec=Vault)
        vault.id = uuid.uuid4()
        vault.name = "Single"
        vault.description = "Only vault"
        vault.container_path = "/data/single"
        vault.host_path = "/data/single"
        vault.created_at = datetime.now(UTC)

        mock_session = MagicMock()
        mock_session.bind.dialect.name = "postgresql"

        mock_query = MagicMock()
        mock_query.group_by.return_value = mock_query
        mock_query.subquery.return_value = MagicMock()
        mock_query.outerjoin.return_value = mock_query
        mock_query.order_by.return_value = mock_query
        mock_query.limit.return_value = mock_query
        mock_query.offset.return_value = mock_query
        mock_query.count.return_value = 1
        mock_query.all.return_value = [(vault, 0)]

        mock_session.query.return_value = mock_query

        result = list_vaults(mock_session)

        assert result.total_count == 1
        assert len(result.results) == 1
        assert result.results[0].name == "Single"
        assert result.has_more is False


class TestGetVault:
    """Test suite for get_vault function."""

    def test_get_vault_by_name(self):
        """Test getting vault by name with document count."""

        vault = MagicMock(spec=Vault)
        vault.id = uuid.uuid4()
        vault.name = "Personal"
        vault.description = "Personal knowledge base"
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
        mock_doc_query.scalar.return_value = 42

        def query_side_effect(model):
            # Use 'is' for identity comparison to avoid SQLAlchemy magic methods
            if model is Vault:
                return mock_vault_query
            # Check for Document model or func.count(Document.id)
            if (
                model is DocumentModel
                or hasattr(model, "element")
                or hasattr(model, "clauses")
            ):
                return mock_doc_query
            return mock_doc_query

        mock_session.query.side_effect = query_side_effect

        result = get_vault(mock_session, vault_name="Personal")

        assert result.name == "Personal"
        assert result.description == "Personal knowledge base"
        assert result.container_path == "/data/personal"
        assert result.host_path == "/home/user/personal"
        assert result.document_count == 42
        assert result.id == vault.id
        assert result.created_at == vault.created_at

    def test_get_vault_by_id(self):
        """Test getting vault by UUID."""

        vault_id = uuid.uuid4()
        vault = MagicMock(spec=Vault)
        vault.id = vault_id
        vault.name = "Work"
        vault.description = "Work notes"
        vault.container_path = "/data/work"
        vault.host_path = "/home/user/work"
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
        mock_doc_query.scalar.return_value = 10

        def query_side_effect(model):
            if model is Vault:
                return mock_vault_query
            if (
                model is DocumentModel
                or hasattr(model, "element")
                or hasattr(model, "clauses")
            ):
                return mock_doc_query
            return mock_doc_query

        mock_session.query.side_effect = query_side_effect

        result = get_vault(mock_session, vault_id=str(vault_id))

        assert result.name == "Work"
        assert result.document_count == 10
        assert result.id == vault_id

    def test_get_vault_not_found_by_name(self):
        """Test ValueError when vault not found by name with available vaults listed."""

        mock_session = MagicMock()
        mock_session.bind.dialect.name = "postgresql"

        # Setup mock for vault lookup (not found)
        mock_vault_query = MagicMock()
        mock_vault_query.filter.return_value = mock_vault_query
        mock_vault_query.first.return_value = None

        # Setup mock for available vaults - first call is for lookup, second is for listing
        available_vault = MagicMock(spec=Vault)
        available_vault.name = "Existing"

        mock_vault_list_query = MagicMock()
        mock_vault_list_query.all.return_value = [available_vault]

        call_count = 0

        def query_side_effect(model):
            nonlocal call_count
            call_count += 1
            # First call is for vault lookup, second for getting available vaults
            if model is Vault:
                if call_count == 1:
                    return mock_vault_query
                return mock_vault_list_query
            return MagicMock()

        mock_session.query.side_effect = query_side_effect

        with pytest.raises(ValueError) as exc_info:
            get_vault(mock_session, vault_name="NonExistent")

        error_msg = str(exc_info.value)
        assert "Vault 'NonExistent' not found" in error_msg
        assert "Available: Existing" in error_msg

    def test_get_vault_not_found_by_id(self):
        """Test ValueError when vault not found by UUID."""

        vault_id = uuid.uuid4()

        mock_session = MagicMock()
        mock_session.bind.dialect.name = "postgresql"

        # Setup mock for vault lookup (not found)
        mock_vault_query = MagicMock()
        mock_vault_query.filter.return_value = mock_vault_query
        mock_vault_query.first.return_value = None

        # Setup mock for available vaults
        available_vault = MagicMock(spec=Vault)
        available_vault.name = "Work"

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
            get_vault(mock_session, vault_id=str(vault_id))

        error_msg = str(exc_info.value)
        assert f"Vault '{vault_id}' not found" in error_msg

    def test_get_vault_neither_name_nor_id(self):
        """Test ValueError when neither name nor vault_id is provided."""

        mock_session = MagicMock()

        with pytest.raises(ValueError) as exc_info:
            get_vault(mock_session)

        assert "Must provide vault_name or vault_id" in str(exc_info.value)

    def test_get_vault_invalid_uuid(self):
        """Test ValueError for malformed UUID string."""

        mock_session = MagicMock()

        with pytest.raises(ValueError) as exc_info:
            get_vault(mock_session, vault_id="not-a-valid-uuid")

        error_msg = str(exc_info.value)
        assert "Invalid vault_id UUID format" in error_msg

    def test_get_vault_name_preferred_over_id(self):
        """Test that name lookup is preferred when both name and id are provided."""

        vault = MagicMock(spec=Vault)
        vault.id = uuid.uuid4()
        vault.name = "ByName"
        vault.description = "Found by name"
        vault.container_path = "/data/name"
        vault.host_path = "/home/user/name"
        vault.created_at = datetime.now(UTC)

        mock_session = MagicMock()
        mock_session.bind.dialect.name = "postgresql"

        # Setup mock for vault lookup by name
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
            if (
                model is DocumentModel
                or hasattr(model, "element")
                or hasattr(model, "clauses")
            ):
                return mock_doc_query
            return mock_doc_query

        mock_session.query.side_effect = query_side_effect

        # Both provided - should use name
        result = get_vault(
            mock_session, vault_name="ByName", vault_id=str(uuid.uuid4())
        )

        assert result.name == "ByName"
        assert result.description == "Found by name"


class TestValidateVaultExists:
    """Test suite for _validate_vault_exists helper."""

    def test_validate_vault_exists_found(self):
        """Returns vault when it exists."""

        vault = MagicMock(spec=Vault)
        vault.name = "Personal"

        mock_session = MagicMock()
        mock_query = MagicMock()
        mock_query.filter.return_value = mock_query
        mock_query.first.return_value = vault
        mock_session.query.return_value = mock_query

        result = _validate_vault_exists(mock_session, "Personal")

        assert result == vault
        mock_session.query.assert_called_once_with(Vault)
        mock_query.filter.assert_called_once()

    def test_validate_vault_exists_not_found_with_available(self):
        """Raises ValueError listing available vaults."""

        available_vault = MagicMock(spec=Vault)
        available_vault.name = "Work"

        mock_session = MagicMock()

        # First query: vault lookup (not found)
        mock_lookup_query = MagicMock()
        mock_lookup_query.filter.return_value = mock_lookup_query
        mock_lookup_query.first.return_value = None

        # Second query: available vaults
        mock_list_query = MagicMock()
        mock_list_query.all.return_value = [available_vault]

        call_count = 0

        def query_side_effect(model):
            nonlocal call_count
            call_count += 1
            if model is Vault:
                if call_count == 1:
                    return mock_lookup_query
                return mock_list_query
            return MagicMock()

        mock_session.query.side_effect = query_side_effect

        with pytest.raises(ValueError) as exc_info:
            _validate_vault_exists(mock_session, "Missing")

        error_msg = str(exc_info.value)
        assert "Vault 'Missing' not found" in error_msg
        assert "Available: Work" in error_msg

    def test_validate_vault_exists_not_found_empty_database(self):
        """Raises ValueError with 'none' when no vaults exist."""

        mock_session = MagicMock()

        # First query: vault lookup (not found)
        mock_lookup_query = MagicMock()
        mock_lookup_query.filter.return_value = mock_lookup_query
        mock_lookup_query.first.return_value = None

        # Second query: available vaults (empty)
        mock_list_query = MagicMock()
        mock_list_query.all.return_value = []

        call_count = 0

        def query_side_effect(model):
            nonlocal call_count
            call_count += 1
            if model is Vault:
                if call_count == 1:
                    return mock_lookup_query
                return mock_list_query
            return MagicMock()

        mock_session.query.side_effect = query_side_effect

        with pytest.raises(ValueError) as exc_info:
            _validate_vault_exists(mock_session, "Missing")

        error_msg = str(exc_info.value)
        assert "Vault 'Missing' not found" in error_msg
        assert "Available: none" in error_msg


class TestGetVaultDefensiveGuards:
    """Test defensive RuntimeError guards in get_vault."""

    def test_get_vault_runtime_error_when_vault_id_none_after_validation(
        self,
    ):
        """Guard raises RuntimeError when vault_id is unexpectedly None."""
        mock_session = MagicMock()

        with (
            patch(
                "obsidian_rag.mcp_server.tools.vaults._validate_get_vault_params",
                return_value=None,
            ),
            patch("obsidian_rag.mcp_server.tools.vaults.log") as mock_log,
        ):
            with pytest.raises(RuntimeError) as exc_info:
                get_vault(mock_session, vault_name=None, vault_id=None)

        expected_msg = "vault_id is None despite validation guarantee"
        assert str(exc_info.value) == expected_msg
        mock_log.error.assert_called_once_with(expected_msg)
