"""Unit tests for MCP vault tools."""

import uuid
from datetime import datetime, UTC
from unittest.mock import MagicMock

import pytest

from obsidian_rag.database.models import Vault


class TestListVaults:
    """Test suite for list_vaults function."""

    def test_list_vaults_basic(self):
        """Test listing all vaults with document counts."""
        from obsidian_rag.mcp_server.tools.vaults import list_vaults

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
        from obsidian_rag.mcp_server.tools.vaults import list_vaults

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
        from obsidian_rag.mcp_server.tools.vaults import list_vaults

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
        from obsidian_rag.mcp_server.tools.vaults import list_vaults

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
        """Test that limit is validated (clamped to max 100)."""
        from obsidian_rag.mcp_server.tools.vaults import list_vaults

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

        result = list_vaults(mock_session, limit=200, offset=0)

        # Should be clamped to 100
        assert len(result.results) == 3  # All vaults returned

    def test_list_vaults_negative_offset(self):
        """Test that negative offset is clamped to 0."""
        from obsidian_rag.mcp_server.tools.vaults import list_vaults

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
        from obsidian_rag.mcp_server.tools.vaults import list_vaults

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
        from obsidian_rag.mcp_server.tools.vaults import list_vaults

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
        from obsidian_rag.mcp_server.tools.vaults import list_vaults

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
        from obsidian_rag.mcp_server.tools.vaults import list_vaults

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
        from obsidian_rag.mcp_server.tools.vaults import list_vaults

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
        from obsidian_rag.mcp_server.tools.vaults import get_vault
        from obsidian_rag.database.models import Document as DocumentModel

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

        result = get_vault(mock_session, name="Personal")

        assert result.name == "Personal"
        assert result.description == "Personal knowledge base"
        assert result.container_path == "/data/personal"
        assert result.host_path == "/home/user/personal"
        assert result.document_count == 42
        assert result.id == vault.id
        assert result.created_at == vault.created_at

    def test_get_vault_by_id(self):
        """Test getting vault by UUID."""
        from obsidian_rag.mcp_server.tools.vaults import get_vault
        from obsidian_rag.database.models import Document as DocumentModel

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
        from obsidian_rag.mcp_server.tools.vaults import get_vault

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
            get_vault(mock_session, name="NonExistent")

        error_msg = str(exc_info.value)
        assert "Vault 'NonExistent' not found" in error_msg
        assert "Available: Existing" in error_msg

    def test_get_vault_not_found_by_id(self):
        """Test ValueError when vault not found by UUID."""
        from obsidian_rag.mcp_server.tools.vaults import get_vault

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
        from obsidian_rag.mcp_server.tools.vaults import get_vault

        mock_session = MagicMock()

        with pytest.raises(ValueError) as exc_info:
            get_vault(mock_session)

        assert "Must provide name or vault_id" in str(exc_info.value)

    def test_get_vault_invalid_uuid(self):
        """Test ValueError for malformed UUID string."""
        from obsidian_rag.mcp_server.tools.vaults import get_vault

        mock_session = MagicMock()

        with pytest.raises(ValueError) as exc_info:
            get_vault(mock_session, vault_id="not-a-valid-uuid")

        error_msg = str(exc_info.value)
        assert "Invalid vault_id UUID format" in error_msg

    def test_get_vault_name_preferred_over_id(self):
        """Test that name lookup is preferred when both name and id are provided."""
        from obsidian_rag.mcp_server.tools.vaults import get_vault
        from obsidian_rag.database.models import Document as DocumentModel

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
        result = get_vault(mock_session, name="ByName", vault_id=str(uuid.uuid4()))

        assert result.name == "ByName"
        assert result.description == "Found by name"


class TestDeleteVault:
    """Test suite for delete_vault function."""

    def test_delete_vault_confirmed(self):
        """Test deleting vault with confirm=True returns success dict."""
        from obsidian_rag.mcp_server.tools.vaults import delete_vault

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

        result = delete_vault(mock_session, name="TestVault", confirm=True)

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
        from obsidian_rag.mcp_server.tools.vaults import delete_vault

        mock_session = MagicMock()

        result = delete_vault(mock_session, name="TestVault", confirm=False)

        assert result["success"] is False
        assert "error" in result
        assert "confirm=True" in result["error"]
        assert "irreversible" in result["error"]
        assert "cascade-delete" in result["error"]

    def test_delete_vault_not_found(self):
        """Test ValueError when vault not found."""
        from obsidian_rag.mcp_server.tools.vaults import delete_vault

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
            delete_vault(mock_session, name="NonExistent", confirm=True)

        error_msg = str(exc_info.value)
        assert "Vault 'NonExistent' not found" in error_msg
        assert "ExistingVault" in error_msg

    def test_delete_vault_with_documents(self):
        """Test cascade counts are correct when vault has documents."""
        from obsidian_rag.mcp_server.tools.vaults import delete_vault

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

        result = delete_vault(mock_session, name="WorkVault", confirm=True)

        assert result["documents_deleted"] == 3
        assert result["tasks_deleted"] == 7
        assert result["chunks_deleted"] == 12

    def test_delete_vault_zero_documents(self):
        """Test deleting vault with zero documents succeeds."""
        from obsidian_rag.mcp_server.tools.vaults import delete_vault

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

        result = delete_vault(mock_session, name="EmptyVault", confirm=True)

        assert result["success"] is True
        assert result["documents_deleted"] == 0
        assert result["tasks_deleted"] == 0
        assert result["chunks_deleted"] == 0

    def test_delete_vault_logs_warning(self, caplog):
        """Test that WARNING log is emitted before deletion."""
        import logging
        from obsidian_rag.mcp_server.tools.vaults import delete_vault

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
            result = delete_vault(mock_session, name="LogTestVault", confirm=True)

        assert result["success"] is True
        # Verify warning log message
        assert "Deleting vault 'LogTestVault'" in caplog.text
        assert "2 documents" in caplog.text
        assert "4 tasks" in caplog.text
        assert "8 chunks" in caplog.text

    def test_delete_vault_response_includes_config_warning(self):
        """Test that response includes warning about config entry."""
        from obsidian_rag.mcp_server.tools.vaults import delete_vault

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

        result = delete_vault(mock_session, name="ConfigWarningVault", confirm=True)

        assert "warning" in result
        assert "config entry" in result["warning"].lower()
        assert "next ingestion" in result["warning"].lower()
        assert "recreate" in result["warning"].lower()


class TestUpdateVault:
    """Test suite for update_vault function."""

    def test_update_vault_description_only(self):
        """Updates description, other fields unchanged."""
        from obsidian_rag.mcp_server.tools.vaults import update_vault
        from obsidian_rag.mcp_server.tools.vaults_params import VaultUpdateParams

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
            name="Personal",
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
        from obsidian_rag.mcp_server.tools.vaults import update_vault
        from obsidian_rag.mcp_server.tools.vaults_params import VaultUpdateParams

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
            name="Personal",
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
        from obsidian_rag.mcp_server.tools.vaults import update_vault
        from obsidian_rag.mcp_server.tools.vaults_params import VaultUpdateParams

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
            name="Personal",
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
        from obsidian_rag.mcp_server.tools.vaults import update_vault
        from obsidian_rag.mcp_server.tools.vaults_params import VaultUpdateParams
        from obsidian_rag.database.models import Document as DocumentModel

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
            name="Personal",
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
        from obsidian_rag.mcp_server.tools.vaults import update_vault
        from obsidian_rag.mcp_server.tools.vaults_params import VaultUpdateParams

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
            name="Personal",
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
        from obsidian_rag.mcp_server.tools.vaults import update_vault
        from obsidian_rag.mcp_server.tools.vaults_params import VaultUpdateParams

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
            name="Personal",
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
        from obsidian_rag.mcp_server.tools.vaults import update_vault
        from obsidian_rag.mcp_server.tools.vaults_params import VaultUpdateParams

        mock_session = MagicMock()
        mock_session.bind.dialect.name = "postgresql"

        mock_vault_query = MagicMock()
        mock_vault_query.filter.return_value = mock_vault_query
        mock_vault_query.first.return_value = None

        mock_session.query.return_value = mock_vault_query

        params = VaultUpdateParams(
            name="NonExistent",
            description="New description",
        )

        with pytest.raises(ValueError) as exc_info:
            update_vault(mock_session, params)

        assert "Vault 'NonExistent' not found" in str(exc_info.value)

    def test_update_vault_multiple_fields(self):
        """Updates description + host_path together."""
        from obsidian_rag.mcp_server.tools.vaults import update_vault
        from obsidian_rag.mcp_server.tools.vaults_params import VaultUpdateParams

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
            name="Personal",
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
        import logging
        from obsidian_rag.mcp_server.tools.vaults import update_vault
        from obsidian_rag.mcp_server.tools.vaults_params import VaultUpdateParams
        from obsidian_rag.database.models import Document as DocumentModel

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
            name="Personal",
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
        from obsidian_rag.mcp_server.tools.vaults import (
            _handle_flush_with_integrity_check,
        )

        mock_session = MagicMock()
        # Simulate a non-integrity error (e.g., database connection error)
        mock_session.flush.side_effect = Exception(
            "(psycopg2.OperationalError) could not connect to server"
        )

        # The error should be re-raised since it's not an integrity error
        with pytest.raises(Exception) as exc_info:
            _handle_flush_with_integrity_check(mock_session, "/data/path")

        assert "could not connect to server" in str(exc_info.value)
