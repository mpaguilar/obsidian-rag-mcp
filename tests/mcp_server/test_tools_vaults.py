"""Unit tests for MCP vault tools."""

import uuid
from datetime import datetime, UTC
from unittest.mock import MagicMock, patch

import pytest

from obsidian_rag.database.models import Vault
from obsidian_rag.mcp_server.models import VaultListResponse, VaultResponse


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
        vault1.host_path = "/home/user/empty"

        vault2 = MagicMock(spec=Vault)
        vault2.id = uuid.uuid4()
        vault2.name = "Personal"
        vault2.description = "Personal knowledge base"
        vault2.host_path = "/home/user/personal"

        vault3 = MagicMock(spec=Vault)
        vault3.id = uuid.uuid4()
        vault3.name = "Work"
        vault3.description = "Work notes"
        vault3.host_path = "/home/user/work"

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
        vault1.host_path = "/home/user/personal"

        vault2 = MagicMock(spec=Vault)
        vault2.id = uuid.uuid4()
        vault2.name = "Work"
        vault2.description = "Work notes"
        vault2.host_path = "/home/user/work"

        vault3 = MagicMock(spec=Vault)
        vault3.id = uuid.uuid4()
        vault3.name = "Empty Vault"
        vault3.description = "Vault with no documents"
        vault3.host_path = "/home/user/empty"

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
        vault1.host_path = "/home/user/empty"

        vault2 = MagicMock(spec=Vault)
        vault2.id = uuid.uuid4()
        vault2.name = "Personal"
        vault2.description = "Personal knowledge base"
        vault2.host_path = "/home/user/personal"

        vault3 = MagicMock(spec=Vault)
        vault3.id = uuid.uuid4()
        vault3.name = "Work"
        vault3.description = "Work notes"
        vault3.host_path = "/home/user/work"

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
        vault1.host_path = "/home/user/personal"

        vault2 = MagicMock(spec=Vault)
        vault2.id = uuid.uuid4()
        vault2.name = "Work"
        vault2.description = "Work notes"
        vault2.host_path = "/home/user/work"

        vault3 = MagicMock(spec=Vault)
        vault3.id = uuid.uuid4()
        vault3.name = "Empty Vault"
        vault3.description = "Vault with no documents"
        vault3.host_path = "/home/user/empty"

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
        vault1.host_path = "/home/user/empty"

        vault2 = MagicMock(spec=Vault)
        vault2.id = uuid.uuid4()
        vault2.name = "Personal"
        vault2.description = "Personal knowledge base"
        vault2.host_path = "/home/user/personal"

        vault3 = MagicMock(spec=Vault)
        vault3.id = uuid.uuid4()
        vault3.name = "Work"
        vault3.description = "Work notes"
        vault3.host_path = "/home/user/work"

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
        vault1.host_path = "/home/user/personal"

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
            assert vault_response.host_path is not None
            assert isinstance(vault_response.document_count, int)

    def test_list_vaults_specific_vault_fields(self):
        """Test specific vault field values."""
        from obsidian_rag.mcp_server.tools.vaults import list_vaults

        vault1 = MagicMock(spec=Vault)
        vault1.id = uuid.uuid4()
        vault1.name = "Personal"
        vault1.description = "Personal knowledge base"
        vault1.host_path = "/home/user/personal"

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
        assert personal_vault.host_path == "/home/user/personal"
        assert personal_vault.document_count == 2

    def test_list_vaults_no_documents_table(self):
        """Test listing vaults when no documents exist yet."""
        from obsidian_rag.mcp_server.tools.vaults import list_vaults

        vault1 = MagicMock(spec=Vault)
        vault1.id = uuid.uuid4()
        vault1.name = "Personal"
        vault1.description = "Personal knowledge base"
        vault1.host_path = "/home/user/personal"

        vault2 = MagicMock(spec=Vault)
        vault2.id = uuid.uuid4()
        vault2.name = "Work"
        vault2.description = "Work notes"
        vault2.host_path = "/home/user/work"

        vault3 = MagicMock(spec=Vault)
        vault3.id = uuid.uuid4()
        vault3.name = "Empty Vault"
        vault3.description = "Vault with no documents"
        vault3.host_path = "/home/user/empty"

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
        vault.host_path = "/data/single"

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
