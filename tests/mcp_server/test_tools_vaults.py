"""Unit tests for MCP vault tools."""

import uuid
from datetime import datetime, UTC
from unittest.mock import MagicMock

import pytest
from sqlalchemy import create_engine, func
from sqlalchemy.orm import sessionmaker

from obsidian_rag.database.models import Base, Document, Vault
from obsidian_rag.mcp_server.tools.vaults import list_vaults


@pytest.fixture
def db_engine():
    """Create a test database engine using SQLite."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    yield engine
    Base.metadata.drop_all(engine)


@pytest.fixture
def db_session(db_engine):
    """Create a test database session."""
    SessionLocal = sessionmaker(bind=db_engine)
    session = SessionLocal()
    yield session
    session.close()


@pytest.fixture
def sample_vaults(db_session):
    """Create sample vaults for testing."""
    vaults = [
        Vault(
            id=uuid.uuid4(),
            name="Personal",
            description="Personal knowledge base",
            container_path="/data/personal",
            host_path="/home/user/personal",
            created_at=datetime.now(UTC),
        ),
        Vault(
            id=uuid.uuid4(),
            name="Work",
            description="Work notes",
            container_path="/data/work",
            host_path="/home/user/work",
            created_at=datetime.now(UTC),
        ),
        Vault(
            id=uuid.uuid4(),
            name="Empty Vault",
            description="Vault with no documents",
            container_path="/data/empty",
            host_path="/home/user/empty",
            created_at=datetime.now(UTC),
        ),
    ]
    for vault in vaults:
        db_session.add(vault)
    db_session.commit()
    return vaults


@pytest.fixture
def sample_documents(db_session, sample_vaults):
    """Create sample documents for testing vault counts."""
    personal_vault = sample_vaults[0]
    work_vault = sample_vaults[1]

    docs = [
        Document(
            id=uuid.uuid4(),
            vault_id=personal_vault.id,
            file_path="projects/project1.md",
            file_name="project1.md",
            content="# Project 1",
            checksum_md5="abc123",
            created_at_fs=datetime.now(),
            modified_at_fs=datetime.now(),
        ),
        Document(
            id=uuid.uuid4(),
            vault_id=personal_vault.id,
            file_path="projects/project2.md",
            file_name="project2.md",
            content="# Project 2",
            checksum_md5="def456",
            created_at_fs=datetime.now(),
            modified_at_fs=datetime.now(),
        ),
        Document(
            id=uuid.uuid4(),
            vault_id=work_vault.id,
            file_path="meetings/weekly.md",
            file_name="weekly.md",
            content="# Weekly Meeting",
            checksum_md5="ghi789",
            created_at_fs=datetime.now(),
            modified_at_fs=datetime.now(),
        ),
    ]
    for doc in docs:
        db_session.add(doc)
    db_session.commit()
    return docs


class TestListVaults:
    """Test suite for list_vaults function."""

    def test_list_vaults_basic(self, db_session, sample_vaults, sample_documents):
        """Test listing all vaults with document counts."""
        result = list_vaults(db_session)

        assert result.total_count == 3
        assert len(result.results) == 3
        assert result.has_more is False
        assert result.next_offset is None

        # Verify vaults are ordered by name
        assert result.results[0].name == "Empty Vault"
        assert result.results[1].name == "Personal"
        assert result.results[2].name == "Work"

    def test_list_vaults_document_counts(
        self, db_session, sample_vaults, sample_documents
    ):
        """Test that document counts are correct."""
        result = list_vaults(db_session)

        # Find vaults by name
        vault_map = {v.name: v for v in result.results}

        assert vault_map["Personal"].document_count == 2
        assert vault_map["Work"].document_count == 1
        assert vault_map["Empty Vault"].document_count == 0

    def test_list_vaults_pagination(self, db_session, sample_vaults):
        """Test pagination with limit and offset."""
        # First page with limit of 2
        result = list_vaults(db_session, limit=2, offset=0)

        assert result.total_count == 3
        assert len(result.results) == 2
        assert result.has_more is True
        assert result.next_offset == 2

        # Second page
        result = list_vaults(db_session, limit=2, offset=2)

        assert result.total_count == 3
        assert len(result.results) == 1
        assert result.has_more is False
        assert result.next_offset is None

    def test_list_vaults_empty_database(self, db_session):
        """Test listing vaults when database is empty."""
        result = list_vaults(db_session)

        assert result.total_count == 0
        assert len(result.results) == 0
        assert result.has_more is False
        assert result.next_offset is None

    def test_list_vaults_limit_validation(self, db_session, sample_vaults):
        """Test that limit is validated (clamped to max 100)."""
        result = list_vaults(db_session, limit=200, offset=0)

        # Should be clamped to 100
        assert len(result.results) == 3  # All vaults returned

    def test_list_vaults_negative_offset(self, db_session, sample_vaults):
        """Test that negative offset is clamped to 0."""
        result = list_vaults(db_session, limit=10, offset=-5)

        # Should start from beginning
        assert len(result.results) == 3
        assert result.results[0].name == "Empty Vault"

    def test_list_vaults_response_fields(
        self, db_session, sample_vaults, sample_documents
    ):
        """Test that all expected fields are in response."""
        result = list_vaults(db_session)

        for vault_response in result.results:
            assert vault_response.id is not None
            assert vault_response.name is not None
            assert vault_response.description is not None
            assert vault_response.host_path is not None
            assert isinstance(vault_response.document_count, int)

    def test_list_vaults_specific_vault_fields(
        self, db_session, sample_vaults, sample_documents
    ):
        """Test specific vault field values."""
        result = list_vaults(db_session)

        personal_vault = next(v for v in result.results if v.name == "Personal")

        assert personal_vault.description == "Personal knowledge base"
        assert personal_vault.host_path == "/home/user/personal"
        assert personal_vault.document_count == 2

    def test_list_vaults_no_documents_table(self, db_session, sample_vaults):
        """Test listing vaults when no documents exist yet."""
        # Don't create any documents
        result = list_vaults(db_session)

        assert result.total_count == 3
        for vault in result.results:
            assert vault.document_count == 0

    def test_list_vaults_offset_beyond_total(self, db_session, sample_vaults):
        """Test pagination with offset beyond total count."""
        result = list_vaults(db_session, limit=10, offset=100)

        assert result.total_count == 3
        assert len(result.results) == 0
        assert result.has_more is False
        assert result.next_offset is None

    def test_list_vaults_single_vault(self, db_session):
        """Test listing with single vault."""
        vault = Vault(
            id=uuid.uuid4(),
            name="Single",
            description="Only vault",
            container_path="/data/single",
            host_path="/data/single",
            created_at=datetime.now(UTC),
        )
        db_session.add(vault)
        db_session.commit()

        result = list_vaults(db_session)

        assert result.total_count == 1
        assert len(result.results) == 1
        assert result.results[0].name == "Single"
        assert result.has_more is False
