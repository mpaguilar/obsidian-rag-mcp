"""Unit tests for MCP document tools."""

import uuid
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from obsidian_rag.database.models import Base, Document
from obsidian_rag.mcp_server.tools.documents import (
    _get_relative_path,
    _glob_to_like,
    get_all_tags,
    get_documents_by_tag,
    query_documents,
)


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


class TestQueryDocuments:
    """Tests for query_documents function."""

    def test_empty_result(self, db_session):
        """Test with no documents in database."""
        query_embedding = [0.1] * 1536
        result = query_documents(db_session, query_embedding)

        assert result.results == []
        assert result.total_count == 0
        assert result.has_more is False
        assert result.next_offset is None

    def test_excludes_documents_without_embeddings(self, db_session):
        """Test documents without embeddings are excluded."""
        doc = Document(
            id=uuid.uuid4(),
            file_path="/test/doc.md",
            file_name="doc.md",
            content="# Test",
            checksum_md5="abc123",
            created_at_fs=datetime.now(),
            modified_at_fs=datetime.now(),
            content_vector=None,  # No embedding
        )
        db_session.add(doc)
        db_session.commit()

        query_embedding = [0.1] * 1536
        result = query_documents(db_session, query_embedding)

        assert result.total_count == 0
        assert len(result.results) == 0

    def test_limit_validation(self, db_session):
        """Test that limit is validated and clamped."""
        query_embedding = [0.1] * 1536

        # Test limit above maximum (should work but be clamped internally)
        result = query_documents(db_session, query_embedding, limit=200)
        assert result.total_count == 0  # No documents

    def test_offset_validation(self, db_session):
        """Test that offset is validated."""
        query_embedding = [0.1] * 1536

        # Test negative offset (should be clamped to 0)
        result = query_documents(db_session, query_embedding, offset=-10)
        assert result.total_count == 0  # No documents


class TestGetRelativePath:
    """Tests for _get_relative_path helper function."""

    def test_with_vault_root(self):
        """Test path calculation with vault_root set."""
        file_path = "/data/vault/folder/note.md"
        vault_root = "/data/vault"
        result = _get_relative_path(file_path, vault_root)
        assert result == "./folder/note.md"

    def test_with_trailing_slash_in_root(self):
        """Test path calculation with trailing slash in root."""
        file_path = "/data/vault/folder/note.md"
        vault_root = "/data/vault/"
        result = _get_relative_path(file_path, vault_root)
        assert result == "./folder/note.md"

    def test_without_vault_root(self):
        """Test path calculation without vault_root returns absolute path."""
        file_path = "/data/vault/folder/note.md"
        result = _get_relative_path(file_path, None)
        assert result == "/data/vault/folder/note.md"

    def test_path_not_under_root(self):
        """Test path not under vault_root returns original path."""
        file_path = "/other/path/note.md"
        vault_root = "/data/vault"
        result = _get_relative_path(file_path, vault_root)
        assert result == "/other/path/note.md"


class TestGlobToLike:
    """Tests for _glob_to_like helper function."""

    def test_star_wildcard(self):
        """Test * becomes %."""
        result = _glob_to_like("tag*")
        assert result == "tag%"

    def test_question_wildcard(self):
        """Test ? becomes _."""
        result = _glob_to_like("ta?")
        assert result == "ta_"

    def test_sql_special_chars_escaped(self):
        """Test SQL special characters are escaped."""
        result = _glob_to_like("100%")
        assert "\\%" in result

    def test_multiple_wildcards(self):
        """Test multiple wildcards in pattern."""
        result = _glob_to_like("*work*")
        assert result == "%work%"


class TestGetDocumentsByTag:
    """Tests for get_documents_by_tag function."""

    def test_empty_result(self, db_session):
        """Test with no documents in database."""
        result = get_documents_by_tag(
            db_session,
            tag="work",
            vault_root=None,
            include_untagged=False,
            limit=20,
            offset=0,
        )

        assert result.results == []
        assert result.total_count == 0
        assert result.has_more is False

    def test_filter_by_tag(self, db_session):
        """Test filtering documents by tag."""
        doc1 = Document(
            id=uuid.uuid4(),
            file_path="/data/vault/work.md",
            file_name="work.md",
            content="# Work",
            checksum_md5="abc123",
            created_at_fs=datetime.now(),
            modified_at_fs=datetime.now(),
            tags=["work", "urgent"],
            vault_root="/data/vault",
        )
        doc2 = Document(
            id=uuid.uuid4(),
            file_path="/data/vault/personal.md",
            file_name="personal.md",
            content="# Personal",
            checksum_md5="def456",
            created_at_fs=datetime.now(),
            modified_at_fs=datetime.now(),
            tags=["personal"],
            vault_root="/data/vault",
        )
        db_session.add_all([doc1, doc2])
        db_session.commit()

        result = get_documents_by_tag(
            db_session,
            tag="work",
            vault_root=None,
            include_untagged=False,
            limit=20,
            offset=0,
        )

        assert result.total_count == 1
        assert len(result.results) == 1
        assert result.results[0].file_name == "work.md"

    def test_include_untagged(self, db_session):
        """Test including untagged documents."""
        doc1 = Document(
            id=uuid.uuid4(),
            file_path="/data/vault/tagged.md",
            file_name="tagged.md",
            content="# Tagged",
            checksum_md5="abc123",
            created_at_fs=datetime.now(),
            modified_at_fs=datetime.now(),
            tags=["work"],
            vault_root="/data/vault",
        )
        doc2 = Document(
            id=uuid.uuid4(),
            file_path="/data/vault/untagged.md",
            file_name="untagged.md",
            content="# Untagged",
            checksum_md5="def456",
            created_at_fs=datetime.now(),
            modified_at_fs=datetime.now(),
            tags=None,
            vault_root="/data/vault",
        )
        db_session.add_all([doc1, doc2])
        db_session.commit()

        result = get_documents_by_tag(
            db_session,
            tag=None,
            vault_root=None,
            include_untagged=True,
            limit=20,
            offset=0,
        )

        assert result.total_count == 1
        assert len(result.results) == 1
        assert result.results[0].file_name == "untagged.md"

    def test_vault_root_filter(self, db_session):
        """Test filtering by vault_root."""
        doc1 = Document(
            id=uuid.uuid4(),
            file_path="/data/vault1/note.md",
            file_name="note.md",
            content="# Note",
            checksum_md5="abc123",
            created_at_fs=datetime.now(),
            modified_at_fs=datetime.now(),
            tags=["work"],
            vault_root="/data/vault1",
        )
        doc2 = Document(
            id=uuid.uuid4(),
            file_path="/data/vault2/note.md",
            file_name="note.md",
            content="# Note",
            checksum_md5="def456",
            created_at_fs=datetime.now(),
            modified_at_fs=datetime.now(),
            tags=["work"],
            vault_root="/data/vault2",
        )
        db_session.add_all([doc1, doc2])
        db_session.commit()

        result = get_documents_by_tag(
            db_session,
            tag="work",
            vault_root="/data/vault1",
            include_untagged=False,
            limit=20,
            offset=0,
        )

        assert result.total_count == 1
        assert result.results[0].file_path.startswith("./")

    def test_relative_path_in_response(self, db_session):
        """Test that response includes relative paths."""
        doc = Document(
            id=uuid.uuid4(),
            file_path="/data/vault/projects/note.md",
            file_name="note.md",
            content="# Note",
            checksum_md5="abc123",
            created_at_fs=datetime.now(),
            modified_at_fs=datetime.now(),
            tags=["work"],
            vault_root="/data/vault",
        )
        db_session.add(doc)
        db_session.commit()

        result = get_documents_by_tag(
            db_session,
            tag="work",
            vault_root=None,
            include_untagged=False,
            limit=20,
            offset=0,
        )

        assert len(result.results) == 1
        assert result.results[0].file_path == "./projects/note.md"

    def test_pagination(self, db_session):
        """Test pagination with limit and offset."""
        for i in range(5):
            doc = Document(
                id=uuid.uuid4(),
                file_path=f"/data/vault/doc{i}.md",
                file_name=f"doc{i}.md",
                content=f"# Doc {i}",
                checksum_md5=f"hash{i}",
                created_at_fs=datetime.now(),
                modified_at_fs=datetime.now(),
                tags=["work"],
                vault_root="/data/vault",
            )
            db_session.add(doc)
        db_session.commit()

        result = get_documents_by_tag(
            db_session,
            tag="work",
            vault_root=None,
            include_untagged=False,
            limit=2,
            offset=0,
        )

        assert result.total_count == 5
        assert len(result.results) == 2
        assert result.has_more is True
        assert result.next_offset == 2


class TestGetAllTags:
    """Tests for get_all_tags function."""

    def test_empty_result(self, db_session):
        """Test with no documents in database."""
        result = get_all_tags(
            db_session,
            pattern=None,
            limit=20,
            offset=0,
        )

        assert result.tags == []
        assert result.total_count == 0
        assert result.has_more is False

    def test_unique_tags(self, db_session):
        """Test extracting unique tags from documents."""
        doc1 = Document(
            id=uuid.uuid4(),
            file_path="/data/vault/doc1.md",
            file_name="doc1.md",
            content="# Doc1",
            checksum_md5="abc123",
            created_at_fs=datetime.now(),
            modified_at_fs=datetime.now(),
            tags=["work", "urgent"],
        )
        doc2 = Document(
            id=uuid.uuid4(),
            file_path="/data/vault/doc2.md",
            file_name="doc2.md",
            content="# Doc2",
            checksum_md5="def456",
            created_at_fs=datetime.now(),
            modified_at_fs=datetime.now(),
            tags=["work", "personal"],
        )
        db_session.add_all([doc1, doc2])
        db_session.commit()

        result = get_all_tags(
            db_session,
            pattern=None,
            limit=20,
            offset=0,
        )

        assert result.total_count == 3
        assert sorted(result.tags) == ["personal", "urgent", "work"]

    def test_pattern_filtering(self, db_session):
        """Test filtering tags with glob pattern."""
        doc = Document(
            id=uuid.uuid4(),
            file_path="/data/vault/doc.md",
            file_name="doc.md",
            content="# Doc",
            checksum_md5="abc123",
            created_at_fs=datetime.now(),
            modified_at_fs=datetime.now(),
            tags=["work", "workplace", "personal", "fun"],
        )
        db_session.add(doc)
        db_session.commit()

        result = get_all_tags(
            db_session,
            pattern="work*",
            limit=20,
            offset=0,
        )

        assert result.total_count == 2
        assert sorted(result.tags) == ["work", "workplace"]

    def test_pagination(self, db_session):
        """Test pagination with limit and offset."""
        doc = Document(
            id=uuid.uuid4(),
            file_path="/data/vault/doc.md",
            file_name="doc.md",
            content="# Doc",
            checksum_md5="abc123",
            created_at_fs=datetime.now(),
            modified_at_fs=datetime.now(),
            tags=["a", "b", "c", "d", "e"],
        )
        db_session.add(doc)
        db_session.commit()

        result = get_all_tags(
            db_session,
            pattern=None,
            limit=2,
            offset=0,
        )

        assert result.total_count == 5
        assert len(result.tags) == 2
        assert result.has_more is True
        assert result.next_offset == 2

    def test_excludes_null_tags(self, db_session):
        """Test that documents with null tags are excluded."""
        doc1 = Document(
            id=uuid.uuid4(),
            file_path="/data/vault/tagged.md",
            file_name="tagged.md",
            content="# Tagged",
            checksum_md5="abc123",
            created_at_fs=datetime.now(),
            modified_at_fs=datetime.now(),
            tags=["work"],
        )
        doc2 = Document(
            id=uuid.uuid4(),
            file_path="/data/vault/untagged.md",
            file_name="untagged.md",
            content="# Untagged",
            checksum_md5="def456",
            created_at_fs=datetime.now(),
            modified_at_fs=datetime.now(),
            tags=None,
        )
        db_session.add_all([doc1, doc2])
        db_session.commit()

        result = get_all_tags(
            db_session,
            pattern=None,
            limit=20,
            offset=0,
        )

        assert result.total_count == 1
        assert result.tags == ["work"]
