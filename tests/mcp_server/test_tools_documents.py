"""Unit tests for MCP document tools with enhanced filtering."""

import uuid
from datetime import datetime
from unittest.mock import MagicMock

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from obsidian_rag.database.models import Base, Document
from obsidian_rag.mcp_server.models import PropertyFilter, TagFilter
from obsidian_rag.mcp_server.tools.documents import (
    _get_relative_path,
    _glob_to_like,
    get_all_tags,
    get_documents_by_property,
    get_documents_by_tag,
    query_documents,
)
from obsidian_rag.mcp_server.tools.documents_params import (
    PaginationParams,
    PropertyFilterParams,
)
from obsidian_rag.mcp_server.tools.documents_filters import (
    get_nested_value as _get_nested_value,
    matches_property_filter as _matches_property_filter,
    validate_property_filters as _validate_property_filters,
)
from obsidian_rag.mcp_server.tools.documents_tags import (
    _matches_all_tags,
    _matches_any_tags,
    _matches_glob,
    matches_tag_filter as _matches_tag_filter,
    validate_tag_filter as _validate_tag_filter,
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


@pytest.fixture
def sample_documents(db_session):
    """Create sample documents for testing."""
    docs = [
        Document(
            id=uuid.uuid4(),
            file_path="/data/vault/work.md",
            file_name="work.md",
            content="# Work Document",
            checksum_md5="abc123",
            created_at_fs=datetime.now(),
            modified_at_fs=datetime.now(),
            tags=["work", "urgent"],
            frontmatter_json={
                "title": "Work",
                "author": {"name": "John", "email": "john@example.com"},
                "status": "draft",
                "priority": 1,
            },
            vault_root="/data/vault",
        ),
        Document(
            id=uuid.uuid4(),
            file_path="/data/vault/personal.md",
            file_name="personal.md",
            content="# Personal Document",
            checksum_md5="def456",
            created_at_fs=datetime.now(),
            modified_at_fs=datetime.now(),
            tags=["personal", "ideas"],
            frontmatter_json={
                "title": "Personal",
                "author": {"name": "Jane", "email": "jane@example.com"},
                "status": "published",
                "priority": 2,
            },
            vault_root="/data/vault",
        ),
        Document(
            id=uuid.uuid4(),
            file_path="/data/vault/mixed.md",
            file_name="mixed.md",
            content="# Mixed Document",
            checksum_md5="ghi789",
            created_at_fs=datetime.now(),
            modified_at_fs=datetime.now(),
            tags=["work", "ideas"],
            frontmatter_json={
                "title": "Mixed",
                "author": {"name": "John", "email": "john2@example.com"},
                "status": "draft",
                "priority": 3,
            },
            vault_root="/data/vault",
        ),
        Document(
            id=uuid.uuid4(),
            file_path="/data/vault/untagged.md",
            file_name="untagged.md",
            content="# Untagged Document",
            checksum_md5="jkl012",
            created_at_fs=datetime.now(),
            modified_at_fs=datetime.now(),
            tags=None,
            frontmatter_json={"title": "Untagged"},
            vault_root="/data/vault",
        ),
    ]
    db_session.add_all(docs)
    db_session.commit()
    return docs


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

    def test_limit_validation(self, db_session):
        """Test that limit is validated and clamped."""
        query_embedding = [0.1] * 1536

        # Test limit above maximum (should work but be clamped internally)
        pagination = PaginationParams(limit=200, offset=0)
        result = query_documents(db_session, query_embedding, pagination=pagination)
        assert result.total_count == 0  # No documents

    def test_offset_validation(self, db_session):
        """Test that offset is validated."""
        query_embedding = [0.1] * 1536

        # Test negative offset (should be clamped to 0)
        pagination = PaginationParams(limit=20, offset=-10)
        result = query_documents(db_session, query_embedding, pagination=pagination)
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


class TestTagFilterHelpers:
    """Tests for tag filter helper functions."""

    def test_matches_all_tags_all_match(self):
        """Test _matches_all_tags when all tags match."""
        doc = MagicMock()
        doc.tags = ["work", "urgent", "ideas"]
        assert _matches_all_tags(doc, ["work", "urgent"]) is True

    def test_matches_all_tags_some_missing(self):
        """Test _matches_all_tags when some tags are missing."""
        doc = MagicMock()
        doc.tags = ["work", "ideas"]
        assert _matches_all_tags(doc, ["work", "urgent"]) is False

    def test_matches_all_tags_empty_list(self):
        """Test _matches_all_tags with empty list."""
        doc = MagicMock()
        doc.tags = ["work"]
        assert _matches_all_tags(doc, []) is True

    def test_matches_all_tags_no_doc_tags(self):
        """Test _matches_all_tags when document has no tags."""
        doc = MagicMock()
        doc.tags = None
        assert _matches_all_tags(doc, ["work"]) is False

    def test_matches_any_tags_one_matches(self):
        """Test _matches_any_tags when one tag matches."""
        doc = MagicMock()
        doc.tags = ["work", "ideas"]
        assert _matches_any_tags(doc, ["work", "urgent"]) is True

    def test_matches_any_tags_none_match(self):
        """Test _matches_any_tags when no tags match."""
        doc = MagicMock()
        doc.tags = ["personal", "ideas"]
        assert _matches_any_tags(doc, ["work", "urgent"]) is False

    def test_matches_any_tags_empty_list(self):
        """Test _matches_any_tags with empty list."""
        doc = MagicMock()
        doc.tags = ["work"]
        assert _matches_any_tags(doc, []) is True


class TestMatchesGlob:
    """Tests for _matches_glob function."""

    def test_matches_glob_star(self):
        """Test glob matching with star."""
        assert _matches_glob("work", "wo*") is True
        assert _matches_glob("work", "*rk") is True
        assert _matches_glob("workplace", "work*") is True

    def test_matches_glob_question(self):
        """Test glob matching with question mark."""
        assert _matches_glob("work", "wo?k") is True
        assert _matches_glob("work", "wor?") is True

    def test_matches_glob_case_insensitive(self):
        """Test glob matching is case-insensitive."""
        assert _matches_glob("WORK", "work*") is True
        assert _matches_glob("work", "WORK*") is True


class TestTagFilterValidation:
    """Tests for tag filter validation."""

    def test_validate_tag_filter_none(self):
        """Test validation with None filter."""
        # Should not raise
        _validate_tag_filter(None)

    def test_validate_tag_filter_valid(self):
        """Test validation with valid filter."""
        tag_filter = TagFilter(include_tags=["work"], exclude_tags=["archived"])
        # Should not raise
        _validate_tag_filter(tag_filter)

    def test_validate_tag_filter_conflicting_tags(self):
        """Test validation with conflicting tags."""
        tag_filter = TagFilter(
            include_tags=["work", "urgent"], exclude_tags=["work", "archived"]
        )
        with pytest.raises(ValueError) as exc_info:
            _validate_tag_filter(tag_filter)
        assert "Conflicting tags" in str(exc_info.value)

    def test_validate_tag_filter_too_many_tags(self):
        """Test validation with too many tags."""
        tag_filter = TagFilter(include_tags=[f"tag{i}" for i in range(60)])
        with pytest.raises(ValueError) as exc_info:
            _validate_tag_filter(tag_filter)
        assert "Maximum" in str(exc_info.value)


class TestPropertyFilterHelpers:
    """Tests for property filter helper functions."""

    def test_get_nested_value_simple(self):
        """Test _get_nested_value with simple path."""
        data = {"name": "John", "age": 30}
        assert _get_nested_value(data, "name") == "John"

    def test_get_nested_value_nested(self):
        """Test _get_nested_value with nested path."""
        data = {"author": {"name": "John", "email": "john@example.com"}}
        assert _get_nested_value(data, "author.name") == "John"

    def test_get_nested_value_missing(self):
        """Test _get_nested_value with missing path."""
        data = {"author": {"name": "John"}}
        assert _get_nested_value(data, "author.email") is None

    def test_get_nested_value_none_data(self):
        """Test _get_nested_value with None data."""
        assert _get_nested_value(None, "name") is None

    def test_matches_property_filter_equals(self):
        """Test _matches_property_filter with equals operator."""
        doc = MagicMock()
        doc.frontmatter_json = {"status": "draft"}
        filter_obj = PropertyFilter(path="status", operator="equals", value="draft")
        assert _matches_property_filter(doc, filter_obj) is True

    def test_matches_property_filter_equals_case_insensitive(self):
        """Test _matches_property_filter with equals operator (case-insensitive)."""
        doc = MagicMock()
        doc.frontmatter_json = {"status": "DRAFT"}
        filter_obj = PropertyFilter(path="status", operator="equals", value="draft")
        assert _matches_property_filter(doc, filter_obj) is True

    def test_matches_property_filter_contains(self):
        """Test _matches_property_filter with contains operator."""
        doc = MagicMock()
        doc.frontmatter_json = {"title": "My Document"}
        filter_obj = PropertyFilter(path="title", operator="contains", value="Doc")
        assert _matches_property_filter(doc, filter_obj) is True

    def test_matches_property_filter_exists(self):
        """Test _matches_property_filter with exists operator."""
        doc = MagicMock()
        doc.frontmatter_json = {"status": "draft"}
        filter_obj = PropertyFilter(path="status", operator="exists")
        assert _matches_property_filter(doc, filter_obj) is True

    def test_matches_property_filter_starts_with(self):
        """Test _matches_property_filter with starts_with operator."""
        doc = MagicMock()
        doc.frontmatter_json = {"title": "Draft Document"}
        filter_obj = PropertyFilter(path="title", operator="starts_with", value="Draft")
        assert _matches_property_filter(doc, filter_obj) is True

    def test_matches_property_filter_regex(self):
        """Test _matches_property_filter with regex operator."""
        doc = MagicMock()
        doc.frontmatter_json = {"email": "john@example.com"}
        filter_obj = PropertyFilter(
            path="email", operator="regex", value=r"@example\.com$"
        )
        assert _matches_property_filter(doc, filter_obj) is True

    def test_matches_property_filter_in(self):
        """Test _matches_property_filter with in operator."""
        doc = MagicMock()
        doc.frontmatter_json = {"status": "draft"}
        filter_obj = PropertyFilter(
            path="status", operator="in", value=["draft", "review"]
        )
        assert _matches_property_filter(doc, filter_obj) is True

    def test_matches_property_filter_nested_path(self):
        """Test _matches_property_filter with nested path."""
        doc = MagicMock()
        doc.frontmatter_json = {"author": {"name": "John"}}
        filter_obj = PropertyFilter(path="author.name", operator="equals", value="John")
        assert _matches_property_filter(doc, filter_obj) is True


class TestPropertyFilterValidation:
    """Tests for property filter validation."""

    def test_validate_property_filters_none(self):
        """Test validation with None filters."""
        # Should not raise
        _validate_property_filters(None)

    def test_validate_property_filters_valid(self):
        """Test validation with valid filters."""
        filters = [PropertyFilter(path="status", operator="equals", value="draft")]
        # Should not raise
        _validate_property_filters(filters)

    def test_validate_property_filters_too_many(self):
        """Test validation with too many filters."""
        filters = [
            PropertyFilter(path=f"field{i}", operator="equals", value="x")
            for i in range(15)
        ]
        with pytest.raises(ValueError) as exc_info:
            _validate_property_filters(filters)
        assert "Maximum" in str(exc_info.value)

    def test_validate_property_filters_invalid_operator(self):
        """Test validation with invalid operator."""
        # Create a filter with valid operator first, then manually set invalid operator
        filter_obj = PropertyFilter(path="status", operator="equals", value="draft")
        # Manually override operator to invalid value for testing
        filter_obj.operator = "invalid"  # type: ignore[assignment]
        filters = [filter_obj]
        with pytest.raises(ValueError) as exc_info:
            _validate_property_filters(filters)
        assert "Invalid operator" in str(exc_info.value)


class TestMatchesTagFilter:
    """Tests for _matches_tag_filter function."""

    def test_matches_tag_filter_none(self):
        """Test with None filter."""
        doc = MagicMock()
        doc.tags = ["work"]
        assert _matches_tag_filter(doc, None) is True

    def test_matches_tag_filter_include_all(self):
        """Test include all tags."""
        doc = MagicMock()
        doc.tags = ["work", "urgent"]
        tag_filter = TagFilter(include_tags=["work", "urgent"], match_mode="all")
        assert _matches_tag_filter(doc, tag_filter) is True

    def test_matches_tag_filter_include_any(self):
        """Test include any tag."""
        doc = MagicMock()
        doc.tags = ["work", "ideas"]
        tag_filter = TagFilter(include_tags=["work", "urgent"], match_mode="any")
        assert _matches_tag_filter(doc, tag_filter) is True

    def test_matches_tag_filter_exclude(self):
        """Test exclude tags."""
        doc = MagicMock()
        doc.tags = ["work", "urgent"]
        tag_filter = TagFilter(exclude_tags=["archived"])
        assert _matches_tag_filter(doc, tag_filter) is True

    def test_matches_tag_filter_exclude_hit(self):
        """Test exclude tags when doc has excluded tag."""
        doc = MagicMock()
        doc.tags = ["work", "archived"]
        tag_filter = TagFilter(exclude_tags=["archived"])
        assert _matches_tag_filter(doc, tag_filter) is False


class TestGetDocumentsByTag:
    """Tests for get_documents_by_tag function with new signature."""

    def test_get_documents_by_tag_include_all(self, db_session, sample_documents):
        """Test filtering with include all tags."""
        tag_filter = TagFilter(include_tags=["work", "urgent"], match_mode="all")
        result = get_documents_by_tag(
            db_session,
            tag_filter=tag_filter,
            limit=20,
            offset=0,
        )
        assert result.total_count == 1
        assert result.results[0].file_name == "work.md"

    def test_get_documents_by_tag_include_any(self, db_session, sample_documents):
        """Test filtering with include any tags."""
        tag_filter = TagFilter(include_tags=["work", "personal"], match_mode="any")
        result = get_documents_by_tag(
            db_session,
            tag_filter=tag_filter,
            limit=20,
            offset=0,
        )
        assert result.total_count == 3  # work.md, personal.md, mixed.md

    def test_get_documents_by_tag_exclude(self, db_session, sample_documents):
        """Test filtering with exclude tags."""
        tag_filter = TagFilter(exclude_tags=["urgent"])
        result = get_documents_by_tag(
            db_session,
            tag_filter=tag_filter,
            limit=20,
            offset=0,
        )
        # Should exclude work.md which has "urgent" tag
        # personal.md, mixed.md, untagged.md don't have "urgent" tag
        assert result.total_count == 3  # personal.md, mixed.md, untagged.md

    def test_get_documents_by_tag_include_and_exclude(
        self, db_session, sample_documents
    ):
        """Test filtering with both include and exclude."""
        tag_filter = TagFilter(
            include_tags=["work"], exclude_tags=["urgent"], match_mode="all"
        )
        result = get_documents_by_tag(
            db_session,
            tag_filter=tag_filter,
            limit=20,
            offset=0,
        )
        # Should include docs with "work" but exclude those with "urgent"
        assert result.total_count == 1  # mixed.md

    def test_get_documents_by_tag_empty_filters(self, db_session, sample_documents):
        """Test filtering with empty filters."""
        tag_filter = TagFilter()
        result = get_documents_by_tag(
            db_session,
            tag_filter=tag_filter,
            limit=20,
            offset=0,
        )
        # Should return all documents
        assert result.total_count == 4

    def test_get_documents_by_tag_pagination(self, db_session, sample_documents):
        """Test pagination."""
        tag_filter = TagFilter(include_tags=["work"], match_mode="any")
        result = get_documents_by_tag(
            db_session,
            tag_filter=tag_filter,
            limit=1,
            offset=0,
        )
        # work.md and mixed.md have "work" tag (exact match)
        assert result.total_count == 2
        assert len(result.results) == 1
        assert result.has_more is True

    def test_get_documents_by_tag_relative_path(self, db_session, sample_documents):
        """Test that response includes relative paths."""
        tag_filter = TagFilter(include_tags=["work"])
        result = get_documents_by_tag(
            db_session,
            tag_filter=tag_filter,
            limit=20,
            offset=0,
        )
        assert len(result.results) > 0
        # Paths should be relative to vault_root
        for doc in result.results:
            assert doc.file_path.startswith("./")


class TestGetDocumentsByProperty:
    """Tests for get_documents_by_property function."""

    def test_get_documents_by_property_equals(self, db_session, sample_documents):
        """Test filtering with equals operator."""
        include_props = [
            PropertyFilter(path="status", operator="equals", value="draft")
        ]
        property_filters = PropertyFilterParams(
            include_filters=include_props, exclude_filters=None
        )
        pagination = PaginationParams(limit=20, offset=0)
        result = get_documents_by_property(
            db_session,
            property_filters=property_filters,
            pagination=pagination,
        )
        assert result.total_count == 2  # work.md, mixed.md

    def test_get_documents_by_property_contains(self, db_session, sample_documents):
        """Test filtering with contains operator."""
        include_props = [
            PropertyFilter(path="author.name", operator="contains", value="John")
        ]
        property_filters = PropertyFilterParams(
            include_filters=include_props, exclude_filters=None
        )
        pagination = PaginationParams(limit=20, offset=0)
        result = get_documents_by_property(
            db_session,
            property_filters=property_filters,
            pagination=pagination,
        )
        assert result.total_count == 2  # work.md, mixed.md

    def test_get_documents_by_property_exists(self, db_session, sample_documents):
        """Test filtering with exists operator."""
        include_props = [PropertyFilter(path="priority", operator="exists")]
        property_filters = PropertyFilterParams(
            include_filters=include_props, exclude_filters=None
        )
        pagination = PaginationParams(limit=20, offset=0)
        result = get_documents_by_property(
            db_session,
            property_filters=property_filters,
            pagination=pagination,
        )
        assert result.total_count == 3  # work.md, personal.md, mixed.md

    def test_get_documents_by_property_exclude(self, db_session, sample_documents):
        """Test filtering with exclude properties."""
        exclude_props = [
            PropertyFilter(path="status", operator="equals", value="draft")
        ]
        property_filters = PropertyFilterParams(
            include_filters=None, exclude_filters=exclude_props
        )
        pagination = PaginationParams(limit=20, offset=0)
        result = get_documents_by_property(
            db_session,
            property_filters=property_filters,
            pagination=pagination,
        )
        # Should exclude work.md and mixed.md (status=draft)
        assert result.total_count == 2  # personal.md, untagged.md

    def test_get_documents_by_property_with_tag_filter(
        self, db_session, sample_documents
    ):
        """Test filtering with both property and tag filters."""
        include_props = [
            PropertyFilter(path="status", operator="equals", value="draft")
        ]
        property_filters = PropertyFilterParams(
            include_filters=include_props, exclude_filters=None
        )
        tag_filter = TagFilter(include_tags=["urgent"])
        pagination = PaginationParams(limit=20, offset=0)
        result = get_documents_by_property(
            db_session,
            property_filters=property_filters,
            tag_filter=tag_filter,
            pagination=pagination,
        )
        # Should include docs with status=draft AND urgent tag
        assert result.total_count == 1  # work.md

    def test_get_documents_by_property_nested_path(self, db_session, sample_documents):
        """Test filtering with nested property path."""
        include_props = [
            PropertyFilter(path="author.name", operator="equals", value="John")
        ]
        property_filters = PropertyFilterParams(
            include_filters=include_props, exclude_filters=None
        )
        pagination = PaginationParams(limit=20, offset=0)
        result = get_documents_by_property(
            db_session,
            property_filters=property_filters,
            pagination=pagination,
        )
        assert result.total_count == 2  # work.md, mixed.md


class TestQueryDocumentsWithFilters:
    """Tests for query_documents with filters."""

    def test_query_documents_with_tag_filter(self, db_session, sample_documents):
        """Test query_documents with tag filter."""
        tag_filter = TagFilter(include_tags=["work"])
        pagination = PaginationParams(limit=20, offset=0)
        result = query_documents(
            db_session,
            query_embedding=[0.1] * 1536,
            tag_filter=tag_filter,
            pagination=pagination,
        )
        # For SQLite, returns filtered docs with similarity_score=0.0
        # work.md and mixed.md have "work" tag (exact match)
        assert result.total_count == 2

    def test_query_documents_with_property_filter(self, db_session, sample_documents):
        """Test query_documents with property filter."""
        include_props = [
            PropertyFilter(path="status", operator="equals", value="draft")
        ]
        property_filters = PropertyFilterParams(
            include_filters=include_props, exclude_filters=None
        )
        pagination = PaginationParams(limit=20, offset=0)
        result = query_documents(
            db_session,
            query_embedding=[0.1] * 1536,
            filter_params=property_filters,
            pagination=pagination,
        )
        # work.md and mixed.md have status="draft"
        assert result.total_count == 2

    def test_query_documents_empty_result(self, db_session):
        """Test query_documents with empty database."""
        tag_filter = TagFilter(include_tags=["work"])
        pagination = PaginationParams(limit=20, offset=0)
        result = query_documents(
            db_session,
            query_embedding=[0.1] * 1536,
            tag_filter=tag_filter,
            pagination=pagination,
        )
        assert result.total_count == 0
        assert result.results == []


class TestGetAllTags:
    """Tests for get_all_tags function."""

    def test_get_all_tags_basic(self, db_session, sample_documents):
        """Test getting all unique tags."""
        result = get_all_tags(db_session, pattern=None, limit=20, offset=0)
        assert result.total_count == 4  # work, urgent, personal, ideas
        assert sorted(result.tags) == ["ideas", "personal", "urgent", "work"]

    def test_get_all_tags_with_pattern(self, db_session, sample_documents):
        """Test getting tags with glob pattern."""
        result = get_all_tags(db_session, pattern="work*", limit=20, offset=0)
        assert result.total_count == 1
        assert result.tags == ["work"]

    def test_get_all_tags_pagination(self, db_session, sample_documents):
        """Test tag pagination."""
        result = get_all_tags(db_session, pattern=None, limit=2, offset=0)
        assert result.total_count == 4
        assert len(result.tags) == 2
        assert result.has_more is True

    def test_get_all_tags_empty_database(self, db_session):
        """Test getting tags with empty database."""
        result = get_all_tags(db_session, pattern=None, limit=20, offset=0)
        assert result.total_count == 0
        assert result.tags == []


class TestGetRelativePathAdditional:
    """Additional tests for _get_relative_path (TASK-096)."""

    def test_get_relative_path_when_already_starts_with_dot_slash(self):
        """Test _get_relative_path when relative path already starts with ./ (lines 70-72)."""
        from obsidian_rag.mcp_server.tools.documents import _get_relative_path

        # If the relative path somehow already starts with ./, it should stay as is
        file_path = "/data/vault/./folder/note.md"
        vault_root = "/data/vault"
        result = _get_relative_path(file_path, vault_root)
        # The function extracts the relative part and ensures it starts with ./
        assert result.startswith("./")

    def test_get_relative_path_root_itself(self):
        """Test _get_relative_path for root path edge case."""
        from obsidian_rag.mcp_server.tools.documents import _get_relative_path

        file_path = "/data/vault/note.md"
        vault_root = "/data/vault"
        result = _get_relative_path(file_path, vault_root)
        assert result == "./note.md"


class TestGetDocumentsByTagAdditional:
    """Additional tests for get_documents_by_tag (TASK-096)."""

    def test_get_documents_by_tag_with_vault_root_filter(
        self, db_session, sample_documents
    ):
        """Test get_documents_by_tag with vault_root filter (line 254)."""
        tag_filter = TagFilter(include_tags=["work"], match_mode="any")
        result = get_documents_by_tag(
            db_session,
            tag_filter=tag_filter,
            vault_root="/data/vault",
            limit=20,
            offset=0,
        )
        # All sample documents have vault_root="/data/vault"
        # work.md and mixed.md have "work" tag
        assert result.total_count == 2


class TestGetDocumentsByPropertyAdditional:
    """Additional tests for get_documents_by_property (TASK-096)."""

    def test_get_documents_by_property_with_vault_root_filter(
        self, db_session, sample_documents
    ):
        """Test get_documents_by_property with vault_root filter (line 356)."""
        # Note: This tests the vault_root parameter is passed correctly
        # The actual filtering happens in the database-specific implementations
        include_props = [
            PropertyFilter(path="status", operator="equals", value="draft")
        ]
        property_filters = PropertyFilterParams(
            include_filters=include_props, exclude_filters=None
        )
        pagination = PaginationParams(limit=20, offset=0)
        result = get_documents_by_property(
            db_session,
            property_filters=property_filters,
            vault_root="/data/vault",
            pagination=pagination,
        )
        # work.md and mixed.md both have status=draft and vault_root=/data/vault
        assert result.total_count == 2
        file_names = {r.file_name for r in result.results}
        assert "work.md" in file_names
        assert "mixed.md" in file_names


class TestExtractTagsPostgresql:
    """Tests for _extract_tags_postgresql function (TASK-096)."""

    def test_extract_tags_postgresql_with_pattern(self):
        """Test _extract_tags_postgresql with pattern filtering (lines 377-390)."""
        from obsidian_rag.mcp_server.tools.documents import _extract_tags_postgresql

        mock_session = MagicMock()

        # Mock the query results
        mock_row1 = MagicMock()
        mock_row1.tag = "work"
        mock_row2 = MagicMock()
        mock_row2.tag = "workplace"

        mock_query = MagicMock()
        mock_query.filter.return_value = mock_query
        mock_query.order_by.return_value = mock_query
        mock_query.all.return_value = [mock_row1, mock_row2]

        mock_session.query.return_value = mock_query

        result = _extract_tags_postgresql(mock_session, pattern="work*")

        assert "work" in result
        assert "workplace" in result
        # Verify filter was called (at least once for the pattern)
        assert mock_query.filter.call_count >= 1


class TestGetAllTagsAdditional:
    """Additional tests for get_all_tags (TASK-096)."""

    def test_get_all_tags_execution_path(self, db_session, sample_documents):
        """Test get_all_tags execution path (line 455)."""
        # This test exercises the full execution path of get_all_tags
        result = get_all_tags(
            db_session,
            pattern=None,
            limit=10,
            offset=0,
        )
        # Should return all unique tags from sample documents
        assert result.total_count == 4
        assert sorted(result.tags) == ["ideas", "personal", "urgent", "work"]
        assert result.has_more is False
        assert result.next_offset is None
