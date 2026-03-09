"""Tests for documents_filters module.

Tests for property filtering utilities in documents_filters.py.
"""

from unittest.mock import MagicMock, Mock, patch

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from obsidian_rag.database.models import Base, Document
from obsidian_rag.mcp_server.models import PropertyFilter
from obsidian_rag.mcp_server.tools.documents_filters import (
    apply_postgresql_property_filter,
    build_equals_condition,
    build_exists_condition,
    build_in_condition,
    build_like_condition,
    build_regex_condition,
    check_contains,
    check_equals,
    check_in_list,
    check_regex,
    check_starts_with,
    get_jsonb_path_expression,
    get_nested_value,
    matches_property_filter,
    validate_property_filter,
    validate_property_path,
)


class TestValidatePropertyPath:
    """Tests for validate_property_path function (TASK-071, TASK-072, TASK-073)."""

    def test_validate_property_path_with_empty_path_error(self):
        """Test validate_property_path with empty path error (TASK-071)."""
        with pytest.raises(ValueError, match="Property path cannot be empty"):
            validate_property_path("")

    def test_validate_property_path_with_too_many_levels_error(self):
        """Test validate_property_path with too many levels error (TASK-072)."""
        with pytest.raises(ValueError, match="Property path cannot exceed 3 levels"):
            validate_property_path("a.b.c.d")

    def test_validate_property_path_with_invalid_characters_error(self):
        """Test validate_property_path with invalid characters error (TASK-073)."""
        with pytest.raises(ValueError, match="Invalid property path segment"):
            validate_property_path("path@invalid")

    def test_validate_property_path_valid(self):
        """Test validate_property_path with valid paths."""
        # Should not raise
        validate_property_path("name")
        validate_property_path("author.name")
        validate_property_path("author.contact.email")
        validate_property_path("snake_case")


class TestValidatePropertyFilter:
    """Tests for validate_property_filter function (TASK-074)."""

    def test_validate_property_filter_with_various_operators(self):
        """Test validate_property_filter with various operators (TASK-074)."""
        from typing import Literal

        operators: list[
            Literal["equals", "contains", "exists", "in", "starts_with", "regex"]
        ] = ["equals", "contains", "exists", "in", "starts_with", "regex"]

        for operator in operators:
            filter_obj = PropertyFilter(path="status", operator=operator, value="draft")
            # Should not raise
            validate_property_filter(filter_obj)

    def test_validate_property_filter_with_invalid_operator(self):
        """Test validate_property_filter with invalid operator."""
        # Create a mock filter with invalid operator since PropertyFilter validates at creation
        mock_filter = MagicMock()
        mock_filter.path = "status"
        mock_filter.operator = "invalid_operator"
        mock_filter.value = "draft"
        with pytest.raises(ValueError, match="Invalid operator"):
            validate_property_filter(mock_filter)


class TestGetJsonbPathExpression:
    """Tests for get_jsonb_path_expression function (TASK-075)."""

    def test_get_jsonb_path_expression_simple(self):
        """Test get_jsonb_path_expression with simple path."""
        result = get_jsonb_path_expression("name")
        assert result == "frontmatter_json->>'name'"

    def test_get_jsonb_path_expression_nested(self):
        """Test get_jsonb_path_expression with nested path (TASK-075)."""
        result = get_jsonb_path_expression("author.name")
        assert result == "frontmatter_json->'author'->>'name'"

    def test_get_jsonb_path_expression_three_levels(self):
        """Test get_jsonb_path_expression with three level path."""
        result = get_jsonb_path_expression("a.b.c")
        assert result == "frontmatter_json->'a'->'b'->>'c'"


class TestBuildExistsCondition:
    """Tests for build_exists_condition function (TASK-077)."""

    def test_build_exists_condition_simple(self):
        """Test build_exists_condition with simple path."""
        result = build_exists_condition("frontmatter_json->>'name'", "name")
        # Should return a SQLAlchemy text object with bindparams
        assert result is not None

    def test_build_exists_condition_nested(self):
        """Test build_exists_condition with nested path."""
        result = build_exists_condition(
            "frontmatter_json->'author'->>'name'", "author.name"
        )
        # Should return a SQLAlchemy text object
        assert result is not None


class TestBuildEqualsCondition:
    """Tests for build_equals_condition function."""

    def test_build_equals_condition_with_value(self):
        """Test build_equals_condition with value."""
        result = build_equals_condition("frontmatter_json->>'status'", "draft")
        assert result is not None

    def test_build_equals_condition_with_none(self):
        """Test build_equals_condition with None value."""
        result = build_equals_condition("frontmatter_json->>'status'", None)
        # Should return IS NULL condition
        assert result is not None


class TestBuildLikeCondition:
    """Tests for build_like_condition function."""

    def test_build_like_condition(self):
        """Test build_like_condition."""
        result = build_like_condition("frontmatter_json->>'title'", "%test%")
        assert result is not None


class TestBuildRegexCondition:
    """Tests for build_regex_condition function (TASK-080)."""

    def test_build_regex_condition_valid_pattern(self):
        """Test build_regex_condition with valid pattern (TASK-080)."""
        result = build_regex_condition("frontmatter_json->>'email'", r"@example\.com$")
        assert result is not None

    def test_build_regex_condition_invalid_pattern(self):
        """Test build_regex_condition with invalid pattern (TASK-080)."""
        with pytest.raises(ValueError, match="Invalid regex pattern"):
            build_regex_condition("frontmatter_json->>'email'", "[invalid")


class TestBuildInCondition:
    """Tests for build_in_condition function (TASK-078, TASK-084)."""

    def test_build_in_condition_with_list(self):
        """Test build_in_condition with list (TASK-078, TASK-084)."""
        result = build_in_condition(
            "frontmatter_json->>'status'", ["draft", "published"]
        )
        assert result is not None

    def test_build_in_condition_with_non_list(self):
        """Test build_in_condition with non-list value (TASK-084)."""
        with pytest.raises(ValueError, match="'in' operator requires a list value"):
            build_in_condition("frontmatter_json->>'status'", "draft")


class TestApplyPostgresqlPropertyFilter:
    """Tests for apply_postgresql_property_filter function (TASK-076, TASK-077, TASK-078, TASK-079, TASK-080)."""

    def test_apply_postgresql_property_filter_contains(self):
        """Test apply_postgresql_property_filter with 'contains' operator (TASK-076)."""
        mock_query = MagicMock()
        mock_query.filter.return_value = mock_query

        filter_obj = PropertyFilter(path="title", operator="contains", value="Test")
        result = apply_postgresql_property_filter(mock_query, filter_obj)

        mock_query.filter.assert_called_once()
        assert result is mock_query

    def test_apply_postgresql_property_filter_exists(self):
        """Test apply_postgresql_property_filter with 'exists' operator (TASK-077)."""
        mock_query = MagicMock()
        mock_query.filter.return_value = mock_query

        filter_obj = PropertyFilter(path="status", operator="exists")
        result = apply_postgresql_property_filter(mock_query, filter_obj)

        mock_query.filter.assert_called_once()
        assert result is mock_query

    def test_apply_postgresql_property_filter_in(self):
        """Test apply_postgresql_property_filter with 'in' operator (TASK-078)."""
        mock_query = MagicMock()
        mock_query.filter.return_value = mock_query

        filter_obj = PropertyFilter(
            path="status", operator="in", value=["draft", "published"]
        )
        result = apply_postgresql_property_filter(mock_query, filter_obj)

        mock_query.filter.assert_called_once()
        assert result is mock_query

    def test_apply_postgresql_property_filter_starts_with(self):
        """Test apply_postgresql_property_filter with 'starts_with' operator (TASK-079)."""
        mock_query = MagicMock()
        mock_query.filter.return_value = mock_query

        filter_obj = PropertyFilter(path="title", operator="starts_with", value="Draft")
        result = apply_postgresql_property_filter(mock_query, filter_obj)

        mock_query.filter.assert_called_once()
        assert result is mock_query

    def test_apply_postgresql_property_filter_regex(self):
        """Test apply_postgresql_property_filter with 'regex' operator (TASK-080)."""
        mock_query = MagicMock()
        mock_query.filter.return_value = mock_query

        filter_obj = PropertyFilter(
            path="email", operator="regex", value=r"@example\.com$"
        )
        result = apply_postgresql_property_filter(mock_query, filter_obj)

        mock_query.filter.assert_called_once()
        assert result is mock_query

    def test_apply_postgresql_property_filter_equals(self):
        """Test apply_postgresql_property_filter with 'equals' operator."""
        mock_query = MagicMock()
        mock_query.filter.return_value = mock_query

        filter_obj = PropertyFilter(path="status", operator="equals", value="draft")
        result = apply_postgresql_property_filter(mock_query, filter_obj)

        mock_query.filter.assert_called_once()
        assert result is mock_query


class TestCheckFunctions:
    """Tests for check_* helper functions."""

    def test_check_equals_case_insensitive(self):
        """Test check_equals is case-insensitive."""
        assert check_equals("DRAFT", "draft") is True
        assert check_equals("draft", "DRAFT") is True
        assert check_equals("draft", "published") is False

    def test_check_equals_with_none(self):
        """Test check_equals with None."""
        assert check_equals(None, None) is True
        assert check_equals("value", None) is False

    def test_check_contains(self):
        """Test check_contains."""
        assert check_contains("Hello World", "world") is True
        assert check_contains("Hello World", "foo") is False
        assert check_contains(None, "test") is False

    def test_check_starts_with(self):
        """Test check_starts_with (TASK-082)."""
        assert check_starts_with("Draft Document", "draft") is True
        assert check_starts_with("Draft Document", "published") is False
        assert check_starts_with(None, "draft") is False

    def test_check_regex(self):
        """Test check_regex (TASK-083)."""
        assert check_regex("john@example.com", r"@example\.com$") is True
        assert check_regex("john@other.com", r"@example\.com$") is False
        assert check_regex(None, r"test") is False

    def test_check_regex_invalid_pattern(self):
        """Test check_regex with invalid pattern."""
        # Should return False for invalid regex
        assert check_regex("test", "[invalid") is False

    def test_check_in_list(self):
        """Test check_in_list (TASK-084)."""
        assert check_in_list("draft", ["draft", "published"]) is True
        assert check_in_list("archived", ["draft", "published"]) is False
        assert check_in_list(None, ["draft", "published"]) is False
        assert check_in_list("draft", "not-a-list") is False


class TestMatchesPropertyFilter:
    """Tests for matches_property_filter function (TASK-081, TASK-082, TASK-083, TASK-084, TASK-085)."""

    def test_matches_property_filter_exists(self):
        """Test matches_property_filter with 'exists' operator (TASK-081)."""
        doc = MagicMock()
        doc.frontmatter_json = {"status": "draft"}

        filter_obj = PropertyFilter(path="status", operator="exists")
        assert matches_property_filter(doc, filter_obj) is True

        doc.frontmatter_json = {}
        assert matches_property_filter(doc, filter_obj) is False

    def test_matches_property_filter_starts_with(self):
        """Test matches_property_filter with 'starts_with' operator (TASK-082)."""
        doc = MagicMock()
        doc.frontmatter_json = {"title": "Draft Document"}

        filter_obj = PropertyFilter(path="title", operator="starts_with", value="Draft")
        assert matches_property_filter(doc, filter_obj) is True

        filter_obj2 = PropertyFilter(
            path="title", operator="starts_with", value="Published"
        )
        assert matches_property_filter(doc, filter_obj2) is False

    def test_matches_property_filter_regex(self):
        """Test matches_property_filter with 'regex' operator (TASK-083)."""
        doc = MagicMock()
        doc.frontmatter_json = {"email": "john@example.com"}

        filter_obj = PropertyFilter(
            path="email", operator="regex", value=r"@example\.com$"
        )
        assert matches_property_filter(doc, filter_obj) is True

    def test_matches_property_filter_in(self):
        """Test matches_property_filter with 'in' operator (TASK-084)."""
        doc = MagicMock()
        doc.frontmatter_json = {"status": "draft"}

        filter_obj = PropertyFilter(
            path="status", operator="in", value=["draft", "published"]
        )
        assert matches_property_filter(doc, filter_obj) is True

        filter_obj2 = PropertyFilter(
            path="status", operator="in", value=["archived", "deleted"]
        )
        assert matches_property_filter(doc, filter_obj2) is False

    def test_matches_property_filter_unknown_operator(self):
        """Test matches_property_filter with unknown operator (TASK-085)."""
        doc = MagicMock()
        doc.frontmatter_json = {"status": "draft"}

        # Create a mock filter with unknown operator since PropertyFilter validates at creation
        mock_filter = MagicMock()
        mock_filter.path = "status"
        mock_filter.operator = "unknown_operator"
        mock_filter.value = "draft"
        assert matches_property_filter(doc, mock_filter) is False


class TestApplyPostgresqlPropertyFilterAdditional:
    """Additional tests for apply_postgresql_property_filter (TASK-103)."""

    def test_apply_postgresql_property_filter_unknown_operator(self):
        """Test apply_postgresql_property_filter with unknown operator (line 226)."""
        from obsidian_rag.mcp_server.tools.documents_filters import (
            apply_postgresql_property_filter,
        )

        mock_query = MagicMock()

        # Create a mock filter with unknown operator
        mock_filter = MagicMock()
        mock_filter.path = "status"
        mock_filter.operator = "unknown_operator"
        mock_filter.value = "draft"

        result = apply_postgresql_property_filter(mock_query, mock_filter)

        # Should return query unchanged when operator is unknown
        assert result is mock_query
        mock_query.filter.assert_not_called()


class TestGetNestedValueAdditional:
    """Additional tests for get_nested_value (TASK-103)."""

    def test_get_nested_value_none_data(self):
        """Test get_nested_value with None data (line 246)."""
        from obsidian_rag.mcp_server.tools.documents_filters import get_nested_value

        result = get_nested_value(None, "status")
        assert result is None

    def test_get_nested_value_non_dict_current(self):
        """Test get_nested_value when current is not a dict (line 247)."""
        from obsidian_rag.mcp_server.tools.documents_filters import get_nested_value

        # Try to access nested path on a string value
        data = {"name": "John"}  # "name" maps to a string, not a dict
        result = get_nested_value(data, "name.first")
        assert result is None


class TestKindFiltering:
    """Tests for filtering documents by kind using property filters."""

    @pytest.fixture
    def db_session(self):
        """Create a test database session."""
        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(engine)
        Session = sessionmaker(bind=engine)
        session = Session()
        yield session
        session.close()
        Base.metadata.drop_all(engine)

    def test_filter_by_kind_equals(self, db_session):
        """Test filtering documents by kind with equals operator."""
        from datetime import datetime
        from obsidian_rag.database.models import Vault
        from obsidian_rag.mcp_server.models import PropertyFilter
        from obsidian_rag.mcp_server.tools.documents_filters import (
            matches_property_filter,
        )

        # Create vault
        vault = Vault(
            name="Test Vault",
            container_path="/test",
            host_path="/test",
        )
        db_session.add(vault)
        db_session.flush()

        # Create documents with different kinds
        doc1 = Document(
            vault_id=vault.id,
            file_path="note1.md",
            file_name="note1.md",
            content="Content 1",
            checksum_md5="abc123",
            created_at_fs=datetime.now(),
            modified_at_fs=datetime.now(),
            frontmatter_json={"kind": "note"},
        )
        doc2 = Document(
            vault_id=vault.id,
            file_path="article1.md",
            file_name="article1.md",
            content="Content 2",
            checksum_md5="def456",
            created_at_fs=datetime.now(),
            modified_at_fs=datetime.now(),
            frontmatter_json={"kind": "article"},
        )
        doc3 = Document(
            vault_id=vault.id,
            file_path="no_kind.md",
            file_name="no_kind.md",
            content="Content 3",
            checksum_md5="ghi789",
            created_at_fs=datetime.now(),
            modified_at_fs=datetime.now(),
            frontmatter_json={},
        )
        db_session.add_all([doc1, doc2, doc3])
        db_session.commit()

        # Filter by kind=note using matches_property_filter
        filter_obj = PropertyFilter(path="kind", operator="equals", value="note")
        results = [
            doc
            for doc in [doc1, doc2, doc3]
            if matches_property_filter(doc, filter_obj)
        ]

        assert len(results) == 1
        assert results[0].file_path == "note1.md"

    def test_filter_by_kind_exists(self, db_session):
        """Test filtering documents by kind with exists operator."""
        from datetime import datetime
        from obsidian_rag.database.models import Vault
        from obsidian_rag.mcp_server.models import PropertyFilter
        from obsidian_rag.mcp_server.tools.documents_filters import (
            matches_property_filter,
        )

        # Create vault and documents
        vault = Vault(
            name="Test Vault",
            container_path="/test",
            host_path="/test",
        )
        db_session.add(vault)
        db_session.flush()

        doc1 = Document(
            vault_id=vault.id,
            file_path="with_kind.md",
            file_name="with_kind.md",
            content="Content",
            checksum_md5="abc123",
            created_at_fs=datetime.now(),
            modified_at_fs=datetime.now(),
            frontmatter_json={"kind": "note"},
        )
        doc2 = Document(
            vault_id=vault.id,
            file_path="without_kind.md",
            file_name="without_kind.md",
            content="Content",
            checksum_md5="def456",
            created_at_fs=datetime.now(),
            modified_at_fs=datetime.now(),
            frontmatter_json={"other": "value"},
        )
        db_session.add_all([doc1, doc2])
        db_session.commit()

        # Filter by kind exists using matches_property_filter
        filter_obj = PropertyFilter(path="kind", operator="exists")
        results = [
            doc for doc in [doc1, doc2] if matches_property_filter(doc, filter_obj)
        ]

        assert len(results) == 1
        assert results[0].file_path == "with_kind.md"

    def test_filter_by_kind_contains(self, db_session):
        """Test filtering documents by kind with contains operator."""
        from datetime import datetime
        from obsidian_rag.database.models import Vault
        from obsidian_rag.mcp_server.models import PropertyFilter
        from obsidian_rag.mcp_server.tools.documents_filters import (
            matches_property_filter,
        )

        # Create vault and documents
        vault = Vault(
            name="Test Vault",
            container_path="/test",
            host_path="/test",
        )
        db_session.add(vault)
        db_session.flush()

        doc1 = Document(
            vault_id=vault.id,
            file_path="daily_note.md",
            file_name="daily_note.md",
            content="Content",
            checksum_md5="abc123",
            created_at_fs=datetime.now(),
            modified_at_fs=datetime.now(),
            frontmatter_json={"kind": "daily_note"},
        )
        doc2 = Document(
            vault_id=vault.id,
            file_path="meeting_note.md",
            file_name="meeting_note.md",
            content="Content",
            checksum_md5="def456",
            created_at_fs=datetime.now(),
            modified_at_fs=datetime.now(),
            frontmatter_json={"kind": "meeting_note"},
        )
        db_session.add_all([doc1, doc2])
        db_session.commit()

        # Filter by kind contains "daily" using matches_property_filter
        filter_obj = PropertyFilter(path="kind", operator="contains", value="daily")
        results = [
            doc for doc in [doc1, doc2] if matches_property_filter(doc, filter_obj)
        ]

        assert len(results) == 1
        assert results[0].file_path == "daily_note.md"
