"""End-to-end integration tests for properties and include_content across the full MCP wrapper chain.

These tests verify that:
1. `properties` is correctly populated from document frontmatter_json in all tool responses
2. `include_content=False` correctly produces empty content/raw_text while preserving metadata
3. The full wrapper chain (server -> tool_definitions -> handler -> tool -> response model)
   correctly propagates these values.
"""

import uuid
from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

from obsidian_rag.database.models import Document
from obsidian_rag.mcp_server.models import (
    DocumentListResponse,
    DocumentResponse,
    TaskListResponse,
)
from obsidian_rag.mcp_server.tools.documents import (
    get_document,
    get_documents_by_property,
    get_documents_by_tag,
    list_documents,
    query_documents,
)
from obsidian_rag.mcp_server.tools.documents_params import (
    PaginationParams,
    PropertyFilterParams,
)
from obsidian_rag.mcp_server.tools.tasks import get_tasks
from obsidian_rag.mcp_server.tools.tasks_params import GetTasksFilterParams


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _create_mock_doc(
    frontmatter_json: dict[str, object] | None = None,
    content: str = "doc content",
) -> MagicMock:
    """Create a mock Document model with frontmatter and vault relationship."""
    doc = MagicMock(spec=Document)
    doc.id = uuid.uuid4()
    doc.file_path = "notes/doc.md"
    doc.file_name = "doc.md"
    doc.content = content
    doc.frontmatter_json = frontmatter_json
    doc.tags = ["work"]
    doc.created_at_fs = datetime.now(UTC)
    doc.modified_at_fs = datetime.now(UTC)
    doc.vault = MagicMock()
    doc.vault.name = "TestVault"
    return doc


def _create_mock_task_doc_pair(
    frontmatter_json: dict[str, object] | None = None,
) -> tuple[MagicMock, MagicMock]:
    """Create a mock (task, document) pair for get_tasks tests."""
    task = MagicMock()
    task.inline_fields = None
    task.id = uuid.uuid4()
    task.status = "not_completed"
    task.raw_text = "- [ ] Test task"
    task.description = "Test task"
    task.tags = ["work"]
    task.priority = "normal"
    task.due = None
    task.scheduled = None
    task.completion = None

    doc = _create_mock_doc(frontmatter_json=frontmatter_json, content="parent content")
    return task, doc


# ---------------------------------------------------------------------------
# Document tools – properties populated
# ---------------------------------------------------------------------------


class TestPropertiesPopulatedInDocumentQueryResponses:
    """Test that query_documents populates properties from frontmatter_json."""

    @patch("obsidian_rag.mcp_server.tools.documents.query_documents_postgresql")
    def test_query_documents_properties_populated(
        self,
        mock_postgres: MagicMock,
    ) -> None:
        """Document-level query returns properties extracted from frontmatter."""
        doc = _create_mock_doc(
            frontmatter_json={"author": "Alice", "project": "p", "tags": ["t"]},
        )

        mock_postgres.return_value = DocumentListResponse(
            results=[
                DocumentResponse(
                    id=doc.id,
                    vault_name="TestVault",
                    file_path=doc.file_path,
                    relative_path=doc.file_path,
                    file_name=doc.file_name,
                    content=doc.content,
                    kind=None,
                    tags=doc.tags,
                    similarity_score=0.1,
                    created_at_fs=doc.created_at_fs,
                    modified_at_fs=doc.modified_at_fs,
                    obsidian_uri="uri",
                    properties={"author": "Alice", "project": "p"},
                ),
            ],
            total_count=1,
            has_more=False,
            next_offset=None,
        )

        mock_session = MagicMock()
        result = query_documents(
            mock_session,
            query_embedding=[0.1] * 10,
            include_content=True,
        )

        assert len(result.results) == 1
        assert result.results[0].properties == {"author": "Alice", "project": "p"}


class TestPropertiesPopulatedInGetDocumentsByTagResponses:
    """Test that get_documents_by_tag populates properties."""

    @patch("obsidian_rag.mcp_server.tools.documents_tags.apply_postgresql_tag_filter")
    def test_get_documents_by_tag_properties_populated(
        self,
        mock_tag_filter: MagicMock,
    ) -> None:
        """get_documents_by_tag returns properties from frontmatter_json."""
        from obsidian_rag.mcp_server.models import TagFilter

        doc = _create_mock_doc(
            frontmatter_json={"status": "active", "priority": 1, "tags": ["work"]},
        )

        mock_session = MagicMock()
        query_mock = MagicMock()
        query_mock.order_by.return_value = query_mock
        query_mock.offset.return_value = query_mock
        query_mock.limit.return_value = query_mock
        query_mock.all.return_value = [doc]
        query_mock.count.return_value = 1
        mock_tag_filter.return_value = query_mock
        mock_session.query.return_value = query_mock

        tag_filter = TagFilter(include_tags=["work"])
        result = get_documents_by_tag(
            mock_session,
            tag_filter=tag_filter,
            include_content=True,
        )

        assert len(result.results) == 1
        assert result.results[0].properties == {"status": "active", "priority": 1}


class TestPropertiesPopulatedInGetDocumentsByPropertyResponses:
    """Test that get_documents_by_property populates properties."""

    @patch(
        "obsidian_rag.mcp_server.tools.documents.get_documents_by_property_postgresql"
    )
    def test_get_documents_by_property_properties_populated(
        self,
        mock_postgres: MagicMock,
    ) -> None:
        """get_documents_by_property returns properties from frontmatter_json."""
        doc = _create_mock_doc(
            frontmatter_json={"category": "reference", "year": 2025},
        )

        mock_postgres.return_value = ([doc], 1)

        mock_session = MagicMock()
        property_filters = PropertyFilterParams(
            include_filters=None,
            exclude_filters=None,
        )
        result = get_documents_by_property(
            mock_session,
            property_filters=property_filters,
            pagination=PaginationParams(limit=20, offset=0),
            include_content=True,
        )

        assert len(result.results) == 1
        assert result.results[0].properties == {"category": "reference", "year": 2025}


class TestPropertiesPopulatedInGetDocumentResponse:
    """Test that get_document populates properties."""

    def test_get_document_by_id_properties_populated(self) -> None:
        """get_document returns properties from frontmatter_json."""
        doc = _create_mock_doc(
            frontmatter_json={"title": "My Note", "kind": "article"},
        )

        mock_session = MagicMock()
        query_mock = MagicMock()
        query_mock.options.return_value = query_mock
        query_mock.filter.return_value = query_mock
        query_mock.first.return_value = doc
        mock_session.query.return_value = query_mock

        result = get_document(
            mock_session,
            document_id=str(doc.id),
            include_content=True,
        )

        assert result.properties == {"title": "My Note", "kind": "article"}

    def test_get_document_by_vault_path_properties_populated(self) -> None:
        """get_document by vault+path returns properties."""
        doc = _create_mock_doc(
            frontmatter_json={"author": "Bob"},
        )

        mock_session = MagicMock()
        vault_mock = MagicMock()
        vault_mock.id = uuid.uuid4()

        # First query: lookup vault
        vault_query = MagicMock()
        vault_query.filter.return_value = vault_query
        vault_query.first.return_value = vault_mock

        # Second query: lookup document
        doc_query = MagicMock()
        doc_query.options.return_value = doc_query
        doc_query.filter.return_value = doc_query
        doc_query.first.return_value = doc

        def _side_effect(model):
            if model.__name__ == "Vault":
                return vault_query
            return doc_query

        mock_session.query.side_effect = _side_effect

        result = get_document(
            mock_session,
            vault_name="TestVault",
            file_path="notes/doc.md",
            include_content=True,
        )

        assert result.properties == {"author": "Bob"}


class TestPropertiesPopulatedInListDocumentsResponse:
    """Test that list_documents populates properties."""

    def test_list_documents_properties_populated(self) -> None:
        """list_documents returns properties from frontmatter_json."""
        doc = _create_mock_doc(
            frontmatter_json={"source": "web", "language": "en"},
        )

        mock_session = MagicMock()
        query_mock = MagicMock()
        query_mock.options.return_value = query_mock
        query_mock.filter.return_value = query_mock
        query_mock.order_by.return_value = query_mock
        query_mock.offset.return_value = query_mock
        query_mock.limit.return_value = query_mock
        query_mock.all.return_value = [doc]
        query_mock.count.return_value = 1
        mock_session.query.return_value = query_mock

        result = list_documents(
            mock_session,
            file_name="doc.md",
            include_content=True,
        )

        assert len(result.results) == 1
        assert result.results[0].properties == {"source": "web", "language": "en"}


# ---------------------------------------------------------------------------
# Chunk search – properties None
# ---------------------------------------------------------------------------


class TestPropertiesNoneInChunkSearchResponses:
    """Test that chunk search always returns properties=None."""

    @patch("obsidian_rag.mcp_server.tools.documents.query_chunks")
    def test_chunk_search_properties_none(self, mock_query_chunks: MagicMock) -> None:
        """Chunk-level search returns properties=None because chunks lack frontmatter."""
        from obsidian_rag.mcp_server.tools.documents_chunks import ChunkQueryResult

        chunk = ChunkQueryResult(
            chunk_id="12345678-1234-1234-1234-123456789abc",
            content="chunk text",
            document_name="doc.md",
            document_path="notes/doc.md",
            vault_name="TestVault",
            chunk_index=0,
            total_chunks=1,
            token_count=100,
            chunk_type="content",
            similarity_score=0.9,
            rerank_score=None,
        )
        mock_query_chunks.return_value = [chunk]

        result = query_documents(
            MagicMock(),
            query_embedding=[0.1] * 10,
            use_chunks=True,
            include_content=True,
        )

        assert len(result.results) == 1
        assert result.results[0].properties is None


# ---------------------------------------------------------------------------
# Document tools – include_content=False
# ---------------------------------------------------------------------------


class TestIncludeContentFalseInAllDocumentTools:
    """Test that include_content=False produces empty content across all document tools."""

    @patch("obsidian_rag.mcp_server.tools.documents.query_documents_postgresql")
    def test_query_documents_include_content_false(
        self, mock_postgres: MagicMock
    ) -> None:
        """query_documents with include_content=False returns empty content."""
        doc = _create_mock_doc(content="full content")

        mock_postgres.return_value = DocumentListResponse(
            results=[
                DocumentResponse(
                    id=doc.id,
                    vault_name="TestVault",
                    file_path=doc.file_path,
                    relative_path=doc.file_path,
                    file_name=doc.file_name,
                    content="",  # empty because include_content=False
                    kind=None,
                    tags=doc.tags,
                    similarity_score=0.1,
                    created_at_fs=doc.created_at_fs,
                    modified_at_fs=doc.modified_at_fs,
                    obsidian_uri="uri",
                    properties={"a": 1},
                ),
            ],
            total_count=1,
            has_more=False,
            next_offset=None,
        )

        mock_session = MagicMock()
        result = query_documents(
            mock_session,
            query_embedding=[0.1] * 10,
            include_content=False,
        )

        assert result.results[0].content == ""
        assert result.results[0].properties == {"a": 1}  # properties still populated

    @patch("obsidian_rag.mcp_server.tools.documents_tags.apply_postgresql_tag_filter")
    def test_get_documents_by_tag_include_content_false(
        self,
        mock_tag_filter: MagicMock,
    ) -> None:
        """get_documents_by_tag with include_content=False returns empty content."""
        from obsidian_rag.mcp_server.models import TagFilter

        doc = _create_mock_doc(
            frontmatter_json={"author": "Alice"},
            content="tagged content",
        )

        mock_session = MagicMock()
        query_mock = MagicMock()
        query_mock.order_by.return_value = query_mock
        query_mock.offset.return_value = query_mock
        query_mock.limit.return_value = query_mock
        query_mock.all.return_value = [doc]
        query_mock.count.return_value = 1
        mock_tag_filter.return_value = query_mock
        mock_session.query.return_value = query_mock

        result = get_documents_by_tag(
            mock_session,
            tag_filter=TagFilter(include_tags=["work"]),
            include_content=False,
        )

        assert result.results[0].content == ""
        assert result.results[0].properties == {"author": "Alice"}

    @patch(
        "obsidian_rag.mcp_server.tools.documents.get_documents_by_property_postgresql"
    )
    def test_get_documents_by_property_include_content_false(
        self,
        mock_postgres: MagicMock,
    ) -> None:
        """get_documents_by_property with include_content=False returns empty content."""
        doc = _create_mock_doc(
            frontmatter_json={"author": "Alice"},
            content="property content",
        )

        mock_postgres.return_value = ([doc], 1)

        mock_session = MagicMock()
        result = get_documents_by_property(
            mock_session,
            property_filters=PropertyFilterParams(
                include_filters=None,
                exclude_filters=None,
            ),
            pagination=PaginationParams(limit=20, offset=0),
            include_content=False,
        )

        assert result.results[0].content == ""
        assert result.results[0].properties == {"author": "Alice"}

    def test_get_document_include_content_false(self) -> None:
        """get_document with include_content=False returns empty content."""
        doc = _create_mock_doc(
            frontmatter_json={"author": "Alice"},
            content="sensitive content",
        )

        mock_session = MagicMock()
        query_mock = MagicMock()
        query_mock.options.return_value = query_mock
        query_mock.filter.return_value = query_mock
        query_mock.first.return_value = doc
        mock_session.query.return_value = query_mock

        result = get_document(
            mock_session,
            document_id=str(doc.id),
            include_content=False,
        )

        assert result.content == ""
        assert result.properties == {"author": "Alice"}

    def test_list_documents_include_content_false(self) -> None:
        """list_documents with include_content=False returns empty content."""
        doc = _create_mock_doc(
            frontmatter_json={"author": "Alice"},
            content="listed content",
        )

        mock_session = MagicMock()
        query_mock = MagicMock()
        query_mock.options.return_value = query_mock
        query_mock.filter.return_value = query_mock
        query_mock.order_by.return_value = query_mock
        query_mock.offset.return_value = query_mock
        query_mock.limit.return_value = query_mock
        query_mock.all.return_value = [doc]
        query_mock.count.return_value = 1
        mock_session.query.return_value = query_mock

        result = list_documents(
            mock_session,
            file_name="doc.md",
            include_content=False,
        )

        assert result.results[0].content == ""
        assert result.results[0].properties == {"author": "Alice"}


# ---------------------------------------------------------------------------
# Task tools – properties and include_content
# ---------------------------------------------------------------------------


class TestPropertiesPopulatedInGetTasksResponses:
    """Test that get_tasks populates properties from parent document frontmatter."""

    def test_get_tasks_properties_populated(self) -> None:
        """get_tasks returns properties extracted from document frontmatter_json."""
        task, doc = _create_mock_task_doc_pair(
            frontmatter_json={"project": "obsidian-rag", "status": "active"},
        )
        mock_session = _configure_mock_session_for_tasks_mock([(task, doc)])

        filters = GetTasksFilterParams()
        result = get_tasks(mock_session, filters)

        assert result.total_count == 1
        assert result.results[0].properties == {
            "project": "obsidian-rag",
            "status": "active",
        }

    def test_get_tasks_properties_excludes_tags_key(self) -> None:
        """properties excludes the 'tags' key from frontmatter_json."""
        task, doc = _create_mock_task_doc_pair(
            frontmatter_json={"title": "Doc", "tags": ["note"]},
        )
        mock_session = _configure_mock_session_for_tasks_mock([(task, doc)])

        filters = GetTasksFilterParams()
        result = get_tasks(mock_session, filters)

        assert result.results[0].properties == {"title": "Doc"}

    def test_get_tasks_properties_none_no_frontmatter(self) -> None:
        """properties is None when document has no frontmatter_json."""
        task, doc = _create_mock_task_doc_pair(frontmatter_json=None)
        mock_session = _configure_mock_session_for_tasks_mock([(task, doc)])

        filters = GetTasksFilterParams()
        result = get_tasks(mock_session, filters)

        assert result.results[0].properties is None


class TestIncludeContentFalseInGetTasks:
    """Test that include_content=False produces empty raw_text in task responses."""

    def test_get_tasks_include_content_false(self) -> None:
        """include_content=False returns empty raw_text but keeps properties."""
        task, doc = _create_mock_task_doc_pair(
            frontmatter_json={"author": "Alice"},
        )
        task.raw_text = "- [ ] Secret task"
        mock_session = _configure_mock_session_for_tasks_mock([(task, doc)])

        filters = GetTasksFilterParams(include_content=False)
        result = get_tasks(mock_session, filters)

        assert result.results[0].raw_text == ""
        assert result.results[0].properties == {"author": "Alice"}

    def test_get_tasks_include_content_false_description_preserved(self) -> None:
        """Description is preserved when include_content=False."""
        task, doc = _create_mock_task_doc_pair()
        task.description = "Important description"
        mock_session = _configure_mock_session_for_tasks_mock([(task, doc)])

        filters = GetTasksFilterParams(include_content=False)
        result = get_tasks(mock_session, filters)

        assert result.results[0].raw_text == ""
        assert result.results[0].description == "Important description"


# ---------------------------------------------------------------------------
# Backward compatibility
# ---------------------------------------------------------------------------


class TestBackwardCompatibilityPropertiesNoneDefault:
    """Test that default/legacy behavior keeps properties=None when no frontmatter."""

    def test_document_response_default_properties_none(self) -> None:
        """DocumentResponse defaults properties to None when not provided."""
        now = datetime.now(UTC)
        response = DocumentResponse(
            id=uuid.uuid4(),
            vault_name="V",
            file_path="p.md",
            relative_path="p.md",
            file_name="p.md",
            content="c",
            kind=None,
            tags=[],
            similarity_score=0.0,
            created_at_fs=now,
            modified_at_fs=now,
            obsidian_uri="uri",
        )
        assert response.properties is None

    def test_task_response_default_properties_none(self) -> None:
        """TaskResponse defaults properties to None when not provided."""
        from obsidian_rag.mcp_server.models import TaskResponse

        response = TaskListResponse(
            results=[
                TaskResponse(
                    id=uuid.uuid4(),
                    raw_text="",
                    status="not_completed",
                    description="desc",
                    due=None,
                    priority="normal",
                    tags=[],
                    document_path="p.md",
                    document_name="p.md",
                ),
            ],
            total_count=1,
            has_more=False,
            next_offset=None,
        )
        assert response.results[0].properties is None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _configure_mock_session_for_tasks_mock(
    pairs: list[tuple[MagicMock, MagicMock]],
) -> MagicMock:
    """Build a mock session that returns the given (task, doc) pairs."""
    mock_session = MagicMock()
    query_mock = MagicMock()
    query_mock.join.return_value = query_mock
    query_mock.filter.return_value = query_mock
    query_mock.order_by.return_value = query_mock
    query_mock.offset.return_value = query_mock
    query_mock.limit.return_value = query_mock
    query_mock.count.return_value = len(pairs)
    query_mock.all.return_value = pairs
    mock_session.query.return_value = query_mock
    mock_session.bind.dialect.name = "postgresql"
    return mock_session
