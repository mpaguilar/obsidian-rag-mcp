"""Tests for include_content parameter in document query tools."""

import uuid as uuid_module
from unittest.mock import MagicMock, patch

from obsidian_rag.mcp_server.models import DocumentResponse, TagFilter
from obsidian_rag.mcp_server.tools.documents import (
    _build_document_list_response,
    get_document,
    get_documents_by_property,
    get_documents_by_tag,
    list_documents,
    query_documents,
)
from obsidian_rag.mcp_server.tools.documents_params import (
    PropertyFilterParams,
)


class TestQueryDocumentsIncludeContent:
    """Tests for query_documents include_content parameter."""

    @patch("obsidian_rag.mcp_server.tools.documents.query_documents_postgresql")
    def test_query_documents_include_content_true_default(self, mock_postgres):
        """Default include_content=True passes through to postgres path."""
        mock_session = MagicMock()
        mock_postgres.return_value = MagicMock()

        query_documents(mock_session, query_embedding=[0.1] * 1536)

        call_args = mock_postgres.call_args
        params = call_args[0][0]
        assert params.pagination.include_content is True

    @patch("obsidian_rag.mcp_server.tools.documents.query_documents_postgresql")
    def test_query_documents_include_content_false_empty_content(self, mock_postgres):
        """include_content=False passes through to postgres path."""
        mock_session = MagicMock()
        mock_postgres.return_value = MagicMock()

        query_documents(
            mock_session,
            query_embedding=[0.1] * 1536,
            include_content=False,
        )

        params = mock_postgres.call_args[0][0]
        assert params.pagination.include_content is False

    @patch("obsidian_rag.mcp_server.tools.documents.query_chunks")
    def test_query_documents_chunk_path_include_content_false(self, mock_query_chunks):
        """Chunk path returns empty content when include_content=False."""
        from obsidian_rag.mcp_server.tools.documents_chunks import ChunkQueryResult

        mock_result = ChunkQueryResult(
            chunk_id="12345678-1234-1234-1234-123456789abc",
            content="Chunk content text",
            document_name="doc.md",
            document_path="path/doc.md",
            vault_name="Vault",
            chunk_index=0,
            total_chunks=2,
            token_count=512,
            chunk_type="content",
            similarity_score=0.8,
            rerank_score=None,
        )
        mock_query_chunks.return_value = [mock_result]

        result = query_documents(
            MagicMock(),
            query_embedding=[0.1] * 1536,
            use_chunks=True,
            include_content=False,
        )

        assert len(result.results) == 1
        assert result.results[0].content == ""

    @patch("obsidian_rag.mcp_server.tools.documents.query_chunks")
    def test_query_documents_chunk_path_properties_none(self, mock_query_chunks):
        """Chunk path always returns properties=None."""
        from obsidian_rag.mcp_server.tools.documents_chunks import ChunkQueryResult

        mock_result = ChunkQueryResult(
            chunk_id="12345678-1234-1234-1234-123456789abc",
            content="Chunk content text",
            document_name="doc.md",
            document_path="path/doc.md",
            vault_name="Vault",
            chunk_index=0,
            total_chunks=2,
            token_count=512,
            chunk_type="content",
            similarity_score=0.8,
            rerank_score=None,
        )
        mock_query_chunks.return_value = [mock_result]

        result = query_documents(
            MagicMock(),
            query_embedding=[0.1] * 1536,
            use_chunks=True,
            include_content=True,
        )

        assert result.results[0].properties is None


class TestGetDocumentsByTagIncludeContent:
    """Tests for get_documents_by_tag include_content parameter."""

    @patch("obsidian_rag.mcp_server.tools.documents_tags.apply_postgresql_tag_filter")
    @patch("obsidian_rag.mcp_server.tools.documents.create_document_response")
    def test_get_documents_by_tag_include_content_true(
        self, mock_response, mock_tag_filter
    ):
        """include_content=True is passed to create_document_response."""
        from datetime import UTC, datetime
        from uuid import uuid4

        mock_session = MagicMock()
        mock_doc = MagicMock()
        mock_session.query.return_value.order_by.return_value.offset.return_value.limit.return_value.all.return_value = [
            mock_doc
        ]
        mock_session.query.return_value.order_by.return_value.count.return_value = 1
        mock_tag_filter.return_value = mock_session.query.return_value
        mock_response.return_value = DocumentResponse(
            id=uuid4(),
            vault_name="Vault",
            file_path="path/doc.md",
            relative_path="path/doc.md",
            file_name="doc.md",
            content="",
            kind=None,
            tags=[],
            similarity_score=0.0,
            matching_chunk=None,
            created_at_fs=datetime.now(UTC),
            modified_at_fs=datetime.now(UTC),
            obsidian_uri="obsidian://open?vault=Vault&file=path/doc.md",
            properties=None,
        )

        tag_filter = TagFilter(include_tags=["work"])
        get_documents_by_tag(
            mock_session,
            tag_filter=tag_filter,
            include_content=True,
        )

        mock_response.assert_called_once_with(mock_doc, 0.0, include_content=True)

    @patch("obsidian_rag.mcp_server.tools.documents_tags.apply_postgresql_tag_filter")
    @patch("obsidian_rag.mcp_server.tools.documents.create_document_response")
    def test_get_documents_by_tag_include_content_false(
        self, mock_response, mock_tag_filter
    ):
        """include_content=False is passed to create_document_response."""
        from datetime import UTC, datetime
        from uuid import uuid4

        mock_session = MagicMock()
        mock_doc = MagicMock()
        mock_session.query.return_value.order_by.return_value.offset.return_value.limit.return_value.all.return_value = [
            mock_doc
        ]
        mock_session.query.return_value.order_by.return_value.count.return_value = 1
        mock_tag_filter.return_value = mock_session.query.return_value
        mock_response.return_value = DocumentResponse(
            id=uuid4(),
            vault_name="Vault",
            file_path="path/doc.md",
            relative_path="path/doc.md",
            file_name="doc.md",
            content="",
            kind=None,
            tags=[],
            similarity_score=0.0,
            matching_chunk=None,
            created_at_fs=datetime.now(UTC),
            modified_at_fs=datetime.now(UTC),
            obsidian_uri="obsidian://open?vault=Vault&file=path/doc.md",
            properties=None,
        )

        tag_filter = TagFilter(include_tags=["work"])
        get_documents_by_tag(
            mock_session,
            tag_filter=tag_filter,
            include_content=False,
        )

        mock_response.assert_called_once_with(mock_doc, 0.0, include_content=False)


class TestGetDocumentsByPropertyIncludeContent:
    """Tests for get_documents_by_property include_content parameter."""

    @patch("obsidian_rag.mcp_server.tools.documents._build_document_list_response")
    def test_get_documents_by_property_include_content_true(self, mock_build):
        """include_content=True is passed to _build_document_list_response."""
        mock_session = MagicMock()
        mock_doc = MagicMock()
        mock_session.query.return_value.filter.return_value.order_by.return_value.all.return_value = [
            mock_doc
        ]
        mock_session.query.return_value.filter.return_value.order_by.return_value.count.return_value = 1
        mock_build.return_value = MagicMock()

        property_filters = PropertyFilterParams(
            include_filters=None,
            exclude_filters=None,
        )
        get_documents_by_property(
            mock_session,
            property_filters=property_filters,
            include_content=True,
        )

        call_args = mock_build.call_args
        assert call_args[1]["include_content"] is True

    @patch("obsidian_rag.mcp_server.tools.documents._build_document_list_response")
    def test_get_documents_by_property_include_content_false(self, mock_build):
        """include_content=False is passed to _build_document_list_response."""
        mock_session = MagicMock()
        mock_doc = MagicMock()
        mock_session.query.return_value.filter.return_value.order_by.return_value.all.return_value = [
            mock_doc
        ]
        mock_session.query.return_value.filter.return_value.order_by.return_value.count.return_value = 1
        mock_build.return_value = MagicMock()

        property_filters = PropertyFilterParams(
            include_filters=None,
            exclude_filters=None,
        )
        get_documents_by_property(
            mock_session,
            property_filters=property_filters,
            include_content=False,
        )

        call_args = mock_build.call_args
        assert call_args[1]["include_content"] is False


class TestGetDocumentIncludeContent:
    """Tests for get_document include_content parameter."""

    @patch("obsidian_rag.mcp_server.tools.documents.create_document_response")
    def test_get_document_include_content_true(self, mock_response):
        """include_content=True is passed to create_document_response."""
        mock_session = MagicMock()
        doc_id = str(uuid_module.uuid4())
        mock_doc = MagicMock()
        mock_doc.id = uuid_module.UUID(doc_id)
        mock_session.query.return_value.options.return_value.filter.return_value.first.return_value = mock_doc
        mock_response.return_value = MagicMock()

        get_document(mock_session, document_id=doc_id, include_content=True)

        mock_response.assert_called_once_with(
            mock_doc,
            similarity_score=0.0,
            include_content=True,
        )

    @patch("obsidian_rag.mcp_server.tools.documents.create_document_response")
    def test_get_document_include_content_false(self, mock_response):
        """include_content=False is passed to create_document_response."""
        mock_session = MagicMock()
        doc_id = str(uuid_module.uuid4())
        mock_doc = MagicMock()
        mock_doc.id = uuid_module.uuid4()
        mock_session.query.return_value.options.return_value.filter.return_value.first.return_value = mock_doc
        mock_response.return_value = MagicMock()

        get_document(mock_session, document_id=doc_id, include_content=False)

        mock_response.assert_called_once_with(
            mock_doc,
            similarity_score=0.0,
            include_content=False,
        )


class TestListDocumentsIncludeContent:
    """Tests for list_documents include_content parameter."""

    @patch("obsidian_rag.mcp_server.tools.documents._build_document_list_response")
    def test_list_documents_include_content_true(self, mock_build):
        """include_content=True is passed to _build_document_list_response."""
        mock_session = MagicMock()
        mock_doc = MagicMock()
        mock_session.query.return_value.options.return_value.filter.return_value.order_by.return_value.offset.return_value.limit.return_value.all.return_value = [
            mock_doc
        ]
        mock_session.query.return_value.options.return_value.filter.return_value.order_by.return_value.count.return_value = 1
        mock_build.return_value = MagicMock()

        list_documents(mock_session, file_name="doc.md", include_content=True)

        call_args = mock_build.call_args
        assert call_args[1]["include_content"] is True

    @patch("obsidian_rag.mcp_server.tools.documents._build_document_list_response")
    def test_list_documents_include_content_false(self, mock_build):
        """include_content=False is passed to _build_document_list_response."""
        mock_session = MagicMock()
        mock_doc = MagicMock()
        mock_session.query.return_value.options.return_value.filter.return_value.order_by.return_value.offset.return_value.limit.return_value.all.return_value = [
            mock_doc
        ]
        mock_session.query.return_value.options.return_value.filter.return_value.order_by.return_value.count.return_value = 1
        mock_build.return_value = MagicMock()

        list_documents(mock_session, file_name="doc.md", include_content=False)

        call_args = mock_build.call_args
        assert call_args[1]["include_content"] is False


class TestBuildDocumentListResponseIncludeContent:
    """Tests for _build_document_list_response include_content parameter."""

    @patch("obsidian_rag.mcp_server.tools.documents.create_document_response")
    def test_build_document_list_response_passes_include_content(self, mock_response):
        """_build_document_list_response passes include_content to create_document_response."""
        from datetime import UTC, datetime
        from uuid import uuid4

        mock_doc = MagicMock()
        mock_response.return_value = DocumentResponse(
            id=uuid4(),
            vault_name="Vault",
            file_path="path/doc.md",
            relative_path="path/doc.md",
            file_name="doc.md",
            content="",
            kind=None,
            tags=[],
            similarity_score=0.0,
            matching_chunk=None,
            created_at_fs=datetime.now(UTC),
            modified_at_fs=datetime.now(UTC),
            obsidian_uri="obsidian://open?vault=Vault&file=path/doc.md",
            properties=None,
        )

        _build_document_list_response(
            [mock_doc],
            total_count=1,
            offset=0,
            limit=20,
            include_content=False,
        )

        mock_response.assert_called_once_with(mock_doc, 0.0, include_content=False)
