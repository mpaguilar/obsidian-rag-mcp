"""Tests for include_content parameter in document query tools."""

import uuid as uuid_module
from unittest.mock import MagicMock, patch

from sqlalchemy.dialects import postgresql

from obsidian_rag.mcp_server.models import DocumentResponse, TagFilter
from obsidian_rag.mcp_server.tools.documents import (
    _build_document_list_response,
    _lookup_document_by_id,
    _lookup_document_by_vault_path,
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


def test_get_documents_by_tag_hardcodes_include_content_false():
    """get_documents_by_tag always calls create_document_response with include_content=False."""
    from datetime import UTC, datetime
    from uuid import uuid4

    mock_session = MagicMock()
    mock_doc = MagicMock()

    # Set up the mock chain after apply_postgresql_tag_filter and order_by
    tag_query = MagicMock()
    tag_query.order_by.return_value.count.return_value = 1
    tag_query.order_by.return_value.offset.return_value.limit.return_value.all.return_value = [
        mock_doc
    ]

    with patch(
        "obsidian_rag.mcp_server.tools.documents_tags.apply_postgresql_tag_filter"
    ) as mock_tag_filter:
        with patch(
            "obsidian_rag.mcp_server.tools.documents.create_document_response"
        ) as mock_response:
            mock_tag_filter.return_value = tag_query
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
            get_documents_by_tag(mock_session, tag_filter=tag_filter)

            mock_response.assert_called_once_with(mock_doc, 0.0, include_content=False)


def test_get_documents_by_property_hardcodes_include_content_false():
    """get_documents_by_property always calls _build_document_list_response with include_content=False."""
    mock_session = MagicMock()
    mock_doc = MagicMock()

    with patch(
        "obsidian_rag.mcp_server.tools.documents.get_documents_by_property_postgresql"
    ) as mock_postgres:
        with patch(
            "obsidian_rag.mcp_server.tools.documents._build_document_list_response"
        ) as mock_build:
            mock_postgres.return_value = ([mock_doc], 1)
            mock_build.return_value = MagicMock()

            property_filters = PropertyFilterParams(
                include_filters=None,
                exclude_filters=None,
            )
            get_documents_by_property(mock_session, property_filters=property_filters)

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


def test_list_documents_hardcodes_include_content_false():
    """list_documents always calls _build_document_list_response with include_content=False."""
    mock_session = MagicMock()
    mock_doc = MagicMock()
    mock_session.query.return_value.options.return_value.filter.return_value.order_by.return_value.offset.return_value.limit.return_value.all.return_value = [
        mock_doc
    ]
    mock_session.query.return_value.options.return_value.filter.return_value.order_by.return_value.count.return_value = 1

    with patch(
        "obsidian_rag.mcp_server.tools.documents._build_document_list_response"
    ) as mock_build:
        mock_build.return_value = MagicMock()

        list_documents(mock_session, file_name="doc.md")

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


class TestLookupDocumentByIdDefer:
    """Tests for _lookup_document_by_id conditional defer."""

    def test_lookup_by_id_defers_content_when_false(self):
        """When include_content=False, defer(Document.content) is added to options."""
        mock_session = MagicMock()
        doc_id = str(uuid_module.uuid4())
        mock_doc = MagicMock()
        mock_doc.id = uuid_module.UUID(doc_id)
        options_query = mock_session.query.return_value.options.return_value
        options_query.filter.return_value.first.return_value = mock_doc
        options_query.filter.return_value.statement.compile.return_value = (
            "SELECT documents.id \nFROM documents"
        )

        result = _lookup_document_by_id(
            mock_session, document_id=doc_id, include_content=False
        )

        assert result is mock_doc
        compiled = str(
            options_query.filter.return_value.statement.compile(
                dialect=postgresql.dialect()
            )
        )
        assert "documents.content" not in compiled

    def test_lookup_by_id_no_defer_when_true(self):
        """When include_content=True, defer is NOT added to options."""
        mock_session = MagicMock()
        doc_id = str(uuid_module.uuid4())
        mock_doc = MagicMock()
        mock_doc.id = uuid_module.UUID(doc_id)
        options_query = mock_session.query.return_value.options.return_value
        options_query.filter.return_value.first.return_value = mock_doc
        options_query.filter.return_value.statement.compile.return_value = (
            "SELECT documents.id, documents.content \nFROM documents"
        )

        result = _lookup_document_by_id(
            mock_session, document_id=doc_id, include_content=True
        )

        assert result is mock_doc
        compiled = str(
            options_query.filter.return_value.statement.compile(
                dialect=postgresql.dialect()
            )
        )
        assert "documents.content" in compiled


class TestLookupDocumentByVaultPathDefer:
    """Tests for _lookup_document_by_vault_path conditional defer."""

    def test_lookup_by_vault_path_defers_content_when_false(self):
        """When include_content=False, defer(Document.content) is added to options."""
        mock_session = MagicMock()
        mock_vault = MagicMock()
        mock_vault.id = uuid_module.uuid4()
        mock_doc = MagicMock()
        # First query (Vault) uses .filter().first()
        mock_session.query.return_value.filter.return_value.first.return_value = (
            mock_vault
        )
        # Second query (Document) uses .options().filter().filter().first()
        doc_options_query = mock_session.query.return_value.options.return_value
        doc_options_query.filter.return_value.filter.return_value.first.return_value = (
            mock_doc
        )
        doc_options_query.filter.return_value.filter.return_value.statement.compile.return_value = "SELECT documents.id \nFROM documents"

        result = _lookup_document_by_vault_path(
            mock_session, vault_name="Vault", file_path="doc.md", include_content=False
        )

        assert result is mock_doc
        compiled = str(
            doc_options_query.filter.return_value.filter.return_value.statement.compile(
                dialect=postgresql.dialect()
            )
        )
        assert "documents.content" not in compiled

    def test_lookup_by_vault_path_no_defer_when_true(self):
        """When include_content=True, defer is NOT added to options."""
        mock_session = MagicMock()
        mock_vault = MagicMock()
        mock_vault.id = uuid_module.uuid4()
        mock_doc = MagicMock()
        mock_session.query.return_value.filter.return_value.first.return_value = (
            mock_vault
        )
        doc_options_query = mock_session.query.return_value.options.return_value
        doc_options_query.filter.return_value.filter.return_value.first.return_value = (
            mock_doc
        )
        doc_options_query.filter.return_value.filter.return_value.statement.compile.return_value = "SELECT documents.id, documents.content \nFROM documents"

        result = _lookup_document_by_vault_path(
            mock_session, vault_name="Vault", file_path="doc.md", include_content=True
        )

        assert result is mock_doc
        compiled = str(
            doc_options_query.filter.return_value.filter.return_value.statement.compile(
                dialect=postgresql.dialect()
            )
        )
        assert "documents.content" in compiled
