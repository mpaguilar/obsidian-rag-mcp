"""Unit tests for MCP server module."""

import json
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner


runner = CliRunner()

runner = CliRunner()


class TestGetDocumentsByTagJsonString:
    """Tests for get_documents_by_tag with JSON string filters parameter."""

    @pytest.fixture
    def setup_registry(self):
        """Set up mock registry for testing."""
        from obsidian_rag.mcp_server.tool_definitions import (
            MCPToolRegistry,
            _set_registry,
        )

        mock_db_manager = MagicMock()
        mock_embedding_provider = MagicMock()
        mock_settings = MagicMock()

        registry = MCPToolRegistry(
            db_manager=mock_db_manager,
            embedding_provider=mock_embedding_provider,
            settings=mock_settings,
        )
        _set_registry(registry)
        yield registry
        _set_registry(None)

    def test_get_documents_by_tag_with_json_string_filters(self, setup_registry):
        """Test get_documents_by_tag accepts filters as JSON string."""
        from obsidian_rag.mcp_server.server import get_documents_by_tag

        with patch(
            "obsidian_rag.mcp_server.server._get_documents_by_tag_handler"
        ) as mock_handler:
            mock_handler.return_value = {
                "results": [{"id": "doc1", "title": "Test"}],
                "total_count": 1,
                "has_more": False,
                "next_offset": None,
            }

            filters_json = '{"include_tags": ["work"], "match_mode": "any"}'
            result = get_documents_by_tag(filters=filters_json)

            assert result["total_count"] == 1
            mock_handler.assert_called_once()
            call_kwargs = mock_handler.call_args[0][1]
            assert call_kwargs["include_tags"] == ["work"]
            assert call_kwargs["match_mode"] == "any"

    def test_get_documents_by_tag_with_dict_filters(self, setup_registry):
        """Test get_documents_by_tag accepts filters as dict."""
        from obsidian_rag.mcp_server.server import get_documents_by_tag

        with patch(
            "obsidian_rag.mcp_server.server._get_documents_by_tag_handler"
        ) as mock_handler:
            mock_handler.return_value = {
                "results": [{"id": "doc1", "title": "Test"}],
                "total_count": 1,
                "has_more": False,
                "next_offset": None,
            }

            filters_dict = {"include_tags": ["personal"], "match_mode": "all"}
            result = get_documents_by_tag(filters=filters_dict)

            assert result["total_count"] == 1
            mock_handler.assert_called_once()
            call_kwargs = mock_handler.call_args[0][1]
            assert call_kwargs["include_tags"] == ["personal"]
            assert call_kwargs["match_mode"] == "all"

    def test_get_documents_by_tag_with_none_filters(self, setup_registry):
        """Test get_documents_by_tag works with None filters (default)."""
        from obsidian_rag.mcp_server.server import get_documents_by_tag

        with patch(
            "obsidian_rag.mcp_server.server._get_documents_by_tag_handler"
        ) as mock_handler:
            mock_handler.return_value = {
                "results": [],
                "total_count": 0,
                "has_more": False,
                "next_offset": None,
            }

            result = get_documents_by_tag(filters=None)

            assert result["total_count"] == 0
            mock_handler.assert_called_once()
            call_kwargs = mock_handler.call_args[0][1]
            assert call_kwargs["include_tags"] == []
            assert call_kwargs["exclude_tags"] == []
            assert call_kwargs["match_mode"] == "all"

    def test_get_documents_by_tag_with_empty_string_filters(self, setup_registry):
        """Test get_documents_by_tag handles empty string as None."""
        from obsidian_rag.mcp_server.server import get_documents_by_tag

        with patch(
            "obsidian_rag.mcp_server.server._get_documents_by_tag_handler"
        ) as mock_handler:
            mock_handler.return_value = {
                "results": [],
                "total_count": 0,
                "has_more": False,
                "next_offset": None,
            }

            result = get_documents_by_tag(filters="")

            assert result["total_count"] == 0
            mock_handler.assert_called_once()
            call_kwargs = mock_handler.call_args[0][1]
            assert call_kwargs["include_tags"] == []
            assert call_kwargs["exclude_tags"] == []

    def test_get_documents_by_tag_with_whitespace_string_filters(self, setup_registry):
        """Test get_documents_by_tag handles whitespace-only string as None."""
        from obsidian_rag.mcp_server.server import get_documents_by_tag

        with patch(
            "obsidian_rag.mcp_server.server._get_documents_by_tag_handler"
        ) as mock_handler:
            mock_handler.return_value = {
                "results": [],
                "total_count": 0,
                "has_more": False,
                "next_offset": None,
            }

            result = get_documents_by_tag(filters="   ")

            assert result["total_count"] == 0
            mock_handler.assert_called_once()
            call_kwargs = mock_handler.call_args[0][1]
            assert call_kwargs["include_tags"] == []
            assert call_kwargs["exclude_tags"] == []


class TestGetDocumentsByPropertyJsonString:
    """Tests for get_documents_by_property with JSON string filters parameter."""

    @pytest.fixture
    def setup_registry(self):
        """Set up mock registry for testing."""
        from obsidian_rag.mcp_server.tool_definitions import (
            MCPToolRegistry,
            _set_registry,
        )

        mock_db_manager = MagicMock()
        mock_embedding_provider = MagicMock()
        mock_settings = MagicMock()

        registry = MCPToolRegistry(
            db_manager=mock_db_manager,
            embedding_provider=mock_embedding_provider,
            settings=mock_settings,
        )
        _set_registry(registry)
        yield registry
        _set_registry(None)

    def test_get_documents_by_property_with_json_string_filters(self, setup_registry):
        """Test get_documents_by_property accepts filters as JSON string."""
        from obsidian_rag.mcp_server.server import get_documents_by_property
        from obsidian_rag.mcp_server.models import DocumentListResponse

        with patch(
            "obsidian_rag.mcp_server.tools.documents.get_documents_by_property"
        ) as mock_tool:
            mock_tool.return_value = DocumentListResponse(
                results=[],
                total_count=0,
                has_more=False,
                next_offset=None,
            )

            filters_json = '{"include_tags": ["work"], "match_mode": "any"}'
            result = get_documents_by_property(filters=filters_json)

            assert result["total_count"] == 0
            mock_tool.assert_called_once()

    def test_get_documents_by_property_with_dict_filters(self, setup_registry):
        """Test get_documents_by_property accepts filters as dict."""
        from obsidian_rag.mcp_server.server import get_documents_by_property
        from obsidian_rag.mcp_server.models import DocumentListResponse

        with patch(
            "obsidian_rag.mcp_server.tools.documents.get_documents_by_property"
        ) as mock_tool:
            mock_tool.return_value = DocumentListResponse(
                results=[],
                total_count=0,
                has_more=False,
                next_offset=None,
            )

            filters_dict = {"include_tags": ["work"], "match_mode": "any"}
            result = get_documents_by_property(filters=filters_dict)

            assert result["total_count"] == 0
            mock_tool.assert_called_once()

    def test_get_documents_by_property_with_none_filters(self, setup_registry):
        """Test get_documents_by_property handles None filters."""
        from obsidian_rag.mcp_server.server import get_documents_by_property
        from obsidian_rag.mcp_server.models import DocumentListResponse

        with patch(
            "obsidian_rag.mcp_server.tools.documents.get_documents_by_property"
        ) as mock_tool:
            mock_tool.return_value = DocumentListResponse(
                results=[],
                total_count=0,
                has_more=False,
                next_offset=None,
            )

            result = get_documents_by_property(filters=None)

            assert result["total_count"] == 0
            mock_tool.assert_called_once()

    def test_get_documents_by_property_with_empty_string_filters(self, setup_registry):
        """Test get_documents_by_property treats empty string as None."""
        from obsidian_rag.mcp_server.server import get_documents_by_property
        from obsidian_rag.mcp_server.models import DocumentListResponse

        with patch(
            "obsidian_rag.mcp_server.tools.documents.get_documents_by_property"
        ) as mock_tool:
            mock_tool.return_value = DocumentListResponse(
                results=[],
                total_count=0,
                has_more=False,
                next_offset=None,
            )

            result = get_documents_by_property(filters="")

            assert result["total_count"] == 0
            mock_tool.assert_called_once()

    def test_get_documents_by_property_with_whitespace_string_filters(
        self, setup_registry
    ):
        """Test get_documents_by_property handles whitespace-only string as None."""
        from obsidian_rag.mcp_server.server import get_documents_by_property
        from obsidian_rag.mcp_server.models import DocumentListResponse

        with patch(
            "obsidian_rag.mcp_server.tools.documents.get_documents_by_property"
        ) as mock_tool:
            mock_tool.return_value = DocumentListResponse(
                results=[],
                total_count=0,
                has_more=False,
                next_offset=None,
            )

            result = get_documents_by_property(filters="   ")

            assert result["total_count"] == 0
            mock_tool.assert_called_once()

    def test_get_documents_by_property_with_complex_json_filters(self, setup_registry):
        """Test get_documents_by_property with complex JSON filters including properties."""
        from obsidian_rag.mcp_server.server import get_documents_by_property
        from obsidian_rag.mcp_server.models import DocumentListResponse

        with patch(
            "obsidian_rag.mcp_server.tools.documents.get_documents_by_property"
        ) as mock_tool:
            mock_tool.return_value = DocumentListResponse(
                results=[],
                total_count=0,
                has_more=False,
                next_offset=None,
            )

            # Complex JSON with property filters
            json_filters = json.dumps(
                {
                    "include_properties": [
                        {"path": "status", "operator": "equals", "value": "active"}
                    ],
                    "exclude_properties": [
                        {"path": "archived", "operator": "equals", "value": True}
                    ],
                    "include_tags": ["work"],
                    "exclude_tags": ["blocked"],
                    "match_mode": "all",
                }
            )

            result = get_documents_by_property(
                filters=json_filters, vault_name="test-vault", limit=50, offset=10
            )

            assert result["total_count"] == 0
            mock_tool.assert_called_once()
            # Verify the call was made with correct vault_name
            call_kwargs = mock_tool.call_args[1]
            assert call_kwargs["vault_name"] == "test-vault"

    def test_get_documents_by_property_invalid_json_raises_error(self, setup_registry):
        """Test get_documents_by_property raises error for invalid JSON."""
        from obsidian_rag.mcp_server.server import get_documents_by_property

        # Invalid JSON string (missing closing bracket)
        invalid_json = '{"include_tags": ["work", "match_mode": "any"}'

        with pytest.raises(json.JSONDecodeError):
            get_documents_by_property(filters=invalid_json)


class TestQueryDocumentsJsonString:
    """Tests for query_documents with AnnotatedQueryFilter type."""

    @pytest.fixture
    def setup_registry(self):
        """Set up mock registry for testing."""
        from obsidian_rag.mcp_server.tool_definitions import (
            MCPToolRegistry,
            _set_registry,
        )

        mock_db_manager = MagicMock()
        mock_embedding_provider = MagicMock()
        mock_settings = MagicMock()

        registry = MCPToolRegistry(
            db_manager=mock_db_manager,
            embedding_provider=mock_embedding_provider,
            settings=mock_settings,
        )
        _set_registry(registry)
        yield registry
        _set_registry(None)

    def test_query_documents_with_dict_filters(self, setup_registry):
        """Test query_documents accepts filters as dict."""
        from obsidian_rag.mcp_server.server import query_documents

        with patch("obsidian_rag.mcp_server.server.query_documents_tool") as mock_tool:
            mock_tool.return_value = {
                "results": [{"id": "doc1", "title": "Test"}],
                "total_count": 1,
                "has_more": False,
                "next_offset": None,
            }

            filters_dict = {"include_tags": ["personal"], "match_mode": "all"}
            result = query_documents(query="test", filters=filters_dict)

            assert result["total_count"] == 1
            mock_tool.assert_called_once()
            call_kwargs = mock_tool.call_args.kwargs
            assert call_kwargs["filters"] is not None

    def test_query_documents_with_none_filters(self, setup_registry):
        """Test query_documents works with None filters (default)."""
        from obsidian_rag.mcp_server.server import query_documents

        with patch("obsidian_rag.mcp_server.server.query_documents_tool") as mock_tool:
            mock_tool.return_value = {
                "results": [],
                "total_count": 0,
                "has_more": False,
                "next_offset": None,
            }

            result = query_documents(query="test", filters=None)

            assert result["total_count"] == 0
            mock_tool.assert_called_once()
            call_kwargs = mock_tool.call_args.kwargs
            assert call_kwargs["filters"] is None

    def test_query_documents_without_filters(self, setup_registry):
        """Test query_documents works without filters parameter."""
        from obsidian_rag.mcp_server.server import query_documents

        with patch("obsidian_rag.mcp_server.server.query_documents_tool") as mock_tool:
            mock_tool.return_value = {
                "results": [],
                "total_count": 0,
                "has_more": False,
                "next_offset": None,
            }

            result = query_documents(query="test")

            assert result["total_count"] == 0
            mock_tool.assert_called_once()
            call_kwargs = mock_tool.call_args.kwargs
            assert call_kwargs["filters"] is None

    def test_query_documents_with_complex_dict_filter(self, setup_registry):
        """Test query_documents handles complex filter dict."""
        from obsidian_rag.mcp_server.server import query_documents

        with patch("obsidian_rag.mcp_server.server.query_documents_tool") as mock_tool:
            mock_tool.return_value = {
                "results": [],
                "total_count": 0,
                "has_more": False,
                "next_offset": None,
            }

            # Complex filter with include_properties
            filters_dict = {
                "include_properties": [
                    {"path": "kind", "operator": "equals", "value": "note"}
                ],
                "include_tags": ["work", "urgent"],
                "match_mode": "any",
            }
            result = query_documents(query="test query", filters=filters_dict)

            assert result["total_count"] == 0
            mock_tool.assert_called_once()
            call_kwargs = mock_tool.call_args.kwargs
            assert call_kwargs["filters"] is not None


class TestGetDocumentToolRegistration:
    """Verify get_document tool is registered."""

    def test_get_document_registered_as_tool(self):
        """Verify tool is registered."""
        from obsidian_rag.mcp_server.server import _register_tools, get_document

        mock_mcp = MagicMock()
        registered = []

        def capture(func):
            registered.append(func)
            return func

        mock_mcp.tool.return_value = capture

        _register_tools(mock_mcp)

        assert get_document in registered

    def test_get_document_tool_call(self):
        """End-to-end through server wrapper."""
        from obsidian_rag.mcp_server.server import get_document

        with patch(
            "obsidian_rag.mcp_server.document_tools._get_registry"
        ) as mock_get_registry:
            mock_registry = MagicMock()
            mock_get_registry.return_value = mock_registry

            with patch(
                "obsidian_rag.mcp_server.document_tools.get_document_tool"
            ) as mock_tool:
                mock_tool.return_value = {
                    "id": "doc-1",
                    "vault_name": "test",
                    "content": "hello",
                }

                result = get_document(vault_name="test", file_path="note.md")

                assert result == {
                    "id": "doc-1",
                    "vault_name": "test",
                    "content": "hello",
                }
                mock_tool.assert_called_once_with(
                    mock_registry.db_manager,
                    vault_name="test",
                    file_path="note.md",
                    document_id=None,
                    include_content=True,
                )

    def test_get_document_error_response(self):
        """Error dict returned for not found."""
        from obsidian_rag.mcp_server.server import get_document

        with patch(
            "obsidian_rag.mcp_server.document_tools._get_registry"
        ) as mock_get_registry:
            mock_registry = MagicMock()
            mock_get_registry.return_value = mock_registry

            with patch(
                "obsidian_rag.mcp_server.document_tools.get_document_tool"
            ) as mock_tool:
                mock_tool.return_value = {
                    "success": False,
                    "error": "Document not found",
                }

                result = get_document(document_id="nonexistent-id")

                assert result["success"] is False
                assert "not found" in result["error"]


class TestListDocumentsToolRegistration:
    """Verify list_documents tool is registered."""

    def test_list_documents_registered_as_tool(self):
        """Verify tool is registered."""
        from obsidian_rag.mcp_server.server import _register_tools, list_documents

        mock_mcp = MagicMock()
        registered = []

        def capture(func):
            registered.append(func)
            return func

        mock_mcp.tool.return_value = capture

        _register_tools(mock_mcp)

        assert list_documents in registered

    def test_list_documents_tool_call(self):
        """End-to-end through server wrapper."""
        from obsidian_rag.mcp_server.server import list_documents

        with patch(
            "obsidian_rag.mcp_server.document_tools._get_registry"
        ) as mock_get_registry:
            mock_registry = MagicMock()
            mock_get_registry.return_value = mock_registry

            with patch(
                "obsidian_rag.mcp_server.document_tools.list_documents_tool"
            ) as mock_tool:
                mock_tool.return_value = {
                    "results": [{"id": "doc-1"}],
                    "total_count": 1,
                }

                result = list_documents(file_name="note.md", vault_name="test")

                assert result["total_count"] == 1
                mock_tool.assert_called_once_with(
                    mock_registry.db_manager,
                    file_name="note.md",
                    vault_name="test",
                    limit=20,
                    offset=0,
                )

    def test_list_documents_empty_results(self):
        """Empty list returned for no matches."""
        from obsidian_rag.mcp_server.server import list_documents

        with patch(
            "obsidian_rag.mcp_server.document_tools._get_registry"
        ) as mock_get_registry:
            mock_registry = MagicMock()
            mock_get_registry.return_value = mock_registry

            with patch(
                "obsidian_rag.mcp_server.document_tools.list_documents_tool"
            ) as mock_tool:
                mock_tool.return_value = {"results": [], "total_count": 0}

                result = list_documents(file_name="nonexistent.md")

                assert result["total_count"] == 0
                assert result["results"] == []


class TestGetDocumentsByPropertyValueError:
    """Tests for get_documents_by_property ValueError handling."""

    @pytest.fixture
    def setup_registry(self):
        """Set up mock registry for testing."""
        from obsidian_rag.mcp_server.tool_definitions import (
            MCPToolRegistry,
            _set_registry,
        )

        mock_db_manager = MagicMock()
        mock_embedding_provider = MagicMock()
        mock_settings = MagicMock()

        registry = MCPToolRegistry(
            db_manager=mock_db_manager,
            embedding_provider=mock_embedding_provider,
            settings=mock_settings,
        )
        _set_registry(registry)
        yield registry
        _set_registry(None)

    def test_get_documents_by_property_value_error_returns_dict(self, setup_registry):
        """Test get_documents_by_property catches ValueError and returns error dict."""
        from obsidian_rag.mcp_server.server import get_documents_by_property

        with patch(
            "obsidian_rag.mcp_server.tools.documents.get_documents_by_property"
        ) as mock_tool:
            mock_tool.side_effect = ValueError("Vault 'MissingVault' not found")

            result = get_documents_by_property(vault_name="MissingVault")

            assert result["success"] is False
            assert "MissingVault" in result["error"]
