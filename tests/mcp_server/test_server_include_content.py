"""Tests for include_content parameter propagation in MCP server wrappers."""

from unittest.mock import MagicMock, patch

import pytest


class TestQueryDocumentsWrapperIncludeContent:
    """Tests that query_documents wrapper correctly passes include_content."""

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

    def test_query_documents_wrapper_include_content_true(self, setup_registry) -> None:
        """Default include_content=True should propagate through pagination."""
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
            pagination = call_kwargs["pagination"]
            assert pagination.include_content is True
            assert call_kwargs["include_content"] is True

    def test_query_documents_wrapper_include_content_false(
        self, setup_registry
    ) -> None:
        """Explicit include_content=False should propagate through pagination."""
        from obsidian_rag.mcp_server.server import query_documents

        with patch("obsidian_rag.mcp_server.server.query_documents_tool") as mock_tool:
            mock_tool.return_value = {
                "results": [],
                "total_count": 0,
                "has_more": False,
                "next_offset": None,
            }

            result = query_documents(query="test", include_content=False)

            assert result["total_count"] == 0
            mock_tool.assert_called_once()
            call_kwargs = mock_tool.call_args.kwargs
            pagination = call_kwargs["pagination"]
            assert pagination.include_content is False
            assert call_kwargs["include_content"] is False


class TestGetDocumentsByTagWrapperIncludeContent:
    """Tests that get_documents_by_tag wrapper correctly passes include_content."""

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

    def test_get_documents_by_tag_wrapper_include_content_true(
        self, setup_registry
    ) -> None:
        """Default include_content=True should be in DocumentTagParams."""
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

            result = get_documents_by_tag()

            assert result["total_count"] == 0
            mock_handler.assert_called_once()
            call_params = mock_handler.call_args[0][1]
            assert call_params["include_content"] is True

    def test_get_documents_by_tag_wrapper_include_content_false(
        self, setup_registry
    ) -> None:
        """Explicit include_content=False should be in DocumentTagParams."""
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

            result = get_documents_by_tag(include_content=False)

            assert result["total_count"] == 0
            mock_handler.assert_called_once()
            call_params = mock_handler.call_args[0][1]
            assert call_params["include_content"] is False


class TestGetDocumentsByPropertyWrapperIncludeContent:
    """Tests that get_documents_by_property wrapper correctly passes include_content."""

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

    def test_get_documents_by_property_wrapper_include_content(
        self, setup_registry
    ) -> None:
        """include_content should propagate through PaginationParams."""
        from obsidian_rag.mcp_server.models import DocumentListResponse
        from obsidian_rag.mcp_server.server import get_documents_by_property

        with patch(
            "obsidian_rag.mcp_server.tools.documents.get_documents_by_property"
        ) as mock_tool:
            mock_tool.return_value = DocumentListResponse(
                results=[],
                total_count=0,
                has_more=False,
                next_offset=None,
            )

            result = get_documents_by_property(include_content=False)

            assert result["total_count"] == 0
            mock_tool.assert_called_once()
            call_kwargs = mock_tool.call_args.kwargs
            pagination = call_kwargs["pagination"]
            assert pagination.include_content is False
            assert call_kwargs["include_content"] is False
