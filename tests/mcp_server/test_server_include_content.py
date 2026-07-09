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


def test_get_documents_by_tag_wrapper_rejects_include_content() -> None:
    """Calling get_documents_by_tag with include_content raises TypeError."""
    from obsidian_rag.mcp_server.server import get_documents_by_tag

    with pytest.raises(TypeError):
        get_documents_by_tag(include_content=False)


def test_get_documents_by_property_wrapper_rejects_include_content() -> None:
    """Calling get_documents_by_property with include_content raises TypeError."""
    from obsidian_rag.mcp_server.server import get_documents_by_property

    with pytest.raises(TypeError):
        get_documents_by_property(include_content=False)


def test_get_documents_by_tag_wrapper_omits_include_content_in_params_dict() -> None:
    """DocumentTagParams dict passed to handler must NOT contain include_content."""
    from obsidian_rag.mcp_server.server import get_documents_by_tag
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

    try:
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
            assert "include_content" not in call_params
    finally:
        _set_registry(None)


def test_get_documents_by_property_wrapper_omits_include_content_in_call() -> None:
    """Handler call must NOT pass include_content keyword argument."""
    from obsidian_rag.mcp_server.server import get_documents_by_property
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

    try:
        with patch(
            "obsidian_rag.mcp_server.server._get_documents_by_property_handler"
        ) as mock_handler:
            mock_handler.return_value = {
                "results": [],
                "total_count": 0,
                "has_more": False,
                "next_offset": None,
            }

            result = get_documents_by_property()

            assert result["total_count"] == 0
            mock_handler.assert_called_once()
            call_kwargs = mock_handler.call_args.kwargs
            assert "include_content" not in call_kwargs
    finally:
        _set_registry(None)
