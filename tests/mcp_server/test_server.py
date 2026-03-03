"""Unit tests for MCP server module."""

from unittest.mock import MagicMock, patch

import pytest

from obsidian_rag.config import Settings


class TestCreateMCPServer:
    """Tests for create_mcp_server function."""

    def test_requires_token(self):
        """Test that server creation fails without token."""
        from obsidian_rag.mcp_server.server import create_mcp_server

        settings = MagicMock()
        settings.mcp = MagicMock()
        settings.mcp.token = None

        with pytest.raises(ValueError, match="MCP token is required"):
            create_mcp_server(settings)

    @patch("obsidian_rag.mcp_server.server.DatabaseManager")
    @patch("obsidian_rag.mcp_server.server.ProviderFactory")
    def test_creates_server_with_token(self, mock_provider_factory, mock_db_manager):
        """Test that server is created with valid token."""
        from obsidian_rag.mcp_server.server import create_mcp_server

        settings = MagicMock()
        settings.mcp = MagicMock()
        settings.mcp.token = "test-token"
        settings.mcp.host = "0.0.0.0"
        settings.mcp.port = 8000
        settings.mcp.enable_health_check = True
        settings.mcp.stateless_http = False
        settings.database = MagicMock()
        settings.database.url = "postgresql://localhost/test"
        settings.endpoints = {}

        mock_db_instance = MagicMock()
        mock_db_manager.return_value = mock_db_instance

        server = create_mcp_server(settings)

        assert server is not None
        mock_db_manager.assert_called_once_with(settings.database.url)


class TestCreateHTTPApp:
    """Tests for create_http_app function."""

    @patch("obsidian_rag.mcp_server.server.create_mcp_server")
    def test_creates_app_with_cors(self, mock_create_server):
        """Test that HTTP app is created with CORS middleware."""
        from obsidian_rag.mcp_server.server import create_http_app

        settings = MagicMock()
        settings.mcp = MagicMock()
        settings.mcp.cors_origins = ["*"]
        settings.mcp.token = "test-token"

        mock_mcp = MagicMock()
        mock_http_app = MagicMock()
        mock_mcp.http_app.return_value = mock_http_app
        mock_create_server.return_value = mock_mcp

        app = create_http_app(settings)

        assert app is mock_http_app
        # Middleware is now passed to http_app() instead of add_middleware
        mock_mcp.http_app.assert_called_once()
        call_args = mock_mcp.http_app.call_args
        assert "middleware" in call_args.kwargs
