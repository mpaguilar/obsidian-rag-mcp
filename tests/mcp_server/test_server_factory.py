"""Tests for create_http_app_factory function in server module."""

from unittest.mock import MagicMock, patch

import pytest


class TestCreateHTTPAppFactory:
    """Tests for create_http_app_factory function."""

    @patch("obsidian_rag.mcp_server.server.create_http_app")
    @patch("obsidian_rag.mcp_server.server.get_settings")
    def test_factory_successful_execution(
        self, mock_get_settings, mock_create_http_app
    ):
        """Test factory function with successful execution."""
        from obsidian_rag.mcp_server.server import create_http_app_factory

        mock_settings = MagicMock()
        mock_settings.mcp.token = "test-token"
        mock_get_settings.return_value = mock_settings

        mock_app = MagicMock()
        mock_create_http_app.return_value = mock_app

        result = create_http_app_factory()

        assert result is mock_app
        mock_get_settings.assert_called_once()
        mock_create_http_app.assert_called_once_with(mock_settings)

    @patch("obsidian_rag.mcp_server.server.get_settings")
    def test_factory_settings_load_failure(self, mock_get_settings):
        """Test factory function when settings fail to load."""
        from obsidian_rag.mcp_server.server import create_http_app_factory

        mock_get_settings.side_effect = Exception("Config error")

        with pytest.raises(SystemExit) as exc_info:
            create_http_app_factory()

        assert exc_info.value.code == 1

    @patch("obsidian_rag.mcp_server.server.get_settings")
    def test_factory_missing_token(self, mock_get_settings):
        """Test factory function when MCP token is not configured."""
        from obsidian_rag.mcp_server.server import create_http_app_factory

        mock_settings = MagicMock()
        mock_settings.mcp.token = None
        mock_get_settings.return_value = mock_settings

        with pytest.raises(SystemExit) as exc_info:
            create_http_app_factory()

        assert exc_info.value.code == 1
