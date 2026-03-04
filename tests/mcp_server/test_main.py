"""Tests for MCP server __main__ module."""

from unittest.mock import MagicMock, patch

import pytest


class TestMainFunction:
    """Tests for main() function."""

    @patch("obsidian_rag.mcp_server.__main__.get_settings")
    @patch("obsidian_rag.mcp_server.__main__.create_http_app")
    def test_main_successful_run(self, mock_create_http_app, mock_get_settings):
        """Test main function with successful execution."""
        import sys

        from obsidian_rag.mcp_server.__main__ import main

        mock_settings = MagicMock()
        mock_settings.mcp.token = "test-token"
        mock_settings.mcp.host = "127.0.0.1"
        mock_settings.mcp.port = 8000
        mock_get_settings.return_value = mock_settings

        mock_app = MagicMock()
        mock_create_http_app.return_value = mock_app

        # Mock uvicorn in sys.modules
        mock_uvicorn = MagicMock()
        original_uvicorn = sys.modules.get("uvicorn")
        sys.modules["uvicorn"] = mock_uvicorn

        try:
            main()  # This should run without raising SystemExit on success

            mock_uvicorn.run.assert_called_once_with(
                mock_app,
                host="127.0.0.1",
                port=8000,
            )
        finally:
            if original_uvicorn is not None:
                sys.modules["uvicorn"] = original_uvicorn
            elif "uvicorn" in sys.modules:
                del sys.modules["uvicorn"]

    @patch("obsidian_rag.mcp_server.__main__.get_settings")
    def test_main_settings_load_failure(self, mock_get_settings):
        """Test main function when settings fail to load."""
        from obsidian_rag.mcp_server.__main__ import main

        mock_get_settings.side_effect = Exception("Config error")

        with pytest.raises(SystemExit) as exc_info:
            main()

        assert exc_info.value.code == 1

    @patch("obsidian_rag.mcp_server.__main__.get_settings")
    def test_main_missing_token(self, mock_get_settings):
        """Test main function when MCP token is not configured."""
        from obsidian_rag.mcp_server.__main__ import main

        mock_settings = MagicMock()
        mock_settings.mcp.token = None
        mock_get_settings.return_value = mock_settings

        with pytest.raises(SystemExit) as exc_info:
            main()

        assert exc_info.value.code == 1

    @patch("obsidian_rag.mcp_server.__main__.get_settings")
    @patch("obsidian_rag.mcp_server.__main__.create_http_app")
    def test_main_create_app_failure(self, mock_create_http_app, mock_get_settings):
        """Test main function when HTTP app creation fails."""
        from obsidian_rag.mcp_server.__main__ import main

        mock_settings = MagicMock()
        mock_settings.mcp.token = "test-token"
        mock_get_settings.return_value = mock_settings

        mock_create_http_app.side_effect = Exception("Creation error")

        with pytest.raises(SystemExit) as exc_info:
            main()

        assert exc_info.value.code == 1

    @patch("obsidian_rag.mcp_server.__main__.get_settings")
    @patch("obsidian_rag.mcp_server.__main__.create_http_app")
    def test_main_missing_uvicorn(self, mock_create_http_app, mock_get_settings):
        """Test main function when uvicorn is not installed."""
        from obsidian_rag.mcp_server.__main__ import main

        mock_settings = MagicMock()
        mock_settings.mcp.token = "test-token"
        mock_get_settings.return_value = mock_settings

        mock_create_http_app.return_value = MagicMock()

        # Simulate uvicorn not being installed by making the import fail
        with patch.dict("sys.modules", {"uvicorn": None}):
            with pytest.raises(SystemExit) as exc_info:
                main()

            assert exc_info.value.code == 1
