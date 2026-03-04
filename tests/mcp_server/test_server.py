"""Unit tests for MCP server module."""

from datetime import date, timedelta
from unittest.mock import MagicMock, Mock, patch

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


class TestCreateEmbeddingProvider:
    """Tests for _create_embedding_provider function."""

    def test_create_embedding_provider_success(self):
        """Test successful creation of embedding provider."""
        from obsidian_rag.mcp_server.server import _create_embedding_provider

        settings = MagicMock()
        settings.endpoints = {"embedding": MagicMock()}
        settings.endpoints["embedding"].provider = "openai"
        settings.endpoints["embedding"].api_key = "test-key"
        settings.endpoints["embedding"].model = "text-embedding-3-small"
        settings.endpoints["embedding"].base_url = None

        with patch("obsidian_rag.mcp_server.server.ProviderFactory") as mock_factory:
            mock_provider = MagicMock()
            mock_factory.create_embedding_provider.return_value = mock_provider

            result = _create_embedding_provider(settings)

            assert result is mock_provider
            mock_factory.create_embedding_provider.assert_called_once()

    def test_create_embedding_provider_no_config(self):
        """Test when no embedding config exists."""
        from obsidian_rag.mcp_server.server import _create_embedding_provider

        settings = MagicMock()
        settings.endpoints = {}

        result = _create_embedding_provider(settings)

        assert result is None

    def test_create_embedding_provider_failure(self):
        """Test when provider creation fails."""
        from obsidian_rag.mcp_server.server import _create_embedding_provider

        settings = MagicMock()
        settings.endpoints = {"embedding": MagicMock()}
        settings.endpoints["embedding"].provider = "openai"

        with patch("obsidian_rag.mcp_server.server.ProviderFactory") as mock_factory:
            mock_factory.create_embedding_provider.side_effect = Exception("Failed")

            result = _create_embedding_provider(settings)

            assert result is None


class TestTaskToolHandlers:
    """Tests for task tool handlers."""

    def test_get_incomplete_tasks_handler(self):
        """Test _get_incomplete_tasks_handler."""
        from obsidian_rag.mcp_server.server import _get_incomplete_tasks_handler

        mock_db_manager = MagicMock()
        mock_session = MagicMock()
        mock_db_manager.get_session.return_value.__enter__ = MagicMock(
            return_value=mock_session
        )
        mock_db_manager.get_session.return_value.__exit__ = MagicMock(
            return_value=False
        )

        with patch(
            "obsidian_rag.mcp_server.server.get_incomplete_tasks_tool"
        ) as mock_tool:
            mock_result = MagicMock()
            mock_result.model_dump.return_value = {"tasks": []}
            mock_tool.return_value = mock_result

            result = _get_incomplete_tasks_handler(mock_db_manager, 20, 0, False)

            assert result == {"tasks": []}
            mock_tool.assert_called_once_with(
                session=mock_session, limit=20, offset=0, include_cancelled=False
            )

    def test_get_tasks_due_this_week_handler(self):
        """Test _get_tasks_due_this_week_handler."""
        from obsidian_rag.mcp_server.server import _get_tasks_due_this_week_handler

        mock_db_manager = MagicMock()
        mock_session = MagicMock()
        mock_db_manager.get_session.return_value.__enter__ = MagicMock(
            return_value=mock_session
        )
        mock_db_manager.get_session.return_value.__exit__ = MagicMock(
            return_value=False
        )

        with patch(
            "obsidian_rag.mcp_server.server.get_tasks_due_this_week_tool"
        ) as mock_tool:
            mock_result = MagicMock()
            mock_result.model_dump.return_value = {"tasks": []}
            mock_tool.return_value = mock_result

            result = _get_tasks_due_this_week_handler(mock_db_manager, 20, 0, True)

            assert result == {"tasks": []}
            mock_tool.assert_called_once_with(
                session=mock_session, limit=20, offset=0, include_completed=True
            )

    def test_get_tasks_by_tag_handler(self):
        """Test _get_tasks_by_tag_handler."""
        from obsidian_rag.mcp_server.server import _get_tasks_by_tag_handler

        mock_db_manager = MagicMock()
        mock_session = MagicMock()
        mock_db_manager.get_session.return_value.__enter__ = MagicMock(
            return_value=mock_session
        )
        mock_db_manager.get_session.return_value.__exit__ = MagicMock(
            return_value=False
        )

        with patch("obsidian_rag.mcp_server.server.get_tasks_by_tag_tool") as mock_tool:
            mock_result = MagicMock()
            mock_result.model_dump.return_value = {"tasks": []}
            mock_tool.return_value = mock_result

            result = _get_tasks_by_tag_handler(mock_db_manager, "test-tag", 20, 0)

            assert result == {"tasks": []}
            mock_tool.assert_called_once_with(
                session=mock_session, tag="test-tag", limit=20, offset=0
            )

    def test_get_completed_tasks_handler_with_date(self):
        """Test _get_completed_tasks_handler with valid date."""
        from obsidian_rag.mcp_server.server import _get_completed_tasks_handler

        mock_db_manager = MagicMock()
        mock_session = MagicMock()
        mock_db_manager.get_session.return_value.__enter__ = MagicMock(
            return_value=mock_session
        )
        mock_db_manager.get_session.return_value.__exit__ = MagicMock(
            return_value=False
        )

        with patch(
            "obsidian_rag.mcp_server.server.get_completed_tasks_tool"
        ) as mock_tool:
            mock_result = MagicMock()
            mock_result.model_dump.return_value = {"tasks": []}
            mock_tool.return_value = mock_result

            result = _get_completed_tasks_handler(mock_db_manager, 20, 0, "2024-01-01")

            assert result == {"tasks": []}
            mock_tool.assert_called_once()
            call_kwargs = mock_tool.call_args.kwargs
            assert call_kwargs["completed_since"] == date(2024, 1, 1)

    def test_get_completed_tasks_handler_with_invalid_date(self):
        """Test _get_completed_tasks_handler with invalid date."""
        from obsidian_rag.mcp_server.server import _get_completed_tasks_handler

        mock_db_manager = MagicMock()
        mock_session = MagicMock()
        mock_db_manager.get_session.return_value.__enter__ = MagicMock(
            return_value=mock_session
        )
        mock_db_manager.get_session.return_value.__exit__ = MagicMock(
            return_value=False
        )

        with patch(
            "obsidian_rag.mcp_server.server.get_completed_tasks_tool"
        ) as mock_tool:
            mock_result = MagicMock()
            mock_result.model_dump.return_value = {"tasks": []}
            mock_tool.return_value = mock_result

            result = _get_completed_tasks_handler(
                mock_db_manager, 20, 0, "invalid-date"
            )

            assert result == {"tasks": []}
            mock_tool.assert_called_once()
            call_kwargs = mock_tool.call_args.kwargs
            assert call_kwargs["completed_since"] is None


class TestToolRegistration:
    """Tests for tool registration functions."""

    def test_register_task_tools(self):
        """Test _register_task_tools."""
        from obsidian_rag.mcp_server.server import _register_task_tools

        mock_mcp = MagicMock()
        mock_db_manager = MagicMock()

        _register_task_tools(mock_mcp, mock_db_manager)

        # Should register 4 task tools
        assert mock_mcp.tool.call_count == 4

    def test_register_document_tools_with_provider(self):
        """Test _register_document_tools with embedding provider."""
        from obsidian_rag.mcp_server.server import _register_document_tools

        mock_mcp = MagicMock()
        mock_db_manager = MagicMock()
        mock_embedding_provider = MagicMock()
        mock_embedding_provider.generate_embedding.return_value = [0.1] * 1536

        _register_document_tools(mock_mcp, mock_db_manager, mock_embedding_provider)

        # Should register 3 document tools (query_documents, get_documents_by_tag, get_all_tags)
        assert mock_mcp.tool.call_count == 3

    def test_register_document_tools_without_provider(self):
        """Test _register_document_tools without embedding provider."""
        from obsidian_rag.mcp_server.server import _register_document_tools

        mock_mcp = MagicMock()
        mock_db_manager = MagicMock()

        _register_document_tools(mock_mcp, mock_db_manager, None)

        # Should register 3 document tools (query_documents, get_documents_by_tag, get_all_tags)
        assert mock_mcp.tool.call_count == 3

    def test_register_health_check(self):
        """Test _register_health_check."""
        from obsidian_rag.mcp_server.server import _register_health_check

        mock_mcp = MagicMock()
        mock_db_manager = MagicMock()

        _register_health_check(mock_mcp, mock_db_manager)

        mock_mcp.custom_route.assert_called_once()
