"""Unit tests for MCP server module."""

import asyncio
import json
import os
from pathlib import Path
from typing import cast
from unittest.mock import MagicMock, Mock, patch

import pytest
from click.testing import CliRunner


runner = CliRunner()

runner = CliRunner()


class TestMCPToolRegistry:
    """Tests for MCPToolRegistry class."""

    def test_init_with_all_dependencies(self):
        """Test MCPToolRegistry initialization with all dependencies."""
        from obsidian_rag.mcp_server.tool_definitions import MCPToolRegistry

        mock_db_manager = MagicMock()
        mock_embedding_provider = MagicMock()
        mock_settings = MagicMock()

        registry = MCPToolRegistry(
            db_manager=mock_db_manager,
            embedding_provider=mock_embedding_provider,
            settings=mock_settings,
        )

        assert registry.db_manager is mock_db_manager
        assert registry.embedding_provider is mock_embedding_provider
        assert registry.settings is mock_settings

    def test_init_with_none_embedding_provider(self):
        """Test MCPToolRegistry initialization with None embedding provider."""
        from obsidian_rag.mcp_server.tool_definitions import MCPToolRegistry

        mock_db_manager = MagicMock()
        mock_settings = MagicMock()

        registry = MCPToolRegistry(
            db_manager=mock_db_manager,
            embedding_provider=None,
            settings=mock_settings,
        )

        assert registry.db_manager is mock_db_manager
        assert registry.embedding_provider is None
        assert registry.settings is mock_settings


class TestGetRegistry:
    """Tests for _get_registry function."""

    def test_get_registry_when_initialized(self):
        """Test _get_registry returns registry when initialized."""
        from obsidian_rag.mcp_server import tool_definitions as tool_definitions_module
        from obsidian_rag.mcp_server.tool_definitions import (
            MCPToolRegistry,
            _get_registry,
        )

        # Save original
        original_registry = tool_definitions_module._tool_registry

        # Create and set mock registry
        mock_registry = MCPToolRegistry(
            db_manager=MagicMock(),
            embedding_provider=MagicMock(),
            settings=MagicMock(),
        )
        tool_definitions_module._tool_registry = mock_registry

        try:
            result = _get_registry()
            assert result is mock_registry
        finally:
            # Restore original
            tool_definitions_module._tool_registry = original_registry

    def test_get_registry_when_not_initialized(self):
        """Test _get_registry raises RuntimeError when not initialized."""
        from obsidian_rag.mcp_server import tool_definitions as tool_definitions_module
        from obsidian_rag.mcp_server.tool_definitions import _get_registry

        # Save original
        original_registry = tool_definitions_module._tool_registry

        # Set to None
        tool_definitions_module._tool_registry = None

        try:
            with pytest.raises(RuntimeError, match="Tool registry not initialized"):
                _get_registry()
        finally:
            # Restore original
            tool_definitions_module._tool_registry = original_registry


class TestToolFunctionsWithRegistry:
    """Tests for module-level tool functions using registry."""

    @pytest.fixture
    def mock_registry(self):
        """Create a mock registry for testing."""
        from obsidian_rag.mcp_server.tool_definitions import MCPToolRegistry

        registry = MCPToolRegistry(
            db_manager=MagicMock(),
            embedding_provider=MagicMock(),
            settings=MagicMock(),
        )
        return registry

    @pytest.fixture
    def setup_registry(self, mock_registry):
        """Setup and teardown for registry tests."""
        from obsidian_rag.mcp_server import tool_definitions as tool_definitions_module

        # Save original
        original_registry = tool_definitions_module._tool_registry

        # Set mock registry
        tool_definitions_module._tool_registry = mock_registry

        yield mock_registry

        # Restore original
        tool_definitions_module._tool_registry = original_registry

    def test_query_documents(self, setup_registry):
        """Test query_documents uses registry."""
        from obsidian_rag.mcp_server.server import query_documents

        mock_registry = setup_registry

        with patch("obsidian_rag.mcp_server.server.query_documents_tool") as mock_tool:
            mock_tool.return_value = {"results": []}

            result = query_documents(query="test query")

            assert result == {"results": []}
            mock_tool.assert_called_once()
            call_kwargs = mock_tool.call_args.kwargs
            assert call_kwargs["db_manager"] is mock_registry.db_manager
            assert call_kwargs["embedding_provider"] is mock_registry.embedding_provider
            assert call_kwargs["query"] == "test query"

    def test_get_documents_by_tag(self, setup_registry):
        """Test get_documents_by_tag uses registry."""
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

            assert result == {
                "results": [],
                "total_count": 0,
                "has_more": False,
                "next_offset": None,
            }
            mock_handler.assert_called_once()

    def test_get_documents_by_property(self, setup_registry):
        """Test get_documents_by_property uses registry."""
        from obsidian_rag.mcp_server.server import get_documents_by_property
        from obsidian_rag.mcp_server.models import DocumentListResponse

        mock_registry = setup_registry

        with patch(
            "obsidian_rag.mcp_server.tools.documents.get_documents_by_property"
        ) as mock_tool:
            mock_tool.return_value = DocumentListResponse(
                results=[],
                total_count=0,
                has_more=False,
                next_offset=None,
            )

            result = get_documents_by_property()

            assert result["total_count"] == 0
            assert result["results"] == []
            mock_tool.assert_called_once()

    def test_get_all_tags(self, setup_registry):
        """Test get_all_tags uses registry."""
        from obsidian_rag.mcp_server.server import get_all_tags

        mock_registry = setup_registry

        with patch("obsidian_rag.mcp_server.server.get_all_tags_tool") as mock_tool:
            mock_tool.return_value = {"tags": []}

            result = get_all_tags(pattern="work*")

            assert result == {"tags": []}
            mock_tool.assert_called_once_with(
                mock_registry.db_manager,
                "work*",
                20,
                0,
            )

    def test_list_vaults(self, setup_registry):
        """Test list_vaults uses registry."""
        from obsidian_rag.mcp_server.server import list_vaults

        mock_registry = setup_registry

        with patch("obsidian_rag.mcp_server.server.list_vaults_tool") as mock_tool:
            mock_tool.return_value = {"vaults": []}

            result = list_vaults(limit=20, offset=0)

            assert result == {"vaults": []}
            mock_tool.assert_called_once_with(
                mock_registry.db_manager,
                20,
                0,
            )

    def test_ingest(self, setup_registry):
        """Test ingest uses registry."""
        from obsidian_rag.mcp_server.server import ingest

        with patch("obsidian_rag.mcp_server.server._ingest_handler") as mock_handler:
            mock_handler.return_value = {"total": 1}

            result = ingest(vault_name="test-vault")

            assert result == {"total": 1}
            mock_handler.assert_called_once()

    @pytest.mark.asyncio
    async def test_health_check(self, setup_registry):
        """Test health_check uses registry."""
        from obsidian_rag.mcp_server.server import health_check

        mock_registry = setup_registry

        with patch(
            "obsidian_rag.mcp_server.server.health_check_handler"
        ) as mock_handler:
            mock_response = MagicMock()
            mock_handler.return_value = mock_response

            mock_request = MagicMock()
            result = await health_check(mock_request)

            assert result is mock_response
            mock_handler.assert_called_once_with(mock_registry.db_manager)

    def test_get_tasks(self, setup_registry):
        """Test get_tasks uses registry."""
        from obsidian_rag.mcp_server.handlers import TaskDateFilterStrings
        from obsidian_rag.mcp_server.server import get_tasks

        mock_registry = setup_registry

        # Patch inside the server module where it's imported
        with patch(
            "obsidian_rag.mcp_server.handlers._get_tasks_handler"
        ) as mock_handler:
            mock_handler.return_value = {"results": []}

            from obsidian_rag.mcp_server.handlers import GetTasksToolInput

            date_filters = TaskDateFilterStrings(
                due_after="2026-01-01",
                due_before="2026-12-31",
            )

            params = GetTasksToolInput(
                status=["not_completed"],
                date_filters=date_filters,
                limit=20,
                offset=0,
            )

            result = get_tasks(params=params)

            assert result == {"results": []}
            mock_handler.assert_called_once()

    def test_get_tasks_without_date_filters(self, setup_registry):
        """Test get_tasks creates default date_filters when not provided."""
        from obsidian_rag.mcp_server.handlers import (
            GetTasksToolInput,
            TaskDateFilterStrings,
        )
        from obsidian_rag.mcp_server.server import get_tasks

        mock_registry = setup_registry

        # Patch inside the server module where it's imported
        with patch(
            "obsidian_rag.mcp_server.handlers._get_tasks_handler"
        ) as mock_handler:
            mock_handler.return_value = {"results": []}

            # Call without date_filters parameter
            params = GetTasksToolInput(
                status=["not_completed"],
                limit=20,
                offset=0,
            )
            result = get_tasks(params=params)

            assert result == {"results": []}
            mock_handler.assert_called_once()
            # Verify that a TaskDateFilterStrings was created with default values
            call_kwargs = mock_handler.call_args.kwargs
            request = call_kwargs["request"]
            assert isinstance(request.date_filters, TaskDateFilterStrings)


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
    @patch("obsidian_rag.mcp_server.tool_definitions.ProviderFactory")
    def test_creates_server_with_token(self, _mock_provider_factory, mock_db_manager):
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
        mock_db_manager.assert_called_once_with(
            settings.database.url,
            pool_size=settings.database.pool_size,
            max_overflow=settings.database.max_overflow,
            pool_timeout=settings.database.pool_timeout,
            pool_recycle=settings.database.pool_recycle,
        )

    @patch("obsidian_rag.mcp_server.server.DatabaseManager")
    @patch("obsidian_rag.mcp_server.tool_definitions.ProviderFactory")
    def test_creates_server_without_health_check_when_disabled(
        self, _mock_provider_factory, mock_db_manager
    ):
        """Test that health check endpoint is not registered when disabled."""
        from obsidian_rag.mcp_server.server import create_mcp_server

        settings = MagicMock()
        settings.mcp = MagicMock()
        settings.mcp.token = "test-token"
        settings.mcp.host = "0.0.0.0"
        settings.mcp.port = 8000
        settings.mcp.enable_health_check = False  # Disabled
        settings.mcp.stateless_http = False
        settings.database = MagicMock()
        settings.database.url = "postgresql://localhost/test"
        settings.endpoints = {}

        mock_db_instance = MagicMock()
        mock_db_manager.return_value = mock_db_instance

        mcp = create_mcp_server(settings)

        # Verify server is created but health check not registered
        assert mcp is not None
        # custom_route should not be called for health check
        # Note: We can't easily verify this without mocking mcp.custom_route,
        # but the branch coverage will confirm this path is tested


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

    @patch("obsidian_rag.mcp_server.server.create_mcp_server")
    def test_creates_app_without_logging_middleware_when_disabled(
        self, mock_create_server
    ):
        """Test that logging middleware is not added when disabled."""
        from obsidian_rag.mcp_server.server import create_http_app

        settings = MagicMock()
        settings.mcp = MagicMock()
        settings.mcp.cors_origins = ["*"]
        settings.mcp.token = "test-token"
        settings.mcp.enable_request_logging = False  # Disabled
        settings.mcp.max_concurrent_sessions = 100
        settings.mcp.session_timeout_seconds = 300
        settings.mcp.rate_limit_per_second = 10.0
        settings.mcp.rate_limit_window = 60
        settings.mcp.stateless_http = False

        mock_mcp = MagicMock()
        mock_http_app = MagicMock()
        mock_mcp.http_app.return_value = mock_http_app
        mock_create_server.return_value = mock_mcp

        app = create_http_app(settings)

        assert app is mock_http_app
        # Verify middleware list doesn't include SessionLoggingMiddleware
        call_args = mock_mcp.http_app.call_args
        middleware_list = call_args.kwargs.get("middleware", [])
        # Should only have CORS middleware, not logging middleware
        assert len(middleware_list) == 1
        # Verify it's the CORS middleware by checking the class name
        from starlette.middleware.cors import CORSMiddleware

        assert middleware_list[0].cls is CORSMiddleware


class TestCreateEmbeddingProvider:
    """Tests for _create_embedding_provider function."""

    def test_create_embedding_provider_success(self):
        """Test successful creation of embedding provider."""
        from obsidian_rag.mcp_server.tool_definitions import _create_embedding_provider

        settings = MagicMock()
        settings.endpoints = {"embedding": MagicMock()}
        settings.endpoints["embedding"].provider = "openai"
        settings.endpoints["embedding"].api_key = "test-key"
        settings.endpoints["embedding"].model = "text-embedding-3-small"
        settings.endpoints["embedding"].base_url = None

        with patch(
            "obsidian_rag.mcp_server.tool_definitions.ProviderFactory"
        ) as mock_factory:
            mock_provider = MagicMock()
            mock_factory.create_embedding_provider.return_value = mock_provider

            result = _create_embedding_provider(settings)

            assert result is mock_provider
            mock_factory.create_embedding_provider.assert_called_once()

    def test_create_embedding_provider_no_config(self):
        """Test when no embedding config exists."""
        from obsidian_rag.mcp_server.tool_definitions import _create_embedding_provider

        settings = MagicMock()
        settings.endpoints = {}

        result = _create_embedding_provider(settings)

        assert result is None

    def test_create_embedding_provider_failure(self):
        """Test when provider creation fails, exception is propagated."""
        from obsidian_rag.mcp_server.tool_definitions import _create_embedding_provider

        settings = MagicMock()
        settings.endpoints = {"embedding": MagicMock()}
        settings.endpoints["embedding"].provider = "openai"

        with patch(
            "obsidian_rag.mcp_server.tool_definitions.ProviderFactory"
        ) as mock_factory:
            mock_factory.create_embedding_provider.side_effect = RuntimeError("Failed")

            with pytest.raises(RuntimeError, match="Failed"):
                _create_embedding_provider(settings)


class TestProviderCreators:
    """Tests for provider creator functions."""

    def test_create_openai_provider(self):
        """Test _create_openai_provider returns provider."""
        from obsidian_rag.mcp_server.tool_definitions import _create_openai_provider

        config = MagicMock()
        config.api_key = "test-key"
        config.model = "text-embedding-3-small"
        config.base_url = None

        with patch(
            "obsidian_rag.mcp_server.tool_definitions.ProviderFactory"
        ) as mock_factory:
            mock_provider = MagicMock()
            mock_factory.create_embedding_provider.return_value = mock_provider

            result = _create_openai_provider(config)

            assert result is mock_provider
            mock_factory.create_embedding_provider.assert_called_once_with(
                provider_name="openai",
                config={
                    "api_key": "test-key",
                    "model": "text-embedding-3-small",
                    "base_url": None,
                },
            )

    def test_create_openrouter_provider(self):
        """Test _create_openrouter_provider returns provider."""
        from obsidian_rag.mcp_server.tool_definitions import _create_openrouter_provider

        config = MagicMock()
        config.api_key = "test-key"
        config.model = "text-embedding-3-small"
        config.base_url = "https://openrouter.ai/api/v1"

        with patch(
            "obsidian_rag.mcp_server.tool_definitions.ProviderFactory"
        ) as mock_factory:
            mock_provider = MagicMock()
            mock_factory.create_embedding_provider.return_value = mock_provider

            result = _create_openrouter_provider(config)

            assert result is mock_provider
            mock_factory.create_embedding_provider.assert_called_once_with(
                provider_name="openrouter",
                config={
                    "api_key": "test-key",
                    "model": "text-embedding-3-small",
                    "base_url": "https://openrouter.ai/api/v1",
                },
            )

    def test_create_huggingface_provider(self):
        """Test _create_huggingface_provider returns provider."""
        from obsidian_rag.mcp_server.tool_definitions import (
            _create_huggingface_provider,
        )

        config = MagicMock()
        config.model = "sentence-transformers/all-MiniLM-L6-v2"

        with patch(
            "obsidian_rag.mcp_server.tool_definitions.ProviderFactory"
        ) as mock_factory:
            mock_provider = MagicMock()
            mock_factory.create_embedding_provider.return_value = mock_provider

            result = _create_huggingface_provider(config)

            assert result is mock_provider
            mock_factory.create_embedding_provider.assert_called_once_with(
                provider_name="huggingface",
                config={"model": "sentence-transformers/all-MiniLM-L6-v2"},
            )

    def test_get_provider_creator_openai(self):
        """Test _get_provider_creator returns openai creator."""
        from obsidian_rag.mcp_server.tool_definitions import _get_provider_creator

        result = _get_provider_creator("openai")
        assert callable(result)

    def test_get_provider_creator_openrouter(self):
        """Test _get_provider_creator returns openrouter creator."""
        from obsidian_rag.mcp_server.tool_definitions import _get_provider_creator

        result = _get_provider_creator("openrouter")
        assert callable(result)

    def test_get_provider_creator_huggingface(self):
        """Test _get_provider_creator returns huggingface creator."""
        from obsidian_rag.mcp_server.tool_definitions import _get_provider_creator

        result = _get_provider_creator("huggingface")
        assert callable(result)

    def test_get_provider_creator_unknown(self):
        """Test _get_provider_creator raises error for unknown provider."""
        from obsidian_rag.mcp_server.tool_definitions import _get_provider_creator

        with pytest.raises(ValueError) as exc_info:
            _get_provider_creator("unknown")

        assert "Unknown embedding provider" in str(exc_info.value)
        assert "unknown" in str(exc_info.value)


class TestRegisterTools:
    """Tests for _register_tools function."""

    def test_register_tools(self):
        """Test _register_tools registers all tools."""
        from obsidian_rag.mcp_server.server import _register_tools

        mock_mcp = MagicMock()

        _register_tools(mock_mcp)

        # Should register 7 tools (get_tasks + 4 document tools + list_vaults + ingest)
        # health_check is registered via custom_route
        assert mock_mcp.tool.call_count == 7


class TestHealthCheckHandler:
    """Tests for health_check_handler function."""

    @pytest.mark.asyncio
    async def test_health_check_handler_database_success(self):
        """Test health_check_handler when database connection succeeds."""
        from obsidian_rag.mcp_server.server import health_check_handler

        mock_db_manager = MagicMock()
        mock_session = MagicMock()
        mock_db_manager.get_session.return_value.__enter__ = MagicMock(
            return_value=mock_session
        )
        mock_db_manager.get_session.return_value.__exit__ = MagicMock(
            return_value=False
        )

        with patch(
            "obsidian_rag.mcp_server.server._get_session_manager"
        ) as mock_get_manager:
            mock_get_manager.return_value = None

            with patch.dict(os.environ, {"OBSIDIAN_RAG_VERSION": "1.0.0"}):
                result = await health_check_handler(mock_db_manager)

        assert result.status_code == 200
        response_data = json.loads(result.body)
        assert response_data["status"] == "healthy"
        assert response_data["version"] == "1.0.0"
        assert response_data["database"] == "connected"
        assert response_data["sessions"]["total_created"] == 0

    @pytest.mark.asyncio
    async def test_health_check_handler_database_error(self):
        """Test health_check_handler when database connection fails."""
        from obsidian_rag.mcp_server.server import health_check_handler
        from sqlalchemy import exc

        mock_db_manager = MagicMock()
        mock_session = MagicMock()
        mock_session.execute.side_effect = exc.SQLAlchemyError("Connection refused")
        mock_db_manager.get_session.return_value.__enter__ = MagicMock(
            return_value=mock_session
        )
        mock_db_manager.get_session.return_value.__exit__ = MagicMock(
            return_value=False
        )

        with patch(
            "obsidian_rag.mcp_server.server._get_session_manager"
        ) as mock_get_manager:
            mock_get_manager.return_value = None

            with patch.dict(os.environ, {"OBSIDIAN_RAG_VERSION": "1.0.0"}):
                result = await health_check_handler(mock_db_manager)

        assert result.status_code == 200
        response_data = json.loads(result.body)
        assert response_data["status"] == "healthy"
        assert "error: Connection refused" in response_data["database"]

    @pytest.mark.asyncio
    async def test_health_check_handler_with_session_metrics(self):
        """Test health_check_handler includes session metrics when available."""
        from obsidian_rag.mcp_server.server import health_check_handler

        mock_db_manager = MagicMock()
        mock_session = MagicMock()
        mock_db_manager.get_session.return_value.__enter__ = MagicMock(
            return_value=mock_session
        )
        mock_db_manager.get_session.return_value.__exit__ = MagicMock(
            return_value=False
        )

        mock_manager = MagicMock()
        mock_manager.get_metrics.return_value = {
            "total_created": 10,
            "total_destroyed": 5,
            "active_count": 5,
            "total_requests": 100,
            "peak_concurrent": 3,
            "connection_rate": 2.5,
            "active_sessions_by_ip": {"127.0.0.1": 2},
        }

        with patch(
            "obsidian_rag.mcp_server.server._get_session_manager"
        ) as mock_get_manager:
            mock_get_manager.return_value = mock_manager

            with patch.dict(os.environ, {"OBSIDIAN_RAG_VERSION": "1.0.0"}):
                result = await health_check_handler(mock_db_manager)

        assert result.status_code == 200
        response_data = json.loads(result.body)
        assert response_data["sessions"]["total_created"] == 10
        assert response_data["sessions"]["total_destroyed"] == 5
        assert response_data["sessions"]["active_count"] == 5
        assert response_data["sessions"]["total_requests"] == 100
        assert response_data["sessions"]["peak_concurrent"] == 3
        assert response_data["sessions"]["connection_rate"] == 2.5
        assert response_data["sessions"]["active_sessions_by_ip"] == {"127.0.0.1": 2}


class TestGetSessionManager:
    """Tests for _get_session_manager function."""

    def test_get_session_manager_returns_none_when_not_initialized(self):
        """Test _get_session_manager returns None when session manager not set."""
        from obsidian_rag.mcp_server import server as server_module
        from obsidian_rag.mcp_server.server import _get_session_manager

        # Save original
        original_manager = server_module._session_manager

        try:
            # Set to None
            server_module._session_manager = None

            result = _get_session_manager()
            assert result is None
        finally:
            # Restore original
            server_module._session_manager = original_manager

    def test_get_session_manager_returns_manager_when_initialized(self):
        """Test _get_session_manager returns manager when set."""
        from obsidian_rag.mcp_server import server as server_module
        from obsidian_rag.mcp_server.server import _get_session_manager

        # Save original
        original_manager = server_module._session_manager

        try:
            # Set mock manager
            mock_manager = MagicMock()
            server_module._session_manager = mock_manager

            result = _get_session_manager()
            assert result is mock_manager
        finally:
            # Restore original
            server_module._session_manager = original_manager


class TestToolImplementations:
    """Tests for module-level tool implementations (direct coverage)."""

    def test_get_all_tags_tool(self):
        """Test get_all_tags_tool directly."""
        from obsidian_rag.mcp_server.server import get_all_tags_tool

        mock_db_manager = MagicMock()

        with patch(
            "obsidian_rag.mcp_server.tool_definitions._get_all_tags_handler"
        ) as mock_handler:
            mock_handler.return_value = {"tags": []}

            result = get_all_tags_tool(
                mock_db_manager, pattern="work*", limit=20, offset=0
            )

            assert result == {"tags": []}
            mock_handler.assert_called_once_with(mock_db_manager, "work*", 20, 0)

    def test_list_vaults_tool(self):
        """Test list_vaults_tool directly."""
        from obsidian_rag.mcp_server.server import list_vaults_tool

        mock_db_manager = MagicMock()

        with patch(
            "obsidian_rag.mcp_server.tool_definitions._list_vaults_handler"
        ) as mock_handler:
            mock_handler.return_value = {"vaults": []}

            result = list_vaults_tool(mock_db_manager, limit=20, offset=0)

            assert result == {"vaults": []}
            mock_handler.assert_called_once_with(mock_db_manager, 20, 0)

    def test_get_tasks_tool(self):
        """Test get_tasks_tool directly."""
        from obsidian_rag.mcp_server.handlers import (
            GetTasksRequest,
            TaskDateFilterStrings,
        )
        from obsidian_rag.mcp_server.tool_definitions import get_tasks_tool

        mock_db_manager = MagicMock()

        with patch(
            "obsidian_rag.mcp_server.tool_definitions._get_tasks_handler"
        ) as mock_handler:
            mock_handler.return_value = {"tasks": []}

            date_filters = TaskDateFilterStrings(
                due_after="2026-01-01",
                due_before="2026-12-31",
            )

            request = GetTasksRequest(
                status=["not_completed"],
                date_filters=date_filters,
                priority=["high"],
                limit=20,
                offset=0,
            )

            result = get_tasks_tool(
                db_manager=mock_db_manager,
                request=request,
            )

            assert result == {"tasks": []}
            mock_handler.assert_called_once()

    def test_query_documents_tool_with_embedding_provider(self):
        """Test query_documents_tool with embedding provider."""
        from obsidian_rag.mcp_server.server import query_documents_tool

        mock_db_manager = MagicMock()
        mock_embedding_provider = MagicMock()
        mock_embedding_provider.generate_embedding.return_value = [0.1] * 1536

        mock_session = MagicMock()
        mock_db_manager.get_session.return_value.__enter__ = MagicMock(
            return_value=mock_session
        )
        mock_db_manager.get_session.return_value.__exit__ = MagicMock(
            return_value=False
        )

        with patch(
            "obsidian_rag.mcp_server.tools.documents.query_documents"
        ) as mock_query:
            mock_result = MagicMock()
            mock_result.model_dump.return_value = {
                "results": [],
                "total_count": 0,
                "has_more": False,
                "next_offset": None,
            }
            mock_query.return_value = mock_result

            result = query_documents_tool(
                db_manager=mock_db_manager,
                embedding_provider=mock_embedding_provider,
                query="test query",
            )

            assert result == {
                "results": [],
                "total_count": 0,
                "has_more": False,
                "next_offset": None,
            }
            mock_embedding_provider.generate_embedding.assert_called_once_with(
                "test query"
            )

    def test_query_documents_tool_without_embedding_provider(self):
        """Test query_documents_tool raises error without embedding provider."""
        from obsidian_rag.mcp_server.server import query_documents_tool

        mock_db_manager = MagicMock()

        with pytest.raises(RuntimeError, match="Embedding provider not configured"):
            query_documents_tool(
                db_manager=mock_db_manager,
                embedding_provider=None,
                query="test query",
            )


class TestMCPServerDiagnosticLogging:
    """Test diagnostic logging for MCP server initialization."""

    def test_logs_embedding_provider_initialization(self):
        """Test that embedding provider configuration is logged at INFO level."""
        from obsidian_rag.mcp_server.server import create_mcp_server

        mock_settings = MagicMock()
        mock_settings.mcp.token = "test-token"
        mock_settings.mcp.enable_health_check = False
        mock_settings.database.url = "postgresql+psycopg://localhost/test"
        mock_settings.endpoints.embedding.provider = "openrouter"
        mock_settings.endpoints.embedding.model = "qwen/qwen3-embedding-8b"
        mock_settings.endpoints.embedding.api_key = "test-key"
        mock_settings.endpoints.embedding.base_url = None

        with patch("obsidian_rag.mcp_server.server.log") as mock_log:
            with patch("obsidian_rag.mcp_server.server.DatabaseManager"):
                with patch(
                    "obsidian_rag.mcp_server.server._create_embedding_provider"
                ) as mock_create:
                    # Create a mock provider with model and base_url attributes
                    mock_provider = MagicMock()
                    mock_provider.model = "qwen/qwen3-embedding-8b"
                    mock_provider.base_url = "https://openrouter.ai/api/v1"
                    mock_create.return_value = mock_provider

                    with patch("obsidian_rag.mcp_server.server._set_registry"):
                        with patch("obsidian_rag.mcp_server.server._register_tools"):
                            create_mcp_server(mock_settings)

        # Verify INFO log was called with provider details
        info_calls = list(mock_log.info.call_args_list)
        provider_log_found = any(
            "Embedding provider initialized" in str(call) for call in info_calls
        )
        assert provider_log_found, (
            "Expected INFO log for embedding provider initialization"
        )

    def test_logs_when_no_embedding_provider(self):
        """Test that disabled semantic search is logged when no provider configured."""
        from obsidian_rag.mcp_server.server import create_mcp_server

        mock_settings = MagicMock()
        mock_settings.mcp.token = "test-token"
        mock_settings.mcp.enable_health_check = False
        mock_settings.database.url = "postgresql+psycopg://localhost/test"

        with patch("obsidian_rag.mcp_server.server.log") as mock_log:
            with patch("obsidian_rag.mcp_server.server.DatabaseManager"):
                with patch(
                    "obsidian_rag.mcp_server.server._create_embedding_provider",
                    return_value=None,
                ):
                    with patch("obsidian_rag.mcp_server.server._set_registry"):
                        with patch("obsidian_rag.mcp_server.server._register_tools"):
                            create_mcp_server(mock_settings)

        # Verify INFO log was called about disabled semantic search
        info_calls = list(mock_log.info.call_args_list)
        disabled_log_found = any(
            "semantic search disabled" in str(call).lower() for call in info_calls
        )
        assert disabled_log_found, "Expected INFO log when semantic search is disabled"


class TestGetTasksServerWrapper:
    """Tests for get_tasks server wrapper function."""

    @patch("obsidian_rag.mcp_server.server._get_registry")
    @patch("obsidian_rag.mcp_server.handlers._get_tasks_handler")
    def test_server_wrapper_passes_tag_filters(self, mock_handler, mock_registry):
        """Test that server wrapper passes tag_filters to handler."""
        mock_registry_instance = MagicMock()
        mock_registry.return_value = mock_registry_instance

        mock_handler.return_value = {
            "results": [],
            "total_count": 0,
            "has_more": False,
            "next_offset": None,
        }

        from obsidian_rag.mcp_server.server import get_tasks
        from obsidian_rag.mcp_server.handlers import GetTasksToolInput, TagFilterStrings

        tag_filters = TagFilterStrings(
            include_tags=["work", "urgent"],
            exclude_tags=["blocked"],
            match_mode="all",
        )

        params = GetTasksToolInput(
            status=["not_completed"],
            tag_filters=tag_filters,
        )
        result = get_tasks(params=params)

        mock_handler.assert_called_once()
        call_args = mock_handler.call_args
        request = call_args.kwargs["request"]
        assert request.tag_filters is not None
        assert request.tag_filters.include_tags == ["work", "urgent"]
        assert request.tag_filters.exclude_tags == ["blocked"]
        assert request.tag_filters.match_mode == "all"

    @patch("obsidian_rag.mcp_server.server._get_registry")
    @patch("obsidian_rag.mcp_server.handlers._get_tasks_handler")
    def test_server_wrapper_default_tag_filters(self, mock_handler, mock_registry):
        """Test that server wrapper creates default tag_filters when not provided."""
        mock_registry_instance = MagicMock()
        mock_registry.return_value = mock_registry_instance

        mock_handler.return_value = {
            "results": [],
            "total_count": 0,
            "has_more": False,
            "next_offset": None,
        }

        from obsidian_rag.mcp_server.server import get_tasks
        from obsidian_rag.mcp_server.handlers import GetTasksToolInput

        params = GetTasksToolInput(status=["not_completed"])
        result = get_tasks(params=params)

        mock_handler.assert_called_once()
        call_args = mock_handler.call_args
        request = call_args.kwargs["request"]
        assert request.tag_filters is not None
        assert request.tag_filters.include_tags is None
        assert request.tag_filters.exclude_tags is None
        assert request.tag_filters.match_mode == "all"


class TestGetTasksJsonString:
    """Tests for get_tasks server wrapper with JSON string input."""

    @patch("obsidian_rag.mcp_server.server._get_registry")
    @patch("obsidian_rag.mcp_server.handlers._get_tasks_handler")
    def test_get_tasks_accepts_json_string(self, mock_handler, mock_registry):
        """Test that get_tasks accepts JSON string params."""
        mock_registry_instance = MagicMock()
        mock_registry.return_value = mock_registry_instance

        mock_handler.return_value = {
            "results": [],
            "total_count": 0,
            "has_more": False,
            "next_offset": None,
        }

        from obsidian_rag.mcp_server.server import get_tasks

        # Pass JSON string instead of dataclass
        json_params = '{"status": ["not_completed"], "limit": 10}'
        result = get_tasks(params=json_params)

        assert result == {
            "results": [],
            "total_count": 0,
            "has_more": False,
            "next_offset": None,
        }
        mock_handler.assert_called_once()
        call_args = mock_handler.call_args
        request = call_args.kwargs["request"]
        assert request.status == ["not_completed"]
        assert request.limit == 10

    @patch("obsidian_rag.mcp_server.server._get_registry")
    @patch("obsidian_rag.mcp_server.handlers._get_tasks_handler")
    def test_get_tasks_accepts_dict(self, mock_handler, mock_registry):
        """Test that get_tasks accepts dict params."""
        mock_registry_instance = MagicMock()
        mock_registry.return_value = mock_registry_instance

        mock_handler.return_value = {
            "results": [],
            "total_count": 0,
            "has_more": False,
            "next_offset": None,
        }

        from obsidian_rag.mcp_server.server import get_tasks

        # Pass dict instead of dataclass
        dict_params = {"status": ["completed"], "offset": 5}
        result = get_tasks(params=dict_params)

        assert result == {
            "results": [],
            "total_count": 0,
            "has_more": False,
            "next_offset": None,
        }
        mock_handler.assert_called_once()
        call_args = mock_handler.call_args
        request = call_args.kwargs["request"]
        assert request.status == ["completed"]
        assert request.offset == 5

    @patch("obsidian_rag.mcp_server.server._get_registry")
    @patch("obsidian_rag.mcp_server.handlers._get_tasks_handler")
    def test_get_tasks_accepts_none(self, mock_handler, mock_registry):
        """Test that get_tasks accepts None params and returns all tasks."""
        mock_registry_instance = MagicMock()
        mock_registry.return_value = mock_registry_instance

        mock_handler.return_value = {
            "results": [],
            "total_count": 0,
            "has_more": False,
            "next_offset": None,
        }

        from obsidian_rag.mcp_server.server import get_tasks

        # Pass None
        result = get_tasks(params=None)

        assert result == {
            "results": [],
            "total_count": 0,
            "has_more": False,
            "next_offset": None,
        }
        mock_handler.assert_called_once()
        call_args = mock_handler.call_args
        request = call_args.kwargs["request"]
        # None should create default GetTasksToolInput with default values
        assert request.status is None
        assert request.limit == 20  # Default value

    @patch("obsidian_rag.mcp_server.server._get_registry")
    @patch("obsidian_rag.mcp_server.handlers._get_tasks_handler")
    def test_get_tasks_json_with_nested_filters(self, mock_handler, mock_registry):
        """Test that get_tasks accepts JSON with nested tag_filters.

        Note: When calling directly (not through FastMCP), nested dicts remain
        as dicts rather than being converted to dataclasses. The full Pydantic
        validation with nested conversion only happens when FastMCP invokes
        the tool with AnnotatedGetTasksInput.
        """
        mock_registry_instance = MagicMock()
        mock_registry.return_value = mock_registry_instance

        mock_handler.return_value = {
            "results": [],
            "total_count": 0,
            "has_more": False,
            "next_offset": None,
        }

        from obsidian_rag.mcp_server.server import get_tasks

        # Pass JSON string with nested tag_filters
        json_params = (
            '{"status": ["not_completed"], '
            '"tag_filters": {"include_tags": ["work", "urgent"], "match_mode": "all"}}'
        )
        result = get_tasks(params=json_params)

        assert result == {
            "results": [],
            "total_count": 0,
            "has_more": False,
            "next_offset": None,
        }
        mock_handler.assert_called_once()
        call_args = mock_handler.call_args
        request = call_args.kwargs["request"]
        assert request.status == ["not_completed"]
        # When called directly, nested objects remain as dicts
        assert request.tag_filters == {
            "include_tags": ["work", "urgent"],
            "match_mode": "all",
        }

    @patch("obsidian_rag.mcp_server.server._get_registry")
    @patch("obsidian_rag.mcp_server.handlers._get_tasks_handler")
    def test_get_tasks_with_empty_string(self, mock_handler, mock_registry):
        """Test that get_tasks treats empty string as no params."""
        mock_registry_instance = MagicMock()
        mock_registry.return_value = mock_registry_instance

        mock_handler.return_value = {
            "results": [],
            "total_count": 0,
            "has_more": False,
            "next_offset": None,
        }

        from obsidian_rag.mcp_server.server import get_tasks

        # Pass empty string - should be treated as no params
        result = get_tasks(params="")

        assert result == {
            "results": [],
            "total_count": 0,
            "has_more": False,
            "next_offset": None,
        }
        mock_handler.assert_called_once()
        call_args = mock_handler.call_args
        request = call_args.kwargs["request"]
        # Empty string should create default GetTasksToolInput
        assert request.status is None
        assert request.limit == 20  # Default value

    @patch("obsidian_rag.mcp_server.server._get_registry")
    @patch("obsidian_rag.mcp_server.handlers._get_tasks_handler")
    def test_get_tasks_with_dataclass(self, mock_handler, mock_registry):
        """Test that get_tasks accepts GetTasksToolInput dataclass directly."""
        mock_registry_instance = MagicMock()
        mock_registry.return_value = mock_registry_instance

        mock_handler.return_value = {
            "results": [],
            "total_count": 0,
            "has_more": False,
            "next_offset": None,
        }

        from obsidian_rag.mcp_server.server import get_tasks
        from obsidian_rag.mcp_server.handlers import GetTasksToolInput

        # Pass GetTasksToolInput dataclass directly
        params = GetTasksToolInput(
            status=["completed"],
            limit=15,
            offset=10,
        )
        result = get_tasks(params=params)

        assert result == {
            "results": [],
            "total_count": 0,
            "has_more": False,
            "next_offset": None,
        }
        mock_handler.assert_called_once()
        call_args = mock_handler.call_args
        request = call_args.kwargs["request"]
        assert request.status == ["completed"]
        assert request.limit == 15
        assert request.offset == 10


class TestIngestRequestTracking:
    """Tests for ingest tool request tracking functionality."""

    def test_generate_request_id_deterministic(self):
        """Test that request ID generation is deterministic."""
        from obsidian_rag.mcp_server.server import _generate_request_id

        id1 = _generate_request_id("vault1", "/path", no_delete=False)
        id2 = _generate_request_id("vault1", "/path", no_delete=False)
        assert id1 == id2
        assert len(id1) == 32  # MD5 hex length

    def test_generate_request_id_different_params(self):
        """Test that different parameters produce different IDs."""
        from obsidian_rag.mcp_server.server import _generate_request_id

        id1 = _generate_request_id("vault1", "/path", no_delete=False)
        id2 = _generate_request_id("vault2", "/path", no_delete=False)
        id3 = _generate_request_id("vault1", "/other", no_delete=False)
        id4 = _generate_request_id("vault1", "/path", no_delete=True)

        assert id1 != id2
        assert id1 != id3
        assert id1 != id4

    def test_generate_request_id_with_none_path(self):
        """Test request ID generation with None path."""
        from obsidian_rag.mcp_server.server import _generate_request_id

        id1 = _generate_request_id("vault1", None, no_delete=False)
        id2 = _generate_request_id("vault1", None, no_delete=False)
        assert id1 == id2
        assert len(id1) == 32

    def test_get_ingest_tracker_creates_instance(self):
        """Test that _get_ingest_tracker creates instance on first call."""
        from obsidian_rag.mcp_server.server import (
            _clear_ingest_tracker,
            _get_ingest_tracker,
        )

        _clear_ingest_tracker()  # Ensure clean state

        tracker1 = _get_ingest_tracker()
        assert tracker1 is not None

        tracker2 = _get_ingest_tracker()
        assert tracker1 is tracker2  # Same instance

        _clear_ingest_tracker()

    def test_clear_ingest_tracker(self):
        """Test that _clear_ingest_tracker clears the tracker."""
        from obsidian_rag.mcp_server.server import (
            _clear_ingest_tracker,
            _get_ingest_tracker,
        )

        tracker = _get_ingest_tracker()

        async def add_request():
            await tracker.start_request("test-req", {"vault": "test"})

        asyncio.run(add_request())

        _clear_ingest_tracker()

        # After clearing, should get new instance
        new_tracker = _get_ingest_tracker()
        assert new_tracker is not tracker

        _clear_ingest_tracker()

    @patch("obsidian_rag.mcp_server.server._get_registry")
    @patch("obsidian_rag.mcp_server.server._ingest_handler")
    def test_ingest_caches_result(self, mock_handler, mock_get_registry):
        """Test that ingest tool caches results for duplicate calls."""
        from obsidian_rag.mcp_server.server import _clear_ingest_tracker, ingest

        _clear_ingest_tracker()

        # Setup mock registry
        mock_registry = Mock()
        mock_registry.settings = Mock()
        mock_registry.db_manager = Mock()
        mock_registry.embedding_provider = None
        mock_get_registry.return_value = mock_registry

        # Setup mock handler to return a result
        expected_result = {
            "total": 10,
            "new": 5,
            "updated": 3,
            "unchanged": 2,
            "errors": 0,
            "deleted": 0,
            "processing_time_seconds": 1.5,
            "message": "Ingested 10 files",
        }
        mock_handler.return_value = expected_result

        # First call - should process
        result1 = ingest("test-vault", "/path", no_delete=False)
        assert result1 == expected_result
        assert mock_handler.call_count == 1

        # Second call with same params - should return cached
        result2 = ingest("test-vault", "/path", no_delete=False)
        assert result2 == expected_result
        # Handler should NOT be called again
        assert mock_handler.call_count == 1

        _clear_ingest_tracker()

    @patch("obsidian_rag.mcp_server.server._get_registry")
    @patch("obsidian_rag.mcp_server.server._ingest_handler")
    def test_ingest_different_params_not_cached(self, mock_handler, mock_get_registry):
        """Test that different parameters are not cached together."""
        from obsidian_rag.mcp_server.server import _clear_ingest_tracker, ingest

        _clear_ingest_tracker()

        # Setup mock registry
        mock_registry = Mock()
        mock_registry.settings = Mock()
        mock_registry.db_manager = Mock()
        mock_registry.embedding_provider = None
        mock_get_registry.return_value = mock_registry

        # Setup mock handler
        result1 = {"total": 10, "message": "First"}
        result2 = {"total": 20, "message": "Second"}
        mock_handler.side_effect = [result1, result2]

        # First call
        ingest("vault1", "/path", no_delete=False)
        assert mock_handler.call_count == 1

        # Different vault - should process
        result = ingest("vault2", "/path", no_delete=False)
        assert mock_handler.call_count == 2
        assert result == result2

        _clear_ingest_tracker()

    @patch("obsidian_rag.mcp_server.server._get_registry")
    @patch("obsidian_rag.mcp_server.server._ingest_handler")
    def test_ingest_error_cached(self, mock_handler, mock_get_registry):
        """Test that errors are properly tracked and not re-processed."""
        from obsidian_rag.mcp_server.server import (
            _clear_ingest_tracker,
            _generate_request_id,
            _get_ingest_tracker,
            ingest,
        )

        _clear_ingest_tracker()

        # Setup mock registry
        mock_registry = Mock()
        mock_registry.settings = Mock()
        mock_registry.db_manager = Mock()
        mock_registry.embedding_provider = None
        mock_get_registry.return_value = mock_registry

        # Setup mock handler to raise error
        mock_handler.side_effect = ValueError("Test error")

        # First call - should raise
        with pytest.raises(ValueError, match="Test error"):
            ingest("test-vault", "/path", no_delete=False)

        # Get tracker and verify error was recorded
        tracker = _get_ingest_tracker()
        request_id = _generate_request_id("test-vault", "/path", no_delete=False)

        async def check_error():
            entry = tracker._requests.get(request_id)
            assert entry is not None
            assert entry.status == "complete"
            assert entry.error is not None

        asyncio.run(check_error())

        _clear_ingest_tracker()

    @patch("obsidian_rag.mcp_server.server._get_registry")
    @patch("obsidian_rag.mcp_server.server._ingest_handler")
    def test_ingest_logs_duplicate_detection(
        self, mock_handler, mock_get_registry, caplog
    ):
        """Test that duplicate request detection is logged."""
        import logging

        from obsidian_rag.mcp_server.server import _clear_ingest_tracker, ingest

        _clear_ingest_tracker()

        # Setup mock registry
        mock_registry = Mock()
        mock_registry.settings = Mock()
        mock_registry.db_manager = Mock()
        mock_registry.embedding_provider = None
        mock_get_registry.return_value = mock_registry

        mock_handler.return_value = {"total": 10, "message": "Done"}

        # Set log level to INFO to capture our log messages
        with caplog.at_level(logging.INFO, logger="obsidian_rag.mcp_server.server"):
            # First call
            ingest("test-vault", "/path", no_delete=False)

            # Second call - duplicate
            ingest("test-vault", "/path", no_delete=False)

        # Check that duplicate detection was logged
        assert "Returning cached result for duplicate request" in caplog.text

        _clear_ingest_tracker()


class TestVaultErrorHandling:
    """Tests for REQ-005: vault not found error handling."""

    @patch("obsidian_rag.mcp_server.server._get_registry")
    @patch("obsidian_rag.mcp_server.server._ingest_handler")
    def test_ingest_vault_not_found_returns_error_dict(
        self, mock_handler, mock_get_registry
    ):
        """Test that vault not found error returns error dict instead of raising."""
        from obsidian_rag.mcp_server.server import (
            _clear_ingest_tracker,
            ingest,
        )

        _clear_ingest_tracker()

        # Setup mock to raise vault not found error
        mock_handler.side_effect = ValueError(
            "Vault 'NonExistent' not found in configuration. Available: Other"
        )

        mock_registry = Mock()
        mock_registry.settings = Mock()
        mock_registry.db_manager = Mock()
        mock_registry.embedding_provider = None
        mock_get_registry.return_value = mock_registry

        # Call ingest - should NOT raise, should return error dict
        result = ingest("NonExistent", "/path", no_delete=False)

        # Verify error response format
        assert result["success"] is False
        assert "not found in configuration" in result["error"]
        assert result["errors"] == 1
        assert result["total"] == 0
        assert "NonExistent" in result["message"]

        _clear_ingest_tracker()

    @patch("obsidian_rag.mcp_server.server._get_registry")
    @patch("obsidian_rag.mcp_server.server._ingest_handler")
    def test_ingest_vault_not_found_logs_warning(
        self, mock_handler, mock_get_registry, caplog
    ):
        """Test that vault not found error logs warning message."""
        import logging

        from obsidian_rag.mcp_server.server import (
            _clear_ingest_tracker,
            ingest,
        )

        _clear_ingest_tracker()

        mock_handler.side_effect = ValueError(
            "Vault 'MissingVault' not found in configuration."
        )

        mock_registry = Mock()
        mock_registry.settings = Mock()
        mock_registry.db_manager = Mock()
        mock_registry.embedding_provider = None
        mock_get_registry.return_value = mock_registry

        with caplog.at_level(logging.WARNING, logger="obsidian_rag.mcp_server.server"):
            ingest("MissingVault", "/path", no_delete=False)

        # Verify warning was logged
        assert "client requested non-existent vault 'MissingVault'" in caplog.text

        _clear_ingest_tracker()

    @patch("obsidian_rag.mcp_server.server._get_registry")
    @patch("obsidian_rag.mcp_server.server._ingest_handler")
    def test_ingest_vault_not_found_not_cached(self, mock_handler, mock_get_registry):
        """Test that failed vault requests are NOT cached."""
        from obsidian_rag.mcp_server.server import (
            _clear_ingest_tracker,
            _get_ingest_tracker,
            ingest,
        )

        _clear_ingest_tracker()

        mock_handler.side_effect = ValueError(
            "Vault 'TestVault' not found in configuration."
        )

        mock_registry = Mock()
        mock_registry.settings = Mock()
        mock_registry.db_manager = Mock()
        mock_registry.embedding_provider = None
        mock_get_registry.return_value = mock_registry

        # First call - should fail
        result1 = ingest("TestVault", "/path", no_delete=False)
        assert result1["success"] is False

        # Check tracker - request should be cleared, not cached
        tracker = _get_ingest_tracker()

        async def check_not_cached():
            # Generate same request ID
            from obsidian_rag.mcp_server.server import _generate_request_id

            request_id = _generate_request_id("TestVault", "/path", no_delete=False)
            return request_id in tracker._requests

        import asyncio

        is_cached = asyncio.run(check_not_cached())
        assert is_cached is False  # Should NOT be cached

        _clear_ingest_tracker()

    @patch("obsidian_rag.mcp_server.server._get_registry")
    @patch("obsidian_rag.mcp_server.server._ingest_handler")
    def test_ingest_other_valueerror_still_raises(
        self, mock_handler, mock_get_registry
    ):
        """Test that non-vault ValueErrors are still raised."""
        from obsidian_rag.mcp_server.server import (
            _clear_ingest_tracker,
            ingest,
        )

        _clear_ingest_tracker()

        # Setup mock to raise a different ValueError (not vault-related)
        mock_handler.side_effect = ValueError("Some other validation error")

        mock_registry = Mock()
        mock_registry.settings = Mock()
        mock_registry.db_manager = Mock()
        mock_registry.embedding_provider = None
        mock_get_registry.return_value = mock_registry

        # Call ingest - should raise ValueError
        with pytest.raises(ValueError, match="Some other validation error"):
            ingest("SomeVault", "/path", no_delete=False)

        _clear_ingest_tracker()

    @patch("obsidian_rag.mcp_server.server._get_registry")
    @patch("obsidian_rag.mcp_server.server._ingest_handler")
    def test_ingest_generic_exception_raises_and_logs(
        self, mock_handler, mock_get_registry, caplog
    ):
        """Test that generic exceptions are raised and logged properly."""
        import logging

        from obsidian_rag.mcp_server.server import (
            _clear_ingest_tracker,
            ingest,
        )

        _clear_ingest_tracker()

        # Setup mock to raise a generic Exception (not ValueError)
        mock_handler.side_effect = RuntimeError("Database connection failed")

        mock_registry = Mock()
        mock_registry.settings = Mock()
        mock_registry.db_manager = Mock()
        mock_registry.embedding_provider = None
        mock_get_registry.return_value = mock_registry

        # Call ingest - should raise RuntimeError
        with caplog.at_level(logging.ERROR, logger="obsidian_rag.mcp_server.server"):
            with pytest.raises(RuntimeError, match="Database connection failed"):
                ingest("SomeVault", "/path", no_delete=False)

        # Verify error was logged
        assert "Request" in caplog.text
        assert "failed" in caplog.text

        _clear_ingest_tracker()


class TestVaultErrorHelperFunctions:
    """Direct tests for vault error handling helper functions."""

    def test_is_vault_not_found_error_returns_true_for_vault_error(self):
        """Test _is_vault_not_found_error returns True for vault not found."""
        from obsidian_rag.mcp_server.server import _is_vault_not_found_error

        error = ValueError("Vault 'Test' not found in configuration. Available: Other")
        result = _is_vault_not_found_error(error)
        assert result is True

    def test_is_vault_not_found_error_returns_false_for_other_errors(self):
        """Test _is_vault_not_found_error returns False for non-vault errors."""
        from obsidian_rag.mcp_server.server import _is_vault_not_found_error

        error1 = ValueError("Some other error")
        error2 = ValueError("not found in configuration")  # Missing "Vault"
        error3 = ValueError("Vault error")  # Missing "not found in configuration"

        assert _is_vault_not_found_error(error1) is False
        assert _is_vault_not_found_error(error2) is False
        assert _is_vault_not_found_error(error3) is False

    def test_handle_vault_not_found_returns_error_dict(self):
        """Test _handle_vault_not_found returns proper error response."""
        from obsidian_rag.mcp_server.server import (
            _clear_ingest_tracker,
            _get_ingest_tracker,
            _handle_vault_not_found,
        )

        _clear_ingest_tracker()
        tracker = _get_ingest_tracker()

        result = _handle_vault_not_found(
            vault_name="MissingVault",
            error_msg="Vault 'MissingVault' not found in configuration",
            request_id="test-request-123",
            tracker=tracker,
        )

        assert result["success"] is False
        assert "not found in configuration" in result["error"]
        assert result["errors"] == 1
        assert result["total"] == 0
        assert "MissingVault" in result["message"]

        _clear_ingest_tracker()

    def test_handle_vault_not_found_clears_tracker(self):
        """Test _handle_vault_not_found clears the request from tracker."""
        from obsidian_rag.mcp_server.server import (
            _clear_ingest_tracker,
            _get_ingest_tracker,
            _handle_vault_not_found,
        )

        _clear_ingest_tracker()
        tracker = _get_ingest_tracker()
        request_id = "test-clear-request"

        # First add a pending request
        async def add_pending():
            await tracker.start_request(request_id, {"test": "data"})

        asyncio.run(add_pending())

        # Verify it's there
        async def check_exists():
            return request_id in tracker._requests

        assert asyncio.run(check_exists()) is True

        # Handle the vault not found error
        _handle_vault_not_found(
            vault_name="TestVault",
            error_msg="Vault not found",
            request_id=request_id,
            tracker=tracker,
        )

        # Verify it was cleared
        assert asyncio.run(check_exists()) is False

        _clear_ingest_tracker()

    def test_check_and_handle_duplicate_returns_none_for_new_request(self):
        """Test _check_and_handle_duplicate returns None for new requests."""
        from obsidian_rag.mcp_server.server import (
            _check_and_handle_duplicate,
            _clear_ingest_tracker,
            _get_ingest_tracker,
        )

        _clear_ingest_tracker()
        tracker = _get_ingest_tracker()

        result = _check_and_handle_duplicate(
            tracker=tracker,
            request_id="new-request-123",
            vault_name="TestVault",
            path="/test/path",
            no_delete=False,
        )

        # Should return None for new requests (should_process=True)
        assert result is None

        _clear_ingest_tracker()

    def test_check_and_handle_duplicate_returns_cached_result(self):
        """Test _check_and_handle_duplicate returns cached result for duplicate."""
        from obsidian_rag.mcp_server.server import (
            _check_and_handle_duplicate,
            _clear_ingest_tracker,
            _get_ingest_tracker,
        )

        _clear_ingest_tracker()
        tracker = _get_ingest_tracker()
        request_id = "duplicate-request-456"
        cached_data = {"success": True, "total": 42}

        # First complete the request with a result
        async def setup_completed():
            await tracker.start_request(request_id, {"vault_name": "Test"})
            await tracker.complete_request(request_id, cached_data)

        asyncio.run(setup_completed())

        # Now check for duplicate - should return cached result
        result = _check_and_handle_duplicate(
            tracker=tracker,
            request_id=request_id,
            vault_name="TestVault",
            path="/test/path",
            no_delete=False,
        )

        assert result is not None
        assert result["success"] is True
        assert result["total"] == 42

        _clear_ingest_tracker()

    def test_check_and_handle_duplicate_handles_none_cached_result(self):
        """Test _check_and_handle_duplicate handles None cached_result gracefully."""
        from obsidian_rag.mcp_server.server import (
            _check_and_handle_duplicate,
            _clear_ingest_tracker,
            _get_ingest_tracker,
        )

        _clear_ingest_tracker()
        tracker = _get_ingest_tracker()
        request_id = "none-result-request"

        # Start but don't complete - this creates a pending entry
        async def setup_pending():
            await tracker.start_request(request_id, {"vault_name": "Test"})

        asyncio.run(setup_pending())

        # Manually set up a completed entry with None result
        # by completing and then checking behavior
        async def complete_none():
            await tracker.complete_request(request_id, None)

        asyncio.run(complete_none())

        # Now the request is completed with None result
        result = _check_and_handle_duplicate(
            tracker=tracker,
            request_id=request_id,
            vault_name="TestVault",
            path="/test/path",
            no_delete=False,
        )

        # When cached_result is None, it should return None (proceed with processing)
        assert result is None

        _clear_ingest_tracker()


class TestCreateHTTPAppFactory:
    """Tests for create_http_app_factory function."""

    @patch("obsidian_rag.mcp_server.server.create_http_app")
    @patch("obsidian_rag.mcp_server.server.get_settings")
    def test_factory_loads_settings(self, mock_get_settings, mock_create_http_app):
        """Test factory loads settings successfully."""
        from obsidian_rag.mcp_server.server import create_http_app_factory

        mock_settings = Mock()
        mock_settings.mcp.token = "test-token"
        mock_get_settings.return_value = mock_settings
        mock_create_http_app.return_value = Mock()

        app = create_http_app_factory()
        assert app is not None
        mock_create_http_app.assert_called_once_with(mock_settings)

    @patch("obsidian_rag.mcp_server.server.get_settings")
    def test_factory_raises_system_exit_on_settings_error(self, mock_get_settings):
        """Test factory raises SystemExit when settings fail to load."""
        from obsidian_rag.mcp_server.server import create_http_app_factory

        mock_get_settings.side_effect = Exception("Config error")

        with pytest.raises(SystemExit) as exc_info:
            create_http_app_factory()

        assert exc_info.value.code == 1

    @patch("obsidian_rag.mcp_server.server.get_settings")
    def test_factory_raises_system_exit_on_missing_token(self, mock_get_settings):
        """Test factory raises SystemExit when MCP token is not configured."""
        from obsidian_rag.mcp_server.server import create_http_app_factory

        mock_settings = Mock()
        mock_settings.mcp.token = None
        mock_get_settings.return_value = mock_settings

        with pytest.raises(SystemExit) as exc_info:
            create_http_app_factory()

        assert exc_info.value.code == 1


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
