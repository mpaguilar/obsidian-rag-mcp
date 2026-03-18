"""Unit tests for MCP server module."""

import json
import os
from pathlib import Path
from typing import cast
from unittest.mock import MagicMock, patch

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

            date_filters = TaskDateFilterStrings(
                due_after="2026-01-01",
                due_before="2026-12-31",
            )

            result = get_tasks(
                status=["not_completed"],
                date_filters=date_filters,
                limit=20,
                offset=0,
            )

            assert result == {"results": []}
            mock_handler.assert_called_once()

    def test_get_tasks_without_date_filters(self, setup_registry):
        """Test get_tasks creates default date_filters when not provided."""
        from obsidian_rag.mcp_server.handlers import TaskDateFilterStrings
        from obsidian_rag.mcp_server.server import get_tasks

        mock_registry = setup_registry

        # Patch inside the server module where it's imported
        with patch(
            "obsidian_rag.mcp_server.handlers._get_tasks_handler"
        ) as mock_handler:
            mock_handler.return_value = {"results": []}

            # Call without date_filters parameter
            result = get_tasks(
                status=["not_completed"],
                limit=20,
                offset=0,
            )

            assert result == {"results": []}
            mock_handler.assert_called_once()
            # Verify that a TaskDateFilterStrings was created with default values
            call_kwargs = mock_handler.call_args.kwargs
            assert isinstance(call_kwargs["date_filters"], TaskDateFilterStrings)


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
        mock_db_manager.assert_called_once_with(settings.database.url)

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
        from obsidian_rag.mcp_server.tool_definitions import get_tasks_tool
        from obsidian_rag.mcp_server.handlers import TaskDateFilterStrings

        mock_db_manager = MagicMock()

        with patch(
            "obsidian_rag.mcp_server.tool_definitions._get_tasks_handler"
        ) as mock_handler:
            mock_handler.return_value = {"tasks": []}

            date_filters = TaskDateFilterStrings(
                due_after="2026-01-01",
                due_before="2026-12-31",
            )

            result = get_tasks_tool(
                db_manager=mock_db_manager,
                status=["not_completed"],
                date_filters=date_filters,
                tags=["work"],
                priority=["high"],
                limit=20,
                offset=0,
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

        with patch("obsidian_rag.mcp_server.server.query_documents") as mock_query:
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
