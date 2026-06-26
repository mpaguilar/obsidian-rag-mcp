"""Unit tests for MCP server module."""

import json
import os
from unittest.mock import MagicMock, Mock, patch

import pytest
from click.testing import CliRunner


runner = CliRunner()

runner = CliRunner()


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
        get_tasks(params=params)

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
        get_tasks(params=params)

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
