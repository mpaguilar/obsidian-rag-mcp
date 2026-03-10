"""Tests for module-level tool functions in server.py.

These tests cover the module-level tool implementations that were extracted
from nested functions to improve testability.
"""

import json
import os
from unittest.mock import MagicMock, patch

import pytest

from obsidian_rag.config import EndpointConfig


class TestGetIncompleteTasksTool:
    """Tests for get_incomplete_tasks_tool function (lines 86-88)."""

    def test_get_incomplete_tasks_tool_basic(self):
        """Test get_incomplete_tasks_tool with default parameters."""
        from obsidian_rag.mcp_server.tool_definitions import get_incomplete_tasks_tool

        mock_db_manager = MagicMock()
        mock_session = MagicMock()
        mock_db_manager.get_session.return_value.__enter__ = MagicMock(
            return_value=mock_session
        )
        mock_db_manager.get_session.return_value.__exit__ = MagicMock(
            return_value=False
        )

        with patch(
            "obsidian_rag.mcp_server.tool_definitions._get_incomplete_tasks_handler"
        ) as mock_handler:
            mock_handler.return_value = {"tasks": []}

            result = get_incomplete_tasks_tool(mock_db_manager, 20, 0)

            assert result == {"tasks": []}
            mock_handler.assert_called_once_with(
                mock_db_manager, 20, 0, include_cancelled=False
            )

    def test_get_incomplete_tasks_tool_with_cancelled(self):
        """Test get_incomplete_tasks_tool with include_cancelled=True."""
        from obsidian_rag.mcp_server.tool_definitions import get_incomplete_tasks_tool

        mock_db_manager = MagicMock()
        mock_session = MagicMock()
        mock_db_manager.get_session.return_value.__enter__ = MagicMock(
            return_value=mock_session
        )
        mock_db_manager.get_session.return_value.__exit__ = MagicMock(
            return_value=False
        )

        with patch(
            "obsidian_rag.mcp_server.tool_definitions._get_incomplete_tasks_handler"
        ) as mock_handler:
            mock_handler.return_value = {"tasks": [{"id": "1"}]}

            result = get_incomplete_tasks_tool(
                mock_db_manager, 50, 10, include_cancelled=True
            )

            assert result == {"tasks": [{"id": "1"}]}
            mock_handler.assert_called_once_with(
                mock_db_manager, 50, 10, include_cancelled=True
            )


class TestGetTasksDueThisWeekTool:
    """Tests for get_tasks_due_this_week_tool function (lines 119-121)."""

    def test_get_tasks_due_this_week_tool_basic(self):
        """Test get_tasks_due_this_week_tool with default parameters."""
        from obsidian_rag.mcp_server.tool_definitions import (
            get_tasks_due_this_week_tool,
        )

        mock_db_manager = MagicMock()

        with patch(
            "obsidian_rag.mcp_server.tool_definitions._get_tasks_due_this_week_handler"
        ) as mock_handler:
            mock_handler.return_value = {"tasks": []}

            result = get_tasks_due_this_week_tool(mock_db_manager, 20, 0)

            assert result == {"tasks": []}
            mock_handler.assert_called_once_with(
                mock_db_manager, 20, 0, include_completed=True
            )

    def test_get_tasks_due_this_week_tool_without_completed(self):
        """Test get_tasks_due_this_week_tool with include_completed=False."""
        from obsidian_rag.mcp_server.tool_definitions import (
            get_tasks_due_this_week_tool,
        )

        mock_db_manager = MagicMock()

        with patch(
            "obsidian_rag.mcp_server.tool_definitions._get_tasks_due_this_week_handler"
        ) as mock_handler:
            mock_handler.return_value = {"tasks": [{"id": "1"}]}

            result = get_tasks_due_this_week_tool(
                mock_db_manager, 30, 5, include_completed=False
            )

            assert result == {"tasks": [{"id": "1"}]}
            mock_handler.assert_called_once_with(
                mock_db_manager, 30, 5, include_completed=False
            )


class TestGetTasksByTagToolImpl:
    """Tests for get_tasks_by_tag_tool_impl function (lines 151-153)."""

    def test_get_tasks_by_tag_tool_impl_basic(self):
        """Test get_tasks_by_tag_tool_impl with basic parameters."""
        from obsidian_rag.mcp_server.tool_definitions import get_tasks_by_tag_tool_impl

        mock_db_manager = MagicMock()

        with patch(
            "obsidian_rag.mcp_server.tool_definitions._get_tasks_by_tag_handler"
        ) as mock_handler:
            mock_handler.return_value = {"tasks": [{"id": "1", "tag": "work"}]}

            result = get_tasks_by_tag_tool_impl(mock_db_manager, "work", 20, 0)

            assert result == {"tasks": [{"id": "1", "tag": "work"}]}
            mock_handler.assert_called_once_with(mock_db_manager, "work", 20, 0)

    def test_get_tasks_by_tag_tool_impl_with_pagination(self):
        """Test get_tasks_by_tag_tool_impl with custom pagination."""
        from obsidian_rag.mcp_server.tool_definitions import get_tasks_by_tag_tool_impl

        mock_db_manager = MagicMock()

        with patch(
            "obsidian_rag.mcp_server.tool_definitions._get_tasks_by_tag_handler"
        ) as mock_handler:
            mock_handler.return_value = {"tasks": []}

            result = get_tasks_by_tag_tool_impl(mock_db_manager, "urgent", 50, 25)

            assert result == {"tasks": []}
            mock_handler.assert_called_once_with(mock_db_manager, "urgent", 50, 25)


class TestGetCompletedTasksTool:
    """Tests for get_completed_tasks_tool function (lines 178-180)."""

    def test_get_completed_tasks_tool_basic(self):
        """Test get_completed_tasks_tool with default parameters."""
        from obsidian_rag.mcp_server.tool_definitions import get_completed_tasks_tool

        mock_db_manager = MagicMock()

        with patch(
            "obsidian_rag.mcp_server.tool_definitions._get_completed_tasks_handler"
        ) as mock_handler:
            mock_handler.return_value = {"tasks": []}

            result = get_completed_tasks_tool(mock_db_manager, 20, 0)

            assert result == {"tasks": []}
            # Check call was made with correct arguments (all positional)
            assert mock_handler.call_args[0] == (mock_db_manager, 20, 0, None)

    def test_get_completed_tasks_tool_with_date(self):
        """Test get_completed_tasks_tool with completed_since date."""
        from obsidian_rag.mcp_server.tool_definitions import get_completed_tasks_tool

        mock_db_manager = MagicMock()

        with patch(
            "obsidian_rag.mcp_server.tool_definitions._get_completed_tasks_handler"
        ) as mock_handler:
            mock_handler.return_value = {"tasks": [{"id": "1"}]}

            result = get_completed_tasks_tool(
                mock_db_manager, 50, 10, completed_since="2024-01-01"
            )

            assert result == {"tasks": [{"id": "1"}]}
            # Check call was made with correct arguments (all positional)
            assert mock_handler.call_args[0] == (mock_db_manager, 50, 10, "2024-01-01")


class TestQueryDocumentsTool:
    """Tests for query_documents_tool function (lines 217-270)."""

    def test_query_documents_tool_success(self):
        """Test query_documents_tool with valid embedding provider."""
        from obsidian_rag.mcp_server.tool_definitions import query_documents_tool

        mock_db_manager = MagicMock()
        mock_session = MagicMock()
        mock_db_manager.get_session.return_value.__enter__ = MagicMock(
            return_value=mock_session
        )
        mock_db_manager.get_session.return_value.__exit__ = MagicMock(
            return_value=False
        )

        mock_embedding_provider = MagicMock()
        mock_embedding_provider.generate_embedding.return_value = [0.1] * 1536

        with patch(
            "obsidian_rag.mcp_server.tools.documents.query_documents"
        ) as mock_query_impl:
            mock_result = MagicMock()
            mock_result.model_dump.return_value = {
                "results": [{"id": "1", "score": 0.9}],
                "total_count": 1,
                "has_more": False,
            }
            mock_query_impl.return_value = mock_result

            result = query_documents_tool(
                db_manager=mock_db_manager,
                embedding_provider=mock_embedding_provider,
                query="test query",
                filters=None,
                tag_match_mode="all",
                limit=20,
                offset=0,
            )

            assert result == {
                "results": [{"id": "1", "score": 0.9}],
                "total_count": 1,
                "has_more": False,
            }
            mock_embedding_provider.generate_embedding.assert_called_once_with(
                "test query"
            )

    def test_query_documents_tool_no_provider(self):
        """Test query_documents_tool raises error when no embedding provider."""
        from obsidian_rag.mcp_server.tool_definitions import query_documents_tool

        mock_db_manager = MagicMock()

        with pytest.raises(RuntimeError, match="Embedding provider not configured"):
            query_documents_tool(
                db_manager=mock_db_manager,
                embedding_provider=None,
                query="test query",
                filters=None,
                tag_match_mode="all",
                limit=20,
                offset=0,
            )

    def test_query_documents_tool_with_filters(self):
        """Test query_documents_tool with filters."""
        from obsidian_rag.mcp_server.tool_definitions import query_documents_tool
        from obsidian_rag.mcp_server.handlers import QueryFilterParams

        mock_db_manager = MagicMock()
        mock_session = MagicMock()
        mock_db_manager.get_session.return_value.__enter__ = MagicMock(
            return_value=mock_session
        )
        mock_db_manager.get_session.return_value.__exit__ = MagicMock(
            return_value=False
        )

        mock_embedding_provider = MagicMock()
        mock_embedding_provider.generate_embedding.return_value = [0.1] * 1536

        with patch(
            "obsidian_rag.mcp_server.tools.documents.query_documents"
        ) as mock_query_impl:
            mock_result = MagicMock()
            mock_result.model_dump.return_value = {"results": [], "total_count": 0}
            mock_query_impl.return_value = mock_result

            filters = QueryFilterParams(
                include_properties=[
                    {"path": "status", "operator": "equals", "value": "draft"}
                ],
                exclude_properties=None,
                include_tags=["work"],
                exclude_tags=None,
            )

            result = query_documents_tool(
                db_manager=mock_db_manager,
                embedding_provider=mock_embedding_provider,
                query="test",
                filters=filters,
                tag_match_mode="all",
                limit=10,
                offset=5,
            )

            assert result == {"results": [], "total_count": 0}


class TestGetAllTagsTool:
    """Tests for get_all_tags_tool function (lines 611-613)."""

    def test_get_all_tags_tool_basic(self):
        """Test get_all_tags_tool with default parameters."""
        from obsidian_rag.mcp_server.tool_definitions import get_all_tags_tool

        mock_db_manager = MagicMock()

        with patch(
            "obsidian_rag.mcp_server.tool_definitions._get_all_tags_handler"
        ) as mock_handler:
            mock_handler.return_value = {"tags": ["work", "personal"], "total_count": 2}

            result = get_all_tags_tool(mock_db_manager, None, 20, 0)

            assert result == {"tags": ["work", "personal"], "total_count": 2}
            mock_handler.assert_called_once_with(mock_db_manager, None, 20, 0)

    def test_get_all_tags_tool_with_pattern(self):
        """Test get_all_tags_tool with pattern filter."""
        from obsidian_rag.mcp_server.tool_definitions import get_all_tags_tool

        mock_db_manager = MagicMock()

        with patch(
            "obsidian_rag.mcp_server.tool_definitions._get_all_tags_handler"
        ) as mock_handler:
            mock_handler.return_value = {"tags": ["work"], "total_count": 1}

            result = get_all_tags_tool(mock_db_manager, "work*", 50, 10)

            assert result == {"tags": ["work"], "total_count": 1}
            mock_handler.assert_called_once_with(mock_db_manager, "work*", 50, 10)


class TestListVaultsTool:
    """Tests for list_vaults_tool function (lines 636-638)."""

    def test_list_vaults_tool_basic(self):
        """Test list_vaults_tool with default parameters."""
        from obsidian_rag.mcp_server.tool_definitions import list_vaults_tool

        mock_db_manager = MagicMock()

        with patch(
            "obsidian_rag.mcp_server.tool_definitions._list_vaults_handler"
        ) as mock_handler:
            mock_handler.return_value = {
                "results": [{"id": "1", "name": "Personal"}],
                "total_count": 1,
            }

            result = list_vaults_tool(mock_db_manager, 20, 0)

            assert result == {
                "results": [{"id": "1", "name": "Personal"}],
                "total_count": 1,
            }
            mock_handler.assert_called_once_with(mock_db_manager, 20, 0)

    def test_list_vaults_tool_with_pagination(self):
        """Test list_vaults_tool with custom pagination."""
        from obsidian_rag.mcp_server.tool_definitions import list_vaults_tool

        mock_db_manager = MagicMock()

        with patch(
            "obsidian_rag.mcp_server.tool_definitions._list_vaults_handler"
        ) as mock_handler:
            mock_handler.return_value = {"results": [], "total_count": 0}

            result = list_vaults_tool(mock_db_manager, 10, 20)

            assert result == {"results": [], "total_count": 0}
            mock_handler.assert_called_once_with(mock_db_manager, 10, 20)


class TestHealthCheckHandler:
    """Tests for health_check_handler function (lines 754-790)."""

    @pytest.mark.asyncio
    async def test_health_check_handler_success(self):
        """Test health_check_handler with healthy database."""
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
            mock_manager = MagicMock()
            mock_manager.get_metrics.return_value = {
                "total_created": 10,
                "total_destroyed": 5,
                "active_count": 5,
                "total_requests": 100,
                "peak_concurrent": 10,
                "connection_rate": 1.5,
                "active_sessions_by_ip": {"127.0.0.1": 2},
            }
            mock_get_manager.return_value = mock_manager

            with patch.dict("os.environ", {"OBSIDIAN_RAG_VERSION": "1.0.0"}):
                result = await health_check_handler(mock_db_manager)

                assert result.status_code == 200
                data = bytes(result.body).decode()
                assert '"status":"healthy"' in data
                assert '"version":"1.0.0"' in data
                assert '"database":"connected"' in data

    @pytest.mark.asyncio
    async def test_health_check_handler_db_error(self):
        """Test health_check_handler with database error."""
        from obsidian_rag.mcp_server.server import health_check_handler
        from sqlalchemy.exc import SQLAlchemyError

        mock_db_manager = MagicMock()
        mock_session = MagicMock()
        mock_session.execute.side_effect = SQLAlchemyError("Connection failed")
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

            with patch.dict("os.environ", {"OBSIDIAN_RAG_VERSION": "1.0.0"}):
                result = await health_check_handler(mock_db_manager)

                assert result.status_code == 200
                data = bytes(result.body).decode()
                assert '"status":"healthy"' in data
                assert '"database":' in data
                assert "error" in data

    @pytest.mark.asyncio
    async def test_health_check_handler_default_version(self):
        """Test health_check_handler uses default version when env var not set."""
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

            # Ensure env var is not set
            with patch.dict(os.environ, {}, clear=True):
                result = await health_check_handler(mock_db_manager)

        response_data = json.loads(result.body)
        assert response_data["version"] == "0.2.3"

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


class TestGetSessionManager:
    """Tests for _get_session_manager function (line 800)."""

    def test_get_session_manager_returns_none_when_not_initialized(self):
        """Test _get_session_manager returns None when not initialized."""
        from obsidian_rag.mcp_server.server import _get_session_manager

        # Ensure global is None
        with patch("obsidian_rag.mcp_server.server._session_manager", None):
            result = _get_session_manager()
            assert result is None

    def test_get_session_manager_returns_manager_when_initialized(self):
        """Test _get_session_manager returns manager when initialized."""
        from obsidian_rag.mcp_server.server import _get_session_manager

        mock_manager = MagicMock()
        with patch("obsidian_rag.mcp_server.server._session_manager", mock_manager):
            result = _get_session_manager()
            assert result is mock_manager


class TestCreateHttpAppBranches:
    """Tests for branch coverage in create_http_app (lines 903->908)."""

    @patch("obsidian_rag.mcp_server.server.create_mcp_server")
    def test_create_http_app_with_request_logging_enabled(self, mock_create_server):
        """Test create_http_app with request logging enabled (line 903->908)."""
        from obsidian_rag.mcp_server.server import create_http_app

        settings = MagicMock()
        settings.mcp = MagicMock()
        settings.mcp.cors_origins = ["*"]
        settings.mcp.token = "test-token"
        settings.mcp.enable_request_logging = True
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
        # Verify middleware was passed with SessionLoggingMiddleware
        call_args = mock_mcp.http_app.call_args
        middleware_list = call_args.kwargs.get("middleware", [])
        assert len(middleware_list) == 2  # CORS + SessionLogging

    @patch("obsidian_rag.mcp_server.server.create_mcp_server")
    def test_create_http_app_with_request_logging_disabled(self, mock_create_server):
        """Test create_http_app with request logging disabled."""
        from obsidian_rag.mcp_server.server import create_http_app

        settings = MagicMock()
        settings.mcp = MagicMock()
        settings.mcp.cors_origins = ["*"]
        settings.mcp.token = "test-token"
        settings.mcp.enable_request_logging = False
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
        # Verify only CORS middleware was passed
        call_args = mock_mcp.http_app.call_args
        middleware_list = call_args.kwargs.get("middleware", [])
        assert len(middleware_list) == 1  # Only CORS


class TestProviderCreatorFunctions:
    """Tests for provider creator functions to ensure full coverage."""

    def test_create_openai_provider_with_all_params(self):
        """Test _create_openai_provider with all parameters."""
        from obsidian_rag.mcp_server.tool_definitions import _create_openai_provider

        config = EndpointConfig(
            provider="openai",
            api_key="test-key",
            model="text-embedding-3-large",
            base_url="https://custom.openai.com",
        )

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
                    "model": "text-embedding-3-large",
                    "base_url": "https://custom.openai.com",
                },
            )

    def test_create_openrouter_provider_with_all_params(self):
        """Test _create_openrouter_provider with all parameters."""
        from obsidian_rag.mcp_server.tool_definitions import _create_openrouter_provider

        config = EndpointConfig(
            provider="openrouter",
            api_key="test-key",
            model="openai/text-embedding-3-small",
            base_url="https://openrouter.ai/api/v1",
        )

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
                    "model": "openai/text-embedding-3-small",
                    "base_url": "https://openrouter.ai/api/v1",
                },
            )

    def test_create_huggingface_provider_with_model(self):
        """Test _create_huggingface_provider with model."""
        from obsidian_rag.mcp_server.tool_definitions import (
            _create_huggingface_provider,
        )

        config = EndpointConfig(
            provider="huggingface",
            model="sentence-transformers/all-MiniLM-L6-v2",
        )

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
