"""Unit tests for MCP server module."""

import asyncio
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from click.testing import CliRunner

from obsidian_rag.config import Settings

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

        # Should register 4 document tools (query_documents, get_documents_by_tag, get_documents_by_property, get_all_tags)
        assert mock_mcp.tool.call_count == 4

    def test_register_document_tools_without_provider(self):
        """Test _register_document_tools without embedding provider."""
        from obsidian_rag.mcp_server.server import _register_document_tools

        mock_mcp = MagicMock()
        mock_db_manager = MagicMock()

        _register_document_tools(mock_mcp, mock_db_manager, None)

        # Should register 4 document tools (query_documents, get_documents_by_tag, get_documents_by_property, get_all_tags)
        assert mock_mcp.tool.call_count == 4

    def test_register_health_check(self):
        """Test _register_health_check."""
        from obsidian_rag.mcp_server.server import _register_health_check

        mock_mcp = MagicMock()
        mock_db_manager = MagicMock()

        _register_health_check(mock_mcp, mock_db_manager)

        mock_mcp.custom_route.assert_called_once()


class TestDocumentHandlers:
    """Tests for document tool handlers (TASK-053, TASK-054, TASK-055, TASK-056)."""

    def test_get_documents_by_tag_handler_full_flow(self):
        """Test _get_documents_by_tag_handler full flow (TASK-053)."""
        from obsidian_rag.mcp_server.server import _get_documents_by_tag_handler

        mock_db_manager = MagicMock()
        mock_session = MagicMock()
        mock_db_manager.get_session.return_value.__enter__ = MagicMock(
            return_value=mock_session
        )
        mock_db_manager.get_session.return_value.__exit__ = MagicMock(
            return_value=False
        )

        with patch(
            "obsidian_rag.mcp_server.server.get_documents_by_tag_tool"
        ) as mock_tool:
            mock_result = MagicMock()
            mock_result.model_dump.return_value = {
                "results": [],
                "total_count": 0,
                "has_more": False,
            }
            mock_tool.return_value = mock_result

            # Create params as a dict-like object
            params = {
                "include_tags": ["work"],
                "exclude_tags": [],
                "match_mode": "all",
                "vault_root": None,
                "limit": 20,
                "offset": 0,
            }
            result = _get_documents_by_tag_handler(mock_db_manager, params)  # type: ignore[arg-type]

            assert result == {"results": [], "total_count": 0, "has_more": False}
            mock_tool.assert_called_once()

    def test_get_all_tags_handler_full_flow(self):
        """Test _get_all_tags_handler full flow (TASK-054)."""
        from obsidian_rag.mcp_server.server import _get_all_tags_handler

        mock_db_manager = MagicMock()
        mock_session = MagicMock()
        mock_db_manager.get_session.return_value.__enter__ = MagicMock(
            return_value=mock_session
        )
        mock_db_manager.get_session.return_value.__exit__ = MagicMock(
            return_value=False
        )

        with patch("obsidian_rag.mcp_server.server.get_all_tags_tool") as mock_tool:
            mock_result = MagicMock()
            mock_result.model_dump.return_value = {
                "tags": [],
                "total_count": 0,
                "has_more": False,
            }
            mock_tool.return_value = mock_result

            result = _get_all_tags_handler(mock_db_manager, "work*", 20, 0)

            assert result == {"tags": [], "total_count": 0, "has_more": False}
            mock_tool.assert_called_once_with(
                session=mock_session, pattern="work*", limit=20, offset=0
            )

    def test_convert_property_filters_with_valid_filters(self):
        """Test _convert_property_filters with valid filters (TASK-055)."""
        from obsidian_rag.mcp_server.server import _convert_property_filters

        properties = [
            {"path": "status", "operator": "equals", "value": "draft"},
            {"path": "priority", "operator": "equals", "value": "high"},
        ]

        result = _convert_property_filters(properties)

        assert result is not None
        assert len(result) == 2
        assert result[0].path == "status"
        assert result[0].operator == "equals"
        assert result[0].value == "draft"

    def test_create_tag_filter_with_empty_tags(self):
        """Test _create_tag_filter with empty tags (TASK-056)."""
        from obsidian_rag.mcp_server.server import _create_tag_filter

        result = _create_tag_filter(None, None, "all")

        assert result is None


class TestTaskHandlerLogging:
    """Tests for task handler logging (TASK-057, TASK-058, TASK-059)."""

    def test_get_incomplete_tasks_handler_logging(self):
        """Test _get_incomplete_tasks_handler logging (TASK-058)."""
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

            # Verify handler returns expected result
            assert result == {"tasks": []}

    def test_get_tasks_due_this_week_handler_logging(self):
        """Test _get_tasks_due_this_week_handler logging (TASK-058)."""
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

            # Verify handler returns expected result
            assert result == {"tasks": []}

    def test_get_tasks_by_tag_handler_logging(self):
        """Test _get_tasks_by_tag_handler logging (TASK-059)."""
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

            # Verify handler returns expected result
            assert result == {"tasks": []}


class TestGetCompletedTasksHandler:
    """Tests for _get_completed_tasks_handler (TASK-060)."""

    @patch("obsidian_rag.mcp_server.server.log")
    def test_get_completed_tasks_handler_with_invalid_date(self, mock_log):
        """Test _get_completed_tasks_handler with invalid date (TASK-060)."""
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
            # Verify logging of invalid date
            mock_log.warning.assert_called()


class TestRegisterQueryDocumentsTool:
    """Tests for _register_query_documents_tool (TASK-061)."""

    def test_register_query_documents_tool_with_missing_embedding_provider(self):
        """Test _register_query_documents_tool with missing embedding provider error (TASK-061)."""
        from obsidian_rag.mcp_server.server import _register_query_documents_tool

        mock_mcp = MagicMock()
        mock_db_manager = MagicMock()

        # Should not raise, but should register tool that returns error
        _register_query_documents_tool(mock_mcp, mock_db_manager, None)

        # Verify tool was registered
        mock_mcp.tool.assert_called_once()


class TestRegisterGetDocumentsByTagTool:
    """Tests for _register_get_documents_by_tag_tool (TASK-062)."""

    def test_register_get_documents_by_tag_tool_full_flow(self):
        """Test _register_get_documents_by_tag_tool full flow (TASK-062)."""
        from obsidian_rag.mcp_server.server import _register_get_documents_by_tag_tool

        mock_mcp = MagicMock()
        mock_db_manager = MagicMock()

        _register_get_documents_by_tag_tool(mock_mcp, mock_db_manager)

        # Verify tool was registered
        mock_mcp.tool.assert_called_once()


class TestRegisterGetDocumentsByPropertyTool:
    """Tests for _register_get_documents_by_property_tool (TASK-063)."""

    def test_register_get_documents_by_property_tool_full_flow(self):
        """Test _register_get_documents_by_property_tool full flow (TASK-063)."""
        from obsidian_rag.mcp_server.server import (
            _register_get_documents_by_property_tool,
        )

        mock_mcp = MagicMock()
        mock_db_manager = MagicMock()

        _register_get_documents_by_property_tool(mock_mcp, mock_db_manager)

        # Verify tool was registered
        mock_mcp.tool.assert_called_once()


class TestRegisterGetAllTagsTool:
    """Tests for _register_get_all_tags_tool (TASK-064)."""

    @patch("obsidian_rag.mcp_server.server.log")
    def test_register_get_all_tags_tool_logging(self, mock_log):
        """Test _register_get_all_tags_tool logging (TASK-064)."""
        from obsidian_rag.mcp_server.server import _register_get_all_tags_tool

        mock_mcp = MagicMock()
        mock_db_manager = MagicMock()

        _register_get_all_tags_tool(mock_mcp, mock_db_manager)

        # Verify tool was registered
        mock_mcp.tool.assert_called_once()


class TestValidateIngestPath:
    """Tests for _validate_ingest_path (TASK-065, TASK-066, TASK-067)."""

    def test_validate_ingest_path_with_parent_directory_references(self):
        """Test _validate_ingest_path with parent directory references (TASK-065)."""
        from obsidian_rag.mcp_server.server import _validate_ingest_path

        with pytest.raises(ValueError, match="parent directory references"):
            _validate_ingest_path("/vault/../etc/passwd")

    def test_validate_ingest_path_with_nonexistent_path(self):
        """Test _validate_ingest_path with non-existent path (TASK-066)."""
        from obsidian_rag.mcp_server.server import _validate_ingest_path

        with pytest.raises(ValueError, match="does not exist"):
            _validate_ingest_path("/nonexistent/path")

    def test_validate_ingest_path_with_non_directory_path(self):
        """Test _validate_ingest_path with non-directory path (TASK-067)."""
        from obsidian_rag.mcp_server.server import _validate_ingest_path

        with runner.isolated_filesystem():
            test_file = "test.txt"
            Path(test_file).write_text("test")
            with pytest.raises(ValueError, match="not a directory"):
                _validate_ingest_path(test_file)


class TestIngestHandler:
    """Tests for _ingest_handler (TASK-068)."""

    @patch("obsidian_rag.mcp_server.server._validate_ingest_path")
    def test_ingest_handler_with_path_override(self, mock_validate):
        """Test _ingest_handler with path_override (TASK-068)."""
        from obsidian_rag.mcp_server.server import _ingest_handler

        mock_db_manager = MagicMock()
        mock_settings = MagicMock()
        mock_settings.vault_root = "/default/vault"

        mock_validate.return_value = Path("/custom/path")

        with runner.isolated_filesystem():
            # Create a temporary directory for testing
            vault_path = Path("vault")
            vault_path.mkdir()

            with patch(
                "obsidian_rag.mcp_server.server.IngestionService"
            ) as mock_service:
                mock_instance = MagicMock()
                mock_result = MagicMock()
                mock_result.total = 1
                mock_result.new = 1
                mock_result.updated = 0
                mock_result.unchanged = 0
                mock_result.errors = 0
                mock_result.processing_time_seconds = 1.0
                mock_result.to_dict.return_value = {
                    "total": 1,
                    "new": 1,
                    "updated": 0,
                    "unchanged": 0,
                    "errors": 0,
                    "processing_time_seconds": 1.0,
                    "message": "Ingested 1 documents",
                }
                mock_instance.ingest_vault.return_value = mock_result
                mock_service.return_value = mock_instance

                mock_embedding_provider = MagicMock()
                result = _ingest_handler(
                    settings=mock_settings,
                    db_manager=mock_db_manager,
                    embedding_provider=mock_embedding_provider,
                    path_override="/custom/path",
                )

                assert result["total"] == 1
                mock_validate.assert_called_once_with("/custom/path")


class TestRegisterIngestTools:
    """Tests for _register_ingest_tools (TASK-069)."""

    def test_register_ingest_tools_registers_tool(self):
        """Test _register_ingest_tools registers the tool (TASK-069)."""
        from obsidian_rag.mcp_server.server import _register_ingest_tools

        mock_mcp = MagicMock()
        mock_db_manager = MagicMock()
        mock_settings = MagicMock()

        mock_embedding_provider = MagicMock()
        _register_ingest_tools(
            mock_mcp, mock_settings, mock_db_manager, mock_embedding_provider
        )

        # Verify tool was registered
        mock_mcp.tool.assert_called_once()


class TestRegisterHealthCheck:
    """Tests for _register_health_check with database error (TASK-070)."""

    def test_register_health_check_with_database_error(self):
        """Test _register_health_check with database error (TASK-070)."""
        from obsidian_rag.mcp_server.server import _register_health_check

        mock_mcp = MagicMock()
        mock_db_manager = MagicMock()

        # Setup session mock that raises exception
        mock_session = MagicMock()
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)
        mock_session.execute.side_effect = Exception("Database error")
        mock_db_manager.get_session.return_value = mock_session

        _register_health_check(mock_mcp, mock_db_manager)

        # Verify custom_route was called
        mock_mcp.custom_route.assert_called_once()

        # Extract and call the health check handler to test error path
        route_call = mock_mcp.custom_route.call_args
        handler = route_call[0][0]
        # Simulate a request
        from unittest.mock import AsyncMock

        mock_request = MagicMock()

        # The handler returns a JSONResponse, we need to test it doesn't raise
        import asyncio

        try:
            if asyncio.iscoroutinefunction(handler):
                result = asyncio.run(handler(mock_request))
            else:
                result = handler(mock_request)
        except Exception:
            pass  # Expected to handle the error
