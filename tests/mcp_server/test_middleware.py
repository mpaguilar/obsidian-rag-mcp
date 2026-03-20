"""Tests for middleware module."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from starlette.requests import Request
from starlette.responses import JSONResponse

from obsidian_rag.mcp_server.middleware import SessionLoggingMiddleware


class TestSessionLoggingMiddleware:
    """Tests for SessionLoggingMiddleware class."""

    @pytest.mark.asyncio
    async def test_dispatch_logs_request(self):
        """Test that dispatch logs incoming requests."""
        app = MagicMock()
        middleware = SessionLoggingMiddleware(app)

        mock_request = MagicMock(spec=Request)
        mock_request.method = "GET"
        mock_request.url.path = "/test"

        mock_response = MagicMock(spec=JSONResponse)
        mock_response.status_code = 200

        mock_call_next = AsyncMock(return_value=mock_response)

        with patch("obsidian_rag.mcp_server.middleware.log") as mock_log:
            response = await middleware.dispatch(mock_request, mock_call_next)

            # Verify request was logged
            debug_calls = [call for call in mock_log.debug.call_args_list]
            assert len(debug_calls) >= 1
            assert "GET /test" in str(debug_calls[0])
            assert response == mock_response

    @pytest.mark.asyncio
    async def test_dispatch_logs_response(self):
        """Test that dispatch logs outgoing responses."""
        app = MagicMock()
        middleware = SessionLoggingMiddleware(app)

        mock_request = MagicMock(spec=Request)
        mock_request.method = "POST"
        mock_request.url.path = "/api/test"

        mock_response = MagicMock(spec=JSONResponse)
        mock_response.status_code = 202

        mock_call_next = AsyncMock(return_value=mock_response)

        with patch("obsidian_rag.mcp_server.middleware.log") as mock_log:
            await middleware.dispatch(mock_request, mock_call_next)

            # Verify response was logged
            debug_calls = [call for call in mock_log.debug.call_args_list]
            assert len(debug_calls) >= 2
            response_log = str(debug_calls[1])
            assert "POST /api/test" in response_log
            assert "202" in response_log

    @pytest.mark.asyncio
    async def test_dispatch_logs_exception(self):
        """Test that dispatch logs exceptions."""
        app = MagicMock()
        middleware = SessionLoggingMiddleware(app)

        mock_request = MagicMock(spec=Request)
        mock_request.method = "GET"
        mock_request.url.path = "/error"

        mock_call_next = AsyncMock(side_effect=ValueError("Test error"))

        with patch("obsidian_rag.mcp_server.middleware.log") as mock_log:
            with pytest.raises(ValueError, match="Test error"):
                await middleware.dispatch(mock_request, mock_call_next)

            # Verify exception was logged
            exception_calls = [call for call in mock_log.exception.call_args_list]
            assert len(exception_calls) >= 1
            exception_log = str(exception_calls[0])
            assert "GET /error" in exception_log

    @pytest.mark.asyncio
    async def test_dispatch_different_methods(self):
        """Test dispatch with different HTTP methods."""
        app = MagicMock()
        middleware = SessionLoggingMiddleware(app)

        for method in ["GET", "POST", "PUT", "DELETE", "PATCH"]:
            mock_request = MagicMock(spec=Request)
            mock_request.method = method
            mock_request.url.path = "/test"

            mock_response = MagicMock(spec=JSONResponse)
            mock_response.status_code = 200

            mock_call_next = AsyncMock(return_value=mock_response)

            with patch("obsidian_rag.mcp_server.middleware.log") as mock_log:
                await middleware.dispatch(mock_request, mock_call_next)

                debug_calls = [call for call in mock_log.debug.call_args_list]
                assert len(debug_calls) >= 2
                assert method in str(debug_calls[0])

    @pytest.mark.asyncio
    async def test_dispatch_different_status_codes(self):
        """Test dispatch with different status codes."""
        app = MagicMock()
        middleware = SessionLoggingMiddleware(app)

        for status in [200, 201, 400, 401, 404, 500]:
            mock_request = MagicMock(spec=Request)
            mock_request.method = "GET"
            mock_request.url.path = "/test"

            mock_response = MagicMock(spec=JSONResponse)
            mock_response.status_code = status

            mock_call_next = AsyncMock(return_value=mock_response)

            with patch("obsidian_rag.mcp_server.middleware.log") as mock_log:
                await middleware.dispatch(mock_request, mock_call_next)

                debug_calls = [call for call in mock_log.debug.call_args_list]
                assert len(debug_calls) >= 2
                assert str(status) in str(debug_calls[1])
