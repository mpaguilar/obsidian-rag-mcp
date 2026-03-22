"""Middleware for MCP server request/response handling.

This module provides middleware for logging, session tracking,
and rate limiting for the MCP server.
"""

import logging
import time
from collections.abc import Callable
from typing import Any, cast

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

log = logging.getLogger(__name__)


class SessionLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware to log HTTP request/response lifecycle.

    Logs incoming requests and outgoing responses at DEBUG level
    to help with debugging client behavior.

    Attributes:
        app: The ASGI application to wrap.

    """

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Any],
    ) -> Response:
        """Process the request/response cycle with logging.

        Args:
            request: The incoming HTTP request.
            call_next: The next handler in the chain.

        Returns:
            The HTTP response.

        """
        _msg = f"HTTP request: {request.method} {request.url.path}"
        log.debug(_msg)

        start_time = time.time()

        try:
            response = await call_next(request)
        except Exception:
            _msg = f"HTTP response exception: {request.method} {request.url.path}"
            log.exception(_msg)
            raise

        duration = time.time() - start_time
        _msg = (
            f"HTTP response: {request.method} {request.url.path} "
            f"{response.status_code} ({duration:.3f}s)"
        )
        log.debug(_msg)

        return cast("Response", response)
