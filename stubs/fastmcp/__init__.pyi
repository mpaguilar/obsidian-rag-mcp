"""Type stubs for fastmcp library.

fastmcp provides FastMCP server implementation for Model Context Protocol.
"""

from collections.abc import Callable
from typing import Any, Protocol

from starlette.applications import Starlette
from starlette.middleware import Middleware

class ASGIApp(Protocol):
    """Protocol for ASGI application."""

    async def __call__(
        self,
        scope: dict[str, Any],
        receive: Callable[[], Any],
        send: Callable[[Any], None],
    ) -> None: ...

class FastMCP:
    """FastMCP server for Model Context Protocol.

    Provides HTTP-based MCP server with tool registration and
    session management capabilities.
    """

    def custom_route(
        self,
        path: str,
        methods: list[str] | None = None,
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Register a custom HTTP route.

        Args:
            path: URL path for the route.
            methods: List of HTTP methods (e.g., ["GET", "POST"]).

        Returns:
            Decorator function to register the route handler.
        """
        ...

    def __init__(
        self,
        name: str,
        instructions: str | None = None,
        auth: Any | None = None,
    ) -> None:
        """Initialize FastMCP server.

        Args:
            name: Server name identifier.
            instructions: Optional instructions for clients.
            auth: Optional authentication provider.
        """
        ...

    def tool(
        self,
        name: str | None = None,
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Decorator to register a function as an MCP tool.

        Args:
            name: Tool name. If None, uses function name.

        Returns:
            Decorator function that registers the tool.
        """
        ...

    def http_app(
        self,
        path: str = "/",
        middleware: list[Middleware] | None = None,
        stateless_http: bool = False,
    ) -> Starlette:
        """Get ASGI application for HTTP transport.

        Args:
            path: URL path for the MCP endpoint.
            middleware: List of Starlette middleware.
            stateless_http: Enable stateless HTTP mode.

        Returns:
            Starlette ASGI app that can be mounted or run directly.
        """
        ...

def create_http_app(
    mcp: FastMCP,
    settings: Any,
) -> ASGIApp:
    """Create HTTP app with CORS middleware.

    Args:
        mcp: Configured FastMCP instance.
        settings: Application settings object.

    Returns:
        ASGI app with middleware configured.
    """
    ...
