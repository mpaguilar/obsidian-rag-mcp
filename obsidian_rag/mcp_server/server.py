"""FastMCP server initialization and configuration."""

import logging
import os
from datetime import date

from fastmcp import FastMCP
from fastmcp.server.auth.providers.jwt import StaticTokenVerifier
from sqlalchemy import text
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

from obsidian_rag.config import Settings
from obsidian_rag.database.engine import DatabaseManager
from obsidian_rag.llm.base import EmbeddingProvider, ProviderFactory
from obsidian_rag.mcp_server.models import HealthResponse
from obsidian_rag.mcp_server.tools.documents import (
    query_documents as query_documents_tool,
)
from obsidian_rag.mcp_server.tools.tasks import (
    get_completed_tasks as get_completed_tasks_tool,
    get_incomplete_tasks as get_incomplete_tasks_tool,
    get_tasks_by_tag as get_tasks_by_tag_tool,
    get_tasks_due_this_week as get_tasks_due_this_week_tool,
)

log = logging.getLogger(__name__)


def _create_embedding_provider(
    settings: Settings,
) -> EmbeddingProvider | None:
    """Create embedding provider for semantic search.

    Args:
        settings: Application settings.

    Returns:
        Embedding provider instance or None if creation fails.

    """
    embedding_config = settings.endpoints.get("embedding")
    if not embedding_config:
        return None

    try:
        provider = ProviderFactory.create_embedding_provider(
            provider_name=embedding_config.provider,
            api_key=embedding_config.api_key,
            model=embedding_config.model,
            base_url=embedding_config.base_url,
        )
        _msg = f"Created embedding provider: {embedding_config.provider}"
        log.info(_msg)
        return provider
    except Exception as e:
        _msg = f"Failed to create embedding provider: {e}"
        log.warning(_msg)
        return None


def _get_incomplete_tasks_handler(
    db_manager: DatabaseManager,
    limit: int,
    offset: int,
    include_cancelled: bool,
) -> dict[str, object]:
    """Handle get_incomplete_tasks tool call."""
    with db_manager.get_session() as session:
        result = get_incomplete_tasks_tool(
            session=session,
            limit=limit,
            offset=offset,
            include_cancelled=include_cancelled,
        )
        return result.model_dump()


def _get_tasks_due_this_week_handler(
    db_manager: DatabaseManager,
    limit: int,
    offset: int,
    include_completed: bool,
) -> dict[str, object]:
    """Handle get_tasks_due_this_week tool call."""
    with db_manager.get_session() as session:
        result = get_tasks_due_this_week_tool(
            session=session,
            limit=limit,
            offset=offset,
            include_completed=include_completed,
        )
        return result.model_dump()


def _get_tasks_by_tag_handler(
    db_manager: DatabaseManager,
    tag: str,
    limit: int,
    offset: int,
) -> dict[str, object]:
    """Handle get_tasks_by_tag tool call."""
    with db_manager.get_session() as session:
        result = get_tasks_by_tag_tool(
            session=session,
            tag=tag,
            limit=limit,
            offset=offset,
        )
        return result.model_dump()


def _get_completed_tasks_handler(
    db_manager: DatabaseManager,
    limit: int,
    offset: int,
    completed_since: str | None,
) -> dict[str, object]:
    """Handle get_completed_tasks tool call."""
    since_date = None
    if completed_since:
        try:
            since_date = date.fromisoformat(completed_since)
        except ValueError:
            _msg = f"Invalid date format: {completed_since}"
            log.warning(_msg)

    with db_manager.get_session() as session:
        result = get_completed_tasks_tool(
            session=session,
            limit=limit,
            offset=offset,
            completed_since=since_date,
        )
        return result.model_dump()


def _register_task_tools(
    mcp: FastMCP,
    db_manager: DatabaseManager,
) -> None:
    """Register task-related tools.

    Args:
        mcp: FastMCP server instance.
        db_manager: Database manager for sessions.

    """

    @mcp.tool()
    def get_incomplete_tasks(
        limit: int = 20,
        offset: int = 0,
        include_cancelled: bool = False,
    ) -> dict[str, object]:
        """Query tasks that are not completed."""
        _msg = "Tool get_incomplete_tasks called"
        log.info(_msg)
        return _get_incomplete_tasks_handler(
            db_manager, limit, offset, include_cancelled
        )

    @mcp.tool()
    def get_tasks_due_this_week(
        limit: int = 20,
        offset: int = 0,
        include_completed: bool = True,
    ) -> dict[str, object]:
        """Query tasks due within the next 7 days."""
        _msg = "Tool get_tasks_due_this_week called"
        log.info(_msg)
        return _get_tasks_due_this_week_handler(
            db_manager, limit, offset, include_completed
        )

    @mcp.tool()
    def get_tasks_by_tag(
        tag: str,
        limit: int = 20,
        offset: int = 0,
    ) -> dict[str, object]:
        """Query tasks by tag (matches task or document level tags)."""
        _msg = f"Tool get_tasks_by_tag called with tag: {tag}"
        log.info(_msg)
        return _get_tasks_by_tag_handler(db_manager, tag, limit, offset)

    @mcp.tool()
    def get_completed_tasks(
        limit: int = 20,
        offset: int = 0,
        completed_since: str | None = None,
    ) -> dict[str, object]:
        """Query completed tasks with optional date filter."""
        _msg = "Tool get_completed_tasks called"
        log.info(_msg)
        return _get_completed_tasks_handler(db_manager, limit, offset, completed_since)


def _register_document_tools(
    mcp: FastMCP,
    db_manager: DatabaseManager,
    embedding_provider: EmbeddingProvider | None,
) -> None:
    """Register document-related tools.

    Args:
        mcp: FastMCP server instance.
        db_manager: Database manager for sessions.
        embedding_provider: Embedding provider for semantic search.

    """

    @mcp.tool()
    def query_documents(
        query: str,
        limit: int = 20,
        offset: int = 0,
    ) -> dict[str, object]:
        """Semantic search over document content.

        Args:
            query: Search query text.
            limit: Maximum number of results (default: 20, max: 100).
            offset: Number of results to skip (default: 0).

        Returns:
            Document list response with pagination and similarity scores.

        Raises:
            RuntimeError: If embedding provider is not available.

        """
        _msg = f"Tool query_documents called with query: {query[:50]}..."
        log.info(_msg)

        if not embedding_provider:
            _msg = "Embedding provider not configured"
            log.error(_msg)
            raise RuntimeError(_msg)

        query_embedding = embedding_provider.generate_embedding(query)

        with db_manager.get_session() as session:
            result = query_documents_tool(
                session=session,
                query_embedding=query_embedding,
                limit=limit,
                offset=offset,
            )
            return result.model_dump()


def _register_health_check(
    mcp: FastMCP,
    db_manager: DatabaseManager,
) -> None:
    """Register health check endpoint.

    Args:
        mcp: FastMCP server instance.
        db_manager: Database manager for sessions.

    """

    @mcp.custom_route("/health", methods=["GET"])
    async def health_check(_request: Request) -> JSONResponse:
        """Health check endpoint.

        Args:
            _request: The incoming request (unused but required by FastMCP).

        Returns:
            Health status response as JSON.

        """
        _msg = "Health check called"
        log.debug(_msg)

        db_status = "connected"
        try:
            with db_manager.get_session() as session:
                session.execute(text("SELECT 1"))
        except Exception as e:
            db_status = f"error: {e}"
            _msg = f"Database health check failed: {e}"
            log.warning(_msg)

        version = os.environ.get("OBSIDIAN_RAG_VERSION", "0.2.3")

        health_data = HealthResponse(
            status="healthy",
            version=version,
            database=db_status,
        ).model_dump()

        return JSONResponse(health_data)


def create_mcp_server(settings: Settings) -> FastMCP:
    """Create and configure the FastMCP server.

    Args:
        settings: Application settings.

    Returns:
        Configured FastMCP server instance.

    Raises:
        ValueError: If MCP token is not configured.

    """
    _msg = "Creating MCP server"
    log.info(_msg)

    if not settings.mcp.token:
        _msg = "MCP token is required but not configured"
        log.error(_msg)
        raise ValueError(_msg)

    auth = StaticTokenVerifier(
        tokens={settings.mcp.token: {"sub": "user"}},
    )
    mcp = FastMCP(
        "Obsidian RAG Server",
        auth=auth,
    )
    db_manager = DatabaseManager(settings.database.url)
    embedding_provider = _create_embedding_provider(settings)

    _register_task_tools(mcp, db_manager)
    _register_document_tools(mcp, db_manager, embedding_provider)

    if settings.mcp.enable_health_check:
        _register_health_check(mcp, db_manager)

    _msg = "MCP server created successfully"
    log.info(_msg)

    return mcp


def create_http_app(settings: Settings) -> Starlette:
    """Create HTTP ASGI app for the MCP server.

    Args:
        settings: Application settings.

    Returns:
        ASGI app with CORS middleware.

    """
    _msg = "Creating HTTP app"
    log.info(_msg)

    mcp = create_mcp_server(settings)

    # Configure CORS middleware with MCP-specific headers for browser clients
    middleware = [
        Middleware(
            CORSMiddleware,
            allow_origins=settings.mcp.cors_origins,
            allow_credentials=True,
            allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
            allow_headers=[
                "mcp-protocol-version",
                "mcp-session-id",
                "Authorization",
                "Content-Type",
            ],
            expose_headers=["mcp-session-id"],
        ),
    ]

    app = mcp.http_app(
        middleware=middleware,
        stateless_http=settings.mcp.stateless_http,
    )

    _msg = "HTTP app created successfully"
    log.info(_msg)

    return app
