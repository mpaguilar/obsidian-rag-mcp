"""FastMCP server initialization and configuration."""

import logging
import os

from fastmcp import FastMCP
from fastmcp.server.auth.providers.jwt import StaticTokenVerifier
from sqlalchemy import exc, text
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

from obsidian_rag.config import Settings
from obsidian_rag.database.engine import DatabaseManager
from obsidian_rag.mcp_server.handlers import (
    DocumentTagParams,
    IngestHandlerParams,
    QueryFilterParams,
    TaskDateFilterStrings,
    _convert_property_filters,
    _create_tag_filter,
    _get_documents_by_tag_handler,
    _ingest_handler,
)
from obsidian_rag.mcp_server.middleware import SessionLoggingMiddleware
from obsidian_rag.mcp_server.models import HealthResponse, SessionMetrics
from obsidian_rag.mcp_server.session_manager import SessionManager
from obsidian_rag.mcp_server.tool_definitions import (
    MCPToolRegistry,
    _create_embedding_provider,
    _get_registry,
    _set_registry,
    get_all_tags_tool,
    list_vaults_tool,
    query_documents_tool,
)

# Global session manager instance
_session_manager: SessionManager | None = None

log = logging.getLogger(__name__)


# ============================================================================
# Tool Wrappers (access dependencies through registry)
# ============================================================================


def query_documents(
    query: str,
    filters: QueryFilterParams | None = None,
    limit: int = 20,
    offset: int = 0,
) -> dict[str, object]:
    """Semantic search over document content with optional filters.

    Args:
        query: Search query text.
        filters: QueryFilterParams with include_properties, exclude_properties,
            include_tags, exclude_tags, and match_mode.
        limit: Maximum number of results (default: 20, max: 100).
        offset: Number of results to skip (default: 0).

    Returns:
        Document list response with pagination and similarity scores.

    """
    registry = _get_registry()
    return query_documents_tool(
        db_manager=registry.db_manager,
        embedding_provider=registry.embedding_provider,
        query=query,
        filters=filters,
        limit=limit,
        offset=offset,
    )


def get_documents_by_tag(
    filters: QueryFilterParams | None = None,
    vault_name: str | None = None,
    limit: int = 20,
    offset: int = 0,
) -> dict[str, object]:
    """Query documents filtered by tags with include/exclude semantics.

    Args:
        filters: QueryFilterParams with include_tags, exclude_tags, and match_mode.
        vault_name: Filter by specific vault name (optional).
        limit: Maximum number of results (default: 20, max: 100).
        offset: Number of results to skip (default: 0).

    Returns:
        Document list response with pagination and relative paths.

    """
    _msg = "Tool get_documents_by_tag called"
    log.info(_msg)

    registry = _get_registry()

    filters = filters or QueryFilterParams(
        include_properties=None,
        exclude_properties=None,
        include_tags=None,
        exclude_tags=None,
        match_mode="all",
    )

    params: DocumentTagParams = {
        "include_tags": filters.include_tags or [],
        "exclude_tags": filters.exclude_tags or [],
        "match_mode": filters.match_mode
        if filters.match_mode in ("all", "any")
        else "all",
        "vault_name": vault_name,
        "limit": limit,
        "offset": offset,
    }
    return _get_documents_by_tag_handler(registry.db_manager, params)


def get_documents_by_property(
    filters: QueryFilterParams | None = None,
    vault_name: str | None = None,
    limit: int = 20,
    offset: int = 0,
) -> dict[str, object]:
    """Query documents filtered by frontmatter properties.

    Args:
        filters: QueryFilterParams with include_properties, exclude_properties,
            include_tags, exclude_tags, and match_mode.
        vault_name: Filter by specific vault name (optional).
        limit: Maximum number of results (default: 20, max: 100).
        offset: Number of results to skip (default: 0).

    Returns:
        Document list response with pagination and relative paths.

    Raises:
        ValueError: If property filter validation fails.

    """
    from obsidian_rag.mcp_server.tools.documents import (
        get_documents_by_property as get_documents_by_property_tool,
    )
    from obsidian_rag.mcp_server.tools.documents_params import (
        PaginationParams,
        PropertyFilterParams,
    )

    _msg = "Tool get_documents_by_property called"
    log.info(_msg)

    registry = _get_registry()

    filters = filters or QueryFilterParams(
        include_properties=None,
        exclude_properties=None,
        include_tags=None,
        exclude_tags=None,
        match_mode="all",
    )

    prop_filters_include = _convert_property_filters(filters.include_properties)
    prop_filters_exclude = _convert_property_filters(filters.exclude_properties)
    tag_filter = _create_tag_filter(filters)

    # Bundle property filters into PropertyFilterParams
    property_filter_params = PropertyFilterParams(
        include_filters=prop_filters_include,
        exclude_filters=prop_filters_exclude,
    )
    pagination = PaginationParams(limit=limit, offset=offset)

    with registry.db_manager.get_session() as session:
        result = get_documents_by_property_tool(
            session=session,
            property_filters=property_filter_params,
            tag_filter=tag_filter,
            vault_name=vault_name,
            pagination=pagination,
        )
        return result.model_dump()


def get_all_tags(
    pattern: str | None = None,
    limit: int = 20,
    offset: int = 0,
) -> dict[str, object]:
    """Query all unique document tags with optional pattern filtering.

    Args:
        pattern: Glob pattern for filtering tags (optional).
            Supports * (any chars), ? (single char), [abc] (char class).
        limit: Maximum number of results (default: 20, max: 100).
        offset: Number of results to skip (default: 0).

    Returns:
        Dictionary with tag list response and pagination info.

    """
    registry = _get_registry()
    return get_all_tags_tool(registry.db_manager, pattern, limit, offset)


def list_vaults(
    limit: int = 20,
    offset: int = 0,
) -> dict[str, object]:
    """List all configured vaults with document counts.

    Args:
        limit: Maximum number of results (default: 20, max: 100).
        offset: Number of results to skip (default: 0).

    Returns:
        Dictionary with vault list response including metadata and document counts.

    """
    registry = _get_registry()
    return list_vaults_tool(registry.db_manager, limit, offset)


def ingest(
    vault_name: str,
    path: str | None = None,
    *,
    no_delete: bool = False,
) -> dict[str, object]:
    """Ingest markdown files from a vault directory into the database.

    Args:
        vault_name: Name of the vault to ingest into (required).
            Must match a vault configured in the config file.
        path: Optional path to vault directory. Uses vault's container_path
            if not provided.
        no_delete: If True, skip deletion of orphaned documents.
            Default is False.

    Returns:
        Dictionary with ingestion statistics including:
        - total: Total files processed
        - new: New documents created
        - updated: Documents updated
        - unchanged: Documents unchanged
        - errors: Files that failed
        - deleted: Orphaned documents deleted
        - processing_time_seconds: Time taken
        - message: Human-readable summary

    Raises:
        ValueError: If vault_name is not found in configuration.

    """
    _msg = f"Tool ingest called with vault: {vault_name}, path: {path}, no_delete: {no_delete}"
    log.info(_msg)

    registry = _get_registry()
    params = IngestHandlerParams(
        settings=registry.settings,
        db_manager=registry.db_manager,
        embedding_provider=registry.embedding_provider,
        vault_name=vault_name,
        path_override=path,
        no_delete=no_delete,
    )
    return _ingest_handler(params)


def get_tasks(  # noqa: PLR0913
    status: list[str] | None = None,
    date_filters: "TaskDateFilterStrings | None" = None,
    tags: list[str] | None = None,
    priority: list[str] | None = None,
    *,
    include_completed: bool = True,
    include_cancelled: bool = False,
    limit: int = 20,
    offset: int = 0,
) -> dict[str, object]:
    """Query tasks with flexible filtering by status, dates, priority, and tags.

    This tool provides comprehensive task filtering with support for date ranges,
    status lists, tag filtering, and priority filtering. All filters are optional
    and combined with AND logic.

    Args:
        status: List of statuses to filter by (e.g., ['not_completed', 'in_progress']).
        date_filters: Date filter parameters with ISO date strings.
        tags: List of tags that tasks must have (all tags required).
        priority: List of priorities to filter by (e.g., ['high', 'highest']).
        include_completed: Whether to include completed tasks (default: True).
        include_cancelled: Whether to include cancelled tasks (default: False).
        limit: Maximum number of results (default: 20, max: 100).
        offset: Number of results to skip (default: 0).

    Returns:
        Dictionary with paginated task list response.

    Notes:
        Date comparisons are inclusive (>= for after, <= for before).
        Multiple filters are combined with AND logic.
        Returns empty results if no tasks match the criteria.

    """
    from obsidian_rag.mcp_server.handlers import _get_tasks_handler

    registry = _get_registry()

    # Create date filters if individual date params provided
    if date_filters is None:
        date_filters = TaskDateFilterStrings()

    return _get_tasks_handler(
        db_manager=registry.db_manager,
        status=status,
        date_filters=date_filters,
        tags=tags,
        priority=priority,
        include_completed=include_completed,
        include_cancelled=include_cancelled,
        limit=limit,
        offset=offset,
    )


async def health_check(_request: Request) -> JSONResponse:
    """Health check endpoint.

    Args:
        _request: HTTP request (unused).

    Returns:
        JSONResponse with health status.

    """
    registry = _get_registry()
    return await health_check_handler(registry.db_manager)


async def health_check_handler(db_manager: DatabaseManager) -> JSONResponse:
    """Health check endpoint handler.

    Args:
        db_manager: Database manager for session management.

    Returns:
        JSONResponse with health status, version, database connection status,
        and session metrics.

    Notes:
        This is a module-level function for testability.
        The @mcp.custom_route() decorator is applied in the registration function.

    """
    _msg = "Health check called"
    log.debug(_msg)

    db_status = "connected"
    try:
        with db_manager.get_session() as session:
            session.execute(text("SELECT 1"))
    except exc.SQLAlchemyError as e:
        db_status = f"error: {e}"
        _msg = f"Database health check failed: {e}"
        log.warning(_msg)

    version = os.environ.get("OBSIDIAN_RAG_VERSION", "0.2.3")

    # Get session metrics if available
    session_metrics = SessionMetrics()
    manager = _get_session_manager()
    if manager:
        metrics_dict = manager.get_metrics()
        session_metrics = SessionMetrics(
            total_created=metrics_dict.get("total_created", 0),
            total_destroyed=metrics_dict.get("total_destroyed", 0),
            active_count=metrics_dict.get("active_count", 0),
            total_requests=metrics_dict.get("total_requests", 0),
            peak_concurrent=metrics_dict.get("peak_concurrent", 0),
            connection_rate=metrics_dict.get("connection_rate", 0.0),
            active_sessions_by_ip=metrics_dict.get("active_sessions_by_ip", {}),
        )

    health_data = HealthResponse(
        status="healthy",
        version=version,
        database=db_status,
        sessions=session_metrics,
    ).model_dump()

    return JSONResponse(health_data)


def _get_session_manager() -> SessionManager | None:
    """Get the global session manager instance.

    Returns:
        SessionManager instance or None if not initialized.

    """
    return _session_manager


# ============================================================================
# Server Creation and Registration
# ============================================================================


def _register_tools(mcp: FastMCP) -> None:
    """Register all MCP tools.

    Tools are defined at module level and access dependencies through
    _get_registry(). This function registers them with the FastMCP instance.

    Args:
        mcp: FastMCP instance to register tools with.

    """
    _msg = "_register_tools starting"
    log.debug(_msg)

    # Register task tools
    mcp.tool()(get_tasks)

    # Register document tools
    mcp.tool()(query_documents)
    mcp.tool()(get_documents_by_tag)
    mcp.tool()(get_documents_by_property)
    mcp.tool()(get_all_tags)

    # Register vault tools
    mcp.tool()(list_vaults)

    # Register ingest tool
    mcp.tool()(ingest)

    _msg = "_register_tools returning"
    log.debug(_msg)


def create_mcp_server(settings: Settings) -> FastMCP:
    """Create and configure the FastMCP server.

    Args:
        settings: Application settings.

    Returns:
        Configured FastMCP instance.

    Raises:
        ValueError: If MCP token is not configured.

    """
    _msg = "Creating MCP server"
    log.info(_msg)

    if not settings.mcp.token:
        _msg = "MCP token is required but not configured"
        log.error(_msg)
        raise ValueError(_msg)

    token_verifier = StaticTokenVerifier(
        tokens={
            settings.mcp.token: {
                "client_id": "obsidian-rag-client",
                "sub": "user",
            },
        },
    )

    # Create FastMCP instance
    mcp = FastMCP("Obsidian RAG Server", auth=token_verifier)

    # Create dependencies
    db_manager = DatabaseManager(settings.database.url)
    embedding_provider = _create_embedding_provider(settings)

    # Log embedding provider configuration for diagnostics
    if embedding_provider is not None:
        _msg = (
            f"Embedding provider initialized: type={type(embedding_provider).__name__}, "
            f"model={getattr(embedding_provider, 'model', 'unknown')}, "
            f"base_url={getattr(embedding_provider, 'base_url', 'default')}"
        )
        log.info(_msg)
    else:
        _msg = "No embedding provider configured - semantic search disabled"
        log.info(_msg)

    # Initialize global registry BEFORE registering tools
    _set_registry(MCPToolRegistry(db_manager, embedding_provider, settings))

    # Register all tools
    _register_tools(mcp)

    # Register health check endpoint if enabled
    if settings.mcp.enable_health_check:
        mcp.custom_route("/health", methods=["GET"])(health_check)

    _msg = "MCP server created successfully"
    log.info(_msg)

    _msg = "create_mcp_server returning"
    log.debug(_msg)
    return mcp


def create_http_app(settings: Settings) -> Starlette:
    """Create HTTP ASGI app for the MCP server.

    Args:
        settings: Application settings.

    Returns:
        Starlette ASGI application.

    """
    _msg = "Creating HTTP app"
    log.info(_msg)

    global _session_manager

    # Create session manager with settings
    _session_manager = SessionManager(
        max_concurrent_sessions=settings.mcp.max_concurrent_sessions,
        session_timeout_seconds=settings.mcp.session_timeout_seconds,
        rate_limit_per_second=settings.mcp.rate_limit_per_second,
        rate_limit_window=settings.mcp.rate_limit_window,
    )

    _msg = (
        f"Session manager created: max_concurrent={settings.mcp.max_concurrent_sessions}, "
        f"timeout={settings.mcp.session_timeout_seconds}s, "
        f"rate_limit={settings.mcp.rate_limit_per_second}/sec"
    )
    log.info(_msg)

    mcp = create_mcp_server(settings)

    middleware: list[Middleware] = [
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

    # Add request logging middleware if enabled
    if settings.mcp.enable_request_logging:
        middleware.append(Middleware(SessionLoggingMiddleware))
        _msg = "Session logging middleware enabled"
        log.info(_msg)

    app = mcp.http_app(
        path="/",
        middleware=middleware,
        stateless_http=settings.mcp.stateless_http,
    )

    _msg = "HTTP app created successfully"
    log.info(_msg)

    return app
