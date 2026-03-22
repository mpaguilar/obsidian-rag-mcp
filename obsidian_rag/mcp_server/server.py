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

from obsidian_rag.config import Settings, get_settings
from obsidian_rag.database.engine import DatabaseManager
from obsidian_rag.mcp_server.handlers import (
    DocumentTagParams,
    GetTasksToolInput,
    IngestHandlerParams,
    QueryFilterParams,
    TagFilterStrings,
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
    *,
    use_chunks: bool = False,
    rerank: bool = False,
) -> dict[str, object]:
    """Semantic search over document content with optional filters.

    Args:
        query: Search query text.
        filters: QueryFilterParams with include_properties, exclude_properties,
            include_tags, exclude_tags, and match_mode.
        limit: Maximum number of results (default: 20, max: 100).
        offset: Number of results to skip (default: 0).
        use_chunks: If True, search at chunk level instead of document level.
            Returns the best matching chunk per document for more precise
            semantic matching in large documents.
        rerank: If True, apply flashrank re-ranking to chunk results.
            Only applies when use_chunks is True.

    Returns:
        Document list response with pagination and similarity scores.
        When use_chunks is True, the content field contains the matching
        chunk text and matching_chunk field indicates chunk search was used.

    """
    from obsidian_rag.mcp_server.tools.documents_params import PaginationParams

    registry = _get_registry()
    pagination = PaginationParams(limit=limit, offset=offset)
    return query_documents_tool(
        db_manager=registry.db_manager,
        embedding_provider=registry.embedding_provider,
        query=query,
        filters=filters,
        pagination=pagination,
        use_chunks=use_chunks,
        rerank=rerank,
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


def get_tasks(
    params: "GetTasksToolInput",
) -> dict[str, object]:
    """Query tasks with flexible filtering by status, dates, priority, and tags.

    This tool provides comprehensive task filtering with support for date ranges,
    status lists, tag filtering with include/exclude semantics, and priority
    filtering. All filters are optional and combined with AND logic by default.

    Valid Status Values:
        - "not_completed": Tasks that are not yet completed
        - "completed": Tasks that have been completed
        - "in_progress": Tasks currently being worked on
        - "cancelled": Tasks that have been cancelled

    Valid Priority Values:
        - "highest": Critical priority tasks
        - "high": High priority tasks
        - "normal": Normal priority tasks (default)
        - "low": Low priority tasks
        - "lowest": Lowest priority tasks

    Tag Filtering:
        Tag filters are specified in the tag_filters object:

        include_tags: Tasks must have these tags (controlled by match_mode).
            - match_mode="all" (default): Task must have ALL include tags
            - match_mode="any": Task must have ANY of the include tags

        exclude_tags: Tasks must NOT have any of these tags (always OR logic).

        Examples:
            - Find tasks with BOTH "work" AND "urgent" tags:
              tag_filters={"include_tags": ["work", "urgent"], "match_mode": "all"}
            - Find tasks with EITHER "work" OR "personal" tag:
              tag_filters={"include_tags": ["work", "personal"], "match_mode": "any"}
            - Find tasks WITHOUT "blocked" tag:
              tag_filters={"exclude_tags": ["blocked"]}
            - Find tasks with "work" but NOT "blocked":
              tag_filters={"include_tags": ["work"], "exclude_tags": ["blocked"]}

        Validation:
            - Same tag cannot appear in both include_tags and exclude_tags
            - Matching is case-insensitive ("Work" matches "work")

    Date Filtering:
        Date filters are specified in the date_filters object:

        Available date fields:
            - due_after: Filter tasks due on or after this date
            - due_before: Filter tasks due on or before this date
            - scheduled_after: Filter tasks scheduled on or after this date
            - scheduled_before: Filter tasks scheduled on or before this date
            - completion_after: Filter tasks completed on or after this date
            - completion_before: Filter tasks completed on or before this date

        match_mode: How to combine date filters
            - "all" (default): AND logic across all date conditions
            - "any": OR logic across all date conditions

    Legacy Support:
        The 'tags' parameter is maintained for backward compatibility and uses
        AND logic (all specified tags required). New code should use
        'tag_filters.include_tags' with 'tag_filters.match_mode' for more flexibility.

    Filter Logic:
        - Multiple status values: OR logic (task matches ANY status)
        - Multiple priority values: OR logic (task matches ANY priority)
        - Legacy tags parameter: AND logic (task must have ALL tags)
        - tag_filters.include_tags with match_mode="all" (default): AND logic
        - tag_filters.include_tags with match_mode="any": OR logic
        - tag_filters.exclude_tags: OR logic (task excluded if it has ANY excluded tag)
        - Date filters: Configurable via date_filters.match_mode
            - "all" (default): AND logic across all date conditions
            - "any": OR logic across all date conditions
        - Different filter types (status, tags, priority, dates): AND logic

    Args:
        params: GetTasksToolInput containing all filter parameters.

    Returns:
        Dictionary with paginated task list response including:
        - results: List of matching tasks with full details
        - total_count: Total number of tasks matching the filters
        - has_more: True if more results are available beyond this page
        - next_offset: The offset value to use for the next page (if has_more)

    Raises:
        ValueError: If tag filter validation fails, such as when the same tag
            appears in both include_tags and exclude_tags lists.

    Notes:
        Date comparisons are inclusive (>= for after, <= for before).
        Returns empty results (total_count=0) if no tasks match the criteria.
        Tag matching is case-insensitive.

    """
    from obsidian_rag.mcp_server.handlers import GetTasksRequest, _get_tasks_handler

    registry = _get_registry()

    # Create default filters if not provided
    tag_filters = params.tag_filters or TagFilterStrings()
    date_filters = params.date_filters or TaskDateFilterStrings()

    request = GetTasksRequest(
        status=params.status,
        tag_filters=tag_filters,
        date_filters=date_filters,
        tags=params.tags,
        priority=params.priority,
        limit=params.limit,
        offset=params.offset,
    )

    return _get_tasks_handler(
        db_manager=registry.db_manager,
        request=request,
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

    # Create dependencies with connection pooling configuration
    db_manager = DatabaseManager(
        settings.database.url,
        pool_size=settings.database.pool_size,
        max_overflow=settings.database.max_overflow,
        pool_timeout=settings.database.pool_timeout,
        pool_recycle=settings.database.pool_recycle,
    )
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


def create_http_app_factory() -> Starlette:
    """Create HTTP ASGI app factory for Gunicorn.

    This factory function loads settings from configuration and creates
    the HTTP ASGI app. It's designed to be used with Gunicorn's app factory
    pattern without requiring command-line arguments.

    Returns:
        Starlette ASGI application configured with settings.

    Raises:
        SystemExit: If settings cannot be loaded or MCP token is not configured.

    """
    _msg = "create_http_app_factory starting"
    log.info(_msg)

    try:
        settings = get_settings()
    except Exception as e:
        _msg = f"Failed to load settings: {e}"
        log.exception(_msg)
        raise SystemExit(1) from e

    # Validate token is set
    if not settings.mcp.token:
        _msg = (
            "MCP token not configured. Set OBSIDIAN_RAG_MCP_TOKEN "
            "environment variable or mcp.token in config file."
        )
        log.error(_msg)
        raise SystemExit(1)

    _msg = "create_http_app_factory returning"
    log.info(_msg)

    return create_http_app(settings)
