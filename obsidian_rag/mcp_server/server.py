"""FastMCP server initialization and configuration."""

import asyncio
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
from obsidian_rag.mcp_server.document_tools import (
    get_document,
    list_documents,
)
from obsidian_rag.mcp_server.handlers import (
    AnnotatedQueryFilter,
    DocumentTagParams,
    IngestHandlerParams,
    QueryFilterParams,
    TagFilterStrings,
    TaskDateFilterStrings,
    _convert_property_filters,
    _create_tag_filter,
    _get_documents_by_property_handler,
    _get_documents_by_tag_handler,
    _ingest_handler,
    parse_json_str,
)
from obsidian_rag.mcp_server.ingest_tracker import IngestRequestTracker
from obsidian_rag.mcp_server.middleware import SessionLoggingMiddleware
from obsidian_rag.mcp_server.models import (
    HealthResponse,
    OutputFileConfig,
    PropertyFilter,
    SessionMetrics,
)
from obsidian_rag.mcp_server.output_file import write_output_file
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
from obsidian_rag.mcp_server.tools.documents_params import PropertyFilterParams
from obsidian_rag.mcp_server.vault_tools import (
    delete_vault,
    get_vault,
    update_vault,
)
from obsidian_rag.services.ingestion_lock import IngestLockError

# Global session manager instance
_session_manager: SessionManager | None = None

# Global request tracker for ingest tool deduplication
_ingest_tracker: IngestRequestTracker | None = None

log = logging.getLogger(__name__)


def _get_ingest_tracker() -> IngestRequestTracker:
    """Get or create the global ingest request tracker.

    Returns:
        IngestRequestTracker instance.

    """
    global _ingest_tracker
    if _ingest_tracker is None:
        _ingest_tracker = IngestRequestTracker()
    return _ingest_tracker


def _clear_ingest_tracker() -> None:
    """Clear the global ingest request tracker (for testing)."""
    global _ingest_tracker
    if _ingest_tracker is not None:
        _ingest_tracker.clear_all()
        _ingest_tracker = None


from obsidian_rag.mcp_server.ingest_helpers import (
    _check_and_handle_duplicate,
    _generate_request_id,
    _handle_ingest_value_error,
)

# Tool Wrappers (access dependencies through registry)


def query_documents(
    query: str,
    filters: AnnotatedQueryFilter = None,
    limit: int = 20,
    offset: int = 0,
    *,
    use_chunks: bool = False,
    rerank: bool = False,
    include_content: bool = True,
    output_file: str | dict | OutputFileConfig | None = None,
    vault_name: str | None = None,
) -> dict[str, object]:
    """Semantic search over document content with optional filters.

    Args:
        query: Search query text.
        filters: QueryFilterParams with include_properties, exclude_properties,
            include_tags, exclude_tags, and match_mode. Can be passed as a
            JSON string or a dict. Examples:
            - Dict: {"include_tags": ["work"], "match_mode": "any"}
            - JSON string: '{"include_tags": ["work"], "match_mode": "any"}'
            - None or "": No filter
        limit: Maximum number of results (default: 20, max: 10000).
        offset: Number of results to skip (default: 0).
        use_chunks: If True, search at chunk level instead of document level.
            Returns the best matching chunk per document for more precise
            semantic matching in large documents.
        rerank: If True, apply flashrank re-ranking to chunk results.
            Only applies when use_chunks is True.
        include_content: If True (default), include full document content in
            responses. When False, the 'content' field is an empty string,
            reducing payload size.
        output_file: Optional output file configuration. When provided, the
            full result is written to the specified target (local or S3) and
            a compact summary is returned instead. Can be passed as a dict,
            JSON string, or OutputFileConfig object.
        vault_name: Optional vault name to scope search results. None means all vaults. Non-existent vault names raise ValueError, caught and returned as error dict.

    Returns:
        Document list response with pagination and similarity scores.
        When use_chunks is True, the content field contains the matching
        chunk text and matching_chunk field indicates chunk search was used.

    """
    from obsidian_rag.mcp_server.tools.documents_params import PaginationParams

    registry = _get_registry()
    pagination = PaginationParams(
        limit=limit, offset=offset, include_content=include_content
    )
    result = query_documents_tool(
        db_manager=registry.db_manager,
        embedding_provider=registry.embedding_provider,
        query=query,
        filters=filters,
        pagination=pagination,
        use_chunks=use_chunks,
        rerank=rerank,
        include_content=include_content,
        vault_name=vault_name,
    )
    parsed_output_file = _parse_output_file(output_file)
    if parsed_output_file is not None:
        return write_output_file(result, parsed_output_file)
    return result


def get_documents_by_tag(
    filters: AnnotatedQueryFilter = None,
    vault_name: str | None = None,
    limit: int = 20,
    offset: int = 0,
    *,
    output_file: str | dict | OutputFileConfig | None = None,
) -> dict[str, object]:
    """Query documents filtered by tags with include/exclude semantics.

    Args:
        filters: Query filter parameters with include_tags, exclude_tags, and
            match_mode. Can be provided as a QueryFilterParams object, a dict,
            a JSON string, or None (default). JSON strings are automatically
            parsed before validation.
            Example dict: {"include_tags": ["work"], "match_mode": "any"}
            Example JSON: '{"include_tags": ["work"], "match_mode": "any"}'
            Tags should NOT include the '#' prefix. Use plain tag names like
            "personal/expenses" instead of "#personal/expenses".
        vault_name: Filter by specific vault name (optional).
        limit: Maximum number of results (default: 20, max: 10000).
        offset: Number of results to skip (default: 0).
        output_file: Optional output file configuration. When provided, the
            full result is written to the specified target (local or S3) and
            a compact summary is returned instead. Can be passed as a dict,
            JSON string, or OutputFileConfig object.

    Returns:
        Document list response with pagination and relative paths.

    """
    _msg = "Tool get_documents_by_tag called"
    log.info(_msg)

    registry = _get_registry()

    # Handle JSON string or dict input (when called directly without FastMCP validation)
    if isinstance(filters, (str, dict)):
        parsed = parse_json_str(filters)
        if parsed is None:
            filters = None
        elif isinstance(parsed, dict):
            filters = QueryFilterParams(**parsed)
        else:
            # Defensive fallback: parse_json_str only returns dict or None for str inputs,
            # but this branch handles the impossible case where it returns a non-dict type.
            filters = parsed  # pragma: no cover

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
    result = _get_documents_by_tag_handler(registry.db_manager, params)
    parsed_output_file = _parse_output_file(output_file)
    if parsed_output_file is not None:
        return write_output_file(result, parsed_output_file)
    return result


def get_documents_by_property(
    filters: AnnotatedQueryFilter = None,
    vault_name: str | None = None,
    limit: int = 20,
    offset: int = 0,
    *,
    output_file: str | dict | OutputFileConfig | None = None,
) -> dict[str, object]:
    """Query documents filtered by frontmatter properties.

    Args:
        filters: QueryFilterParams with include_properties, exclude_properties,
            include_tags, exclude_tags, and match_mode. Can be provided as a
            dict object or a JSON-encoded string. JSON strings are automatically
            parsed before validation. Empty or whitespace-only strings are
            treated as None (no filters).
        vault_name: Filter by specific vault name (optional).
        limit: Maximum number of results (default: 20, max: 10000).
        offset: Number of results to skip (default: 0).
        output_file: Optional output file configuration. When provided, the
            full result is written to the specified target (local or S3) and
            a compact summary is returned instead. Can be passed as a dict,
            JSON string, or OutputFileConfig object.

    Returns:
        Document list response with pagination and relative paths.

    Raises:
        ValueError: If property filter validation fails or JSON parsing fails. Non-existent vault names raise ValueError, caught and returned as error dict.

    Examples:
        Dict input:
            filters={"include_tags": ["work"], "match_mode": "any"}
        JSON string input:
            filters='{"include_tags": ["work"], "match_mode": "any"}'
        No filters:
            filters=None

    """
    _msg = "Tool get_documents_by_property called"
    log.info(_msg)

    registry = _get_registry()

    # Parse JSON string filters if needed (handles clients that pass JSON strings)
    if isinstance(filters, str):
        filters = parse_json_str(filters)

    # Convert dict to QueryFilterParams if needed
    if isinstance(filters, dict):
        filters = QueryFilterParams(**filters)

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

    result = _get_documents_by_property_handler(
        db_manager=registry.db_manager,
        property_filters=property_filter_params,
        tag_filter=tag_filter,
        vault_name=vault_name,
        limit=limit,
        offset=offset,
    )

    parsed_output_file = _parse_output_file(output_file)
    if parsed_output_file is not None:
        return write_output_file(result, parsed_output_file)
    return result


def get_all_tags(
    pattern: str | None = None,
    limit: int = 20,
    offset: int = 0,
    *,
    output_file: str | dict | OutputFileConfig | None = None,
    vault_name: str | None = None,
) -> dict[str, object]:
    """Query all unique document tags with optional pattern filtering.

    Args:
        pattern: Glob pattern for filtering tags (optional).
            Supports * (any chars), ? (single char), [abc] (char class).
        limit: Maximum number of results (default: 20, max: 10000).
        offset: Number of results to skip (default: 0).
        output_file: Optional output file configuration. When provided, the
            full result is written to the specified target (local or S3) and
            a compact summary is returned instead. Can be passed as a dict,
            JSON string, or OutputFileConfig object.
        vault_name: Optional vault name to scope tag extraction. None means all vaults. Non-existent vault names raise ValueError, caught and returned as error dict.

    Returns:
        Dictionary with tag list response and pagination info.

    """
    registry = _get_registry()
    result = get_all_tags_tool(
        registry.db_manager, pattern, limit, offset, vault_name=vault_name
    )
    parsed_output_file = _parse_output_file(output_file)
    if parsed_output_file is not None:
        return write_output_file(result, parsed_output_file)
    return result


def list_vaults(
    limit: int = 20,
    offset: int = 0,
) -> dict[str, object]:
    """List all configured vaults with document counts.

    Args:
        limit: Maximum number of results (default: 20, max: 10000).
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
    no_delete: bool | None = None,
    force: bool = False,
) -> dict[str, object]:
    """Ingest markdown files from a vault directory into the database.

    Args:
        vault_name: Name of the vault to ingest into (required).
            Must match a vault configured in the config file.
        path: Optional path to vault directory. Uses vault's container_path
            if not provided.
        no_delete: If True, skip deletion of orphaned documents. If None
            (default, not specified), the system auto-sets True when `path`
            is a subdirectory of `container_path` (incremental ingestion),
            and False for full-vault ingestion. Explicitly passing False with
            an incremental path honors the client's choice (deletion proceeds
            — the client accepts the risk of losing documents outside the
            scanned subdirectory).
        force: If True, re-ingest all documents regardless of checksums.
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

    Notes:
        This tool is idempotent - duplicate calls with the same parameters
        within the same session will return the cached result from the
        first call without re-processing files.

    """
    _msg = f"Tool ingest called with vault: {vault_name}, path: {path}, no_delete: {no_delete}, force: {force}"
    log.info(_msg)

    # Generate deterministic request ID
    request_id = _generate_request_id(
        vault_name, path, no_delete=no_delete, force=force
    )
    _msg = f"Generated request ID: {request_id}"
    log.debug(_msg)

    # Get the request tracker
    tracker = _get_ingest_tracker()

    # Check if this request is already being processed or completed
    cached_result = _check_and_handle_duplicate(
        tracker, request_id, vault_name, path, no_delete=no_delete, force=force
    )
    if cached_result is not None:
        return cached_result
    # Process the ingest request
    try:
        registry = _get_registry()
        params = IngestHandlerParams(
            settings=registry.settings,
            db_manager=registry.db_manager,
            embedding_provider=registry.embedding_provider,
            vault_name=vault_name,
            path_override=path,
            no_delete=no_delete,
            force=force,
        )
        result = _ingest_handler(params)

        asyncio.run(tracker.complete_request(request_id, result))
        _msg = f"Request {request_id} completed successfully"
        log.info(_msg)

        return result

    except IngestLockError as e:
        # Recoverable: do NOT cache — a retry after the running ingest finishes should succeed
        asyncio.run(tracker.clear_request(request_id))
        _msg = f"Request {request_id} skipped due to ingest lock: {e}"
        log.info(_msg)
        return {"success": False, "error": str(e), "total": 0, "skipped": True}

    except ValueError as e:
        return _handle_ingest_value_error(tracker, request_id, vault_name, e)

    except Exception as e:
        asyncio.run(tracker.fail_request(request_id, e))
        _msg = f"Request {request_id} failed: {e}"
        log.exception(_msg)
        raise


def _parse_tag_filters(
    tag_filters: str | dict | TagFilterStrings | None,
) -> TagFilterStrings:
    """Parse tag_filters from str/dict/dataclass to TagFilterStrings."""
    if isinstance(tag_filters, (str, dict)):
        parsed = parse_json_str(tag_filters)
        if isinstance(parsed, dict):
            return TagFilterStrings(**parsed)
        return TagFilterStrings()
    if tag_filters is None:
        return TagFilterStrings()
    return tag_filters


def _parse_date_filters(
    date_filters: str | dict | TaskDateFilterStrings | None,
) -> TaskDateFilterStrings:
    """Parse date_filters from str/dict/dataclass to TaskDateFilterStrings."""
    if isinstance(date_filters, (str, dict)):
        parsed = parse_json_str(date_filters)
        if isinstance(parsed, dict):
            return TaskDateFilterStrings(**parsed)
        return TaskDateFilterStrings()
    if date_filters is None:
        return TaskDateFilterStrings()
    return date_filters


def _parse_inline_filters_str_or_dict(
    inline_filters: str | dict,
) -> list[PropertyFilter] | None:
    """Parse inline_filters from str/dict to list[PropertyFilter].

    Args:
        inline_filters: Filter input as JSON string or dict.

    Returns:
        List of PropertyFilter objects, or None if input is not
        parseable as dict or list.

    """
    parsed = parse_json_str(inline_filters)
    if isinstance(parsed, dict):
        return [PropertyFilter(**parsed)]
    if isinstance(parsed, list):
        return [PropertyFilter(**f) for f in parsed]
    return None


def _parse_inline_filters(
    inline_filters: str | dict | list[PropertyFilter] | None,
) -> list[PropertyFilter] | None:
    """Parse inline_filters from str/dict/list to list[PropertyFilter].

    Args:
        inline_filters: Filter input as JSON string, dict, list of
            PropertyFilter objects, or None.

    Returns:
        List of PropertyFilter objects, or None if input is empty/None.

    Notes:
        Follows the same pattern as _parse_tag_filters() and
        _parse_date_filters(). Uses parse_json_str() for str/dict inputs.

    """
    if inline_filters is None:
        return None
    if isinstance(inline_filters, list):
        return inline_filters
    return _parse_inline_filters_str_or_dict(inline_filters)


def _parse_output_file_str_or_dict(
    output_file: str | dict,
) -> OutputFileConfig | None:
    """Parse output_file from str/dict to OutputFileConfig.

    Args:
        output_file: Filter input as JSON string or dict.

    Returns:
        OutputFileConfig object, or None if input is not parseable as dict.
    """
    parsed = parse_json_str(output_file)
    if isinstance(parsed, dict):
        return OutputFileConfig(**parsed)
    return None


def _parse_output_file(
    output_file: str | dict | OutputFileConfig | None,
) -> OutputFileConfig | None:
    """Parse output_file from str/dict/model to OutputFileConfig or None.

    Args:
        output_file: Output file config as JSON string, dict, OutputFileConfig
            object, or None.

    Returns:
        OutputFileConfig object, or None if input is None/empty.

    Notes:
        Follows the same pattern as _parse_tag_filters() and
        _parse_date_filters(). Uses parse_json_str() for str/dict inputs.

    """
    if output_file is None:
        return None
    if isinstance(output_file, OutputFileConfig):
        return output_file
    return _parse_output_file_str_or_dict(output_file)


def get_tasks(
    status: list[str] | None = None,
    tag_filters: str | dict | TagFilterStrings | None = None,
    date_filters: str | dict | TaskDateFilterStrings | None = None,
    priority: list[str] | None = None,
    inline_filters: str | dict | list[PropertyFilter] | None = None,
    *,
    include_content: bool = True,
    limit: int = 20,
    offset: int = 0,
    output_file: str | dict | OutputFileConfig | None = None,
    vault_name: str | None = None,
) -> dict[str, object]:
    """Query tasks with flexible filtering by status, dates, priority, and tags.

    Args:
        status: List of statuses to filter by. Valid values: "not_completed",
            "completed", "in_progress", "cancelled".
        tag_filters: Tag filter parameters. Can be provided as a TagFilterStrings
            object, a dict, or a JSON string. Dict/JSON string examples:
            {"include_tags": ["work"], "match_mode": "any"}
            '{"include_tags": ["work"], "match_mode": "any"}'
            Tags should NOT include the '#' prefix.
        date_filters: Date filter parameters. Can be provided as a
            TaskDateFilterStrings object, a dict, or a JSON string.
            Dict/JSON string examples:
            {"due_after": "2026-01-01", "due_before": "2026-12-31"}
            '{"due_after": "2026-01-01"}'
        priority: List of priorities to filter by. Valid values: "highest",
            "high", "normal", "low", "lowest".
        inline_filters: Inline field filter parameters. Can be provided as
            a list of PropertyFilter objects, a dict, or a JSON string.
            Dict/JSON string examples:
            {"path": "vendor", "operator": "equals", "value": "Amazon"}
            '[{"path": "vendor", "operator": "equals", "value": "Amazon"}]'
            Inline fields are flat key-value pairs (no dot notation in path).
            Maximum 10 inline filters per query.
        include_content: If True (default), include the raw task line text
            (`raw_text`) in each task response. When False, `raw_text` is an
            empty string, reducing payload size.
        limit: Maximum number of results (default: 20, max: 10000).
        offset: Number of results to skip (default: 0).
        output_file: Optional output file configuration. When provided, the
            full result is written to the specified target (local or S3) and
            a compact summary is returned instead. Can be passed as a dict,
            JSON string, or OutputFileConfig object.
        vault_name: Optional vault name to scope task results. None means all vaults. Non-existent vault names raise ValueError, caught and returned as error dict.

    Returns:
        Dictionary with paginated task list response.

    Raises:
        ValueError: If tag filter validation fails or JSON parsing fails.

    """
    from obsidian_rag.mcp_server.handlers import (
        GetTasksRequest,
        _get_tasks_handler,
    )

    registry = _get_registry()

    parsed_tag_filters = _parse_tag_filters(tag_filters)
    parsed_date_filters = _parse_date_filters(date_filters)
    parsed_inline_filters = _parse_inline_filters(inline_filters)

    request = GetTasksRequest(
        status=status,
        tag_filters=parsed_tag_filters,
        date_filters=parsed_date_filters,
        priority=priority,
        inline_filters=parsed_inline_filters,
        include_content=include_content,
        limit=limit,
        offset=offset,
        vault_name=vault_name,
    )

    result = _get_tasks_handler(
        db_manager=registry.db_manager,
        request=request,
    )
    parsed_output_file = _parse_output_file(output_file)
    if parsed_output_file is not None:
        return write_output_file(result, parsed_output_file)
    return result


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
    ).model_dump(mode="json")

    return JSONResponse(health_data)


def _get_session_manager() -> SessionManager | None:
    """Get the global session manager instance.

    Returns:
        SessionManager instance or None if not initialized.

    """
    return _session_manager


# Server Creation and Registration


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

    # Register document retrieval tools
    mcp.tool()(get_document)
    mcp.tool()(list_documents)

    # Register vault tools
    mcp.tool()(list_vaults)
    mcp.tool()(get_vault)
    mcp.tool()(update_vault)
    mcp.tool()(delete_vault)

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

    # host_origin_protection is disabled because:
    # 1. The server is already protected by Bearer token auth (StaticTokenVerifier),
    #    so the DNS-rebinding threat model (unauthenticated localhost browser
    #    exploitation) does not apply.
    # 2. FastMCP's HostOriginGuardMiddleware defaults to on with a localhost-only
    #    allowlist (127.0.0.1, localhost, ::1), which rejects external clients
    #    with HTTP 421 "Misdirected Request" when the server is bound to 0.0.0.0
    #    or deployed behind a reverse proxy.
    app = mcp.http_app(
        path="/",
        middleware=middleware,
        stateless_http=settings.mcp.stateless_http,
        host_origin_protection=False,
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
