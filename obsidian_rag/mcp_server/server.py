"""FastMCP server initialization and configuration."""

import logging
import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Literal, TypedDict, cast

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
from obsidian_rag.llm.base import EmbeddingProvider
from obsidian_rag.llm.providers import ProviderFactory
from obsidian_rag.mcp_server.models import (
    HealthResponse,
    PropertyFilter,
    TagFilter,
)
from obsidian_rag.mcp_server.tools.documents import (
    get_all_tags as get_all_tags_tool,
)
from obsidian_rag.mcp_server.tools.documents import (
    get_documents_by_property as get_documents_by_property_tool,
)
from obsidian_rag.mcp_server.tools.documents import (
    get_documents_by_tag as get_documents_by_tag_tool,
)
from obsidian_rag.mcp_server.tools.documents import (
    query_documents as query_documents_tool,
)
from obsidian_rag.mcp_server.tools.documents_params import (
    PaginationParams,
    PropertyFilterParams,
)
from obsidian_rag.mcp_server.tools.tasks import (
    get_completed_tasks as get_completed_tasks_tool,
)
from obsidian_rag.mcp_server.tools.tasks import (
    get_incomplete_tasks as get_incomplete_tasks_tool,
)
from obsidian_rag.mcp_server.tools.tasks import (
    get_tasks_by_tag as get_tasks_by_tag_tool,
)
from obsidian_rag.mcp_server.tools.tasks import (
    get_tasks_due_this_week as get_tasks_due_this_week_tool,
)
from obsidian_rag.services.ingestion import IngestionService


class DocumentTagParams(TypedDict, total=False):
    """Parameters for get_documents_by_tag tool."""

    include_tags: list[str]
    exclude_tags: list[str]
    match_mode: str
    vault_root: str | None
    limit: int
    offset: int


@dataclass
class QueryFilterParams:
    """Parameters for query_documents filter configuration."""

    include_properties: list[dict] | None
    exclude_properties: list[dict] | None
    include_tags: list[str] | None
    exclude_tags: list[str] | None


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
    _msg = "_create_embedding_provider starting"
    log.debug(_msg)

    embedding_config = settings.endpoints.get("embedding")
    if not embedding_config:
        _msg = "_create_embedding_provider returning"
        log.debug(_msg)
        return None

    provider = None
    try:
        provider = ProviderFactory.create_embedding_provider(
            provider_name=embedding_config.provider,
            api_key=embedding_config.api_key,
            model=embedding_config.model,
            base_url=embedding_config.base_url,
        )
    except (ValueError, ImportError, RuntimeError) as e:
        _msg = f"Failed to create embedding provider: {e}"
        log.warning(_msg)

    _msg = "_create_embedding_provider returning"
    log.debug(_msg)
    return provider


def _get_incomplete_tasks_handler(
    db_manager: DatabaseManager,
    limit: int,
    offset: int,
    *,
    include_cancelled: bool,
) -> dict[str, object]:
    """Handle get_incomplete_tasks tool call."""
    _msg = "_get_incomplete_tasks_handler starting"
    log.debug(_msg)
    with db_manager.get_session() as session:
        result = get_incomplete_tasks_tool(
            session=session,
            limit=limit,
            offset=offset,
            include_cancelled=include_cancelled,
        )
        _msg = "_get_incomplete_tasks_handler returning"
        log.debug(_msg)
        return result.model_dump()


def _get_tasks_due_this_week_handler(
    db_manager: DatabaseManager,
    limit: int,
    offset: int,
    *,
    include_completed: bool,
) -> dict[str, object]:
    """Handle get_tasks_due_this_week tool call."""
    _msg = "_get_tasks_due_this_week_handler starting"
    log.debug(_msg)
    with db_manager.get_session() as session:
        result = get_tasks_due_this_week_tool(
            session=session,
            limit=limit,
            offset=offset,
            include_completed=include_completed,
        )
        _msg = "_get_tasks_due_this_week_handler returning"
        log.debug(_msg)
        return result.model_dump()


def _get_tasks_by_tag_handler(
    db_manager: DatabaseManager,
    tag: str,
    limit: int,
    offset: int,
) -> dict[str, object]:
    """Handle get_tasks_by_tag tool call."""
    _msg = "_get_tasks_by_tag_handler starting"
    log.debug(_msg)
    with db_manager.get_session() as session:
        result = get_tasks_by_tag_tool(
            session=session,
            tag=tag,
            limit=limit,
            offset=offset,
        )
        _msg = "_get_tasks_by_tag_handler returning"
        log.debug(_msg)
        return result.model_dump()


def _get_completed_tasks_handler(
    db_manager: DatabaseManager,
    limit: int,
    offset: int,
    completed_since: str | None,
) -> dict[str, object]:
    """Handle get_completed_tasks tool call."""
    _msg = "_get_completed_tasks_handler starting"
    log.debug(_msg)
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
        _msg = "_get_completed_tasks_handler returning"
        log.debug(_msg)
        return result.model_dump()


def _get_documents_by_tag_handler(
    db_manager: DatabaseManager,
    params: DocumentTagParams,
) -> dict[str, object]:
    """Handle get_documents_by_tag tool call.

    Args:
        db_manager: Database manager for sessions.
        params: Dictionary with include_tags, exclude_tags, match_mode, vault_root, limit, offset.

    Returns:
        Document list response as dictionary.

    """
    _msg = "_get_documents_by_tag_handler starting"
    log.debug(_msg)

    # Create TagFilter from params with proper type casting
    match_mode_value = params.get("match_mode", "all")
    match_mode_casted = cast("Literal['all', 'any']", match_mode_value)
    tag_filter = TagFilter(
        include_tags=params.get("include_tags", []),
        exclude_tags=params.get("exclude_tags", []),
        match_mode=match_mode_casted,
    )

    with db_manager.get_session() as session:
        result = get_documents_by_tag_tool(
            session=session,
            tag_filter=tag_filter,
            vault_root=params.get("vault_root"),
            limit=params.get("limit", 20),
            offset=params.get("offset", 0),
        )
        _msg = "_get_documents_by_tag_handler returning"
        log.debug(_msg)
        return result.model_dump()


def _get_all_tags_handler(
    db_manager: DatabaseManager,
    pattern: str | None,
    limit: int,
    offset: int,
) -> dict[str, object]:
    """Handle get_all_tags tool call."""
    _msg = "_get_all_tags_handler starting"
    log.debug(_msg)
    with db_manager.get_session() as session:
        result = get_all_tags_tool(
            session=session,
            pattern=pattern,
            limit=limit,
            offset=offset,
        )
        _msg = "_get_all_tags_handler returning"
        log.debug(_msg)
        return result.model_dump()


def _convert_property_filters(
    properties: list[dict] | None,
) -> list[PropertyFilter] | None:
    """Convert list of dict property filters to PropertyFilter objects.

    Args:
        properties: List of property filter dicts.

    Returns:
        List of PropertyFilter objects or None.

    """
    _msg = "_convert_property_filters starting"
    log.debug(_msg)
    if not properties:
        _msg = "_convert_property_filters returning"
        log.debug(_msg)
        return None
    result = [PropertyFilter(**prop) for prop in properties]
    _msg = "_convert_property_filters returning"
    log.debug(_msg)
    return result


def _create_tag_filter(
    include_tags: list[str] | None,
    exclude_tags: list[str] | None,
    match_mode: str,
) -> TagFilter | None:
    """Build TagFilter from parameters.

    Args:
        include_tags: List of tags documents must have.
        exclude_tags: List of tags documents must NOT have.
        match_mode: Whether to match "all" or "any" of include_tags.

    Returns:
        TagFilter or None if no tags specified.

    """
    _msg = "_create_tag_filter starting"
    log.debug(_msg)
    if not include_tags and not exclude_tags:
        _msg = "_create_tag_filter returning"
        log.debug(_msg)
        return None

    valid_match_mode = match_mode if match_mode in ("all", "any") else "all"
    result = TagFilter(
        include_tags=include_tags or [],
        exclude_tags=exclude_tags or [],
        match_mode=cast("Literal['all', 'any']", valid_match_mode),
    )
    _msg = "_create_tag_filter returning"
    log.debug(_msg)
    return result


def _register_task_tools(
    mcp: FastMCP,
    db_manager: DatabaseManager,
) -> None:
    """Register task-related tools."""

    @mcp.tool()
    def get_incomplete_tasks(
        limit: int = 20,
        offset: int = 0,
        *,
        include_cancelled: bool = False,
    ) -> dict[str, object]:
        """Query tasks that are not completed."""
        _msg = "Tool get_incomplete_tasks called"
        log.info(_msg)
        return _get_incomplete_tasks_handler(
            db_manager,
            limit,
            offset,
            include_cancelled=include_cancelled,
        )

    @mcp.tool()
    def get_tasks_due_this_week(
        limit: int = 20,
        offset: int = 0,
        *,
        include_completed: bool = True,
    ) -> dict[str, object]:
        """Query tasks due within the next 7 days."""
        _msg = "Tool get_tasks_due_this_week called"
        log.info(_msg)
        return _get_tasks_due_this_week_handler(
            db_manager,
            limit,
            offset,
            include_completed=include_completed,
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


def _register_query_documents_tool(
    mcp: FastMCP,
    db_manager: DatabaseManager,
    embedding_provider: EmbeddingProvider | None,
) -> None:
    """Register query_documents tool."""

    @mcp.tool()
    def query_documents(
        query: str,
        filters: QueryFilterParams | None = None,
        tag_match_mode: str = "all",
        limit: int = 20,
        offset: int = 0,
    ) -> dict[str, object]:
        """Semantic search over document content with optional filters.

        Args:
            query: Search query text.
            filters: QueryFilterParams with include_properties, exclude_properties,
                include_tags, exclude_tags.
            tag_match_mode: Whether document must have ALL or ANY of include_tags.
            limit: Maximum number of results (default: 20, max: 100).
            offset: Number of results to skip (default: 0).

        Returns:
            Document list response with pagination and similarity scores.

        Raises:
            RuntimeError: If embedding provider is not available.
            ValueError: If filter validation fails.

        """
        _msg = f"Tool query_documents called with query: {query[:50]}..."
        log.info(_msg)

        if not embedding_provider:
            _msg = "Embedding provider not configured"
            log.error(_msg)
            raise RuntimeError(_msg)

        query_embedding = embedding_provider.generate_embedding(query)

        # Ensure filters is a QueryFilterParams dataclass
        query_filters = filters or QueryFilterParams(
            include_properties=None,
            exclude_properties=None,
            include_tags=None,
            exclude_tags=None,
        )

        prop_filters_include = _convert_property_filters(
            query_filters.include_properties,
        )
        prop_filters_exclude = _convert_property_filters(
            query_filters.exclude_properties,
        )
        tag_filter = _create_tag_filter(
            query_filters.include_tags,
            query_filters.exclude_tags,
            tag_match_mode,
        )

        # Bundle property filters into PropertyFilterParams
        property_filter_params = PropertyFilterParams(
            include_filters=prop_filters_include,
            exclude_filters=prop_filters_exclude,
        )
        pagination = PaginationParams(limit=limit, offset=offset)

        with db_manager.get_session() as session:
            result = query_documents_tool(
                session=session,
                query_embedding=query_embedding,
                filter_params=property_filter_params,
                tag_filter=tag_filter,
                pagination=pagination,
            )
            return result.model_dump()


def _register_get_documents_by_tag_tool(
    mcp: FastMCP,
    db_manager: DatabaseManager,
) -> None:
    """Register get_documents_by_tag tool."""

    @mcp.tool()
    def get_documents_by_tag(
        filters: QueryFilterParams | None = None,
        tag_match_mode: str = "all",
        vault_root: str | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> dict[str, object]:
        """Query documents filtered by tags with include/exclude semantics.

        Args:
            filters: QueryFilterParams with include_tags, exclude_tags.
            tag_match_mode: Whether document must have ALL or ANY of include_tags.
            vault_root: Filter by specific vault root path (optional).
            limit: Maximum number of results (default: 20, max: 100).
            offset: Number of results to skip (default: 0).

        Returns:
            Document list response with pagination and relative paths.

        """
        _msg = "Tool get_documents_by_tag called"
        log.info(_msg)

        filters = filters or QueryFilterParams(
            include_properties=None,
            exclude_properties=None,
            include_tags=None,
            exclude_tags=None,
        )

        valid_match_mode = tag_match_mode if tag_match_mode in ("all", "any") else "all"
        params: DocumentTagParams = {
            "include_tags": filters.include_tags or [],
            "exclude_tags": filters.exclude_tags or [],
            "match_mode": valid_match_mode,
            "vault_root": vault_root,
            "limit": limit,
            "offset": offset,
        }
        return _get_documents_by_tag_handler(db_manager, params)


def _register_get_documents_by_property_tool(
    mcp: FastMCP,
    db_manager: DatabaseManager,
) -> None:
    """Register get_documents_by_property tool."""

    @mcp.tool()
    def get_documents_by_property(
        filters: QueryFilterParams | None = None,
        tag_match_mode: str = "all",
        vault_root: str | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> dict[str, object]:
        """Query documents filtered by frontmatter properties.

        Args:
            filters: QueryFilterParams with include_properties, exclude_properties,
                include_tags, exclude_tags.
            tag_match_mode: Whether document must have ALL or ANY of include_tags.
            vault_root: Filter by specific vault root path (optional).
            limit: Maximum number of results (default: 20, max: 100).
            offset: Number of results to skip (default: 0).

        Returns:
            Document list response with pagination and relative paths.

        Raises:
            ValueError: If property filter validation fails.

        """
        _msg = "Tool get_documents_by_property called"
        log.info(_msg)

        filters = filters or QueryFilterParams(
            include_properties=None,
            exclude_properties=None,
            include_tags=None,
            exclude_tags=None,
        )

        prop_filters_include = _convert_property_filters(filters.include_properties)
        prop_filters_exclude = _convert_property_filters(filters.exclude_properties)
        tag_filter = _create_tag_filter(
            filters.include_tags,
            filters.exclude_tags,
            tag_match_mode,
        )

        # Bundle property filters into PropertyFilterParams
        property_filter_params = PropertyFilterParams(
            include_filters=prop_filters_include,
            exclude_filters=prop_filters_exclude,
        )
        pagination = PaginationParams(limit=limit, offset=offset)

        with db_manager.get_session() as session:
            result = get_documents_by_property_tool(
                session=session,
                property_filters=property_filter_params,
                tag_filter=tag_filter,
                vault_root=vault_root,
                pagination=pagination,
            )
            return result.model_dump()


def _register_get_all_tags_tool(
    mcp: FastMCP,
    db_manager: DatabaseManager,
) -> None:
    """Register get_all_tags tool."""

    @mcp.tool()
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
            Tag list response with pagination.

        """
        _msg = "Tool get_all_tags called"
        log.info(_msg)
        return _get_all_tags_handler(db_manager, pattern, limit, offset)


def _register_document_tools(
    mcp: FastMCP,
    db_manager: DatabaseManager,
    embedding_provider: EmbeddingProvider | None,
) -> None:
    """Register document-related tools."""
    _msg = "_register_document_tools starting"
    log.debug(_msg)
    _register_query_documents_tool(mcp, db_manager, embedding_provider)
    _register_get_documents_by_tag_tool(mcp, db_manager)
    _register_get_documents_by_property_tool(mcp, db_manager)
    _register_get_all_tags_tool(mcp, db_manager)
    _msg = "_register_document_tools returning"
    log.debug(_msg)


def _validate_ingest_path(ingest_path: str) -> Path:
    """Validate the ingest path."""
    _msg = "_validate_ingest_path starting"
    log.debug(_msg)

    if ".." in ingest_path:
        _msg = "Path cannot contain parent directory references (..)"
        log.error(_msg)
        raise ValueError(_msg)

    path = Path(ingest_path)

    if not path.exists():
        _msg = f"Data directory '{ingest_path}' does not exist. Please ensure the volume is mounted."
        log.error(_msg)
        raise ValueError(_msg)

    if not path.is_dir():
        _msg = f"Path '{ingest_path}' exists but is not a directory"
        log.error(_msg)
        raise ValueError(_msg)

    _msg = "_validate_ingest_path returning"
    log.debug(_msg)
    return path


def _ingest_handler(
    settings: Settings,
    db_manager: DatabaseManager,
    embedding_provider: EmbeddingProvider | None,
    path_override: str | None,
) -> dict[str, object]:
    """Handle ingest tool call."""
    _msg = "ingest handler starting"
    log.debug(_msg)

    ingest_path = path_override if path_override else settings.mcp.ingest_path
    path = _validate_ingest_path(ingest_path)

    ingestion_service = IngestionService(
        db_manager=db_manager,
        embedding_provider=embedding_provider,
        settings=settings,
    )

    result = ingestion_service.ingest_vault(
        vault_path=path,
        dry_run=False,
    )

    _msg = f"ingest handler completed: {result.message}"
    log.info(_msg)

    return result.to_dict()


def _register_ingest_tools(
    mcp: FastMCP,
    settings: Settings,
    db_manager: DatabaseManager,
    embedding_provider: EmbeddingProvider | None,
) -> None:
    """Register ingest-related tools."""

    @mcp.tool()
    def ingest(
        path: str | None = None,
    ) -> dict[str, object]:
        """Ingest markdown files from the data directory into the database."""
        _msg = f"Tool ingest called with path: {path}"
        log.info(_msg)
        return _ingest_handler(settings, db_manager, embedding_provider, path)


def _register_health_check(
    mcp: FastMCP,
    db_manager: DatabaseManager,
) -> None:
    """Register health check endpoint."""

    @mcp.custom_route("/health", methods=["GET"])
    async def health_check(_request: Request) -> JSONResponse:
        """Health check endpoint."""
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

        health_data = HealthResponse(
            status="healthy",
            version=version,
            database=db_status,
        ).model_dump()

        return JSONResponse(health_data)


def create_mcp_server(settings: Settings) -> FastMCP:
    """Create and configure the FastMCP server."""
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

    mcp = FastMCP("Obsidian RAG Server", auth=token_verifier)
    db_manager = DatabaseManager(settings.database.url)
    embedding_provider = _create_embedding_provider(settings)

    _register_task_tools(mcp, db_manager)
    _register_document_tools(mcp, db_manager, embedding_provider)
    _register_ingest_tools(mcp, settings, db_manager, embedding_provider)

    if settings.mcp.enable_health_check:
        _register_health_check(mcp, db_manager)

    _msg = "MCP server created successfully"
    log.info(_msg)

    _msg = "create_mcp_server returning"
    log.debug(_msg)
    return mcp


def create_http_app(settings: Settings) -> Starlette:
    """Create HTTP ASGI app for the MCP server."""
    _msg = "Creating HTTP app"
    log.info(_msg)

    mcp = create_mcp_server(settings)

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
        path="/",
        middleware=middleware,
        stateless_http=settings.mcp.stateless_http,
    )

    _msg = "HTTP app created successfully"
    log.info(_msg)

    return app
