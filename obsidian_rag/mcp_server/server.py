"""FastMCP server initialization and configuration."""

import logging
import os
from datetime import date
from pathlib import Path
from typing import TypedDict

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
    get_all_tags as get_all_tags_tool,
    get_documents_by_tag as get_documents_by_tag_tool,
    query_documents as query_documents_tool,
)
from obsidian_rag.mcp_server.tools.tasks import (
    get_completed_tasks as get_completed_tasks_tool,
    get_incomplete_tasks as get_incomplete_tasks_tool,
    get_tasks_by_tag as get_tasks_by_tag_tool,
    get_tasks_due_this_week as get_tasks_due_this_week_tool,
)


class DocumentTagParams(TypedDict, total=False):
    """Parameters for get_documents_by_tag tool."""

    tag: str | None
    vault_root: str | None
    include_untagged: bool
    limit: int
    offset: int


from obsidian_rag.parsing.scanner import scan_markdown_files

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


def _get_documents_by_tag_handler(
    db_manager: DatabaseManager,
    params: DocumentTagParams,
) -> dict[str, object]:
    """Handle get_documents_by_tag tool call.

    Args:
        db_manager: Database manager for sessions.
        params: Dictionary with tag, vault_root, include_untagged, limit, offset.

    Returns:
        Document list response as dictionary.

    """
    with db_manager.get_session() as session:
        result = get_documents_by_tag_tool(
            session=session,
            tag=params.get("tag"),
            vault_root=params.get("vault_root"),
            include_untagged=params.get("include_untagged", False),
            limit=params.get("limit", 20),
            offset=params.get("offset", 0),
        )
        return result.model_dump()


def _get_all_tags_handler(
    db_manager: DatabaseManager,
    pattern: str | None,
    limit: int,
    offset: int,
) -> dict[str, object]:
    """Handle get_all_tags tool call."""
    with db_manager.get_session() as session:
        result = get_all_tags_tool(
            session=session,
            pattern=pattern,
            limit=limit,
            offset=offset,
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

    @mcp.tool()
    def get_documents_by_tag(
        tag: str | None = None,
        vault_root: str | None = None,
        include_untagged: bool = False,
        limit: int = 20,
        offset: int = 0,
    ) -> dict[str, object]:
        """Query documents filtered by tag.

        Args:
            tag: Tag to filter by (optional, case-insensitive substring match).
            vault_root: Filter by specific vault root path (optional).
            include_untagged: Include documents with no tags when True.
            limit: Maximum number of results (default: 20, max: 100).
            offset: Number of results to skip (default: 0).

        Returns:
            Document list response with pagination and relative paths.

        """
        _msg = f"Tool get_documents_by_tag called with tag: {tag}"
        log.info(_msg)
        params: DocumentTagParams = {
            "tag": tag,
            "vault_root": vault_root,
            "include_untagged": include_untagged,
            "limit": limit,
            "offset": offset,
        }
        return _get_documents_by_tag_handler(db_manager, params)

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


def _validate_ingest_path(ingest_path: str) -> Path:
    """Validate the ingest path.

    Args:
        ingest_path: Path to validate.

    Returns:
        Validated Path object.

    Raises:
        ValueError: If path is invalid or inaccessible.

    """
    # Validate path is absolute
    if ".." in ingest_path:
        _msg = "Path cannot contain parent directory references (..)"
        log.error(_msg)
        raise ValueError(_msg)

    path = Path(ingest_path)

    # Check if path exists
    if not path.exists():
        _msg = f"Data directory '{ingest_path}' does not exist. Please ensure the volume is mounted."
        log.error(_msg)
        raise ValueError(_msg)

    # Check if path is a directory
    if not path.is_dir():
        _msg = f"Path '{ingest_path}' exists but is not a directory"
        log.error(_msg)
        raise ValueError(_msg)

    return path


def _scan_files_for_ingest(path: Path) -> list:
    """Scan directory for markdown files.

    Args:
        path: Directory path to scan.

    Returns:
        List of files found.

    Raises:
        ValueError: If scanning fails.

    """
    try:
        return scan_markdown_files(path)
    except PermissionError as e:
        _msg = f"Permission denied accessing data directory: {e}"
        log.exception(_msg)
        raise ValueError(_msg) from e
    except Exception as e:
        _msg = f"Error scanning data directory: {e}"
        log.exception(_msg)
        raise ValueError(_msg) from e


def _ingest_handler(
    settings: Settings,
    path_override: str | None,
) -> dict[str, object]:
    """Handle ingest tool call.

    Args:
        settings: Application settings.
        path_override: Optional path override from tool call.

    Returns:
        Dictionary with ingestion statistics.

    Raises:
        ValueError: If path is invalid or inaccessible.

    """
    _msg = "ingest handler starting"
    log.debug(_msg)

    # Determine the path to use
    ingest_path = path_override if path_override else settings.mcp.ingest_path

    # Validate path
    path = _validate_ingest_path(ingest_path)

    # Scan for markdown files
    files = _scan_files_for_ingest(path)
    total = len(files)

    if total == 0:
        return {
            "total": 0,
            "new": 0,
            "updated": 0,
            "unchanged": 0,
            "errors": 0,
            "message": "No markdown files found in data directory",
        }

    # Return statistics (actual ingestion would require more refactoring)
    _msg = f"ingest handler found {total} files"
    log.info(_msg)

    return {
        "total": total,
        "new": 0,
        "updated": 0,
        "unchanged": total,
        "errors": 0,
        "message": f"Found {total} markdown files. Use CLI ingest for processing.",
    }


def _register_ingest_tools(
    mcp: FastMCP,
    settings: Settings,
) -> None:
    """Register ingest-related tools.

    Args:
        mcp: FastMCP server instance.
        settings: Application settings.

    """

    @mcp.tool()
    def ingest(
        path: str | None = None,
    ) -> dict[str, object]:
        """Check data directory and return ingestion statistics.

        Args:
            path: Optional path override (uses config default if not provided).

        Returns:
            Dictionary with statistics:
            - total: Total files found
            - new: New files (0 - use CLI for processing)
            - updated: Updated files (0 - use CLI for processing)
            - unchanged: Unchanged files
            - errors: Number of errors
            - message: Status message

        Raises:
            ValueError: If data directory doesn't exist or is inaccessible.

        """
        _msg = f"Tool ingest called with path: {path}"
        log.info(_msg)
        return _ingest_handler(settings, path)


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

    # Configure static bearer token authentication
    # StaticTokenVerifier accepts a dict mapping tokens to their claims
    token_verifier = StaticTokenVerifier(
        tokens={
            settings.mcp.token: {
                "client_id": "obsidian-rag-client",
                "sub": "user",
            }
        }
    )

    mcp = FastMCP("Obsidian RAG Server", auth=token_verifier)
    db_manager = DatabaseManager(settings.database.url)
    embedding_provider = _create_embedding_provider(settings)

    _register_task_tools(mcp, db_manager)
    _register_document_tools(mcp, db_manager, embedding_provider)
    _register_ingest_tools(mcp, settings)

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
        path="/",
        middleware=middleware,
        stateless_http=settings.mcp.stateless_http,
    )

    _msg = "HTTP app created successfully"
    log.info(_msg)

    return app
