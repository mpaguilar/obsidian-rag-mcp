"""Entry point for running the MCP server."""

import logging
import sys
from typing import Any

from obsidian_rag.config import Settings, get_settings
from obsidian_rag.mcp_server.server import create_http_app

# Optional dependency - will be None if not installed
uvicorn: Any = None
try:
    import uvicorn as _uvicorn

    uvicorn = _uvicorn
except ImportError:
    pass

log = logging.getLogger(__name__)


def _setup_logging() -> None:
    """Configure logging for the MCP server."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Set library logging to INFO for better log analysis
    logging.getLogger("httpx").setLevel(logging.INFO)
    logging.getLogger("httpcore").setLevel(logging.INFO)
    logging.getLogger("asyncio").setLevel(logging.INFO)
    logging.getLogger("openai").setLevel(logging.INFO)


def _log_config(settings: Settings) -> None:
    """Log configuration details for diagnostics."""
    _msg = f"MCP configuration: host={settings.mcp.host!r}, port={settings.mcp.port}"
    log.info(_msg)

    # Log database URL (mask credentials)
    db_url = settings.database.url
    if "@" in db_url:
        _msg = f"Database URL configured: {db_url.split('@')[0]}@***"
    else:
        _msg = "Database URL configured (custom format)"
    log.info(_msg)


def _run_server(app: object, settings: Settings) -> None:
    """Run uvicorn server with error handling.

    Args:
        app: The ASGI application to run.
        settings: Server configuration settings.

    Raises:
        SystemExit: If uvicorn exits with error.
        OSError: If bind fails or network error occurs.

    """
    _msg = f"Starting MCP server on {settings.mcp.host}:{settings.mcp.port}"
    log.info(_msg)

    try:
        uvicorn.run(
            app,
            host=settings.mcp.host,
            port=settings.mcp.port,
        )
    except SystemExit as e:
        # SystemExit from uvicorn - could be bind failure or other startup error
        _msg = (
            f"MCP server exited with code {e.code}. "
            f"Common causes: port already in use, permission denied, "
            f"or invalid host '{settings.mcp.host}'"
        )
        log.error(_msg)
        raise
    except OSError as e:
        # OSError typically means bind failure or network issue
        _msg = (
            f"Failed to start MCP server: {e}. "
            f"Check if host '{settings.mcp.host}' is valid and "
            f"port {settings.mcp.port} is available"
        )
        log.exception(_msg)
        raise


def main() -> None:
    """Run the MCP server."""
    _setup_logging()

    _msg = "Starting Obsidian RAG MCP server"
    log.info(_msg)

    try:
        settings = get_settings()
    except Exception as e:
        _msg = f"Failed to load settings: {e}"
        log.exception(_msg)
        sys.exit(1)

    # Validate token is set
    if not settings.mcp.token:
        _msg = (
            "MCP token not configured. Set OBSIDIAN_RAG_MCP_TOKEN "
            "environment variable or mcp.token in config file."
        )
        log.error(_msg)
        sys.exit(1)

    _log_config(settings)

    # Create HTTP app
    try:
        app = create_http_app(settings)
    except Exception as e:
        _msg = f"Failed to create MCP server: {e}"
        log.exception(_msg)
        sys.exit(1)

    # Check uvicorn is available
    if uvicorn is None:
        _msg = "uvicorn is required to run the MCP server. Install with: pip install uvicorn"
        log.error(_msg)
        sys.exit(1)

    _run_server(app, settings)


if __name__ == "__main__":
    main()
