"""Entry point for running the MCP server."""

import logging
import sys
from typing import Any

from obsidian_rag.config import get_settings
from obsidian_rag.mcp_server.server import create_http_app

# Optional dependency - will be None if not installed
uvicorn: Any = None
try:
    import uvicorn as _uvicorn

    uvicorn = _uvicorn
except ImportError:
    pass


def main() -> None:
    """Run the MCP server."""
    # Setup basic logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    log = logging.getLogger(__name__)
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

    # Run server
    _msg = f"Starting MCP server on {settings.mcp.host}:{settings.mcp.port}"
    log.info(_msg)

    uvicorn.run(
        app,
        host=settings.mcp.host,
        port=settings.mcp.port,
    )


if __name__ == "__main__":
    main()
