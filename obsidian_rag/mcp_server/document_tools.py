"""MCP tool wrappers for document retrieval operations."""

import logging
from typing import cast

from obsidian_rag.mcp_server.tool_definitions import (
    _get_registry,
    get_document_tool,
    list_documents_tool,
)

log = logging.getLogger(__name__)


def get_document(
    *,
    vault_name: str | None = None,
    file_path: str | None = None,
    document_id: str | None = None,
) -> dict[str, object]:
    """Get a single document by exact file_path within a vault, or by UUID document_id.

    Retrieves full document content, metadata, tags, and obsidian_uri.
    Either vault_name+file_path or document_id must be provided.

    Args:
        vault_name: Vault name (required when using file_path).
        file_path: Relative file path from vault root.
        document_id: Document UUID string (use document_id, not id,
            to avoid shadowing the built-in id function).

    Returns:
        Document response as dictionary on success, or error dict on failure:
        - Success: {"id": ..., "vault_name": ..., "content": ...}
        - Error: {"success": False, "error": "..."}

    Notes:
        This wrapper accesses dependencies through _get_registry() and delegates
        to get_document_tool in tool_definitions.py.

    """
    _msg = "Tool wrapper get_document starting"
    log.debug(_msg)

    registry = _get_registry()
    result = get_document_tool(
        registry.db_manager,
        vault_name=vault_name,
        file_path=file_path,
        document_id=document_id,
    )

    _msg = "Tool wrapper get_document returning"
    log.debug(_msg)
    return cast("dict[str, object]", result)


def list_documents(
    file_name: str | None = None,
    vault_name: str | None = None,
    limit: int = 20,
    offset: int = 0,
) -> dict[str, object]:
    """List documents by file_name with optional vault scope.

    Returns all documents matching the exact file_name. When vault_name
    is provided, results are scoped to that vault only.

    Args:
        file_name: Document file name to search for (required).
        vault_name: Filter by specific vault name (optional).
        limit: Maximum number of results (default: 20, max: 100).
        offset: Number of results to skip (default: 0).

    Returns:
        Document list response with pagination, or error dict if no
        file_name provided.

    Notes:
        This wrapper accesses dependencies through _get_registry() and delegates
        to list_documents_tool in tool_definitions.py.

    """
    _msg = "Tool wrapper list_documents starting"
    log.debug(_msg)

    registry = _get_registry()
    result = list_documents_tool(
        registry.db_manager,
        file_name=file_name,
        vault_name=vault_name,
        limit=limit,
        offset=offset,
    )

    _msg = "Tool wrapper list_documents returning"
    log.debug(_msg)
    return cast("dict[str, object]", result)
