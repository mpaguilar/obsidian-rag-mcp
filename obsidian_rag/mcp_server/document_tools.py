"""MCP tool wrappers for document retrieval operations."""

import logging
from typing import cast

from obsidian_rag.mcp_server.handlers import parse_json_str
from obsidian_rag.mcp_server.models import OutputFileConfig
from obsidian_rag.mcp_server.output_file import write_output_file
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
    include_content: bool = True,
    output_file: str | dict | OutputFileConfig | None = None,
) -> dict[str, object]:
    """Get a single document by exact file_path within a vault, or by UUID document_id.

    Retrieves full document content, metadata, tags, and obsidian_uri.
    Either vault_name+file_path or document_id must be provided.

    Args:
        vault_name: Vault name (required when using file_path).
        file_path: Relative file path from vault root.
        document_id: Document UUID string (use document_id, not id,
            to avoid shadowing the built-in id function).
        include_content: Whether to include document content in the response.
        output_file: Optional output file configuration. When provided, the
            full result is written to the specified file and a compact summary
            is returned instead.

            The S3 output_file config now accepts an optional `region` field
            (SigV4 signing region override). Region resolution precedence:
            per-call `region` -> OBSIDIAN_RAG_MCP_OUTPUT_FILE_S3_REGION env
            var -> GetBucketLocation probe (non-AWS endpoints only) ->
            URL-derived region -> us-east-1 fallback (with WARNING log). Set
            `region` explicitly for Garage/MinIO/Ceph endpoints whose
            configured region is not derivable from the hostname.

    Returns:
        Document response as dictionary on success, or error dict on failure:
        - Success: {"id": ..., "vault_name": ..., "content": ...}
        - Error: {"success": False, "error": "..."}
        - With output_file: {"output_file": {...}} summary

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
        include_content=include_content,
    )

    parsed_output_file = _parse_output_file_from_wrapper(output_file)
    if parsed_output_file is not None:
        return write_output_file(
            result,
            parsed_output_file,
            app_default_region=registry.settings.mcp.output_file_s3_region,
        )

    _msg = "Tool wrapper get_document returning"
    log.debug(_msg)
    return cast("dict[str, object]", result)


def list_documents(
    file_name: str | None = None,
    vault_name: str | None = None,
    limit: int = 20,
    offset: int = 0,
    *,
    output_file: str | dict | OutputFileConfig | None = None,
) -> dict[str, object]:
    """List documents by file_name with optional vault scope.

    Returns all documents matching the exact file_name. When vault_name
    is provided, results are scoped to that vault only.

    Args:
        file_name: Document file name to search for (required).
        vault_name: Filter by specific vault name (optional).
        limit: Maximum number of results (default: 20, max: 10000).
        offset: Number of results to skip (default: 0).
        output_file: Optional output file configuration. When provided, the
            full result is written to the specified file and a compact summary
            is returned instead.

            The S3 output_file config now accepts an optional `region` field
            (SigV4 signing region override). Region resolution precedence:
            per-call `region` -> OBSIDIAN_RAG_MCP_OUTPUT_FILE_S3_REGION env
            var -> GetBucketLocation probe (non-AWS endpoints only) ->
            URL-derived region -> us-east-1 fallback (with WARNING log). Set
            `region` explicitly for Garage/MinIO/Ceph endpoints whose
            configured region is not derivable from the hostname.

    Returns:
        Document list response with pagination, or error dict if no
        file_name provided.
        - With output_file: {"output_file": {...}} summary

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

    parsed_output_file = _parse_output_file_from_wrapper(output_file)
    if parsed_output_file is not None:
        return write_output_file(
            result,
            parsed_output_file,
            app_default_region=registry.settings.mcp.output_file_s3_region,
        )

    _msg = "Tool wrapper list_documents returning"
    log.debug(_msg)
    return cast("dict[str, object]", result)


def _parse_output_file_from_wrapper(
    output_file: str | dict | OutputFileConfig | None,
) -> OutputFileConfig | None:
    """Parse output_file for document_tools wrappers.

    Same logic as _parse_output_file in server.py but standalone
    to avoid circular imports. Uses parse_json_str from handlers.

    Args:
        output_file: Output file config as JSON string, dict, OutputFileConfig
            object, or None.

    Returns:
        OutputFileConfig object, or None if input is None/empty.
    """
    if output_file is None:
        return None
    if isinstance(output_file, OutputFileConfig):
        return output_file
    parsed = parse_json_str(output_file)
    if isinstance(parsed, dict):
        return OutputFileConfig(**parsed)
    return None
