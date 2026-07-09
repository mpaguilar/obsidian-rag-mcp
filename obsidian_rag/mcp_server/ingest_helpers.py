"""Ingest helper functions for MCP server."""

import asyncio
import hashlib
import json
import logging
from typing import Any

from obsidian_rag.mcp_server.ingest_tracker import IngestRequestTracker

log = logging.getLogger(__name__)


def _generate_request_id(
    vault_name: str,
    path: str | None,
    *,
    no_delete: bool | None,
    force: bool,
) -> str:
    """Generate a deterministic request ID from parameters.

    Args:
        vault_name: Name of the vault.
        path: Optional path override.
        no_delete: Whether to skip deletion (None if unspecified by client).
        force: Whether to force re-ingestion.

    Returns:
        MD5 hash string uniquely identifying this request.

    Notes:
        Uses MD5 of sorted JSON for deterministic ID generation.
        Same parameters always produce the same ID.

    """
    _msg = "_generate_request_id starting"
    log.debug(_msg)

    params: dict[str, Any] = {
        "vault_name": vault_name,
        "path": path,
        "no_delete": no_delete,
        "force": force,
    }
    # Sort keys for deterministic serialization
    params_json = json.dumps(params, sort_keys=True, separators=(",", ":"))
    request_id = hashlib.md5(params_json.encode(), usedforsecurity=False).hexdigest()

    _msg = f"_generate_request_id returning: {request_id}"
    log.debug(_msg)
    return request_id


def _create_vault_error_response(error_msg: str) -> dict[str, object]:
    """Create error response dict for vault-not-found errors.

    Args:
        error_msg: The error message from the ValueError.

    Returns:
        Error response dictionary with all required fields.

    """
    return {
        "success": False,
        "error": error_msg,
        "total": 0,
        "new": 0,
        "updated": 0,
        "unchanged": 0,
        "errors": 1,
        "deleted": 0,
        "chunks_created": 0,
        "empty_documents": 0,
        "total_chunks": 0,
        "avg_chunk_tokens": 0,
        "task_chunk_count": 0,
        "content_chunk_count": 0,
        "processing_time_seconds": 0.0,
        "message": f"Failed to ingest: {error_msg}",
    }


def _is_vault_not_found_error(error: ValueError) -> bool:
    """Check if a ValueError is a vault-not-found error.

    Args:
        error: The ValueError to check.

    Returns:
        True if the error indicates a vault was not found in configuration.

    """
    error_msg = str(error)
    return "not found in configuration" in error_msg and "Vault" in error_msg


def _handle_vault_not_found(
    vault_name: str,
    error_msg: str,
    request_id: str,
    tracker: "IngestRequestTracker",
) -> dict[str, object]:
    """Handle vault not found error by returning error response dict.

    Args:
        vault_name: Name of the vault that was not found.
        error_msg: The error message from the exception.
        request_id: The request ID for tracking.
        tracker: The ingest request tracker instance.

    Returns:
        Error response dictionary with success=False.

    """
    _msg = f"client requested non-existent vault '{vault_name}'"
    log.warning(_msg)

    error_response: dict[str, object] = {
        "success": False,
        "error": error_msg,
        "total": 0,
        "new": 0,
        "updated": 0,
        "unchanged": 0,
        "errors": 1,
        "deleted": 0,
        "chunks_created": 0,
        "empty_documents": 0,
        "total_chunks": 0,
        "avg_chunk_tokens": 0,
        "task_chunk_count": 0,
        "content_chunk_count": 0,
        "processing_time_seconds": 0.0,
        "message": f"Failed to ingest: {error_msg}",
    }

    # Do NOT cache failed vault requests in tracker
    asyncio.run(tracker.clear_request(request_id))

    return error_response


def _handle_ingest_value_error(
    tracker: "IngestRequestTracker",
    request_id: str,
    vault_name: str,
    error: ValueError,
) -> dict[str, object]:
    """Handle ValueError from ingest handler.

    Returns error response dict for vault-not-found errors.
    Re-raises the error for all other ValueErrors after marking
    the request as failed in the tracker.

    Args:
        tracker: The ingest request tracker.
        request_id: The request ID for tracking.
        vault_name: Name of the vault being ingested.
        error: The ValueError raised by the ingest handler.

    Returns:
        Error response dictionary for vault-not-found errors.

    Raises:
        ValueError: Re-raises the original error for non-vault-not-found errors.

    """
    if _is_vault_not_found_error(error):
        return _handle_vault_not_found(vault_name, str(error), request_id, tracker)
    asyncio.run(tracker.fail_request(request_id, error))
    raise error


def _check_and_handle_duplicate(
    tracker: "IngestRequestTracker",
    request_id: str,
    vault_name: str,
    path: str | None,
    *,
    no_delete: bool | None,
    force: bool,
) -> dict[str, object] | None:
    """Check for duplicate requests and return cached result if available.

    Args:
        tracker: The ingest request tracker.
        request_id: The generated request ID.
        vault_name: Name of the vault.
        path: Optional path override.
        no_delete: Whether to skip deletion (None if unspecified by client).
        force: Whether to force re-ingestion.

    Returns:
        Cached result dict if duplicate with cached data, None otherwise.

    """
    should_process, cached_result = asyncio.run(
        tracker.start_request(
            request_id,
            {
                "vault_name": vault_name,
                "path": path,
                "no_delete": no_delete,
                "force": force,
            },
        )
    )

    if not should_process:
        _msg = f"Returning cached result for duplicate request {request_id}"
        log.info(_msg)
        if cached_result is not None:
            return cached_result
        _msg = "Cached result was None, proceeding with processing"
        log.warning(_msg)

    return None
