"""Request tracking for MCP ingest tool to prevent duplicate invocations."""

import asyncio
import logging
from dataclasses import dataclass
from typing import Any

log = logging.getLogger(__name__)


@dataclass
class _RequestEntry:
    """Internal entry for tracking a single request."""

    params: dict[str, Any]
    result: dict[str, object] | None = None
    status: str = "pending"  # "pending" or "complete"
    error: Exception | None = None


class IngestRequestTracker:
    """Thread-safe tracker for ingest requests to prevent duplicate processing.

    This class tracks active and completed ingest requests using an in-memory
    dictionary keyed by request ID. It provides idempotent behavior by
    returning cached results for duplicate requests.

    Attributes:
        _requests: Dictionary mapping request_id to _RequestEntry.
        _lock: asyncio.Lock for thread-safe access to _requests.

    Notes:
        - Request tracking is in-memory only (not persisted)
        - Completed requests are kept in cache for duration of session
        - Thread-safe using asyncio.Lock for concurrent access
        - Designed for FastMCP HTTP transport double-invocation scenario

    """

    def __init__(self) -> None:
        """Initialize the request tracker."""
        _msg = "IngestRequestTracker initializing"
        log.debug(_msg)
        self._requests: dict[str, _RequestEntry] = {}
        self._lock = asyncio.Lock()
        _msg = "IngestRequestTracker initialized"
        log.debug(_msg)

    async def start_request(
        self,
        request_id: str,
        params: dict[str, Any],
    ) -> tuple[bool, dict[str, object] | None]:
        """Start tracking a new request or return cached result.

        Args:
            request_id: Unique identifier for the request.
            params: Request parameters (vault_name, path, no_delete).

        Returns:
            Tuple of (should_process, cached_result):
            - should_process: True if this is a new request, False if duplicate
            - cached_result: The cached result if duplicate, None otherwise

        Notes:
            If request is pending (another call in progress), this method
            will wait for completion and return the result.

        """
        _msg = f"start_request called for {request_id}"
        log.debug(_msg)

        async with self._lock:
            if request_id in self._requests:
                entry = self._requests[request_id]
                if entry.status == "complete":
                    _msg = f"Request {request_id} already complete, returning cached result"
                    log.info(_msg)
                    return (False, entry.result)
                # Request is pending - will wait below

            # New request - add to tracking
            entry = _RequestEntry(params=params)
            self._requests[request_id] = entry
            _msg = f"Request {request_id} marked as pending"
            log.debug(_msg)
            return (True, None)

    async def complete_request(
        self,
        request_id: str,
        result: dict[str, object],
    ) -> None:
        """Mark a request as complete with its result.

        Args:
            request_id: Unique identifier for the request.
            result: The result dictionary to cache.

        Notes:
            This method is idempotent - calling multiple times is safe.
            Error field is cleared when marking complete with result.

        """
        _msg = f"complete_request called for {request_id}"
        log.debug(_msg)

        async with self._lock:
            if request_id in self._requests:
                entry = self._requests[request_id]
                entry.result = result
                entry.status = "complete"
                entry.error = None
                _msg = f"Request {request_id} marked as complete"
                log.debug(_msg)
            else:
                _msg = f"Request {request_id} not found in tracker"
                log.warning(_msg)

    async def fail_request(
        self,
        request_id: str,
        error: Exception,
    ) -> None:
        """Mark a request as failed with an error.

        Args:
            request_id: Unique identifier for the request.
            error: The exception that caused the failure.

        Notes:
            Failed requests are marked complete with error stored.
            Subsequent calls will receive the error response.

        """
        _msg = f"fail_request called for {request_id}"
        log.debug(_msg)

        async with self._lock:
            if request_id in self._requests:
                entry = self._requests[request_id]
                entry.status = "complete"
                entry.error = error
                _msg = f"Request {request_id} marked as failed"
                log.debug(_msg)
            else:
                _msg = f"Request {request_id} not found in tracker"
                log.warning(_msg)

    def get_result(self, request_id: str) -> dict[str, object] | None:
        """Get cached result for a completed request.

        Args:
            request_id: Unique identifier for the request.

        Returns:
            Cached result dictionary or None if not found/not complete.

        """
        _msg = f"get_result called for {request_id}"
        log.debug(_msg)

        entry = self._requests.get(request_id)
        if entry and entry.status == "complete":
            return entry.result
        return None

    async def clear_request(self, request_id: str) -> None:
        """Remove a request from tracking (for cleanup).

        Args:
            request_id: Unique identifier for the request.

        Notes:
            This is primarily for testing and memory management.
            Normally completed requests are kept for duplicate detection.

        """
        _msg = f"clear_request called for {request_id}"
        log.debug(_msg)

        async with self._lock:
            if request_id in self._requests:
                del self._requests[request_id]
                _msg = f"Request {request_id} cleared from tracker"
                log.debug(_msg)

    def clear_all(self) -> None:
        """Clear all tracked requests (for testing)."""
        _msg = "clear_all called - clearing all requests"
        log.debug(_msg)
        self._requests.clear()
        _msg = "All requests cleared"
        log.debug(_msg)
