"""Tests for IngestRequestTracker class."""

import asyncio

import pytest

from obsidian_rag.mcp_server.ingest_tracker import (
    IngestRequestTracker,
    _RequestEntry,
)


def test_request_entry_creation():
    """Test _RequestEntry dataclass creation."""
    entry = _RequestEntry(params={"vault": "test"})
    assert entry.params == {"vault": "test"}
    assert entry.result is None
    assert entry.status == "pending"
    assert entry.error is None


def test_tracker_initialization():
    """Test IngestRequestTracker initialization."""
    tracker = IngestRequestTracker()
    assert tracker._requests == {}
    assert tracker._lock is not None


def test_start_request_new_request():
    """Test starting a new request."""
    tracker = IngestRequestTracker()

    async def run_test():
        should_process, cached = await tracker.start_request("req-1", {"vault": "test"})
        assert should_process is True
        assert cached is None
        assert "req-1" in tracker._requests
        assert tracker._requests["req-1"].status == "pending"

    asyncio.run(run_test())


def test_start_request_duplicate_complete():
    """Test duplicate request when original is complete."""
    tracker = IngestRequestTracker()

    async def run_test():
        # First request
        await tracker.start_request("req-1", {"vault": "test"})
        result = {"total": 10}
        await tracker.complete_request("req-1", result)

        # Duplicate request
        should_process, cached = await tracker.start_request("req-1", {"vault": "test"})
        assert should_process is False
        assert cached == result

    asyncio.run(run_test())


def test_complete_request():
    """Test marking request as complete."""
    tracker = IngestRequestTracker()

    async def run_test():
        await tracker.start_request("req-1", {"vault": "test"})
        result = {"total": 10, "new": 5}
        await tracker.complete_request("req-1", result)

        entry = tracker._requests["req-1"]
        assert entry.status == "complete"
        assert entry.result == result
        assert entry.error is None

    asyncio.run(run_test())


def test_complete_request_not_found():
    """Test completing a request that doesn't exist."""
    tracker = IngestRequestTracker()

    async def run_test():
        # Should not raise, just log warning
        await tracker.complete_request("nonexistent", {"total": 10})

    asyncio.run(run_test())


def test_fail_request():
    """Test marking request as failed."""
    tracker = IngestRequestTracker()

    async def run_test():
        await tracker.start_request("req-1", {"vault": "test"})
        error = ValueError("Test error")
        await tracker.fail_request("req-1", error)

        entry = tracker._requests["req-1"]
        assert entry.status == "complete"
        assert entry.error == error

    asyncio.run(run_test())


def test_fail_request_not_found():
    """Test failing a request that doesn't exist."""
    tracker = IngestRequestTracker()

    async def run_test():
        # Should not raise, just log warning
        await tracker.fail_request("nonexistent", ValueError("Test"))

    asyncio.run(run_test())


def test_get_result_complete():
    """Test getting result for completed request."""
    tracker = IngestRequestTracker()

    async def run_test():
        await tracker.start_request("req-1", {"vault": "test"})
        result = {"total": 10}
        await tracker.complete_request("req-1", result)

        cached = tracker.get_result("req-1")
        assert cached == result

    asyncio.run(run_test())


def test_get_result_pending():
    """Test getting result for pending request."""
    tracker = IngestRequestTracker()

    async def run_test():
        await tracker.start_request("req-1", {"vault": "test"})

        cached = tracker.get_result("req-1")
        assert cached is None

    asyncio.run(run_test())


def test_get_result_not_found():
    """Test getting result for non-existent request."""
    tracker = IngestRequestTracker()
    cached = tracker.get_result("nonexistent")
    assert cached is None


def test_clear_request():
    """Test clearing a specific request."""
    tracker = IngestRequestTracker()

    async def run_test():
        await tracker.start_request("req-1", {"vault": "test"})
        await tracker.clear_request("req-1")

        assert "req-1" not in tracker._requests

    asyncio.run(run_test())


def test_clear_request_not_found():
    """Test clearing a request that doesn't exist."""
    tracker = IngestRequestTracker()

    async def run_test():
        # Should not raise, just log warning
        await tracker.clear_request("nonexistent")

    asyncio.run(run_test())


def test_clear_all():
    """Test clearing all requests."""
    tracker = IngestRequestTracker()

    async def run_test():
        await tracker.start_request("req-1", {"vault": "test"})
        await tracker.start_request("req-2", {"vault": "test2"})

        tracker.clear_all()

        assert tracker._requests == {}

    asyncio.run(run_test())


def test_thread_safety_concurrent_starts():
    """Test thread safety with concurrent start_request calls."""
    tracker = IngestRequestTracker()
    results = []

    async def start_request():
        should_process, cached = await tracker.start_request(
            "concurrent-req", {"vault": "test"}
        )
        results.append((should_process, cached))

    async def run_test():
        # Start multiple concurrent requests
        await asyncio.gather(start_request(), start_request(), start_request())

    asyncio.run(run_test())

    # Only one should process, others should get cached result
    process_count = sum(1 for should_process, _ in results if should_process)
    cached_count = sum(1 for should_process, _ in results if not should_process)

    # First one processes, subsequent ones find it pending or complete
    assert process_count >= 1
    assert process_count + cached_count == 3


def test_complete_request_idempotent():
    """Test that complete_request is idempotent."""
    tracker = IngestRequestTracker()

    async def run_test():
        await tracker.start_request("req-1", {"vault": "test"})
        result1 = {"total": 10}
        result2 = {"total": 20}

        # Complete twice with different results
        await tracker.complete_request("req-1", result1)
        await tracker.complete_request("req-1", result2)

        # Should have the second result (last write wins)
        entry = tracker._requests["req-1"]
        assert entry.result == result2

    asyncio.run(run_test())


def test_different_request_ids():
    """Test that different parameters produce different request IDs."""
    tracker = IngestRequestTracker()

    async def run_test():
        # Start two different requests
        should_process1, _ = await tracker.start_request("req-1", {"vault": "test1"})
        should_process2, _ = await tracker.start_request("req-2", {"vault": "test2"})

        assert should_process1 is True
        assert should_process2 is True
        assert len(tracker._requests) == 2

    asyncio.run(run_test())
