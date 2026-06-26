"""Unit tests for MCP server module."""

import asyncio
from unittest.mock import Mock, patch

import pytest
from click.testing import CliRunner


runner = CliRunner()

runner = CliRunner()


class TestIngestRequestTracking:
    """Tests for ingest tool request tracking functionality."""

    def test_generate_request_id_deterministic(self):
        """Test that request ID generation is deterministic."""
        from obsidian_rag.mcp_server.ingest_helpers import _generate_request_id

        id1 = _generate_request_id("vault1", "/path", no_delete=False, force=False)
        id2 = _generate_request_id("vault1", "/path", no_delete=False, force=False)
        assert id1 == id2
        assert len(id1) == 32  # MD5 hex length

    def test_generate_request_id_different_params(self):
        """Test that different parameters produce different IDs."""
        from obsidian_rag.mcp_server.ingest_helpers import _generate_request_id

        id1 = _generate_request_id("vault1", "/path", no_delete=False, force=False)
        id2 = _generate_request_id("vault2", "/path", no_delete=False, force=False)
        id3 = _generate_request_id("vault1", "/other", no_delete=False, force=False)
        id4 = _generate_request_id("vault1", "/path", no_delete=True, force=False)

        assert id1 != id2
        assert id1 != id3
        assert id1 != id4

    def test_generate_request_id_with_none_path(self):
        """Test request ID generation with None path."""
        from obsidian_rag.mcp_server.ingest_helpers import _generate_request_id

        id1 = _generate_request_id("vault1", None, no_delete=False, force=False)
        id2 = _generate_request_id("vault1", None, no_delete=False, force=False)
        assert id1 == id2
        assert len(id1) == 32

    def test_get_ingest_tracker_creates_instance(self):
        """Test that _get_ingest_tracker creates instance on first call."""
        from obsidian_rag.mcp_server.server import (
            _clear_ingest_tracker,
            _get_ingest_tracker,
        )

        _clear_ingest_tracker()  # Ensure clean state

        tracker1 = _get_ingest_tracker()
        assert tracker1 is not None

        tracker2 = _get_ingest_tracker()
        assert tracker1 is tracker2  # Same instance

        _clear_ingest_tracker()

    def test_clear_ingest_tracker(self):
        """Test that _clear_ingest_tracker clears the tracker."""
        from obsidian_rag.mcp_server.server import (
            _clear_ingest_tracker,
            _get_ingest_tracker,
        )

        tracker = _get_ingest_tracker()

        async def add_request():
            await tracker.start_request("test-req", {"vault": "test"})

        asyncio.run(add_request())

        _clear_ingest_tracker()

        # After clearing, should get new instance
        new_tracker = _get_ingest_tracker()
        assert new_tracker is not tracker

        _clear_ingest_tracker()

    @patch("obsidian_rag.mcp_server.server._get_registry")
    @patch("obsidian_rag.mcp_server.server._ingest_handler")
    def test_ingest_caches_result(self, mock_handler, mock_get_registry):
        """Test that ingest tool caches results for duplicate calls."""
        from obsidian_rag.mcp_server.server import _clear_ingest_tracker, ingest

        _clear_ingest_tracker()

        # Setup mock registry
        mock_registry = Mock()
        mock_registry.settings = Mock()
        mock_registry.db_manager = Mock()
        mock_registry.embedding_provider = None
        mock_get_registry.return_value = mock_registry

        # Setup mock handler to return a result
        expected_result = {
            "total": 10,
            "new": 5,
            "updated": 3,
            "unchanged": 2,
            "errors": 0,
            "deleted": 0,
            "processing_time_seconds": 1.5,
            "message": "Ingested 10 files",
        }
        mock_handler.return_value = expected_result

        # First call - should process
        result1 = ingest("test-vault", "/path", no_delete=False)
        assert result1 == expected_result
        assert mock_handler.call_count == 1

        # Second call with same params - should return cached
        result2 = ingest("test-vault", "/path", no_delete=False)
        assert result2 == expected_result
        # Handler should NOT be called again
        assert mock_handler.call_count == 1

        _clear_ingest_tracker()

    @patch("obsidian_rag.mcp_server.server._get_registry")
    @patch("obsidian_rag.mcp_server.server._ingest_handler")
    def test_ingest_different_params_not_cached(self, mock_handler, mock_get_registry):
        """Test that different parameters are not cached together."""
        from obsidian_rag.mcp_server.server import _clear_ingest_tracker, ingest

        _clear_ingest_tracker()

        # Setup mock registry
        mock_registry = Mock()
        mock_registry.settings = Mock()
        mock_registry.db_manager = Mock()
        mock_registry.embedding_provider = None
        mock_get_registry.return_value = mock_registry

        # Setup mock handler
        result1 = {"total": 10, "message": "First"}
        result2 = {"total": 20, "message": "Second"}
        mock_handler.side_effect = [result1, result2]

        # First call
        ingest("vault1", "/path", no_delete=False)
        assert mock_handler.call_count == 1

        # Different vault - should process
        result = ingest("vault2", "/path", no_delete=False)
        assert mock_handler.call_count == 2
        assert result == result2

        _clear_ingest_tracker()

    @patch("obsidian_rag.mcp_server.server._get_registry")
    @patch("obsidian_rag.mcp_server.server._ingest_handler")
    def test_ingest_error_cached(self, mock_handler, mock_get_registry):
        """Test that errors are properly tracked and not re-processed."""
        from obsidian_rag.mcp_server.ingest_helpers import _generate_request_id
        from obsidian_rag.mcp_server.server import (
            _clear_ingest_tracker,
            _get_ingest_tracker,
            ingest,
        )

        _clear_ingest_tracker()

        # Setup mock registry
        mock_registry = Mock()
        mock_registry.settings = Mock()
        mock_registry.db_manager = Mock()
        mock_registry.embedding_provider = None
        mock_get_registry.return_value = mock_registry

        # Setup mock handler to raise error
        mock_handler.side_effect = ValueError("Test error")

        # First call - should raise
        with pytest.raises(ValueError, match="Test error"):
            ingest("test-vault", "/path", no_delete=False)

        # Get tracker and verify error was recorded
        tracker = _get_ingest_tracker()
        request_id = _generate_request_id(
            "test-vault", "/path", no_delete=False, force=False
        )

        async def check_error():
            entry = tracker._requests.get(request_id)
            assert entry is not None
            assert entry.status == "complete"
            assert entry.error is not None

        asyncio.run(check_error())

        _clear_ingest_tracker()

    @patch("obsidian_rag.mcp_server.server._get_registry")
    @patch("obsidian_rag.mcp_server.server._ingest_handler")
    def test_ingest_logs_duplicate_detection(
        self, mock_handler, mock_get_registry, caplog
    ):
        """Test that duplicate request detection is logged."""
        import logging

        from obsidian_rag.mcp_server.server import _clear_ingest_tracker, ingest

        _clear_ingest_tracker()

        # Setup mock registry
        mock_registry = Mock()
        mock_registry.settings = Mock()
        mock_registry.db_manager = Mock()
        mock_registry.embedding_provider = None
        mock_get_registry.return_value = mock_registry

        mock_handler.return_value = {"total": 10, "message": "Done"}

        # Set log level to INFO to capture our log messages
        with caplog.at_level(
            logging.INFO, logger="obsidian_rag.mcp_server.ingest_helpers"
        ):
            # First call
            ingest("test-vault", "/path", no_delete=False)

            # Second call - duplicate
            ingest("test-vault", "/path", no_delete=False)

        # Check that duplicate detection was logged
        assert "Returning cached result for duplicate request" in caplog.text

        _clear_ingest_tracker()


class TestVaultErrorHandling:
    """Tests for REQ-005: vault not found error handling."""

    @patch("obsidian_rag.mcp_server.server._get_registry")
    @patch("obsidian_rag.mcp_server.server._ingest_handler")
    def test_ingest_vault_not_found_returns_error_dict(
        self, mock_handler, mock_get_registry
    ):
        """Test that vault not found error returns error dict instead of raising."""
        from obsidian_rag.mcp_server.server import (
            _clear_ingest_tracker,
            ingest,
        )

        _clear_ingest_tracker()

        # Setup mock to raise vault not found error
        mock_handler.side_effect = ValueError(
            "Vault 'NonExistent' not found in configuration. Available: Other"
        )

        mock_registry = Mock()
        mock_registry.settings = Mock()
        mock_registry.db_manager = Mock()
        mock_registry.embedding_provider = None
        mock_get_registry.return_value = mock_registry

        # Call ingest - should NOT raise, should return error dict
        result = ingest("NonExistent", "/path", no_delete=False)

        # Verify error response format
        assert result["success"] is False
        assert "not found in configuration" in result["error"]
        assert result["errors"] == 1
        assert result["total"] == 0
        assert "NonExistent" in result["message"]

        _clear_ingest_tracker()

    @patch("obsidian_rag.mcp_server.server._get_registry")
    @patch("obsidian_rag.mcp_server.server._ingest_handler")
    def test_ingest_vault_not_found_logs_warning(
        self, mock_handler, mock_get_registry, caplog
    ):
        """Test that vault not found error logs warning message."""
        import logging

        from obsidian_rag.mcp_server.server import (
            _clear_ingest_tracker,
            ingest,
        )

        _clear_ingest_tracker()

        mock_handler.side_effect = ValueError(
            "Vault 'MissingVault' not found in configuration."
        )

        mock_registry = Mock()
        mock_registry.settings = Mock()
        mock_registry.db_manager = Mock()
        mock_registry.embedding_provider = None
        mock_get_registry.return_value = mock_registry

        with caplog.at_level(logging.WARNING, logger="obsidian_rag.mcp_server.server"):
            ingest("MissingVault", "/path", no_delete=False)

        # Verify warning was logged
        assert "client requested non-existent vault 'MissingVault'" in caplog.text

        _clear_ingest_tracker()

    @patch("obsidian_rag.mcp_server.server._get_registry")
    @patch("obsidian_rag.mcp_server.server._ingest_handler")
    def test_ingest_vault_not_found_not_cached(self, mock_handler, mock_get_registry):
        """Test that failed vault requests are NOT cached."""
        from obsidian_rag.mcp_server.server import (
            _clear_ingest_tracker,
            _get_ingest_tracker,
            ingest,
        )

        _clear_ingest_tracker()

        mock_handler.side_effect = ValueError(
            "Vault 'TestVault' not found in configuration."
        )

        mock_registry = Mock()
        mock_registry.settings = Mock()
        mock_registry.db_manager = Mock()
        mock_registry.embedding_provider = None
        mock_get_registry.return_value = mock_registry

        # First call - should fail
        result1 = ingest("TestVault", "/path", no_delete=False)
        assert result1["success"] is False

        # Check tracker - request should be cleared, not cached
        tracker = _get_ingest_tracker()

        async def check_not_cached():
            # Generate same request ID
            from obsidian_rag.mcp_server.ingest_helpers import _generate_request_id

            request_id = _generate_request_id(
                "TestVault", "/path", no_delete=False, force=False
            )
            return request_id in tracker._requests

        import asyncio

        is_cached = asyncio.run(check_not_cached())
        assert is_cached is False  # Should NOT be cached

        _clear_ingest_tracker()

    @patch("obsidian_rag.mcp_server.server._get_registry")
    @patch("obsidian_rag.mcp_server.server._ingest_handler")
    def test_ingest_other_valueerror_still_raises(
        self, mock_handler, mock_get_registry
    ):
        """Test that non-vault ValueErrors are still raised."""
        from obsidian_rag.mcp_server.server import (
            _clear_ingest_tracker,
            ingest,
        )

        _clear_ingest_tracker()

        # Setup mock to raise a different ValueError (not vault-related)
        mock_handler.side_effect = ValueError("Some other validation error")

        mock_registry = Mock()
        mock_registry.settings = Mock()
        mock_registry.db_manager = Mock()
        mock_registry.embedding_provider = None
        mock_get_registry.return_value = mock_registry

        # Call ingest - should raise ValueError
        with pytest.raises(ValueError, match="Some other validation error"):
            ingest("SomeVault", "/path", no_delete=False)

        _clear_ingest_tracker()

    @patch("obsidian_rag.mcp_server.server._get_registry")
    @patch("obsidian_rag.mcp_server.server._ingest_handler")
    def test_ingest_generic_exception_raises_and_logs(
        self, mock_handler, mock_get_registry, caplog
    ):
        """Test that generic exceptions are raised and logged properly."""
        import logging

        from obsidian_rag.mcp_server.server import (
            _clear_ingest_tracker,
            ingest,
        )

        _clear_ingest_tracker()

        # Setup mock to raise a generic Exception (not ValueError)
        mock_handler.side_effect = RuntimeError("Database connection failed")

        mock_registry = Mock()
        mock_registry.settings = Mock()
        mock_registry.db_manager = Mock()
        mock_registry.embedding_provider = None
        mock_get_registry.return_value = mock_registry

        # Call ingest - should raise RuntimeError
        with caplog.at_level(logging.ERROR, logger="obsidian_rag.mcp_server.server"):
            with pytest.raises(RuntimeError, match="Database connection failed"):
                ingest("SomeVault", "/path", no_delete=False)

        # Verify error was logged
        assert "Request" in caplog.text
        assert "failed" in caplog.text

        _clear_ingest_tracker()


class TestVaultErrorHelperFunctions:
    """Direct tests for vault error handling helper functions."""

    def test_is_vault_not_found_error_returns_true_for_vault_error(self):
        """Test _is_vault_not_found_error returns True for vault not found."""
        from obsidian_rag.mcp_server.ingest_helpers import _is_vault_not_found_error

        error = ValueError("Vault 'Test' not found in configuration. Available: Other")
        result = _is_vault_not_found_error(error)
        assert result is True

    def test_is_vault_not_found_error_returns_false_for_other_errors(self):
        """Test _is_vault_not_found_error returns False for non-vault errors."""
        from obsidian_rag.mcp_server.ingest_helpers import _is_vault_not_found_error

        error1 = ValueError("Some other error")
        error2 = ValueError("not found in configuration")  # Missing "Vault"
        error3 = ValueError("Vault error")  # Missing "not found in configuration"

        assert _is_vault_not_found_error(error1) is False
        assert _is_vault_not_found_error(error2) is False
        assert _is_vault_not_found_error(error3) is False

    def test_handle_vault_not_found_returns_error_dict(self):
        """Test _handle_vault_not_found returns proper error response."""
        from obsidian_rag.mcp_server.ingest_helpers import _handle_vault_not_found
        from obsidian_rag.mcp_server.server import (
            _clear_ingest_tracker,
            _get_ingest_tracker,
        )

        _clear_ingest_tracker()
        tracker = _get_ingest_tracker()

        result = _handle_vault_not_found(
            vault_name="MissingVault",
            error_msg="Vault 'MissingVault' not found in configuration",
            request_id="test-request-123",
            tracker=tracker,
        )

        assert result["success"] is False
        assert "not found in configuration" in result["error"]
        assert result["errors"] == 1
        assert result["total"] == 0
        assert "MissingVault" in result["message"]

        _clear_ingest_tracker()

    def test_handle_vault_not_found_clears_tracker(self):
        """Test _handle_vault_not_found clears the request from tracker."""
        from obsidian_rag.mcp_server.ingest_helpers import _handle_vault_not_found
        from obsidian_rag.mcp_server.server import (
            _clear_ingest_tracker,
            _get_ingest_tracker,
        )

        _clear_ingest_tracker()
        tracker = _get_ingest_tracker()
        request_id = "test-clear-request"

        # First add a pending request
        async def add_pending():
            await tracker.start_request(request_id, {"test": "data"})

        asyncio.run(add_pending())

        # Verify it's there
        async def check_exists():
            return request_id in tracker._requests

        assert asyncio.run(check_exists()) is True

        # Handle the vault not found error
        _handle_vault_not_found(
            vault_name="TestVault",
            error_msg="Vault not found",
            request_id=request_id,
            tracker=tracker,
        )

        # Verify it was cleared
        assert asyncio.run(check_exists()) is False

        _clear_ingest_tracker()

    def test_check_and_handle_duplicate_returns_none_for_new_request(self):
        """Test _check_and_handle_duplicate returns None for new requests."""
        from obsidian_rag.mcp_server.ingest_helpers import _check_and_handle_duplicate
        from obsidian_rag.mcp_server.server import (
            _clear_ingest_tracker,
            _get_ingest_tracker,
        )

        _clear_ingest_tracker()
        tracker = _get_ingest_tracker()

        result = _check_and_handle_duplicate(
            tracker=tracker,
            request_id="new-request-123",
            vault_name="TestVault",
            path="/test/path",
            no_delete=False,
            force=False,
        )

        # Should return None for new requests (should_process=True)
        assert result is None

        _clear_ingest_tracker()

    def test_check_and_handle_duplicate_returns_cached_result(self):
        """Test _check_and_handle_duplicate returns cached result for duplicate."""
        from obsidian_rag.mcp_server.ingest_helpers import _check_and_handle_duplicate
        from obsidian_rag.mcp_server.server import (
            _clear_ingest_tracker,
            _get_ingest_tracker,
        )

        _clear_ingest_tracker()
        tracker = _get_ingest_tracker()
        request_id = "duplicate-request-456"
        cached_data = {"success": True, "total": 42}

        # First complete the request with a result
        async def setup_completed():
            await tracker.start_request(request_id, {"vault_name": "Test"})
            await tracker.complete_request(request_id, cached_data)

        asyncio.run(setup_completed())

        # Now check for duplicate - should return cached result
        result = _check_and_handle_duplicate(
            tracker=tracker,
            request_id=request_id,
            vault_name="TestVault",
            path="/test/path",
            no_delete=False,
            force=False,
        )

        assert result is not None
        assert result["success"] is True
        assert result["total"] == 42

        _clear_ingest_tracker()

    def test_check_and_handle_duplicate_handles_none_cached_result(self):
        """Test _check_and_handle_duplicate handles None cached_result gracefully."""
        from obsidian_rag.mcp_server.ingest_helpers import _check_and_handle_duplicate
        from obsidian_rag.mcp_server.server import (
            _clear_ingest_tracker,
            _get_ingest_tracker,
        )

        _clear_ingest_tracker()
        tracker = _get_ingest_tracker()
        request_id = "none-result-request"

        # Start but don't complete - this creates a pending entry
        async def setup_pending():
            await tracker.start_request(request_id, {"vault_name": "Test"})

        asyncio.run(setup_pending())

        # Manually set up a completed entry with None result
        # by completing and then checking behavior
        async def complete_none():
            await tracker.complete_request(request_id, None)

        asyncio.run(complete_none())

        # Now the request is completed with None result
        result = _check_and_handle_duplicate(
            tracker=tracker,
            request_id=request_id,
            vault_name="TestVault",
            path="/test/path",
            no_delete=False,
            force=False,
        )

        # When cached_result is None, it should return None (proceed with processing)
        assert result is None

        _clear_ingest_tracker()
