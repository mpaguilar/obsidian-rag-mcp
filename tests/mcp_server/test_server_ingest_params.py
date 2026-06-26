"""Unit tests for MCP server module."""

import asyncio
from unittest.mock import MagicMock, patch

from click.testing import CliRunner


runner = CliRunner()

runner = CliRunner()


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


class TestIngestForceParameter:
    """Tests for force parameter in ingest tool."""

    def test_generate_request_id_force_produces_different_id(self):
        """Test force=True vs force=False produces different request IDs."""
        from obsidian_rag.mcp_server.ingest_helpers import _generate_request_id

        id_no_force = _generate_request_id(
            "test-vault", None, no_delete=False, force=False
        )
        id_force = _generate_request_id("test-vault", None, no_delete=False, force=True)

        assert id_no_force != id_force

    def test_generate_request_id_force_same_params_same_id(self):
        """Test identical params including force produce same ID."""
        from obsidian_rag.mcp_server.ingest_helpers import _generate_request_id

        id1 = _generate_request_id("test-vault", "/path", no_delete=True, force=True)
        id2 = _generate_request_id("test-vault", "/path", no_delete=True, force=True)

        assert id1 == id2

    def test_check_and_handle_duplicate_accepts_force(self):
        """Test _check_and_handle_duplicate accepts force parameter."""
        from obsidian_rag.mcp_server.ingest_helpers import _check_and_handle_duplicate
        from obsidian_rag.mcp_server.server import (
            _clear_ingest_tracker,
            _get_ingest_tracker,
        )

        _clear_ingest_tracker()
        tracker = _get_ingest_tracker()
        result = _check_and_handle_duplicate(
            tracker,
            "test-id",
            "test-vault",
            None,
            no_delete=False,
            force=True,
        )
        assert result is None
        _clear_ingest_tracker()

    @patch("obsidian_rag.mcp_server.server._get_registry")
    @patch("obsidian_rag.mcp_server.server._ingest_handler")
    def test_ingest_tool_accepts_force_parameter(self, mock_handler, mock_get_registry):
        """Test ingest() accepts force parameter and passes to IngestHandlerParams."""
        from obsidian_rag.mcp_server.handlers import IngestHandlerParams
        from obsidian_rag.mcp_server.server import (
            _clear_ingest_tracker,
            ingest,
        )

        _clear_ingest_tracker()

        mock_registry = MagicMock()
        mock_registry.settings = MagicMock()
        mock_registry.db_manager = MagicMock()
        mock_registry.embedding_provider = None
        mock_get_registry.return_value = mock_registry

        mock_handler.return_value = {"total": 0}

        ingest("test-vault", force=True)

        # Verify _ingest_handler was called
        mock_handler.assert_called_once()
        call_params = mock_handler.call_args.args[0]
        assert isinstance(call_params, IngestHandlerParams)
        assert call_params.force is True

        _clear_ingest_tracker()

    @patch("obsidian_rag.mcp_server.server._get_registry")
    @patch("obsidian_rag.mcp_server.server._ingest_handler")
    def test_ingest_tool_force_generates_different_request_id(
        self, mock_handler, mock_get_registry
    ):
        """Test ingest with force=True vs False generates different request IDs."""
        import hashlib
        import json

        from obsidian_rag.mcp_server.server import (
            _clear_ingest_tracker,
            ingest,
        )

        _clear_ingest_tracker()

        mock_registry = MagicMock()
        mock_registry.settings = MagicMock()
        mock_registry.db_manager = MagicMock()
        mock_registry.embedding_provider = None
        mock_get_registry.return_value = mock_registry

        mock_handler.return_value = {"total": 0}

        generated_ids = []

        def capture_id(*args, **kwargs):
            params = {
                "vault_name": kwargs.get("vault_name") or args[0],
                "path": kwargs.get("path") if "path" in kwargs else args[1],
                "no_delete": kwargs.get("no_delete"),
                "force": kwargs.get("force"),
            }
            params_json = json.dumps(params, sort_keys=True, separators=(",", ":"))
            request_id = hashlib.md5(params_json.encode()).hexdigest()
            generated_ids.append(request_id)
            return request_id

        with patch(
            "obsidian_rag.mcp_server.server._generate_request_id",
            side_effect=capture_id,
        ):
            ingest("test-vault", force=False)
            ingest("test-vault", force=True)

        assert len(generated_ids) == 2
        assert generated_ids[0] != generated_ids[1]

        _clear_ingest_tracker()

    def test_ingest_force_not_cached_for_non_force(self):
        """Test force=True request does not return non-force cached result."""
        from obsidian_rag.mcp_server.ingest_helpers import _generate_request_id
        from obsidian_rag.mcp_server.ingest_tracker import IngestRequestTracker

        # First call without force
        id_no_force = _generate_request_id(
            "test-vault", None, no_delete=False, force=False
        )
        tracker = IngestRequestTracker()
        asyncio.run(tracker.complete_request(id_no_force, {"cached": True}))

        # Second call with force should have different ID
        id_force = _generate_request_id("test-vault", None, no_delete=False, force=True)
        assert id_no_force != id_force

        # Verify tracker doesn't return cached result for force request
        should_process, cached = asyncio.run(tracker.start_request(id_force, {}))
        assert should_process is True
        assert cached is None

    def test_ingest_handler_params_force_field(self):
        """Test IngestHandlerParams with force=True."""
        from obsidian_rag.mcp_server.handlers import IngestHandlerParams

        params = IngestHandlerParams(
            settings=MagicMock(),
            db_manager=MagicMock(),
            embedding_provider=None,
            vault_name="test",
            path_override=None,
            no_delete=False,
            force=True,
        )
        assert params.force is True


class TestIngestNoDeleteNone:
    """Tests for TASK-003: no_delete parameter accepts None (unspecified)."""

    @patch("obsidian_rag.mcp_server.server._get_registry")
    @patch("obsidian_rag.mcp_server.server._ingest_handler")
    def test_ingest_no_delete_none_propagates_to_handler_params(
        self, mock_handler: MagicMock, mock_get_registry: MagicMock
    ):
        """Test no_delete=None is passed through to IngestHandlerParams."""
        from obsidian_rag.mcp_server.server import (
            _clear_ingest_tracker,
            ingest,
        )

        _clear_ingest_tracker()

        mock_registry = MagicMock()
        mock_get_registry.return_value = mock_registry
        mock_handler.return_value = {"total": 0}

        result = ingest("vault", "/path", no_delete=None)

        call_params = mock_handler.call_args.args[0]
        assert call_params.no_delete is None
        assert result == {"total": 0}

        _clear_ingest_tracker()

    @patch("obsidian_rag.mcp_server.server._check_and_handle_duplicate")
    @patch("obsidian_rag.mcp_server.server._get_registry")
    @patch("obsidian_rag.mcp_server.server._ingest_handler")
    def test_ingest_no_delete_none_propagates_to_request_id(
        self,
        mock_handler: MagicMock,
        mock_get_registry: MagicMock,
        mock_dup_check: MagicMock,
    ):
        """Test no_delete=None flows to _generate_request_id."""
        from obsidian_rag.mcp_server.server import (
            _clear_ingest_tracker,
            ingest,
        )

        _clear_ingest_tracker()

        mock_dup_check.return_value = None
        mock_registry = MagicMock()
        mock_get_registry.return_value = mock_registry
        mock_handler.return_value = {"total": 0}

        with patch("obsidian_rag.mcp_server.server._generate_request_id") as mock_gen:
            mock_gen.return_value = "test-id"
            ingest("vault", "/path")
            mock_gen.assert_called_once_with(
                "vault", "/path", no_delete=None, force=False
            )

        _clear_ingest_tracker()

    @patch("obsidian_rag.mcp_server.server._check_and_handle_duplicate")
    @patch("obsidian_rag.mcp_server.server._get_registry")
    @patch("obsidian_rag.mcp_server.server._ingest_handler")
    def test_ingest_no_delete_none_propagates_to_duplicate_check(
        self,
        mock_handler: MagicMock,
        mock_get_registry: MagicMock,
        mock_dup_check: MagicMock,
    ):
        """Test no_delete=None flows to _check_and_handle_duplicate."""
        from obsidian_rag.mcp_server.server import (
            _clear_ingest_tracker,
            ingest,
        )

        _clear_ingest_tracker()

        mock_dup_check.return_value = None
        mock_registry = MagicMock()
        mock_get_registry.return_value = mock_registry
        mock_handler.return_value = {"total": 0}

        ingest("vault", "/path")

        mock_dup_check.assert_called_once()
        call_kwargs = mock_dup_check.call_args.kwargs
        assert call_kwargs["no_delete"] is None

        _clear_ingest_tracker()

    @patch("obsidian_rag.mcp_server.server._get_registry")
    @patch("obsidian_rag.mcp_server.server._ingest_handler")
    def test_ingest_no_delete_not_passed_defaults_to_none(
        self, mock_handler: MagicMock, mock_get_registry: MagicMock
    ):
        """Test omitting no_delete defaults to None."""
        from obsidian_rag.mcp_server.server import (
            _clear_ingest_tracker,
            ingest,
        )

        _clear_ingest_tracker()

        mock_registry = MagicMock()
        mock_get_registry.return_value = mock_registry
        mock_handler.return_value = {"total": 0}

        ingest("vault", "/path")

        call_params = mock_handler.call_args.args[0]
        assert call_params.no_delete is None

        _clear_ingest_tracker()

    @patch("obsidian_rag.mcp_server.server._get_registry")
    @patch("obsidian_rag.mcp_server.server._ingest_handler")
    def test_ingest_no_delete_false_still_works(
        self, mock_handler: MagicMock, mock_get_registry: MagicMock
    ):
        """Test explicit False is still accepted."""
        from obsidian_rag.mcp_server.server import (
            _clear_ingest_tracker,
            ingest,
        )

        _clear_ingest_tracker()

        mock_registry = MagicMock()
        mock_get_registry.return_value = mock_registry
        mock_handler.return_value = {"total": 0}

        ingest("vault", "/path", no_delete=False)

        call_params = mock_handler.call_args.args[0]
        assert call_params.no_delete is False

        _clear_ingest_tracker()

    @patch("obsidian_rag.mcp_server.server._get_registry")
    @patch("obsidian_rag.mcp_server.server._ingest_handler")
    def test_ingest_no_delete_true_still_works(
        self, mock_handler: MagicMock, mock_get_registry: MagicMock
    ):
        """Test explicit True is still accepted."""
        from obsidian_rag.mcp_server.server import (
            _clear_ingest_tracker,
            ingest,
        )

        _clear_ingest_tracker()

        mock_registry = MagicMock()
        mock_get_registry.return_value = mock_registry
        mock_handler.return_value = {"total": 0}

        ingest("vault", "/path", no_delete=True)

        call_params = mock_handler.call_args.args[0]
        assert call_params.no_delete is True

        _clear_ingest_tracker()
