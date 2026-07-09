"""Tests for IngestLockError handling in MCP ingest pipeline."""

import asyncio
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest
from click.testing import CliRunner

from obsidian_rag.services.ingestion import IngestionResult
from obsidian_rag.services.ingestion_lock import IngestLockError

runner = CliRunner()


def test_ingest_handler_propagates_ingest_lock_error():
    """Test that _ingest_handler propagates IngestLockError without catching it."""
    from obsidian_rag.mcp_server.handlers import IngestHandlerParams, _ingest_handler

    mock_settings = MagicMock()
    vault_config = MagicMock()
    vault_config.container_path = "/test/vault"
    mock_settings.get_vault.return_value = vault_config

    mock_db_manager = MagicMock()
    mock_embedding_provider = MagicMock()

    params = IngestHandlerParams(
        settings=mock_settings,
        db_manager=mock_db_manager,
        embedding_provider=mock_embedding_provider,
        vault_name="TestVault",
        path_override=None,
    )

    with patch(
        "obsidian_rag.mcp_server.handlers._validate_ingest_path"
    ) as mock_validate:
        mock_validate.return_value = Path("/test/vault")

        with patch("obsidian_rag.mcp_server.handlers.IngestionService") as mock_service:
            mock_instance = MagicMock()
            mock_instance.ingest_vault.side_effect = IngestLockError(
                "Ingest already in progress"
            )
            mock_service.return_value = mock_instance

            with pytest.raises(IngestLockError, match="Ingest already in progress"):
                _ingest_handler(params)


def test_ingest_handler_passes_skip_result_through():
    """Test that _ingest_handler passes synthetic skip IngestionResult through unchanged."""
    from obsidian_rag.mcp_server.handlers import IngestHandlerParams, _ingest_handler

    mock_settings = MagicMock()
    vault_config = MagicMock()
    vault_config.container_path = "/test/vault"
    mock_settings.get_vault.return_value = vault_config

    mock_db_manager = MagicMock()
    mock_embedding_provider = MagicMock()

    params = IngestHandlerParams(
        settings=mock_settings,
        db_manager=mock_db_manager,
        embedding_provider=mock_embedding_provider,
        vault_name="TestVault",
        path_override=None,
    )

    synthetic_result = IngestionResult(
        total=0,
        new=0,
        updated=0,
        unchanged=0,
        errors=0,
        deleted=0,
        chunks_created=0,
        empty_documents=0,
        processing_time_seconds=0.0,
        message="Skipped: force re-ingest already in progress for vault 'TestVault'",
    )

    with patch(
        "obsidian_rag.mcp_server.handlers._validate_ingest_path"
    ) as mock_validate:
        mock_validate.return_value = Path("/test/vault")

        with patch("obsidian_rag.mcp_server.handlers.IngestionService") as mock_service:
            mock_instance = MagicMock()
            mock_instance.ingest_vault.return_value = synthetic_result
            mock_service.return_value = mock_instance

            result = _ingest_handler(params)

            assert result["total"] == 0
            assert (
                result["message"]
                == "Skipped: force re-ingest already in progress for vault 'TestVault'"
            )
            assert result["errors"] == 0


def test_ingest_wrapper_returns_error_dict_on_lock_error():
    """Test that ingest() wrapper returns error dict on IngestLockError."""
    from obsidian_rag.mcp_server.server import _clear_ingest_tracker, ingest

    _clear_ingest_tracker()

    mock_registry = Mock()
    mock_registry.settings = Mock()
    mock_registry.db_manager = Mock()
    mock_registry.embedding_provider = None

    with patch("obsidian_rag.mcp_server.server._get_registry") as mock_get_registry:
        mock_get_registry.return_value = mock_registry

        with patch("obsidian_rag.mcp_server.server._ingest_handler") as mock_handler:
            mock_handler.side_effect = IngestLockError("Ingest already in progress")

            result = ingest("test-vault", "/path", no_delete=False)

            assert result["success"] is False
            assert "Ingest already in progress" in result["error"]
            assert result["total"] == 0
            assert result["skipped"] is True

    _clear_ingest_tracker()


def test_ingest_wrapper_clears_tracker_on_lock_error():
    """Test that ingest() wrapper clears tracker on IngestLockError, not fail_request."""
    from obsidian_rag.mcp_server.ingest_helpers import _generate_request_id
    from obsidian_rag.mcp_server.server import (
        _clear_ingest_tracker,
        _get_ingest_tracker,
        ingest,
    )

    _clear_ingest_tracker()

    mock_registry = Mock()
    mock_registry.settings = Mock()
    mock_registry.db_manager = Mock()
    mock_registry.embedding_provider = None

    tracker = _get_ingest_tracker()
    request_id = _generate_request_id(
        "test-vault", "/path", no_delete=False, force=False
    )

    with patch("obsidian_rag.mcp_server.server._get_registry") as mock_get_registry:
        mock_get_registry.return_value = mock_registry

        with patch("obsidian_rag.mcp_server.server._ingest_handler") as mock_handler:
            mock_handler.side_effect = IngestLockError("Ingest already in progress")

            ingest("test-vault", "/path", no_delete=False)

            # Verify the request was cleared from tracker, not failed
            async def check_cleared():
                return request_id not in tracker._requests

            assert asyncio.run(check_cleared()) is True

    _clear_ingest_tracker()


def test_ingest_wrapper_does_not_cache_lock_error():
    """Test that IngestLockError is not cached and second call re-invokes handler."""
    from obsidian_rag.mcp_server.server import _clear_ingest_tracker, ingest

    _clear_ingest_tracker()

    mock_registry = Mock()
    mock_registry.settings = Mock()
    mock_registry.db_manager = Mock()
    mock_registry.embedding_provider = None

    with patch("obsidian_rag.mcp_server.server._get_registry") as mock_get_registry:
        mock_get_registry.return_value = mock_registry

        with patch("obsidian_rag.mcp_server.server._ingest_handler") as mock_handler:
            # First call raises IngestLockError
            mock_handler.side_effect = [
                IngestLockError("Ingest already in progress"),
                {"total": 10, "message": "Success"},
            ]

            # First call - should return error dict (not raise)
            result1 = ingest("test-vault", "/path", no_delete=False)
            assert result1["success"] is False
            assert mock_handler.call_count == 1

            # Second call - should invoke handler again because error was not cached
            result2 = ingest("test-vault", "/path", no_delete=False)
            assert result2 == {"total": 10, "message": "Success"}
            assert mock_handler.call_count == 2

    _clear_ingest_tracker()


def test_ingest_wrapper_normal_path_still_works():
    """Test that normal ingest path still works and calls tracker.complete_request."""
    from obsidian_rag.mcp_server.ingest_helpers import _generate_request_id
    from obsidian_rag.mcp_server.server import (
        _clear_ingest_tracker,
        _get_ingest_tracker,
        ingest,
    )

    _clear_ingest_tracker()

    mock_registry = Mock()
    mock_registry.settings = Mock()
    mock_registry.db_manager = Mock()
    mock_registry.embedding_provider = None

    tracker = _get_ingest_tracker()
    request_id = _generate_request_id(
        "test-vault", "/path", no_delete=False, force=False
    )

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

    with patch("obsidian_rag.mcp_server.server._get_registry") as mock_get_registry:
        mock_get_registry.return_value = mock_registry

        with patch("obsidian_rag.mcp_server.server._ingest_handler") as mock_handler:
            mock_handler.return_value = expected_result

            result = ingest("test-vault", "/path", no_delete=False)

            assert result == expected_result

            # Verify tracker has the completed result cached
            async def check_completed():
                entry = tracker._requests.get(request_id)
                if entry is None:
                    return False
                return entry.status == "complete" and entry.result == expected_result

            assert asyncio.run(check_completed()) is True

    _clear_ingest_tracker()


def test_ingest_wrapper_clears_tracker_on_noop_skip() -> None:
    """A skipped result dict from _ingest_handler clears the tracker (REQ-001, REQ-003)."""
    from obsidian_rag.mcp_server.ingest_helpers import _generate_request_id
    from obsidian_rag.mcp_server.server import (
        _clear_ingest_tracker,
        _get_ingest_tracker,
        ingest,
    )

    _clear_ingest_tracker()

    mock_registry = Mock()
    mock_registry.settings = Mock()
    mock_registry.db_manager = Mock()
    mock_registry.embedding_provider = None

    tracker = _get_ingest_tracker()
    request_id = _generate_request_id(
        "test-vault", "/path", no_delete=False, force=False
    )

    skip_result = {
        "total": 0,
        "new": 0,
        "updated": 0,
        "unchanged": 0,
        "errors": 0,
        "deleted": 0,
        "chunks_created": 0,
        "empty_documents": 0,
        "processing_time_seconds": 0.0,
        "message": "Skipped: force re-ingest already in progress for vault 'test-vault'",
        "skipped": True,
    }

    with patch("obsidian_rag.mcp_server.server._get_registry") as mock_get_registry:
        mock_get_registry.return_value = mock_registry
        with patch("obsidian_rag.mcp_server.server._ingest_handler") as mock_handler:
            mock_handler.return_value = skip_result

            result = ingest("test-vault", "/path", no_delete=False)

            assert result == skip_result

            async def check_cleared():
                return request_id not in tracker._requests

            assert asyncio.run(check_cleared()) is True

    _clear_ingest_tracker()


def test_ingest_wrapper_does_not_cache_noop_skip() -> None:
    """Skip result is not cached; second identical call re-invokes handler (REQ-001)."""
    from obsidian_rag.mcp_server.server import _clear_ingest_tracker, ingest

    _clear_ingest_tracker()

    mock_registry = Mock()
    mock_registry.settings = Mock()
    mock_registry.db_manager = Mock()
    mock_registry.embedding_provider = None

    skip_result = {
        "total": 0,
        "new": 0,
        "updated": 0,
        "unchanged": 0,
        "errors": 0,
        "deleted": 0,
        "chunks_created": 0,
        "empty_documents": 0,
        "processing_time_seconds": 0.0,
        "message": "Skipped: force re-ingest already in progress for vault 'test-vault'",
        "skipped": True,
    }
    success_result = {"total": 10, "message": "Success"}

    with patch("obsidian_rag.mcp_server.server._get_registry") as mock_get_registry:
        mock_get_registry.return_value = mock_registry

        with patch("obsidian_rag.mcp_server.server._ingest_handler") as mock_handler:
            mock_handler.side_effect = [skip_result, success_result]

            result1 = ingest("test-vault", "/path", no_delete=False)
            assert result1 == skip_result
            assert mock_handler.call_count == 1

            result2 = ingest("test-vault", "/path", no_delete=False)
            assert result2 == success_result
            assert mock_handler.call_count == 2

    _clear_ingest_tracker()


def test_ingest_wrapper_returns_skip_result_unchanged() -> None:
    """Skip dict is returned verbatim without success/error wrapping (REQ-003)."""
    from obsidian_rag.mcp_server.server import _clear_ingest_tracker, ingest

    _clear_ingest_tracker()

    mock_registry = Mock()
    mock_registry.settings = Mock()
    mock_registry.db_manager = Mock()
    mock_registry.embedding_provider = None

    skip_result = {
        "total": 0,
        "message": "Skipped: force re-ingest already in progress",
        "skipped": True,
    }

    with patch("obsidian_rag.mcp_server.server._get_registry") as mock_get_registry:
        mock_get_registry.return_value = mock_registry
        with patch("obsidian_rag.mcp_server.server._ingest_handler") as mock_handler:
            mock_handler.return_value = skip_result

            result = ingest("test-vault", "/path", no_delete=False)
            assert result == skip_result
            assert "success" not in result
            assert "error" not in result

    _clear_ingest_tracker()


def test_handle_skip_result_helper_returns_false_for_non_skip() -> None:
    """_handle_skip_result returns False when result has no skipped key."""
    from obsidian_rag.mcp_server.ingest_tracker import _handle_skip_result
    from obsidian_rag.mcp_server.server import (
        _clear_ingest_tracker,
        _get_ingest_tracker,
    )

    _clear_ingest_tracker()
    tracker = _get_ingest_tracker()
    request_id = "test-req-id"
    tracker._requests[request_id] = Mock()

    result = _handle_skip_result(tracker, request_id, {"total": 1})
    assert result is False
    assert request_id in tracker._requests

    _clear_ingest_tracker()


def test_handle_skip_result_helper_returns_true_and_clears_for_skip() -> None:
    """_handle_skip_result returns True and clears tracker when skipped=True."""
    from obsidian_rag.mcp_server.ingest_tracker import _handle_skip_result
    from obsidian_rag.mcp_server.server import (
        _clear_ingest_tracker,
        _get_ingest_tracker,
    )

    _clear_ingest_tracker()
    tracker = _get_ingest_tracker()
    request_id = "test-req-id"
    tracker._requests[request_id] = Mock()

    result = _handle_skip_result(tracker, request_id, {"skipped": True})
    assert result is True

    async def check_cleared():
        return request_id not in tracker._requests

    assert asyncio.run(check_cleared()) is True

    _clear_ingest_tracker()


def test_ingest_wrapper_normal_path_still_caches_when_skipped_absent() -> None:
    """Result dict without skipped key is still cached (REQ-004 regression)."""
    from obsidian_rag.mcp_server.ingest_helpers import _generate_request_id
    from obsidian_rag.mcp_server.server import (
        _clear_ingest_tracker,
        _get_ingest_tracker,
        ingest,
    )

    _clear_ingest_tracker()

    mock_registry = Mock()
    mock_registry.settings = Mock()
    mock_registry.db_manager = Mock()
    mock_registry.embedding_provider = None

    tracker = _get_ingest_tracker()
    request_id = _generate_request_id(
        "test-vault", "/path", no_delete=False, force=False
    )

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

    with patch("obsidian_rag.mcp_server.server._get_registry") as mock_get_registry:
        mock_get_registry.return_value = mock_registry
        with patch("obsidian_rag.mcp_server.server._ingest_handler") as mock_handler:
            mock_handler.return_value = expected_result

            result = ingest("test-vault", "/path", no_delete=False)
            assert result == expected_result

            async def check_completed():
                entry = tracker._requests.get(request_id)
                if entry is None:
                    return False
                return entry.status == "complete" and entry.result == expected_result

            assert asyncio.run(check_completed()) is True

    _clear_ingest_tracker()


def test_ingest_wrapper_normal_path_still_caches_when_skipped_false() -> None:
    """Result dict with explicit skipped=False is still cached (REQ-004 regression)."""
    from obsidian_rag.mcp_server.ingest_helpers import _generate_request_id
    from obsidian_rag.mcp_server.server import (
        _clear_ingest_tracker,
        _get_ingest_tracker,
        ingest,
    )

    _clear_ingest_tracker()

    mock_registry = Mock()
    mock_registry.settings = Mock()
    mock_registry.db_manager = Mock()
    mock_registry.embedding_provider = None

    tracker = _get_ingest_tracker()
    request_id = _generate_request_id(
        "test-vault", "/path", no_delete=False, force=False
    )

    expected_result = {
        "total": 10,
        "new": 5,
        "updated": 3,
        "unchanged": 2,
        "errors": 0,
        "deleted": 0,
        "processing_time_seconds": 1.5,
        "message": "Ingested 10 files",
        "skipped": False,
    }

    with patch("obsidian_rag.mcp_server.server._get_registry") as mock_get_registry:
        mock_get_registry.return_value = mock_registry
        with patch("obsidian_rag.mcp_server.server._ingest_handler") as mock_handler:
            mock_handler.return_value = expected_result

            result = ingest("test-vault", "/path", no_delete=False)
            assert result == expected_result

            async def check_completed():
                entry = tracker._requests.get(request_id)
                if entry is None:
                    return False
                return entry.status == "complete" and entry.result == expected_result

            assert asyncio.run(check_completed()) is True

    _clear_ingest_tracker()
