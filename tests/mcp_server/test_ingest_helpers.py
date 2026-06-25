"""Tests for ingest_helpers functions."""

from unittest.mock import MagicMock, patch

from obsidian_rag.mcp_server.ingest_helpers import (
    _check_and_handle_duplicate,
    _create_vault_error_response,
    _generate_request_id,
    _handle_vault_not_found,
    _is_vault_not_found_error,
)


def test_generate_request_id_with_none_no_delete():
    """Test that None no_delete produces distinct IDs from True and False."""
    id_none = _generate_request_id("vault", "/path", no_delete=None, force=False)
    id_false = _generate_request_id("vault", "/path", no_delete=False, force=False)
    id_true = _generate_request_id("vault", "/path", no_delete=True, force=False)

    assert id_none
    assert id_none != id_false
    assert id_none != id_true
    assert id_false != id_true


def test_generate_request_id_none_serializes_as_null():
    """Test that None serializes as null in JSON, distinct from false."""
    id_none = _generate_request_id("vault", "/path", no_delete=None, force=False)
    id_false = _generate_request_id("vault", "/path", no_delete=False, force=False)

    assert id_none != id_false


def test_generate_request_id_none_is_deterministic():
    """Test that None no_delete produces identical IDs for identical params."""
    id1 = _generate_request_id("vault", "/path", no_delete=None, force=False)
    id2 = _generate_request_id("vault", "/path", no_delete=None, force=False)

    assert id1 == id2


def test_create_vault_error_response():
    """Test that vault error response contains all required fields."""
    result = _create_vault_error_response("vault not found")

    assert result["success"] is False
    assert result["error"] == "vault not found"
    assert result["total"] == 0
    assert result["message"] == "Failed to ingest: vault not found"


def test_is_vault_not_found_error_true():
    """Test vault not found detection for matching error."""
    error = ValueError("Vault 'Test' not found in configuration")
    assert _is_vault_not_found_error(error) is True


def test_is_vault_not_found_error_false():
    """Test vault not found detection for non-matching error."""
    error = ValueError("Some other error")
    assert _is_vault_not_found_error(error) is False


def test_handle_vault_not_found():
    """Test vault not found handler returns error response and clears tracker."""
    tracker = MagicMock()
    tracker.clear_request.return_value = None

    with patch("obsidian_rag.mcp_server.ingest_helpers.asyncio.run") as mock_run:
        mock_run.return_value = None
        result = _handle_vault_not_found(
            "TestVault",
            "Vault 'TestVault' not found in configuration",
            "request-id",
            tracker,
        )

    assert result["success"] is False
    assert result["error"] == "Vault 'TestVault' not found in configuration"
    tracker.clear_request.assert_called_once_with("request-id")


def test_check_and_handle_duplicate_with_none_no_delete():
    """Test that no_delete=None is passed through in params dict."""
    tracker = MagicMock()
    tracker.start_request.return_value = (True, None)

    with patch("obsidian_rag.mcp_server.ingest_helpers.asyncio.run") as mock_run:
        mock_run.return_value = (True, None)
        result = _check_and_handle_duplicate(
            tracker,
            "request-id",
            "vault",
            "/path",
            no_delete=None,
            force=False,
        )

    assert result is None
    call_args = tracker.start_request.call_args
    params = call_args[0][1]
    assert params["no_delete"] is None


def test_check_and_handle_duplicate_returns_cached_with_none():
    """Test that cached result is returned when no_delete=None."""
    tracker = MagicMock()
    cached = {"cached": True}

    with patch("obsidian_rag.mcp_server.ingest_helpers.asyncio.run") as mock_run:
        mock_run.return_value = (False, cached)
        result = _check_and_handle_duplicate(
            tracker,
            "request-id",
            "vault",
            "/path",
            no_delete=None,
            force=False,
        )

    assert result == cached
    call_args = tracker.start_request.call_args
    params = call_args[0][1]
    assert params["no_delete"] is None


def test_check_and_handle_duplicate_returns_none_when_cached_is_none():
    """Test that None is returned when should_process=False and cached_result is None."""
    tracker = MagicMock()

    with patch("obsidian_rag.mcp_server.ingest_helpers.asyncio.run") as mock_run:
        mock_run.return_value = (False, None)
        result = _check_and_handle_duplicate(
            tracker,
            "request-id",
            "vault",
            "/path",
            no_delete=None,
            force=False,
        )

    assert result is None
