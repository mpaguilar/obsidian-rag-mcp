"""Tests for type guards in get_document."""

from unittest.mock import MagicMock, patch

import pytest

from obsidian_rag.mcp_server.tools.documents import get_document


def _noop_validate(*_args: object, **_kwargs: object) -> None:
    """No-op replacement for parameter validation."""


def test_get_document_raises_runtime_error_when_vault_name_is_none() -> None:
    """Guard raises RuntimeError and logs when vault_name is unexpectedly None."""
    mock_session = MagicMock()

    with (
        patch(
            "obsidian_rag.mcp_server.tools.documents._validate_get_document_params",
            _noop_validate,
        ),
        patch("obsidian_rag.mcp_server.tools.documents.log") as mock_log,
    ):
        with pytest.raises(RuntimeError) as exc_info:
            get_document(
                mock_session,
                vault_name=None,
                file_path="notes.md",
                document_id=None,
            )

    expected_msg = "vault_name is None despite validation guarantee"
    assert str(exc_info.value) == expected_msg
    mock_log.error.assert_called_once_with(expected_msg)


def test_get_document_raises_runtime_error_when_file_path_is_none() -> None:
    """Guard raises RuntimeError and logs when file_path is unexpectedly None."""
    mock_session = MagicMock()

    with (
        patch(
            "obsidian_rag.mcp_server.tools.documents._validate_get_document_params",
            _noop_validate,
        ),
        patch("obsidian_rag.mcp_server.tools.documents.log") as mock_log,
    ):
        with pytest.raises(RuntimeError) as exc_info:
            get_document(
                mock_session,
                vault_name="Personal",
                file_path=None,
                document_id=None,
            )

    expected_msg = "file_path is None despite validation guarantee"
    assert str(exc_info.value) == expected_msg
    mock_log.error.assert_called_once_with(expected_msg)
