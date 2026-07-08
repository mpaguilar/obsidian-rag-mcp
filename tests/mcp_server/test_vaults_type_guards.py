"""Tests for type guards in vaults.py."""

from unittest.mock import MagicMock, patch

import pytest

from obsidian_rag.mcp_server.tools.vaults import (
    VaultUpdateParams,
    _apply_vault_updates,
    get_vault,
)


def _noop_validate(*_args: object, **_kwargs: object) -> None:
    """No-op replacement for parameter validation."""


def test_get_vault_raises_runtime_error_when_vault_id_is_none() -> None:
    """Guard raises RuntimeError and logs when vault_id is unexpectedly None."""
    mock_session = MagicMock()

    with (
        patch(
            "obsidian_rag.mcp_server.tools.vaults._validate_get_vault_params",
            _noop_validate,
        ),
        patch("obsidian_rag.mcp_server.tools.vaults.log") as mock_log,
    ):
        with pytest.raises(RuntimeError) as exc_info:
            get_vault(mock_session, vault_name=None, vault_id=None)

    expected_msg = "vault_id is None despite validation guarantee"
    assert str(exc_info.value) == expected_msg
    mock_log.error.assert_called_once_with(expected_msg)


def test_apply_vault_updates_raises_runtime_error_when_container_path_is_none() -> None:
    """Guard raises RuntimeError and logs when container_path is unexpectedly None."""
    mock_session = MagicMock()
    mock_vault = MagicMock()
    mock_vault.id = "vault-uuid"

    params = VaultUpdateParams(
        vault_name="Test",
        container_path=None,
        force=True,
    )

    with (
        patch(
            "obsidian_rag.mcp_server.tools.vaults._is_container_path_changing",
            return_value=True,
        ),
        patch("obsidian_rag.mcp_server.tools.vaults._delete_vault_documents"),
        patch("obsidian_rag.mcp_server.tools.vaults.log") as mock_log,
    ):
        with pytest.raises(RuntimeError) as exc_info:
            _apply_vault_updates(mock_vault, params, mock_session)

    expected_msg = "params.container_path is None despite validation guarantee"
    assert str(exc_info.value) == expected_msg
    mock_log.error.assert_called_once_with(expected_msg)
