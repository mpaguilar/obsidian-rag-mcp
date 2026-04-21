"""MCP tool wrappers for vault operations.

This module contains the tool wrapper functions that access dependencies
through the registry and delegate to the corresponding handlers in
tool_definitions.py. These wrappers are registered with the FastMCP server.
"""

import logging

from obsidian_rag.mcp_server.tool_definitions import (
    _get_registry,
    delete_vault_tool,
    get_vault_tool,
    update_vault_tool,
)
from obsidian_rag.mcp_server.tools.vaults_params import VaultUpdateParams

log = logging.getLogger(__name__)


def get_vault(
    *,
    name: str | None = None,
    vault_id: str | None = None,
) -> dict[str, object]:
    """Get a single vault by name or ID.

    Retrieves vault details including document count. Either name or vault_id
    must be provided, with name taking precedence if both are given.

    Args:
        name: Vault name to lookup (preferred if both provided).
        vault_id: Vault UUID string to lookup (use vault_id, not id,
            to avoid shadowing the built-in id function).

    Returns:
        Vault response as dictionary on success, or error dict on failure:
        - Success: {"id": ..., "name": ..., "description": ...}
        - Error: {"success": False, "error": "..."}

    Notes:
        This wrapper accesses dependencies through _get_registry() and delegates
        to get_vault_tool in tool_definitions.py.

    """
    _msg = "Tool wrapper get_vault starting"
    log.debug(_msg)

    registry = _get_registry()
    result = get_vault_tool(
        registry.db_manager,
        name=name,
        vault_id=vault_id,
    )

    _msg = "Tool wrapper get_vault returning"
    log.debug(_msg)
    return result


def update_vault(
    name: str,
    *,
    description: str | None = None,
    host_path: str | None = None,
    container_path: str | None = None,
    force: bool = False,
) -> dict[str, object]:
    """Update a vault's properties.

    The name field is used for lookup only and cannot be changed.
    Changing container_path requires force=True as it deletes all documents,
    tasks, and chunks for the vault.

    Args:
        name: Vault name for lookup (required, not updatable).
        description: New description (optional).
        host_path: New host path (optional).
        container_path: New container path (optional, requires force).
        force: Required when changing container_path to confirm deletion.

    Returns:
        Vault response as dictionary on success, or error dict on failure:
        - Success: {"id": ..., "name": ..., "description": ...}
        - Error: {"success": False, "error": "..."}

    Notes:
        This wrapper accesses dependencies through _get_registry() and delegates
        to update_vault_tool in tool_definitions.py.
        Changing container_path is destructive and requires force=True.

    """
    _msg = "Tool wrapper update_vault starting"
    log.debug(_msg)

    registry = _get_registry()
    params = VaultUpdateParams(
        name=name,
        description=description,
        host_path=host_path,
        container_path=container_path,
        force=force,
    )
    result = update_vault_tool(registry.db_manager, params)

    _msg = "Tool wrapper update_vault returning"
    log.debug(_msg)
    return result


def delete_vault(
    name: str,
    *,
    confirm: bool = False,
) -> dict[str, object]:
    """Delete a vault and all associated data.

    This operation is irreversible and cascade-deletes all associated documents,
    tasks, and chunks. Requires explicit confirmation via confirm=True parameter.

    Args:
        name: Vault name to delete (required).
        confirm: Must be True to proceed with deletion. If False, returns
            an error dict explaining the requirement.

    Returns:
        Success dict with deletion counts if confirmed:
        {"success": True, "name": ..., "documents_deleted": ..., ...}
        Error dict if not confirmed or vault not found:
        {"success": False, "error": "..."}

    Notes:
        This wrapper accesses dependencies through _get_registry() and delegates
        to delete_vault_tool in tool_definitions.py.
        The vault configuration entry in the config file is NOT deleted.
        If not removed from config, the next ingestion will recreate the vault.

    """
    _msg = "Tool wrapper delete_vault starting"
    log.debug(_msg)

    registry = _get_registry()
    result = delete_vault_tool(
        registry.db_manager,
        name=name,
        confirm=confirm,
    )

    _msg = "Tool wrapper delete_vault returning"
    log.debug(_msg)
    return result
