"""CLI ingest command helpers for obsidian-rag.

This module contains path resolution and validation functions for the ingest
command, separated from cli_commands.py to keep file size under 1000 lines.
"""

import logging
from pathlib import Path

import click

from obsidian_rag.config import Settings

log = logging.getLogger(__name__)


def _validate_path_matches_vault(path: str, vault: str, container_path: str) -> None:
    """Validate that explicit path matches vault's container_path.

    Args:
        path: Explicit path provided by user.
        vault: Vault name.
        container_path: Vault's configured container_path.

    Raises:
        click.BadParameter: If path does not match container_path.

    """
    _msg = "_validate_path_matches_vault starting"
    log.debug(_msg)

    normalized_input = Path(path).resolve()
    normalized_config = Path(container_path).resolve()
    if normalized_input != normalized_config:
        _msg = (
            f"Path '{path}' does not match vault '{vault}' container path "
            f"'{container_path}'. "
            "The path must match the configured container_path exactly."
        )
        raise click.BadParameter(_msg, param_hint="'PATH'")

    _msg = "_validate_path_matches_vault returning"
    log.debug(_msg)


def _resolve_ingest_path(
    settings: Settings,
    path: str | None,
    vault: str,
) -> str:
    """Resolve the ingest path from explicit path or vault configuration.

    If an explicit path is provided, it takes precedence (REQ-009).
    If no path is provided, the vault's container_path from configuration
    is used (REQ-008). Validates that the vault exists in configuration
    and that the resolved path matches the vault's container_path.

    Args:
        settings: Application settings containing vault configurations.
        path: Explicit path argument, or None if not provided.
        vault: Vault name for configuration lookup.

    Returns:
        Resolved path string to use for ingestion.

    Raises:
        click.BadParameter: If vault not found in config and no path provided,
            if the resolved path does not exist on the filesystem,
            or if an explicit path does not match the vault's container_path.

    """
    _msg = "_resolve_ingest_path starting"
    log.debug(_msg)

    vault_config = settings.get_vault(vault)
    if vault_config is None:
        available = settings.get_vault_names()
        vaults_list = ", ".join(available) if available else "none configured"
        _msg = (
            f"Vault '{vault}' not found in configuration. "
            f"Available vaults: {vaults_list}"
        )
        raise click.BadParameter(_msg, param_hint="'--vault'")

    if path is not None:
        _validate_path_matches_vault(path, vault, vault_config.container_path)
        resolved = path
    else:
        resolved = vault_config.container_path
        _msg = f"Using vault container_path: {resolved}"
        log.debug(_msg)

    resolved_path = Path(resolved)
    if not resolved_path.exists():
        _msg = (
            f"Path '{resolved}' does not exist. "
            "Please verify the path or provide an alternative."
        )
        raise click.BadParameter(_msg, param_hint="'PATH'")

    if not resolved_path.is_dir():
        _msg = f"Path '{resolved}' exists but is not a directory."
        raise click.BadParameter(_msg, param_hint="'PATH'")

    _msg = f"_resolve_ingest_path returning: {resolved}"
    log.debug(_msg)
    return resolved
