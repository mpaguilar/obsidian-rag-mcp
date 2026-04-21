"""CLI vault command implementations for obsidian-rag.

This module contains vault-related CLI command implementations,
extracted from cli_commands.py to keep file sizes under 1000 lines.
"""

import json
import logging
import uuid

import click
from sqlalchemy import func
from sqlalchemy.orm import Session

from obsidian_rag.database.engine import DatabaseManager
from obsidian_rag.database.models import Document, Vault
from obsidian_rag.mcp_server.tools.vaults import (
    VaultUpdateParams,
    _apply_vault_updates,
    _check_container_path_update,
    _count_vault_cascade_targets,
    _count_vault_documents,
    _handle_flush_with_integrity_check,
    _has_vault_changed,
    _lookup_vault_by_name,
)

log = logging.getLogger(__name__)

MAX_VAULT_LIMIT = 100


def _validate_limit(limit: int) -> int:
    """Validate and normalize limit parameter.

    Args:
        limit: The requested limit value.

    Returns:
        Validated limit (clamped to 1-MAX_VAULT_LIMIT range).

    """
    if limit < 1:
        return 1
    if limit > MAX_VAULT_LIMIT:
        return MAX_VAULT_LIMIT
    return limit


def _validate_offset(offset: int) -> int:
    """Validate and normalize offset parameter.

    Args:
        offset: The requested offset value.

    Returns:
        Validated offset (minimum 0).

    """
    if offset < 0:
        return 0
    return offset


def vault_list_command(
    ctx: click.Context,
    *,
    output_format: str,
    limit: int,
    offset: int,
) -> None:
    """Execute the vault list command.

    Args:
        ctx: Click context containing settings.
        output_format: Output format ("table" or "json").
        limit: Maximum number of results.
        offset: Number of results to skip.

    """
    _msg = "vault_list_command starting"
    log.debug(_msg)

    settings = ctx.obj["settings"]

    # Validate parameters
    limit = _validate_limit(limit)
    offset = _validate_offset(offset)

    db_manager = DatabaseManager(
        settings.database.url,
        pool_size=settings.database.pool_size,
        max_overflow=settings.database.max_overflow,
        pool_timeout=settings.database.pool_timeout,
        pool_recycle=settings.database.pool_recycle,
    )

    with db_manager.get_session() as session:
        # Build query with document counts using subquery
        document_counts = (
            session.query(
                Document.vault_id,
                func.count(Document.id).label("doc_count"),
            )
            .group_by(Document.vault_id)
            .subquery()
        )

        # Main query joining vaults with document counts
        query = (
            session.query(
                Vault,
                func.coalesce(document_counts.c.doc_count, 0).label("document_count"),
            )
            .outerjoin(
                document_counts,
                Vault.id == document_counts.c.vault_id,
            )
            .order_by(Vault.name)
        )

        # Get paginated results
        results = query.offset(offset).limit(limit).all()

        if not results:
            click.echo("No vaults found.")
            _msg = "vault_list_command returning - no vaults"
            log.debug(_msg)
            return

        # Format and display results
        if output_format == "json":
            output = _format_vault_list_json(results)
        else:
            output = _format_vault_list_table(results)

        click.echo(output)

    _msg = "vault_list_command returning"
    log.debug(_msg)


def _format_vault_list_json(results: list) -> str:
    """Format vault list results as JSON.

    Args:
        results: List of (Vault, document_count) tuples.

    Returns:
        JSON string with vault data.

    """
    _msg = "_format_vault_list_json starting"
    log.debug(_msg)

    output = [
        {
            "id": str(vault.id),
            "name": vault.name,
            "description": vault.description,
            "container_path": vault.container_path,
            "host_path": vault.host_path,
            "document_count": doc_count,
            "created_at": vault.created_at.isoformat() if vault.created_at else None,
        }
        for vault, doc_count in results
    ]

    result = json.dumps(output, indent=2)
    _msg = "_format_vault_list_json returning"
    log.debug(_msg)
    return result


def _format_vault_list_table(results: list) -> str:
    """Format vault list results as a table.

    Args:
        results: List of (Vault, document_count) tuples.

    Returns:
        Formatted table string.

    """
    _msg = "_format_vault_list_table starting"
    log.debug(_msg)

    lines = [f"Found {len(results)} vault(s):\n"]

    for vault, doc_count in results:
        lines.append(f"Name: {vault.name}")
        if vault.description:
            lines.append(f"Description: {vault.description}")
        lines.append(f"Container Path: {vault.container_path}")
        lines.append(f"Host Path: {vault.host_path}")
        lines.append(f"Document Count: {doc_count}")
        lines.append(
            f"Created: {vault.created_at.isoformat() if vault.created_at else 'N/A'}"
        )
        lines.append("")

    result = "\n".join(lines)
    _msg = "_format_vault_list_table returning"
    log.debug(_msg)
    return result


def _lookup_vault_by_id(
    session: Session,
    vault_id: str,
) -> Vault:
    """Lookup a vault by ID.

    Args:
        session: Database session.
        vault_id: Vault UUID string to look up.

    Returns:
        Vault instance if found.

    Raises:
        click.ClickException: If vault not found or invalid UUID.

    """
    _msg = f"Looking up vault by ID: {vault_id}"
    log.debug(_msg)
    try:
        vault_uuid = uuid.UUID(vault_id)
    except ValueError as e:
        _msg = f"Invalid UUID format: {vault_id}"
        log.exception(_msg)
        click.echo(f"Error: {_msg}", err=True)
        raise click.ClickException(_msg) from e

    vault = session.query(Vault).filter(Vault.id == vault_uuid).first()
    if not vault:
        _msg = f"Vault not found: {vault_id}"
        click.echo(f"Error: {_msg}", err=True)
        raise click.ClickException(_msg)
    return vault


def _lookup_vault_by_name_cli(
    session: Session,
    name: str,
) -> Vault:
    """Lookup a vault by name.

    Args:
        session: Database session.
        name: Vault name to look up.

    Returns:
        Vault instance if found.

    Raises:
        click.ClickException: If vault not found.

    """
    _msg = f"Looking up vault by name: {name}"
    log.debug(_msg)
    vault = session.query(Vault).filter(Vault.name == name).first()
    if not vault:
        _msg = f"Vault not found: {name}"
        click.echo(f"Error: {_msg}", err=True)
        raise click.ClickException(_msg)
    return vault


def vault_get_command(
    ctx: click.Context,
    *,
    name: str | None,
    vault_id: str | None,
) -> None:
    """Execute the vault get command.

    Args:
        ctx: Click context containing settings.
        name: Vault name to look up (preferred if both provided).
        vault_id: Vault UUID string to look up.

    Raises:
        click.ClickException: If vault not found or invalid parameters.

    """
    _msg = "vault_get_command starting"
    log.debug(_msg)

    # Validate that at least one lookup criteria is provided
    if not name and not vault_id:
        _msg = "Either --name or --id must be provided"
        click.echo(f"Error: {_msg}", err=True)
        raise click.ClickException(_msg)

    settings = ctx.obj["settings"]

    db_manager = DatabaseManager(
        settings.database.url,
        pool_size=settings.database.pool_size,
        max_overflow=settings.database.max_overflow,
        pool_timeout=settings.database.pool_timeout,
        pool_recycle=settings.database.pool_recycle,
    )

    with db_manager.get_session() as session:
        # Prefer name lookup if provided
        if name:
            vault = _lookup_vault_by_name_cli(session, name)
        else:
            # vault_id is guaranteed to be not None here because we checked above
            vault = _lookup_vault_by_id(session, vault_id)  # type: ignore[arg-type]

        # Count documents in the vault
        doc_count = (
            session.query(func.count(Document.id))
            .filter(Document.vault_id == vault.id)
            .scalar()
        ) or 0

        # Display vault details
        output = _format_vault_details(vault, doc_count)
        click.echo(output)

    _msg = "vault_get_command returning"
    log.debug(_msg)


def _format_vault_details(vault: Vault, doc_count: int) -> str:
    """Format vault details for display.

    Args:
        vault: The Vault instance to display.
        doc_count: Number of documents in the vault.

    Returns:
        Formatted vault details string.

    """
    _msg = "_format_vault_details starting"
    log.debug(_msg)

    lines = ["Vault Details:\n"]
    lines.append(f"ID: {vault.id}")
    lines.append(f"Name: {vault.name}")
    if vault.description:
        lines.append(f"Description: {vault.description}")
    lines.append(f"Container Path: {vault.container_path}")
    lines.append(f"Host Path: {vault.host_path}")
    lines.append(f"Document Count: {doc_count}")
    lines.append(
        f"Created: {vault.created_at.isoformat() if vault.created_at else 'N/A'}"
    )

    result = "\n".join(lines)
    _msg = "_format_vault_details returning"
    log.debug(_msg)
    return result


def vault_update_command(
    ctx: click.Context,
    name: str,
    description: str | None,
    host_path: str | None,
    container_path: str | None,
    *,
    force: bool,
) -> None:
    """Execute the vault update command.

    Args:
        ctx: Click context containing settings.
        name: Vault name to update.
        description: New vault description (optional).
        host_path: New host path (optional).
        container_path: New container path (optional, requires force).
        force: Whether to confirm destructive container_path change.

    Raises:
        click.ClickException: If vault not found or invalid update parameters.

    """
    _msg = "vault_update_command starting"
    log.debug(_msg)

    settings = ctx.obj["settings"]

    db_manager = DatabaseManager(
        settings.database.url,
        pool_size=settings.database.pool_size,
        max_overflow=settings.database.max_overflow,
        pool_timeout=settings.database.pool_timeout,
        pool_recycle=settings.database.pool_recycle,
    )

    with db_manager.get_session() as session:
        # Look up the vault
        vault = _lookup_vault_by_name(session, name=name)
        if not vault:
            _msg = f"Vault '{name}' not found"
            click.echo(f"Error: {_msg}", err=True)
            raise click.ClickException(_msg)

        # Create update params
        params = VaultUpdateParams(
            name=name,
            description=description,
            host_path=host_path,
            container_path=container_path,
            force=force,
        )

        # Check if any fields need updating
        if not _has_vault_changed(vault, params):
            doc_count = _count_vault_documents(session, vault_id=vault.id)
            _msg = f"No changes to apply to vault '{name}' ({doc_count} documents)"
            click.echo(_msg)
            return

        # Check container_path change requirements
        container_error = _check_container_path_update(params, vault)
        if container_error:
            error_msg = str(
                container_error.get("error", "Container path change requires --force")
            )
            _msg = f"Error: {error_msg}"
            click.echo(_msg, err=True)
            raise click.ClickException(error_msg)

        # Apply updates
        _apply_vault_updates(vault, params, session)

        # Check for integrity errors
        integrity_error = _handle_flush_with_integrity_check(
            session, params.container_path or vault.container_path
        )
        if integrity_error:
            error_msg = str(integrity_error.get("error", "Update failed"))
            _msg = f"Error: {error_msg}"
            click.echo(_msg, err=True)
            raise click.ClickException(error_msg)

        # Get updated document count
        doc_count = _count_vault_documents(session, vault_id=vault.id)

        _msg = f"Updated vault '{name}' ({doc_count} documents)"
        click.echo(_msg)

    _msg = "vault_update_command returning"
    log.debug(_msg)


def vault_delete_command(
    ctx: click.Context,
    name: str,
    *,
    confirm: bool,
) -> None:
    """Execute the vault delete command.

    Args:
        ctx: Click context containing settings.
        name: Vault name to delete.
        confirm: Whether deletion is confirmed (required).

    Raises:
        click.ClickException: If vault not found or confirm not provided.

    """
    _msg = "vault_delete_command starting"
    log.debug(_msg)

    # Check confirmation first
    if not confirm:
        _error_msg = (
            "--confirm is required to delete a vault. "
            "This action is irreversible and will cascade-delete all "
            "associated documents, tasks, and chunks."
        )
        click.echo(f"Error: {_error_msg}", err=True)
        raise click.ClickException(_error_msg)

    settings = ctx.obj["settings"]

    db_manager = DatabaseManager(
        settings.database.url,
        pool_size=settings.database.pool_size,
        max_overflow=settings.database.max_overflow,
        pool_timeout=settings.database.pool_timeout,
        pool_recycle=settings.database.pool_recycle,
    )

    with db_manager.get_session() as session:
        # Look up the vault
        vault = _lookup_vault_by_name(session, name=name)
        if not vault:
            _msg = f"Vault '{name}' not found"
            click.echo(f"Error: {_msg}", err=True)
            raise click.ClickException(_msg)

        # Count cascade targets
        doc_count, task_count, chunk_count = _count_vault_cascade_targets(
            session, vault_id=vault.id
        )

        # Log warning
        _warning_msg = (
            f"Deleting vault '{name}' with {doc_count} documents, "
            f"{task_count} tasks, {chunk_count} chunks"
        )
        log.warning(_warning_msg)

        # Delete the vault (cascade handles documents, tasks, chunks)
        session.delete(vault)
        session.flush()

        # Report success
        _msg = (
            f"Deleted vault '{name}' ("
            f"{doc_count} documents, {task_count} tasks, {chunk_count} chunks)"
        )
        click.echo(_msg)

        _warning_msg2 = (
            "Note: Vault configuration entry still exists in config file. "
            "Next ingestion will recreate this vault in the database."
        )
        click.echo(_warning_msg2)

    _msg = "vault_delete_command returning"
    log.debug(_msg)
