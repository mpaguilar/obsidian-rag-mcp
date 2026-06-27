"""CLI exact document query implementations.

This module contains the exact document lookup functions,
extracted from cli_commands.py to keep file size under 1000 lines.
"""

import logging
import sys
from typing import TYPE_CHECKING

import click

from obsidian_rag.database.engine import DatabaseManager
from obsidian_rag.mcp_server.models import DocumentListResponse, DocumentResponse
from obsidian_rag.mcp_server.tools.documents import (
    get_document as get_document_impl,
)
from obsidian_rag.mcp_server.tools.documents import (
    list_documents as list_documents_impl,
)

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

log = logging.getLogger(__name__)


def _display_single_document(document: DocumentResponse) -> None:
    """Display a single document in table format.

    Args:
        document: Document response to display.

    """
    click.echo(f"Vault: {document.vault_name}")
    click.echo(f"Path: {document.file_path}")
    click.echo(f"Tags: {', '.join(document.tags) if document.tags else 'none'}")
    if document.properties:
        click.echo("Properties:")
        for key, value in document.properties.items():
            click.echo(f"  {key}: {value}")
    click.echo()
    click.echo(document.content)


def _display_document_item(doc: DocumentResponse) -> None:
    """Display a single document item in a list.

    Args:
        doc: Document response to display.

    """
    click.echo(f"File: {doc.file_name}")
    click.echo(f"Path: {doc.file_path}")
    click.echo(f"Vault: {doc.vault_name}")
    if doc.tags:
        click.echo(f"Tags: {', '.join(doc.tags)}")
    if doc.properties:
        click.echo("Properties:")
        for key, value in doc.properties.items():
            click.echo(f"  {key}: {value}")
    click.echo("")


def _display_document_list(documents: DocumentListResponse) -> None:
    """Display a list of documents in table format.

    Args:
        documents: Document list response to display.

    """
    if not documents.results:
        click.echo("No matching documents found.")
        return

    click.echo(f"Found {documents.total_count} results:\n")
    for doc in documents.results:
        _display_document_item(doc)


def _execute_get_document_lookup(
    session: "Session",
    *,
    vault: str | None,
    path: str | None,
    document_id: str | None,
    output_format: str,
) -> None:
    """Execute single document lookup and display results.

    Args:
        session: Database session.
        vault: Vault name filter.
        path: Document file path.
        document_id: Document UUID.
        output_format: Output format (table or json).

    """
    try:
        result = get_document_impl(
            session,
            vault_name=vault,
            file_path=path,
            document_id=document_id,
        )
        if output_format == "json":
            click.echo(result.model_dump_json(indent=2))
        else:
            _display_single_document(result)
    except ValueError as e:
        _msg = f"Error: {e}"
        click.echo(_msg, err=True)
        sys.exit(1)


def _execute_list_documents_lookup(
    session: "Session",
    *,
    vault: str | None,
    name: str,
    limit: int,
    output_format: str,
) -> None:
    """Execute document list lookup and display results.

    Args:
        session: Database session.
        vault: Vault name filter.
        name: Document file name.
        limit: Maximum number of results.
        output_format: Output format (table or json).

    """
    try:
        result = list_documents_impl(
            session,
            file_name=name,
            vault_name=vault,
            limit=limit,
        )
        if output_format == "json":
            click.echo(result.model_dump_json(indent=2))
        else:
            _display_document_list(result)
    except ValueError as e:
        _msg = f"Error: {e}"
        click.echo(_msg, err=True)
        sys.exit(1)


def _run_exact_query_command(
    ctx: click.Context,
    *,
    vault: str | None,
    path: str | None,
    name: str | None,
    document_id: str | None,
    limit: int,
    output_format: str,
) -> None:
    """Execute the exact query command.

    Args:
        ctx: Click context.
        vault: Vault name filter.
        path: Document file path (requires vault).
        name: Document file name.
        document_id: Document UUID.
        limit: Maximum number of results (for name lookups).
        output_format: Output format (table or json).

    """
    _msg = "_run_exact_query_command starting"
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
        if document_id is not None or path is not None:
            _execute_get_document_lookup(
                session,
                vault=vault,
                path=path,
                document_id=document_id,
                output_format=output_format,
            )
        elif name is not None:
            _execute_list_documents_lookup(
                session,
                vault=vault,
                name=name,
                limit=limit,
                output_format=output_format,
            )

    _msg = "_run_exact_query_command returning"
    log.debug(_msg)
