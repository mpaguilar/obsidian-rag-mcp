"""CLI entry point for obsidian-rag."""

import logging

import click

from obsidian_rag.cli_commands import (
    TaskDateFilters,
    TaskFilterOptions,
    _execute_tasks_query,
    _run_ingest_command,
    _run_query_command,
    setup_logging,
)
from obsidian_rag.cli_dates import parse_cli_date
from obsidian_rag.cli_vault_commands import (
    vault_delete_command,
    vault_get_command,
    vault_list_command,
    vault_update_command,
)
from obsidian_rag.config import get_settings

log = logging.getLogger(__name__)


@click.group()
@click.option(
    "--verbose",
    is_flag=True,
    help="Enable verbose output (equivalent to --log-level DEBUG).",
)
@click.option(
    "--log-level",
    help="Set logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL). Overrides --verbose.",
)
@click.option("--config-file", type=click.Path(), help="Path to config file.")
@click.pass_context
def cli(
    ctx: click.Context,
    *,
    verbose: bool,
    log_level: str | None,
    config_file: str | None,
) -> None:
    """Obsidian RAG - CLI tool for ingesting Obsidian documents with vector search."""
    _msg = "Starting obsidian-rag CLI"
    log.debug(_msg)

    # Log config file path if provided (config loading is handled by get_settings)
    if config_file:
        _msg = f"Config file specified: {config_file}"
        log.debug(_msg)

    # Determine logging level: --log-level takes precedence over --verbose
    effective_log_level = None
    if log_level:
        effective_log_level = log_level.upper()
    elif verbose:
        effective_log_level = "DEBUG"

    # Load settings with explicit verbose flag and optional logging config
    if effective_log_level:
        settings = get_settings(
            verbose=verbose,
            logging={"level": effective_log_level},
        )
    else:
        settings = get_settings(verbose=verbose)

    # Setup logging
    setup_logging(settings.logging.level, settings.logging.format)

    # Store settings in context
    ctx.ensure_object(dict)
    ctx.obj["settings"] = settings


@cli.command()
@click.argument("path", required=False, type=click.Path())
@click.option(
    "--vault",
    required=True,
    help=(
        "Name of the vault to ingest into (as configured in config file). "
        "When PATH is not provided, the vault's container_path is used."
    ),
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Show what would happen without writing to database.",
)
@click.option(
    "--no-delete",
    is_flag=True,
    help="Skip deletion of documents not found in filesystem.",
)
@click.option("--verbose", is_flag=True, help="Show detailed progress.")
def ingest(
    path: str | None,
    *,
    vault: str,
    dry_run: bool,
    no_delete: bool,
    verbose: bool,
) -> None:
    """Ingest documents from an Obsidian vault.

    PATH is the path to the Obsidian vault directory. If not provided,
    the container_path from the vault's configuration will be used.
    When both PATH and --vault are provided, PATH takes precedence.

    By default, documents that exist in the database but not on the filesystem
    will be deleted. Use --no-delete to preserve orphaned documents.
    """
    if path:
        _msg = f"Starting ingestion from: {path}"
        log.info(_msg)
    else:
        _msg = "Starting ingestion using vault container_path"
        log.info(_msg)

    ctx = click.get_current_context()
    _run_ingest_command(
        ctx,
        path,
        vault=vault,
        dry_run=dry_run,
        no_delete=no_delete,
        verbose=verbose,
    )


@cli.command()
@click.argument("query_text")
@click.option("--limit", default=10, help="Maximum number of results.")
@click.option(
    "--format", "output_format", default="table", help="Output format (table or json)."
)
@click.option("--vault", help="Vault name to search within.")
@click.option(
    "--chunks", is_flag=True, help="Search document chunks instead of full documents."
)
@click.option("--rerank", is_flag=True, help="Rerank chunk results using FlashRank.")
def query(
    query_text: str,
    *,
    limit: int,
    output_format: str,
    vault: str | None,
    chunks: bool,
    rerank: bool,
) -> None:
    """Search documents using semantic similarity."""
    _msg = f"Searching for: {query_text}"
    log.info(_msg)

    ctx = click.get_current_context()
    _run_query_command(
        ctx,
        query_text,
        limit=limit,
        output_format=output_format,
        vault=vault,
        chunks=chunks,
        rerank=rerank,
    )


@cli.command()
@click.option(
    "--status",
    type=click.Choice(["completed", "not_completed", "in_progress", "cancelled"]),
)
@click.option(
    "--due-before",
    type=str,
    help="Filter tasks due on or before date (YYYY-MM-DD).",
)
@click.option(
    "--due-after",
    type=str,
    help="Filter tasks due on or after date (YYYY-MM-DD).",
)
@click.option(
    "--scheduled-before",
    type=str,
    help="Filter tasks scheduled on or before date (YYYY-MM-DD).",
)
@click.option(
    "--scheduled-after",
    type=str,
    help="Filter tasks scheduled on or after date (YYYY-MM-DD).",
)
@click.option(
    "--completion-before",
    type=str,
    help="Filter tasks completed on or before date (YYYY-MM-DD).",
)
@click.option(
    "--completion-after",
    type=str,
    help="Filter tasks completed on or after date (YYYY-MM-DD).",
)
@click.option(
    "--include-tags",
    multiple=True,
    help="Filter by tags tasks must have. Can be specified multiple times.",
)
@click.option(
    "--exclude-tags",
    multiple=True,
    help="Filter by tags tasks must NOT have. Can be specified multiple times.",
)
@click.option("--limit", default=20, help="Maximum number of results.")
@click.pass_context
def tasks(
    ctx: click.Context,
    status: str | None,
    due_before: str | None,
    due_after: str | None,
    scheduled_before: str | None,
    scheduled_after: str | None,
    completion_before: str | None,
    completion_after: str | None,
    include_tags: tuple[str, ...],
    exclude_tags: tuple[str, ...],
    limit: int,
) -> None:
    """Query and filter tasks."""
    _msg = "Querying tasks"
    log.info(_msg)

    # Parse all date options into a filters object
    date_filters = TaskDateFilters(
        due_before=parse_cli_date(due_before),
        due_after=parse_cli_date(due_after),
        scheduled_before=parse_cli_date(scheduled_before),
        scheduled_after=parse_cli_date(scheduled_after),
        completion_before=parse_cli_date(completion_before),
        completion_after=parse_cli_date(completion_after),
    )

    options = TaskFilterOptions(
        status=status,
        include_tags=list(include_tags) if include_tags else None,
        exclude_tags=list(exclude_tags) if exclude_tags else None,
        limit=limit,
        date_filters=date_filters,
    )

    _execute_tasks_query(ctx, options)


@cli.group()
def vault() -> None:
    """Manage Obsidian vaults."""


@vault.command(name="list")
@click.option(
    "--format",
    "output_format",
    default="table",
    help="Output format (table or json).",
)
@click.option("--limit", default=20, help="Maximum number of results.")
@click.option("--offset", default=0, help="Number of results to skip.")
@click.pass_context
def vault_list(
    ctx: click.Context,
    *,
    output_format: str,
    limit: int,
    offset: int,
) -> None:
    """List all vaults."""
    vault_list_command(ctx, output_format=output_format, limit=limit, offset=offset)


@vault.command(name="get")
@click.option("--name", help="Vault name to look up.")
@click.option("--id", "vault_id", help="Vault UUID to look up.")
@click.pass_context
def vault_get(
    ctx: click.Context,
    *,
    name: str | None,
    vault_id: str | None,
) -> None:
    """Get a single vault by name or ID."""
    vault_get_command(ctx, name=name, vault_id=vault_id)


@vault.command(name="update")
@click.option("--name", required=True, help="Vault name to update.")
@click.option("--description", help="New vault description.")
@click.option("--host-path", help="New host path.")
@click.option("--container-path", help="New container path (requires --force).")
@click.option(
    "--force",
    is_flag=True,
    help="Confirm destructive container_path change (deletes all documents).",
)
@click.pass_context
def vault_update(
    ctx: click.Context,
    *,
    name: str,
    description: str | None,
    host_path: str | None,
    container_path: str | None,
    force: bool,
) -> None:
    """Update a vault's properties.

    Changing the container_path requires --force as it will delete all documents,
    tasks, and chunks for this vault, requiring re-ingestion.
    """
    vault_update_command(
        ctx,
        name,
        description,
        host_path,
        container_path,
        force=force,
    )


@vault.command(name="delete")
@click.option("--name", required=True, help="Vault name to delete.")
@click.option(
    "--confirm",
    is_flag=True,
    help="Confirm vault deletion (required - deletes all associated data).",
)
@click.pass_context
def vault_delete(ctx: click.Context, *, name: str, confirm: bool) -> None:
    """Delete a vault and all associated data.

    This action is irreversible and will cascade-delete all documents,
    tasks, and chunks associated with the vault. Requires --confirm flag.
    """
    vault_delete_command(ctx, name, confirm=confirm)


def main() -> None:
    """Main entry point."""
    cli()


if __name__ == "__main__":
    main()
