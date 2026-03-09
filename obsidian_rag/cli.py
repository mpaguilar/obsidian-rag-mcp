"""CLI entry point for obsidian-rag."""

import json
import logging
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import partial
from pathlib import Path

import click
from sqlalchemy.orm import Session

from obsidian_rag.config import Settings, get_settings
from obsidian_rag.database.engine import DatabaseManager
from obsidian_rag.database.models import Document, Task
from obsidian_rag.llm.base import EmbeddingProvider
from obsidian_rag.llm.providers import ProviderFactory
from obsidian_rag.parsing.scanner import (
    process_files_in_batches,
    scan_markdown_files,
)
from obsidian_rag.services.ingestion import IngestionService, IngestVaultOptions

log = logging.getLogger(__name__)


@dataclass
class IngestOptions:
    """Options for the ingest command."""

    vault: str
    dry_run: bool
    no_delete: bool
    verbose: bool


def _setup_logging(level: str, format_type: str) -> None:
    """Configure logging.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        format_type: Format type ('text' or 'json').

    Raises:
        SystemExit: If the provided log level is invalid.

    """
    numeric_level = getattr(logging, level.upper(), None)
    if numeric_level is None:
        _msg = f"Invalid log level '{level}'. Valid levels: DEBUG, INFO, WARNING, ERROR, CRITICAL"
        click.echo(f"Error: {_msg}", err=True)
        sys.exit(1)

    if format_type == "json":
        # Simple JSON format
        formatter = logging.Formatter(
            '{"timestamp": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s"}',
        )
    else:
        # Text format
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    root_logger.handlers = []
    root_logger.addHandler(handler)


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
    _setup_logging(settings.logging.level, settings.logging.format)

    # Store settings in context
    ctx.ensure_object(dict)
    ctx.obj["settings"] = settings


def _get_embedding_provider(settings: Settings) -> EmbeddingProvider:
    """Create embedding provider from settings."""
    _msg = "_get_embedding_provider starting"
    log.debug(_msg)
    embedding_config = settings.get_endpoint_config("embedding")
    if embedding_config:
        result = ProviderFactory.create_embedding_provider(
            embedding_config.provider,  # type: ignore[call-overload]
            api_key=embedding_config.api_key,
            model=embedding_config.model,
            base_url=embedding_config.base_url,
        )
        _msg = "_get_embedding_provider returning"
        log.debug(_msg)
        return result
    _msg = "No embedding configuration found, using default OpenAI provider"
    log.warning(_msg)
    result = ProviderFactory.create_embedding_provider("openai")
    _msg = "_get_embedding_provider returning"
    log.debug(_msg)
    return result


def _scan_vault(vault_path: Path) -> list:
    """Scan vault for markdown files."""
    _msg = "_scan_vault starting"
    log.debug(_msg)
    try:
        result = scan_markdown_files(vault_path)
    except Exception as e:
        _msg = f"Failed to scan vault: {e}"
        log.exception(_msg)
        click.echo(f"Error: {_msg}", err=True)
        sys.exit(1)
    else:
        _msg = "_scan_vault returning"
        log.debug(_msg)
        return result


def progress_callback(
    current: int,
    total: int,
    successes: int,
    errors: int,
    *,
    verbose: bool = False,
) -> None:
    """Report progress during file processing.

    Args:
        current: Current file number being processed.
        total: Total number of files to process.
        successes: Number of successfully processed files.
        errors: Number of files that failed processing.
        verbose: Whether to print progress messages.

    """
    _msg = "progress_callback starting"
    log.debug(_msg)

    if verbose:
        _msg = f"Progress: {current}/{total} files processed ({successes} successful, {errors} errors)"
        click.echo(_msg)

    _msg = "progress_callback returning"
    log.debug(_msg)


def _report_ingest_results(
    total: int,
    stats: dict[str, int],
    elapsed_time: float,
    deleted: int,
    no_delete: bool,  # noqa: FBT001
) -> None:
    """Report ingestion results.

    Args:
        total: Total number of files processed.
        stats: Dictionary with 'new', 'updated', 'unchanged', 'errors' counts.
        elapsed_time: Time taken to process in seconds.
        deleted: Number of orphaned documents deleted.
        no_delete: Whether deletion was skipped.

    """
    _msg = "_report_ingest_results starting"
    log.debug(_msg)

    # Build the base message
    if no_delete:
        result_msg = (
            f"\nSuccessfully ingested {total} documents "
            f"({stats['new']} new, {stats['updated']} updated, "
            f"{stats['unchanged']} unchanged, {stats['errors']} errors, deletion skipped)"
        )
    else:
        result_msg = (
            f"\nSuccessfully ingested {total} documents "
            f"({stats['new']} new, {stats['updated']} updated, "
            f"{stats['unchanged']} unchanged, {stats['errors']} errors, {deleted} deleted)"
        )

    click.echo(result_msg)
    click.echo(f"Completed in {elapsed_time:.1f} seconds")
    _msg = "_report_ingest_results returning"
    log.debug(_msg)


def _validate_vault_config(settings: Settings, vault: str, path: str) -> None:
    """Validate vault configuration and path match.

    Args:
        settings: Application settings.
        vault: Vault name.
        path: Vault path.

    Raises:
        SystemExit: If validation fails.

    """
    vault_path = Path(path)
    vault_config = settings.get_vault(vault)

    if vault_config is None:
        available = settings.get_vault_names()
        _msg = (
            f"Vault '{vault}' not found in configuration. "
            f"Available vaults: {', '.join(available)}"
        )
        click.echo(f"Error: {_msg}", err=True)
        sys.exit(1)

    # Validate path matches vault container_path (with trailing slash normalization)
    normalized_input = vault_path.resolve()
    normalized_config = Path(vault_config.container_path).resolve()

    if normalized_input != normalized_config:
        _msg = (
            f"Path '{path}' does not match vault '{vault}' container path "
            f"'{vault_config.container_path}'. "
            "The path must match the configured container_path exactly."
        )
        click.echo(f"Error: {_msg}", err=True)
        sys.exit(1)


def _run_ingestion(
    settings: Settings,
    vault_path: Path,
    vault: str,
    options: IngestOptions,
) -> None:
    """Run the ingestion process.

    Args:
        settings: Application settings.
        vault_path: Path to vault.
        vault: Vault name.
        options: Ingestion options.

    """
    db_manager = DatabaseManager(settings.database.url)

    if options.dry_run:
        click.echo("DRY RUN: No changes will be written to the database")

    if options.no_delete:
        click.echo("Deletion phase skipped (--no-delete flag)")

    files = _scan_vault(vault_path)
    if not files:
        click.echo("No markdown files found in the specified path.")
        return

    click.echo(f"Found {len(files)} markdown files")

    start_time = time.time()
    file_infos = process_files_in_batches(
        files,
        batch_size=settings.ingestion.batch_size,
        progress_interval=settings.ingestion.progress_interval,
        progress_callback=partial(progress_callback, verbose=options.verbose),
    )

    embedding_provider = _get_embedding_provider(settings)
    ingestion_service = IngestionService(
        db_manager=db_manager,
        embedding_provider=embedding_provider,
        settings=settings,
    )

    ingest_options = IngestVaultOptions(
        vault=vault,
        file_infos=file_infos,
        dry_run=options.dry_run,
        progress_callback=partial(progress_callback, verbose=options.verbose),
        no_delete=options.no_delete,
    )
    result = ingestion_service.ingest_vault(vault_path, ingest_options)

    elapsed_time = time.time() - start_time
    stats = {
        "new": result.new,
        "updated": result.updated,
        "unchanged": result.unchanged,
        "errors": result.errors,
    }
    _report_ingest_results(
        result.total, stats, elapsed_time, result.deleted, options.no_delete
    )


@cli.command()
@click.argument("path", type=click.Path(exists=True))
@click.option(
    "--vault",
    required=True,
    help="Name of the vault to ingest into (as configured in config file).",
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
    path: str,
    *,
    vault: str,
    dry_run: bool,
    no_delete: bool,
    verbose: bool,
) -> None:
    """Ingest documents from an Obsidian vault.

    PATH is the path to the Obsidian vault directory. This must match the
    container_path configured for the specified vault.

    By default, documents that exist in the database but not on the filesystem
    will be deleted. Use --no-delete to preserve orphaned documents.
    """
    _msg = f"Starting ingestion from: {path}"
    log.info(_msg)

    ctx = click.get_current_context()
    settings = ctx.obj["settings"]
    vault_path = Path(path)
    options = IngestOptions(
        vault=vault,
        dry_run=dry_run,
        no_delete=no_delete,
        verbose=verbose,
    )

    _validate_vault_config(settings, vault, path)
    _run_ingestion(settings, vault_path, vault, options)


def _format_query_results_json(results: list) -> str:
    """Format query results as JSON."""
    _msg = "_format_query_results_json starting"
    log.debug(_msg)
    output = [
        {
            "file_path": str(doc.file_path),
            "file_name": doc.file_name,
            "distance": float(dist),
            "kind": doc.frontmatter_json.get("kind") if doc.frontmatter_json else None,
            "tags": doc.tags,
        }
        for doc, dist in results
    ]
    result = json.dumps(output, indent=2)
    _msg = "_format_query_results_json returning"
    log.debug(_msg)
    return result


def _format_query_results_table(results: list) -> str:
    """Format query results as table/text."""
    _msg = "_format_query_results_table starting"
    log.debug(_msg)
    lines = [f"Found {len(results)} results:\n"]
    for doc, dist in results:
        lines.append(f"File: {doc.file_name}")
        lines.append(f"Path: {doc.file_path}")
        lines.append(f"Distance: {float(dist):.4f}")
        kind = doc.frontmatter_json.get("kind") if doc.frontmatter_json else None
        if kind:
            lines.append(f"Kind: {kind}")
        if doc.tags:
            lines.append(f"Tags: {', '.join(doc.tags)}")
        lines.append("")
    result = "\n".join(lines)
    _msg = "_format_query_results_table returning"
    log.debug(_msg)
    return result


def _search_documents(
    session: Session,
    query_embedding: list[float],
    limit: int,
) -> list:
    """Search documents using semantic similarity.

    Args:
        session: Database session for queries.
        query_embedding: Vector embedding of the search query.
        limit: Maximum number of results to return.

    Returns:
        List of tuples containing (Document, distance) pairs.

    Notes:
        Performs database query using pgvector cosine distance.
        Requires content_vector to be populated for documents.

    """
    _msg = "_search_documents starting"
    log.debug(_msg)
    result = (
        session.query(
            Document,
            Document.content_vector.cosine_distance(query_embedding).label("distance"),
        )
        .filter(Document.content_vector.isnot(None))
        .order_by("distance")
        .limit(limit)
        .all()
    )
    _msg = "_search_documents returning"
    log.debug(_msg)
    return result


@cli.command()
@click.argument("query_text")
@click.option("--limit", default=10, help="Maximum number of results.")
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["table", "json"]),
    default="table",
)
@click.pass_context
def query(ctx: click.Context, query_text: str, limit: int, output_format: str) -> None:
    """Search documents using semantic similarity."""
    _msg = f"Searching for: {query_text}"
    log.info(_msg)

    settings = ctx.obj["settings"]
    db_manager = DatabaseManager(settings.database.url)
    embedding_provider = _get_embedding_provider(settings)

    # Generate query embedding
    try:
        query_embedding = embedding_provider.generate_embedding(query_text)
    except Exception as e:
        _msg = f"Failed to generate query embedding: {e}"
        log.exception(_msg)
        click.echo(f"Error: {_msg}", err=True)
        sys.exit(1)

    # Search documents
    with db_manager.get_session() as session:
        results = _search_documents(session, query_embedding, limit)

        if not results:
            click.echo("No matching documents found.")
            return

        if output_format == "json":
            click.echo(_format_query_results_json(results))
        else:
            click.echo(_format_query_results_table(results))


@cli.command()
@click.option(
    "--status",
    type=click.Choice(["completed", "not_completed", "in_progress", "cancelled"]),
)
@click.option(
    "--due-before",
    type=str,
    help="Filter tasks due before date (YYYY-MM-DD).",
)
@click.option("--tag", help="Filter by tag.")
@click.option("--limit", default=20, help="Maximum number of results.")
@click.pass_context
def tasks(
    ctx: click.Context,
    status: str | None,
    due_before: str | None,
    tag: str | None,
    limit: int,
) -> None:
    """Query and filter tasks."""
    _msg = "Querying tasks"
    log.info(_msg)

    settings = ctx.obj["settings"]
    db_manager = DatabaseManager(settings.database.url)

    with db_manager.get_session() as session:
        query = _build_tasks_query(session, status, due_before, tag, limit)
        results = query.all()

        if not results:
            click.echo("No tasks found matching the criteria.")
            return

        click.echo(_format_task_results(results))


def _build_tasks_query(
    session: Session,
    status: str | None,
    due_before: str | None,
    tag: str | None,
    limit: int,
):
    """Build the tasks query with filters.

    Args:
        session: Database session for queries.
        status: Filter by task status (optional).
        due_before: Filter by due date before this date (optional).
        tag: Filter by tag (optional).
        limit: Maximum number of results to return.

    Returns:
        SQLAlchemy query object for tasks.

    Notes:
        Performs database query with joins to documents table.
        Date parsing uses strptime with YYYY-MM-DD format.

    """
    _msg = "_build_tasks_query starting"
    log.debug(_msg)
    query = session.query(Task).join(Document)

    if status:
        query = query.filter(Task.status == status)

    if due_before:
        try:
            parsed = datetime.strptime(due_before, "%Y-%m-%d").replace(tzinfo=UTC)
            due_date = parsed.date()
            query = query.filter(Task.due <= due_date)
        except ValueError:
            click.echo("Error: Invalid date format. Use YYYY-MM-DD.", err=True)
            sys.exit(1)

    if tag:
        query = query.filter(Task.tags.contains([tag]))

    query = query.order_by(Task.priority, Task.due)
    query = query.limit(limit)
    _msg = "_build_tasks_query returning"
    log.debug(_msg)
    return query


def _format_task_results(results: list) -> str:
    """Format task results for display.

    Args:
        results: List of Task objects with joined Document.

    Returns:
        Formatted string with one task per line.

    """
    _msg = "_format_task_results starting"
    log.debug(_msg)
    lines = [f"Found {len(results)} tasks:\n"]
    status_indicator = {
        "completed": "[x]",
        "not_completed": "[ ]",
        "in_progress": "[/]",
        "cancelled": "[-]",
    }

    for task in results:
        indicator = status_indicator.get(task.status, "[ ]")
        # Build optional metadata parts
        parts = [f"File: {task.document.file_name}"]
        if task.due:
            parts.append(f"Due: {task.due}")
        if task.priority != "normal":
            parts.append(f"Priority: {task.priority}")
        if task.tags:
            parts.append(f"Tags: {', '.join(task.tags)}")

        # Combine into one line
        metadata = ", ".join(parts)
        lines.append(f"{indicator} {task.description} ({metadata})")

    result = "\n".join(lines)
    _msg = "_format_task_results returning"
    log.debug(_msg)
    return result


def main() -> None:
    """Main entry point."""
    cli()


if __name__ == "__main__":
    main()
