"""CLI entry point for obsidian-rag."""

import logging
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

import click
from sqlalchemy.orm import Session

from obsidian_rag.config import Settings, get_settings
from obsidian_rag.database.engine import DatabaseManager
from obsidian_rag.database.models import Document, Task
from obsidian_rag.llm.base import EmbeddingProvider, ProviderFactory
from obsidian_rag.parsing.scanner import (
    process_files_in_batches,
    scan_markdown_files,
)
from obsidian_rag.services.ingestion import IngestionService

log = logging.getLogger(__name__)


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
            '{"timestamp": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s"}'
        )
    else:
        # Text format
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
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
    ctx: click.Context, verbose: bool, log_level: str | None, config_file: str | None
) -> None:
    """Obsidian RAG - CLI tool for ingesting Obsidian documents with vector search."""
    _msg = "Starting obsidian-rag CLI"
    log.debug(_msg)

    # Log config file path if provided (config loading is handled by get_settings)
    if config_file:
        _msg = f"Config file specified: {config_file}"
        log.debug(_msg)

    # Load settings
    settings_kwargs = {}

    # Determine logging level: --log-level takes precedence over --verbose
    effective_log_level = None
    if log_level:
        effective_log_level = log_level.upper()
    elif verbose:
        effective_log_level = "DEBUG"

    if effective_log_level:
        settings_kwargs["logging"] = {"level": effective_log_level}

    settings = get_settings(**settings_kwargs)

    # Setup logging
    _setup_logging(settings.logging.level, settings.logging.format)

    # Store settings in context
    ctx.ensure_object(dict)
    ctx.obj["settings"] = settings


def _get_embedding_provider(settings: Settings) -> EmbeddingProvider:
    """Create embedding provider from settings."""
    embedding_config = settings.get_endpoint_config("embedding")
    if embedding_config:
        return ProviderFactory.create_embedding_provider(
            embedding_config.provider,
            api_key=embedding_config.api_key,
            model=embedding_config.model,
            base_url=embedding_config.base_url,
        )
    else:
        _msg = "No embedding configuration found, using default OpenAI provider"
        log.warning(_msg)
        return ProviderFactory.create_embedding_provider("openai")


def _scan_vault(vault_path: Path) -> list:
    """Scan vault for markdown files."""
    try:
        return scan_markdown_files(vault_path)
    except Exception as e:
        _msg = f"Failed to scan vault: {e}"
        log.exception(_msg)
        click.echo(f"Error: {_msg}", err=True)
        import sys

        sys.exit(1)


def _create_progress_callback(verbose: bool):
    """Create progress callback function."""

    def callback(current: int, total: int, successes: int, errors: int) -> None:
        if verbose:
            click.echo(
                f"Progress: {current}/{total} files processed ({successes} successful, {errors} errors)"
            )

    return callback


def _report_ingest_results(
    total: int, stats: dict[str, int], elapsed_time: float
) -> None:
    """Report ingestion results."""
    click.echo(
        f"\nSuccessfully ingested {total} documents "
        f"({stats['new']} new, {stats['updated']} updated, {stats['unchanged']} unchanged)"
    )
    if stats["errors"] > 0:
        click.echo(f"Errors: {stats['errors']} files")
    click.echo(f"Completed in {elapsed_time:.1f} seconds")


@cli.command()
@click.argument("path", type=click.Path(exists=True))
@click.option(
    "--dry-run",
    is_flag=True,
    help="Show what would happen without writing to database.",
)
@click.option("--verbose", is_flag=True, help="Show detailed progress.")
@click.pass_context
def ingest(ctx: click.Context, path: str, dry_run: bool, verbose: bool) -> None:
    """Ingest documents from an Obsidian vault.

    PATH is the path to the Obsidian vault directory.
    """
    _msg = f"Starting ingestion from: {path}"
    log.info(_msg)

    settings = ctx.obj["settings"]
    vault_path = Path(path)
    db_manager = DatabaseManager(settings.database.url)

    if dry_run:
        click.echo("DRY RUN: No changes will be written to the database")

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
        progress_callback=_create_progress_callback(verbose),
    )

    embedding_provider = _get_embedding_provider(settings)
    ingestion_service = IngestionService(
        db_manager=db_manager,
        embedding_provider=embedding_provider,
        settings=settings,
    )

    result = ingestion_service.ingest_vault(
        vault_path=vault_path,
        file_infos=file_infos,
        dry_run=dry_run,
        progress_callback=_create_progress_callback(verbose),
    )

    elapsed_time = time.time() - start_time
    stats = {
        "new": result.new,
        "updated": result.updated,
        "unchanged": result.unchanged,
        "errors": result.errors,
    }
    _report_ingest_results(result.total, stats, elapsed_time)


def _format_query_results_json(results: list) -> str:
    """Format query results as JSON."""
    import json

    output = [
        {
            "file_path": str(doc.file_path),
            "file_name": doc.file_name,
            "distance": float(dist),
            "kind": doc.kind,
            "tags": doc.tags,
        }
        for doc, dist in results
    ]
    return json.dumps(output, indent=2)


def _format_query_results_table(results: list) -> str:
    """Format query results as table/text."""
    lines = [f"Found {len(results)} results:\n"]
    for doc, dist in results:
        lines.append(f"File: {doc.file_name}")
        lines.append(f"Path: {doc.file_path}")
        lines.append(f"Distance: {float(dist):.4f}")
        if doc.kind:
            lines.append(f"Kind: {doc.kind}")
        if doc.tags:
            lines.append(f"Tags: {', '.join(doc.tags)}")
        lines.append("")
    return "\n".join(lines)


def _search_documents(
    session: Session, query_embedding: list[float], limit: int
) -> list:
    """Search documents using semantic similarity."""
    return (
        session.query(
            Document,
            Document.content_vector.cosine_distance(query_embedding).label("distance"),
        )
        .filter(Document.content_vector.isnot(None))
        .order_by("distance")
        .limit(limit)
        .all()
    )


@cli.command()
@click.argument("query_text")
@click.option("--limit", default=10, help="Maximum number of results.")
@click.option(
    "--format", "output_format", type=click.Choice(["table", "json"]), default="table"
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
    "--due-before", type=str, help="Filter tasks due before date (YYYY-MM-DD)."
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
    """Build the tasks query with filters."""
    from datetime import datetime

    query = session.query(Task).join(Document)

    if status:
        query = query.filter(Task.status == status)

    if due_before:
        try:
            due_date = datetime.strptime(due_before, "%Y-%m-%d").date()  # noqa: DTZ007
            query = query.filter(Task.due <= due_date)
        except ValueError:
            click.echo("Error: Invalid date format. Use YYYY-MM-DD.", err=True)
            import sys

            sys.exit(1)

    if tag:
        query = query.filter(Task.tags.contains([tag]))

    query = query.order_by(Task.priority, Task.due)
    query = query.limit(limit)

    return query


def _format_task_results(results: list) -> str:
    """Format task results for display.

    Args:
        results: List of Task objects with joined Document.

    Returns:
        Formatted string with one task per line.

    """
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

    return "\n".join(lines)


def main() -> None:
    """Main entry point."""
    cli()


if __name__ == "__main__":
    main()
