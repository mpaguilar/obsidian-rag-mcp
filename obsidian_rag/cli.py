"""CLI entry point for obsidian-rag."""

import logging
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import click
from sqlalchemy.orm import Session

from obsidian_rag.config import get_settings
from obsidian_rag.database.engine import DatabaseManager
from obsidian_rag.database.models import Document, Task
from obsidian_rag.llm.base import ProviderFactory
from obsidian_rag.parsing.frontmatter import parse_frontmatter
from obsidian_rag.parsing.scanner import (
    FileInfo,
    process_files_in_batches,
    scan_markdown_files,
)
from obsidian_rag.parsing.tasks import parse_tasks_from_content

if TYPE_CHECKING:
    pass

log = logging.getLogger(__name__)


def _setup_logging(level: str, format_type: str) -> None:
    """Configure logging.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR).
        format_type: Format type ('text' or 'json').

    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)

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

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    root_logger.handlers = []
    root_logger.addHandler(handler)


@click.group()
@click.option("--verbose", is_flag=True, help="Enable verbose output.")
@click.option("--config-file", type=click.Path(), help="Path to config file.")
@click.pass_context
def cli(ctx: click.Context, verbose: bool, config_file: str | None) -> None:
    """Obsidian RAG - CLI tool for ingesting Obsidian documents with vector search."""
    _msg = "Starting obsidian-rag CLI"
    log.debug(_msg)

    # Log config file path if provided (config loading is handled by get_settings)
    if config_file:
        _msg = f"Config file specified: {config_file}"
        log.debug(_msg)

    # Load settings
    settings_kwargs = {}
    if verbose:
        settings_kwargs["logging"] = {"level": "DEBUG"}

    settings = get_settings(**settings_kwargs)

    # Setup logging
    _setup_logging(settings.logging.level, settings.logging.format)

    # Store settings in context
    ctx.ensure_object(dict)
    ctx.obj["settings"] = settings


def _get_embedding_provider(settings: Any) -> Any:
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


def _update_stats(stats: dict[str, int], result: str) -> None:
    """Update statistics based on processing result."""
    if result in stats:
        stats[result] += 1


class ProcessingContext:
    """Context for file processing operations."""

    def __init__(
        self,
        db_manager: DatabaseManager,
        embedding_provider: Any,
        dry_run: bool,
        verbose: bool,
        stats: dict[str, int],
    ) -> None:
        self.db_manager = db_manager
        self.embedding_provider = embedding_provider
        self.dry_run = dry_run
        self.verbose = verbose
        self.stats = stats


def _process_single_file_safe(
    file_info: FileInfo,
    ctx: ProcessingContext,
) -> None:
    """Process a single file with error handling."""
    try:
        result = _process_single_file(
            db_manager=ctx.db_manager,
            file_info=file_info,
            embedding_provider=ctx.embedding_provider,
            dry_run=ctx.dry_run,
        )
        _update_stats(ctx.stats, result)
    except Exception as e:
        _msg = f"Error processing file {file_info.path}: {e}"
        log.exception(_msg)
        ctx.stats["errors"] += 1
        if ctx.verbose:
            click.echo(f"Warning: {file_info.name} - {e}", err=True)


def _process_files(
    file_infos: list[FileInfo],
    db_manager: DatabaseManager,
    embedding_provider: Any,
    dry_run: bool,
    verbose: bool,
) -> dict[str, int]:
    """Process all files and return statistics."""
    stats = {"new": 0, "updated": 0, "unchanged": 0, "errors": 0}
    ctx = ProcessingContext(db_manager, embedding_provider, dry_run, verbose, stats)

    for file_info in file_infos:
        _process_single_file_safe(file_info, ctx)

    return stats


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
    file_stats = _process_files(
        file_infos, db_manager, embedding_provider, dry_run, verbose
    )

    elapsed_time = time.time() - start_time
    _report_ingest_results(len(files), file_stats, elapsed_time)


def _process_single_file(
    db_manager: DatabaseManager,
    file_info: FileInfo,
    embedding_provider: Any,
    dry_run: bool,
) -> str:
    """Process a single file.

    Returns:
        'new', 'updated', 'unchanged', or raises exception.

    """
    # Parse frontmatter and content
    kind, tags, metadata, content = parse_frontmatter(file_info.content)

    # Parse tasks
    parsed_tasks = parse_tasks_from_content(content)

    if dry_run:
        return "new"  # Simulate new document in dry run

    with db_manager.get_session() as session:
        # Check if document exists
        existing = (
            session.query(Document).filter_by(file_path=str(file_info.path)).first()
        )

        if existing:
            # Check if document has changed
            if existing.checksum_md5 == file_info.checksum:
                return "unchanged"

            # Update existing document
            parsed_data = (kind, tags, metadata, content)
            _update_document(existing, file_info, parsed_data)
            _update_tasks(session, existing, parsed_tasks)
            result = "updated"
        else:
            # Create new document
            parsed_data = (kind, tags, metadata, content)
            document = _create_document(file_info, parsed_data, embedding_provider)
            session.add(document)
            session.flush()  # Get document ID
            _create_tasks(session, document, parsed_tasks)
            result = "new"

        return result


def _create_document(
    file_info: FileInfo,
    parsed_data: tuple[str | None, list[str] | None, dict[str, Any], str],
    embedding_provider: Any,
) -> Document:
    """Create a new Document instance."""
    kind, tags, metadata, content = parsed_data

    # Generate embedding
    try:
        embedding = embedding_provider.generate_embedding(content)
    except Exception as e:
        _msg = f"Failed to generate embedding: {e}"
        log.warning(_msg)
        embedding = None

    return Document(
        file_path=str(file_info.path),
        file_name=file_info.name,
        content=content,
        content_vector=embedding,
        checksum_md5=file_info.checksum,
        created_at_fs=file_info.created_at,
        modified_at_fs=file_info.modified_at,
        kind=kind,
        tags=tags,
        frontmatter_json=metadata,
    )


def _update_document(
    document: Document,
    file_info: FileInfo,
    parsed_data: tuple[str | None, list[str] | None, dict[str, Any], str],
) -> None:
    """Update an existing Document instance."""
    kind, tags, metadata, content = parsed_data

    document.content = content
    document.checksum_md5 = file_info.checksum
    document.modified_at_fs = file_info.modified_at
    document.kind = kind
    document.tags = tags
    document.frontmatter_json = metadata


def _create_tasks(
    session: Any,
    document: Document,
    parsed_tasks: list[tuple[int, Any]],
) -> None:
    """Create Task instances for a document."""
    for line_number, parsed_task in parsed_tasks:
        task = Task(
            document_id=document.id,
            line_number=line_number,
            raw_text=parsed_task.raw_text,
            status=parsed_task.status,
            description=parsed_task.description,
            tags=parsed_task.tags,
            repeat=parsed_task.repeat,
            scheduled=parsed_task.scheduled,
            due=parsed_task.due,
            completion=parsed_task.completion,
            priority=parsed_task.priority,
            custom_metadata=parsed_task.custom_metadata,
        )
        session.add(task)


def _update_tasks(
    session: Any,
    document: Document,
    parsed_tasks: list[tuple[int, Any]],
) -> None:
    """Update tasks for a document (delete old, create new)."""
    # Delete existing tasks
    session.query(Task).filter_by(document_id=document.id).delete()

    # Create new tasks
    _create_tasks(session, document, parsed_tasks)


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
    """Format task results for display."""
    lines = [f"Found {len(results)} tasks:\n"]
    status_emoji = {
        "completed": "[x]",
        "not_completed": "[ ]",
        "in_progress": "[/]",
        "cancelled": "[-]",
    }

    for task in results:
        emoji = status_emoji.get(task.status, "[ ]")
        lines.append(f"{emoji} {task.description}")
        lines.append(f"   File: {task.document.file_name}")
        if task.due:
            lines.append(f"   Due: {task.due}")
        if task.priority != "normal":
            lines.append(f"   Priority: {task.priority}")
        if task.tags:
            lines.append(f"   Tags: {', '.join(task.tags)}")
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    """Main entry point."""
    cli()


if __name__ == "__main__":
    main()
