"""CLI command implementations for obsidian-rag.

This module contains the actual implementations of CLI commands,
separated from cli.py to keep file size under 1000 lines.
"""

import json
import logging
import sys
import time
from dataclasses import dataclass
from datetime import date
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import click
from sqlalchemy import not_, or_
from sqlalchemy.orm import Session

if TYPE_CHECKING:
    from sqlalchemy.orm import Query

from obsidian_rag.cli_ingest import _resolve_ingest_path
from obsidian_rag.config import Settings
from obsidian_rag.database.engine import DatabaseManager
from obsidian_rag.database.models import Document, Task
from obsidian_rag.llm.base import EmbeddingProvider
from obsidian_rag.llm.providers import ProviderFactory
from obsidian_rag.mcp_server.tools.documents_chunks import (
    query_chunks,
    rerank_chunk_results,
)
from obsidian_rag.mcp_server.tools.tasks import _strip_tag_list
from obsidian_rag.parsing.scanner import (
    process_files_in_batches,
    scan_markdown_files,
)
from obsidian_rag.services.ingestion import IngestionService, IngestVaultOptions

log = logging.getLogger(__name__)


def setup_logging(level: str, format_type: str) -> None:
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

    # Set library logging to INFO for better log analysis
    logging.getLogger("httpx").setLevel(logging.INFO)
    logging.getLogger("httpcore").setLevel(logging.INFO)
    logging.getLogger("asyncio").setLevel(logging.INFO)
    logging.getLogger("openai").setLevel(logging.INFO)


@dataclass
class IngestOptions:
    """Options for the ingest command."""

    vault: str
    dry_run: bool
    no_delete: bool
    verbose: bool


@dataclass
class TaskDateFilters:
    """Date filter parameters for tasks command.

    Attributes:
        due_before: Filter tasks due on or before this date.
        due_after: Filter tasks due on or after this date.
        scheduled_before: Filter tasks scheduled on or before this date.
        scheduled_after: Filter tasks scheduled on or after this date.
        completion_before: Filter tasks completed on or before this date.
        completion_after: Filter tasks completed on or after this date.

    """

    due_before: date | None = None
    due_after: date | None = None
    scheduled_before: date | None = None
    scheduled_after: date | None = None
    completion_before: date | None = None
    completion_after: date | None = None


@dataclass
class TaskFilterOptions:
    """Combined filter options for tasks command.

    This dataclass bundles all filter options to comply with
    the 5 argument limit per function (PLR0913).

    Attributes:
        status: Filter by task status.
        include_tags: List of tags that tasks must have.
        exclude_tags: List of tags that tasks must NOT have.
        limit: Maximum number of results.
        date_filters: Date filter parameters.

    """

    status: str | None = None
    include_tags: list[str] | None = None
    exclude_tags: list[str] | None = None
    limit: int = 20
    date_filters: TaskDateFilters | None = None


def _get_embedding_provider(settings: Settings) -> EmbeddingProvider:
    """Create embedding provider from settings."""
    _msg = "_get_embedding_provider starting"
    log.debug(_msg)
    embedding_config = settings.get_endpoint_config("embedding")
    if embedding_config:
        result = ProviderFactory.create_embedding_provider(
            embedding_config.provider,
            config={
                "api_key": embedding_config.api_key,
                "model": embedding_config.model,
                "base_url": embedding_config.base_url,
            },
        )
        _msg = "_get_embedding_provider returning"
        log.debug(_msg)
        return result
    _msg = "No embedding configuration found, using default OpenAI provider"
    log.warning(_msg)
    result = ProviderFactory.create_embedding_provider("openai", config={})
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
    *,
    no_delete: bool,
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
    db_manager = DatabaseManager(
        settings.database.url,
        pool_size=settings.database.pool_size,
        max_overflow=settings.database.max_overflow,
        pool_timeout=settings.database.pool_timeout,
        pool_recycle=settings.database.pool_recycle,
    )

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
        result.total, stats, elapsed_time, result.deleted, no_delete=options.no_delete
    )


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


def _format_chunk_results_json(results: list) -> str:
    """Format chunk query results as JSON.

    Args:
        results: List of ChunkQueryResult objects.

    Returns:
        JSON string with chunk results.

    """
    _msg = "_format_chunk_results_json starting"
    log.debug(_msg)
    output = [
        {
            "chunk_id": r.chunk_id,
            "content": r.content,
            "document_name": r.document_name,
            "document_path": r.document_path,
            "vault_name": r.vault_name,
            "chunk_index": r.chunk_index,
            "total_chunks": r.total_chunks,
            "token_count": r.token_count,
            "chunk_type": r.chunk_type,
            "similarity_score": r.similarity_score,
            "rerank_score": r.rerank_score,
        }
        for r in results
    ]
    result = json.dumps(output, indent=2)
    _msg = "_format_chunk_results_json returning"
    log.debug(_msg)
    return result


def _format_chunk_results_table(results: list) -> str:
    """Format chunk query results as table/text.

    Args:
        results: List of ChunkQueryResult objects.

    Returns:
        Formatted string with chunk results.

    """
    _msg = "_format_chunk_results_table starting"
    log.debug(_msg)
    lines = [f"Found {len(results)} results:\n"]
    for r in results:
        lines.append(
            f"\n{r.document_name} (chunk {r.chunk_index + 1}/{r.total_chunks})"
        )
        lines.append(f"Path: {r.document_path}")
        lines.append(f"Vault: {r.vault_name}")
        if r.token_count:
            lines.append(f"Tokens: {r.token_count}")
        lines.append(f"Similarity: {r.similarity_score:.3f}")
        if r.rerank_score:
            lines.append(f"Rerank: {r.rerank_score:.3f}")
        lines.append(f"Content: {r.content[:200]}...")
        lines.append("")
    result = "\n".join(lines)
    _msg = "_format_chunk_results_table returning"
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
    return cast("list", result)


def _execute_chunk_search(
    session: "Session",
    query_embedding: list[float],
    query_text: str,
    vault: str | None,
    limit: int,
    *,
    rerank: bool,
    flashrank_model: str,
    flashrank_top_k: int,
) -> list:
    """Execute chunk-level search with optional reranking.

    Args:
        session: Database session.
        query_embedding: Query embedding vector.
        query_text: Original query text for reranking.
        vault: Optional vault filter.
        limit: Maximum results.
        rerank: Whether to apply reranking.
        flashrank_model: Model for reranking.
        flashrank_top_k: Top-k for reranking.

    Returns:
        List of chunk results.

    """
    chunk_results = query_chunks(
        session,
        query_embedding,
        vault_name=vault,
        limit=limit,
    )

    if rerank and chunk_results:
        chunk_results = rerank_chunk_results(
            query_text,
            chunk_results,
            flashrank_model,
            128,
            flashrank_top_k,
        )

    return chunk_results


def _execute_document_search(
    session: "Session",
    query_embedding: list[float],
    limit: int,
) -> list:
    """Execute document-level search.

    Args:
        session: Database session.
        query_embedding: Query embedding vector.
        limit: Maximum results.

    Returns:
        List of document results.

    """
    return _search_documents(session, query_embedding, limit)


def _display_chunk_results(
    chunk_results: list,
    output_format: str,
) -> None:
    """Display chunk search results.

    Args:
        chunk_results: List of chunk results.
        output_format: Output format (json or table).

    """
    if not chunk_results:
        click.echo("No matching chunks found.")
        return

    if output_format == "json":
        click.echo(_format_chunk_results_json(chunk_results))
    else:
        click.echo(_format_chunk_results_table(chunk_results))


def _display_document_results(
    results: list,
    output_format: str,
) -> None:
    """Display document search results.

    Args:
        results: List of document results.
        output_format: Output format (json or table).

    """
    if not results:
        click.echo("No matching documents found.")
        return

    if output_format == "json":
        click.echo(_format_query_results_json(results))
    else:
        click.echo(_format_query_results_table(results))


def _apply_status_filter_cli(query: "Query[Any]", status: str | None) -> "Query[Any]":
    """Apply status filter to CLI tasks query.

    Args:
        query: The base query to filter.
        status: Status to filter by.

    Returns:
        Query with status filter applied.

    """
    if status:
        return query.filter(Task.status == status)
    return query


def _apply_due_date_filters_cli(
    query: "Query[Any]",
    due_before: date | None,
    due_after: date | None,
) -> "Query[Any]":
    """Apply due date filters to CLI tasks query.

    Args:
        query: The base query to filter.
        due_before: Filter tasks due on or before this date.
        due_after: Filter tasks due on or after this date.

    Returns:
        Query with due date filters applied.

    """
    if due_before is not None:
        query = query.filter(Task.due <= due_before)
    if due_after is not None:
        query = query.filter(Task.due >= due_after)
    return query


def _apply_scheduled_date_filters_cli(
    query: "Query[Any]",
    scheduled_before: date | None,
    scheduled_after: date | None,
) -> "Query[Any]":
    """Apply scheduled date filters to CLI tasks query.

    Args:
        query: The base query to filter.
        scheduled_before: Filter tasks scheduled on or before this date.
        scheduled_after: Filter tasks scheduled on or after this date.

    Returns:
        Query with scheduled date filters applied.

    """
    if scheduled_before is not None:
        query = query.filter(Task.scheduled <= scheduled_before)
    if scheduled_after is not None:
        query = query.filter(Task.scheduled >= scheduled_after)
    return query


def _apply_completion_date_filters_cli(
    query: "Query[Any]",
    completion_before: date | None,
    completion_after: date | None,
) -> "Query[Any]":
    """Apply completion date filters to CLI tasks query.

    Args:
        query: The base query to filter.
        completion_before: Filter tasks completed on or before this date.
        completion_after: Filter tasks completed on or after this date.

    Returns:
        Query with completion date filters applied.

    """
    if completion_before is not None:
        query = query.filter(Task.completion <= completion_before)
    if completion_after is not None:
        query = query.filter(Task.completion >= completion_after)
    return query


def _apply_include_tags_cli(
    query: "Query[Any]", include_tags: list[str] | None
) -> "Query[Any]":
    """Apply include tag filter to CLI tasks query.

    Args:
        query: The base query to filter.
        include_tags: List of tags that tasks must have.

    Returns:
        Query with include tag filter applied.

    """
    if not include_tags:
        return query
    stripped = _strip_tag_list(include_tags)
    if not stripped:  # pragma: no cover - defensive edge case
        return query
    for tag in stripped:
        query = query.filter(Task.tags.contains([tag.lower()]))
    return query


def _apply_exclude_tags_cli(
    query: "Query[Any]", exclude_tags: list[str] | None
) -> "Query[Any]":
    """Apply exclude tag filter to CLI tasks query.

    Args:
        query: The base query to filter.
        exclude_tags: List of tags that tasks must NOT have.

    Returns:
        Query with exclude tag filter applied.

    """
    if not exclude_tags:
        return query
    stripped = _strip_tag_list(exclude_tags)
    if not stripped:  # pragma: no cover - defensive edge case
        return query
    # Exclude tasks that have ANY of the excluded tags
    conditions = [Task.tags.contains([tag.lower()]) for tag in stripped]
    return query.filter(not_(or_(*conditions)))


def _build_tasks_query(
    session: Session,
    status: str | None,
    date_filters: TaskDateFilters,
    include_tags: list[str] | None,
    exclude_tags: list[str] | None,
    limit: int,
) -> "Query":
    """Build the tasks query with filters.

    Args:
        session: Database session for queries.
        status: Filter by task status (optional).
        date_filters: Date filter parameters.
        include_tags: List of tags that tasks must have.
        exclude_tags: List of tags that tasks must NOT have.
        limit: Maximum number of results to return.

    Returns:
        SQLAlchemy query object for tasks.

    Notes:
        Performs database query with joins to documents table.
        Multiple filters are combined with AND logic.
        Date comparisons are inclusive (>= for after, <= for before).

    """
    _msg = "_build_tasks_query starting"
    log.debug(_msg)
    query = session.query(Task).join(Document)

    # Apply all filters
    query = _apply_status_filter_cli(query, status)
    query = _apply_due_date_filters_cli(
        query, date_filters.due_before, date_filters.due_after
    )
    query = _apply_scheduled_date_filters_cli(
        query, date_filters.scheduled_before, date_filters.scheduled_after
    )
    query = _apply_completion_date_filters_cli(
        query, date_filters.completion_before, date_filters.completion_after
    )
    query = _apply_include_tags_cli(query, include_tags)
    query = _apply_exclude_tags_cli(query, exclude_tags)

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


def _execute_tasks_query(ctx: click.Context, options: TaskFilterOptions) -> None:
    """Execute the tasks query with the given filter options.

    Args:
        ctx: Click context.
        options: Filter options for the query.

    """
    settings = ctx.obj["settings"]
    db_manager = DatabaseManager(
        settings.database.url,
        pool_size=settings.database.pool_size,
        max_overflow=settings.database.max_overflow,
        pool_timeout=settings.database.pool_timeout,
        pool_recycle=settings.database.pool_recycle,
    )

    with db_manager.get_session() as session:
        query = _build_tasks_query(
            session=session,
            status=options.status,
            date_filters=options.date_filters or TaskDateFilters(),
            include_tags=options.include_tags,
            exclude_tags=options.exclude_tags,
            limit=options.limit,
        )
        results = query.all()

        if not results:
            click.echo("No tasks found matching the criteria.")
            return

        click.echo(_format_task_results(results))


def _run_query_command(
    ctx: click.Context,
    query_text: str,
    *,
    limit: int,
    output_format: str,
    vault: str | None,
    chunks: bool,
    rerank: bool,
) -> None:
    """Execute the query command.

    Args:
        ctx: Click context.
        query_text: Search query text.
        limit: Maximum number of results.
        output_format: Output format (table or json).
        vault: Optional vault name filter.
        chunks: Whether to search chunks instead of documents.
        rerank: Whether to rerank chunk results.

    """
    settings = ctx.obj["settings"]
    db_manager = DatabaseManager(
        settings.database.url,
        pool_size=settings.database.pool_size,
        max_overflow=settings.database.max_overflow,
        pool_timeout=settings.database.pool_timeout,
        pool_recycle=settings.database.pool_recycle,
    )
    embedding_provider = _get_embedding_provider(settings)

    try:
        query_embedding = embedding_provider.generate_embedding(query_text)
    except Exception as e:
        _msg = f"Failed to generate query embedding: {e}"
        log.exception(_msg)
        click.echo(f"Error: {_msg}", err=True)
        sys.exit(1)

    with db_manager.get_session() as session:
        if chunks:
            chunk_results = _execute_chunk_search(
                session,
                query_embedding,
                query_text,
                vault,
                limit,
                rerank=rerank,
                flashrank_model=settings.chunking.flashrank_model,
                flashrank_top_k=settings.chunking.flashrank_top_k,
            )
            _display_chunk_results(chunk_results, output_format)
        else:
            results = _execute_document_search(session, query_embedding, limit)
            _display_document_results(results, output_format)


def _run_ingest_command(
    ctx: click.Context,
    path: str | None,
    *,
    vault: str,
    dry_run: bool,
    no_delete: bool,
    verbose: bool,
) -> None:
    """Execute the ingest command.

    Args:
        ctx: Click context.
        path: Path to the vault directory, or None to use vault config.
        vault: Vault name.
        dry_run: Whether to perform a dry run.
        no_delete: Whether to skip deletion.
        verbose: Whether to show verbose output.

    """
    settings = ctx.obj["settings"]
    resolved_path = _resolve_ingest_path(settings, path, vault)
    vault_path = Path(resolved_path)
    options = IngestOptions(
        vault=vault,
        dry_run=dry_run,
        no_delete=no_delete,
        verbose=verbose,
    )

    _run_ingestion(settings, vault_path, vault, options)
