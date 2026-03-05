"""Ingestion service for processing markdown documents."""

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from sqlalchemy.orm import Session

from obsidian_rag.database.engine import DatabaseManager
from obsidian_rag.database.models import Document, Task
from obsidian_rag.llm.base import EmbeddingProvider
from obsidian_rag.parsing.frontmatter import parse_frontmatter
from obsidian_rag.parsing.scanner import (
    FileInfo,
    process_files_in_batches,
    scan_markdown_files,
)
from obsidian_rag.parsing.tasks import parse_tasks_from_content

if TYPE_CHECKING:
    from obsidian_rag.config import Settings
    from obsidian_rag.parsing.tasks import ParsedTask

log = logging.getLogger(__name__)


@dataclass
class IngestionResult:
    """Result of an ingestion operation.

    Attributes:
        total: Total number of files processed.
        new: Number of new documents created.
        updated: Number of existing documents updated.
        unchanged: Number of unchanged documents.
        errors: Number of files that failed processing.
        processing_time_seconds: Time taken to process all files.
        message: Human-readable summary message.

    """

    total: int
    new: int
    updated: int
    unchanged: int
    errors: int
    processing_time_seconds: float
    message: str

    def to_dict(self) -> dict[str, object]:
        """Convert result to dictionary."""
        return {
            "total": self.total,
            "new": self.new,
            "updated": self.updated,
            "unchanged": self.unchanged,
            "errors": self.errors,
            "processing_time_seconds": self.processing_time_seconds,
            "message": self.message,
        }


class IngestionService:
    """Service for ingesting markdown documents into the database.

    This service provides a reusable interface for document ingestion
    that can be used by both CLI and MCP server.

    Attributes:
        db_manager: Database manager for sessions.
        embedding_provider: Provider for generating embeddings (optional).
        settings: Application settings.

    """

    def __init__(
        self,
        db_manager: DatabaseManager,
        embedding_provider: EmbeddingProvider | None,
        settings: "Settings",
    ) -> None:
        """Initialize the ingestion service.

        Args:
            db_manager: Database manager for sessions.
            embedding_provider: Provider for generating embeddings (optional).
            settings: Application settings.

        """
        _msg = "IngestionService initializing"
        log.debug(_msg)

        self.db_manager = db_manager
        self.embedding_provider = embedding_provider
        self.settings = settings

        _msg = "IngestionService initialized"
        log.debug(_msg)

    def _get_file_info_list(
        self,
        vault_path: Path,
        file_infos: list[FileInfo] | None,
        progress_callback: Callable[[int, int, int, int], None] | None,
    ) -> tuple[list[FileInfo], int]:
        """Get list of FileInfo objects to process.

        Args:
            vault_path: Path to the vault directory.
            file_infos: Optional pre-scanned file info objects.
            progress_callback: Optional progress callback.

        Returns:
            Tuple of (file_info_list, total_count).

        """
        if file_infos is not None:
            _msg = f"Using provided file_infos with {len(file_infos)} files"
            log.debug(_msg)
            return file_infos, len(file_infos)

        _msg = "Scanning for markdown files"
        log.debug(_msg)
        files = scan_markdown_files(vault_path)
        total = len(files)

        if total == 0:
            return [], 0

        _msg = f"Found {total} markdown files to process"
        log.info(_msg)

        file_info_list = list(
            process_files_in_batches(
                files,
                batch_size=self.settings.ingestion.batch_size,
                progress_interval=self.settings.ingestion.progress_interval,
                progress_callback=progress_callback,
            )
        )

        return file_info_list, total

    def _process_files_with_stats(
        self,
        file_info_list: list[FileInfo],
        dry_run: bool,
        progress_callback: Callable[[int, int, int, int], None] | None,
    ) -> dict[str, int]:
        """Process files and collect statistics.

        Args:
            file_info_list: List of FileInfo objects to process.
            dry_run: If True, don't write to database.
            progress_callback: Optional progress callback.

        Returns:
            Dictionary with statistics.

        """
        stats = {"new": 0, "updated": 0, "unchanged": 0, "errors": 0}
        total = len(file_info_list)

        for idx, file_info in enumerate(file_info_list):
            try:
                result = self._ingest_single_file(file_info, dry_run=dry_run)
                stats[result] += 1
            except Exception as e:
                _msg = f"Error processing file {file_info.path}: {e}"
                log.exception(_msg)
                stats["errors"] += 1

            if progress_callback:
                successes = stats["new"] + stats["updated"] + stats["unchanged"]
                progress_callback(idx + 1, total, successes, stats["errors"])

        return stats

    def ingest_vault(
        self,
        vault_path: Path,
        dry_run: bool = False,
        progress_callback: Callable[[int, int, int, int], None] | None = None,
        file_infos: list[FileInfo] | None = None,
    ) -> IngestionResult:
        """Ingest all markdown files from a vault directory.

        Args:
            vault_path: Path to the vault directory.
            dry_run: If True, don't write to database (default: False).
            progress_callback: Optional callback(current, total, successes, errors).
            file_infos: Optional pre-scanned file info objects. If provided,
                vault_path is used only for reference and file_infos are processed directly.

        Returns:
            IngestionResult with statistics about the operation.

        Raises:
            ValueError: If vault_path is invalid.

        """
        _msg = f"ingest_vault starting for path: {vault_path}"
        log.info(_msg)

        start_time = time.time()

        file_info_list, total = self._get_file_info_list(
            vault_path, file_infos, progress_callback
        )

        if total == 0:
            elapsed = time.time() - start_time
            return IngestionResult(
                total=0,
                new=0,
                updated=0,
                unchanged=0,
                errors=0,
                processing_time_seconds=elapsed,
                message="No markdown files found in directory",
            )

        stats = self._process_files_with_stats(
            file_info_list, dry_run, progress_callback
        )

        elapsed_time = time.time() - start_time

        result = IngestionResult(
            total=total,
            new=stats["new"],
            updated=stats["updated"],
            unchanged=stats["unchanged"],
            errors=stats["errors"],
            processing_time_seconds=elapsed_time,
            message=(
                f"Ingested {total} files: {stats['new']} new, "
                f"{stats['updated']} updated, {stats['unchanged']} unchanged, "
                f"{stats['errors']} errors"
            ),
        )

        _msg = f"ingest_vault completed: {result.message}"
        log.info(_msg)

        return result

    def _ingest_single_file(
        self,
        file_info: FileInfo,
        dry_run: bool = False,
    ) -> str:
        """Ingest a single file.

        Args:
            file_info: File information including path, content, checksum.
            dry_run: If True, don't write to database.

        Returns:
            Status string: 'new', 'updated', 'unchanged', or raises exception.

        Raises:
            Exception: If processing fails.

        """
        _msg = f"_ingest_single_file starting for: {file_info.path}"
        log.debug(_msg)

        # Parse frontmatter and content
        kind, tags, metadata, content = parse_frontmatter(file_info.content)

        # Parse tasks
        parsed_tasks = parse_tasks_from_content(content)

        if dry_run:
            _msg = "Dry run mode - simulating new document"
            log.debug(_msg)
            return "new"

        # Process in a single transaction for this file
        with self.db_manager.get_session() as session:
            # Check if document exists
            existing = (
                session.query(Document).filter_by(file_path=str(file_info.path)).first()
            )

            if existing:
                # Check if document has changed
                if existing.checksum_md5 == file_info.checksum:
                    _msg = "Document unchanged - skipping"
                    log.debug(_msg)
                    return "unchanged"

                # Update existing document
                _msg = "Updating existing document"
                log.debug(_msg)
                parsed_data = (kind, tags, metadata, content)
                self._update_document(existing, file_info, parsed_data)
                self._update_tasks(session, existing, parsed_tasks)
                result = "updated"
            else:
                # Create new document
                _msg = "Creating new document"
                log.debug(_msg)
                parsed_data = (kind, tags, metadata, content)
                document = self._create_document(file_info, parsed_data)
                session.add(document)
                session.flush()  # Get document ID
                self._create_tasks(session, document, parsed_tasks)
                result = "new"

        _msg = f"_ingest_single_file returning: {result}"
        log.debug(_msg)

        return result

    def _create_document(
        self,
        file_info: FileInfo,
        parsed_data: tuple[str | None, list[str] | None, dict[str, Any], str],
    ) -> Document:
        """Create a new Document instance.

        Args:
            file_info: File information.
            parsed_data: Tuple of (kind, tags, metadata, content).

        Returns:
            New Document instance.

        """
        _msg = "_create_document starting"
        log.debug(_msg)

        kind, tags, metadata, content = parsed_data

        # Generate embedding
        embedding = None
        if self.embedding_provider:
            try:
                _msg = "Generating embedding for document"
                log.debug(_msg)
                embedding = self.embedding_provider.generate_embedding(content)
            except Exception as e:
                _msg = f"Failed to generate embedding: {e}"
                log.warning(_msg)
                embedding = None

        document = Document(
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

        _msg = "_create_document returning"
        log.debug(_msg)

        return document

    def _update_document(
        self,
        document: Document,
        file_info: FileInfo,
        parsed_data: tuple[str | None, list[str] | None, dict[str, Any], str],
    ) -> None:
        """Update an existing Document instance.

        Args:
            document: Existing document to update.
            file_info: File information.
            parsed_data: Tuple of (kind, tags, metadata, content).

        """
        _msg = "_update_document starting"
        log.debug(_msg)

        kind, tags, metadata, content = parsed_data

        document.content = content
        document.checksum_md5 = file_info.checksum
        document.modified_at_fs = file_info.modified_at
        document.kind = kind
        document.tags = tags
        document.frontmatter_json = metadata

        _msg = "_update_document completed"
        log.debug(_msg)

    def _create_tasks(
        self,
        session: Session,
        document: Document,
        parsed_tasks: list[tuple[int, "ParsedTask"]],
    ) -> None:
        """Create Task instances for a document.

        Args:
            session: Database session.
            document: Parent document.
            parsed_tasks: List of (line_number, ParsedTask) tuples.

        """
        _msg = f"_create_tasks starting for {len(parsed_tasks)} tasks"
        log.debug(_msg)

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

        _msg = "_create_tasks completed"
        log.debug(_msg)

    def _update_tasks(
        self,
        session: Session,
        document: Document,
        parsed_tasks: list[tuple[int, "ParsedTask"]],
    ) -> None:
        """Update tasks for a document (delete old, create new).

        Args:
            session: Database session.
            document: Parent document.
            parsed_tasks: List of (line_number, ParsedTask) tuples.

        """
        _msg = "_update_tasks starting"
        log.debug(_msg)

        # Delete existing tasks
        _msg = "Deleting existing tasks"
        log.debug(_msg)
        session.query(Task).filter_by(document_id=document.id).delete()

        # Create new tasks
        self._create_tasks(session, document, parsed_tasks)

        _msg = "_update_tasks completed"
        log.debug(_msg)
