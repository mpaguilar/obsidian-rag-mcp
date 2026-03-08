"""Ingestion service for processing markdown documents."""

import logging
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from sqlalchemy.orm import Session

from obsidian_rag.config import VaultConfig
from obsidian_rag.database.engine import DatabaseManager
from obsidian_rag.database.models import Document, Task, Vault
from obsidian_rag.llm.base import EmbeddingError, EmbeddingProvider
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
class IngestVaultOptions:
    """Options for ingest_vault method.

    Attributes:
        vault: Vault configuration or vault name.
        dry_run: If True, don't write to database.
        progress_callback: Optional callback for progress updates.
        file_infos: Optional pre-scanned file info objects.
        no_delete: If True, skip deletion of orphaned documents.

    """

    vault: VaultConfig | str
    dry_run: bool = False
    progress_callback: Callable[[int, int, int, int], None] | None = None
    file_infos: list[FileInfo] | None = None
    no_delete: bool = False


@dataclass
class IngestionResult:
    """Result of an ingestion operation.

    Attributes:
        total: Total number of files processed.
        new: Number of new documents created.
        updated: Number of existing documents updated.
        unchanged: Number of unchanged documents.
        errors: Number of files that failed processing.
        deleted: Number of orphaned documents deleted from database.
        processing_time_seconds: Time taken to process all files.
        message: Human-readable summary message.

    """

    total: int
    new: int
    updated: int
    unchanged: int
    errors: int
    deleted: int
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
            "deleted": self.deleted,
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

    def _resolve_vault_config(self, vault: VaultConfig | str) -> VaultConfig:
        """Resolve vault configuration from name or config object.

        Args:
            vault: VaultConfig instance or vault name string.

        Returns:
            VaultConfig instance.

        Raises:
            ValueError: If vault name not found in configuration.

        """
        _msg = "_resolve_vault_config starting"
        log.debug(_msg)

        if isinstance(vault, VaultConfig):
            _msg = "_resolve_vault_config returning VaultConfig instance"
            log.debug(_msg)
            return vault

        # Look up vault by name
        vault_config = self.settings.get_vault(vault)
        if vault_config is None:
            available = self.settings.get_vault_names()
            _msg = (
                f"Vault '{vault}' not found. Available vaults: {', '.join(available)}"
            )
            raise ValueError(_msg)

        _msg = f"_resolve_vault_config returning config for vault: {vault}"
        log.debug(_msg)
        return vault_config

    def _get_or_create_vault(
        self,
        vault_config: VaultConfig,
        *,
        dry_run: bool = False,
    ) -> uuid.UUID | None:
        """Get existing vault record or create new one.

        Args:
            vault_config: Vault configuration.
            dry_run: If True, don't write to database.

        Returns:
            Vault ID (UUID) or None if dry_run.

        """
        _msg = "_get_or_create_vault starting"
        log.debug(_msg)

        if dry_run:
            _msg = "_get_or_create_vault returning None (dry_run)"
            log.debug(_msg)
            return None

        # Find vault by container_path (unique identifier)
        with self.db_manager.get_session() as session:
            vault = (
                session.query(Vault)
                .filter_by(container_path=vault_config.container_path)
                .first()
            )

            if vault is None:
                # Create new vault record
                _msg = f"Creating new vault record for: {vault_config.container_path}"
                log.info(_msg)

                # Find vault name from config
                vault_name = "Unknown"
                for name, config in self.settings.vaults.items():
                    if config.container_path == vault_config.container_path:
                        vault_name = name
                        break

                vault = Vault(
                    name=vault_name,
                    description=vault_config.description,
                    container_path=vault_config.container_path,
                    host_path=vault_config.host_path or vault_config.container_path,
                )
                session.add(vault)
                session.commit()
                vault_id = vault.id
                _msg = f"Created vault record with ID: {vault_id}"
                log.info(_msg)
            else:
                vault_id = vault.id
                _msg = f"Found existing vault record: {vault_id}"
                log.debug(_msg)

            return vault_id

    def _compute_relative_path(self, file_path: Path, container_path: str) -> str:
        """Compute relative path from file to vault root.

        Args:
            file_path: Absolute path to the file.
            container_path: Vault container path.

        Returns:
            Relative path using forward slashes.

        """
        _msg = "_compute_relative_path starting"
        log.debug(_msg)

        container = Path(container_path).resolve()
        resolved_file = file_path.resolve()

        try:
            relative = resolved_file.relative_to(container)
            # Always use forward slashes for cross-platform consistency
            result = str(relative).replace("\\", "/")
        except ValueError:
            # File is not under container path - should not happen if validated
            result = str(file_path).replace("\\", "/")

        _msg = f"_compute_relative_path returning: {result}"
        log.debug(_msg)
        return result

    def _validate_files_in_vault(
        self,
        file_info_list: list[FileInfo],
        vault_config: VaultConfig,
    ) -> None:
        """Validate all files are within the vault container path.

        Args:
            file_info_list: List of file info objects.
            vault_config: Vault configuration.

        Raises:
            ValueError: If any file is outside the vault.

        """
        _msg = "_validate_files_in_vault starting"
        log.debug(_msg)

        container = Path(vault_config.container_path).resolve()

        for file_info in file_info_list:
            file_path = file_info.path.resolve()

            # Check for path traversal
            if ".." in str(file_info.path):
                _msg = f"Path traversal detected: {file_info.path}"
                raise ValueError(_msg)

            # Check file is within vault
            try:
                file_path.relative_to(container)
            except ValueError as _err:
                _msg = (
                    f"File {file_info.path} is outside vault container path "
                    f"{vault_config.container_path}. All files must be within the vault directory."
                )
                raise ValueError(_msg) from _err

        _msg = f"_validate_files_in_vault completed for {len(file_info_list)} files"
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
            ),
        )

        return file_info_list, total

    def _process_files_with_stats(
        self,
        file_info_list: list[FileInfo],
        *,
        vault_id: uuid.UUID | None,
        vault_config: VaultConfig,
        dry_run: bool,
        progress_callback: Callable[[int, int, int, int], None] | None,
    ) -> dict[str, int]:
        """Process files and collect statistics.

        Args:
            file_info_list: List of FileInfo objects to process.
            vault_id: Vault ID (UUID) or None if dry_run.
            vault_config: Vault configuration.
            dry_run: If True, don't write to database.
            progress_callback: Optional progress callback.

        Returns:
            Dictionary with statistics.

        """
        stats = {"new": 0, "updated": 0, "unchanged": 0, "errors": 0}
        total = len(file_info_list)

        for idx, file_info in enumerate(file_info_list):
            try:
                result = self._ingest_single_file(
                    file_info,
                    vault_id=vault_id,
                    vault_config=vault_config,
                    dry_run=dry_run,
                )
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
        options: IngestVaultOptions,
    ) -> IngestionResult:
        """Ingest all markdown files from a vault directory.

        Args:
            vault_path: Path to the vault directory.
            options: Ingestion options including vault config, dry_run, etc.

        Returns:
            IngestionResult with statistics about the operation.

        Raises:
            ValueError: If vault_path is invalid or files are outside vault.

        Notes:
            Performs database operations for document and task creation/update.
            Uses file checksums to detect changes and avoid unnecessary updates.
            Each file is processed in its own database transaction.
            Orphaned documents (in DB but not on filesystem) are deleted by default.
            Files outside the vault container_path are rejected.

        """
        _msg = f"ingest_vault starting for path: {vault_path}"
        log.info(_msg)

        # Resolve vault configuration
        vault_config = self._resolve_vault_config(options.vault)
        vault_id = self._get_or_create_vault(vault_config, dry_run=options.dry_run)

        start_time = time.time()

        file_info_list, total = self._get_file_info_list(
            vault_path,
            options.file_infos,
            options.progress_callback,
        )

        if total == 0:
            elapsed = time.time() - start_time
            return IngestionResult(
                total=0,
                new=0,
                updated=0,
                unchanged=0,
                errors=0,
                deleted=0,
                processing_time_seconds=elapsed,
                message="No markdown files found in directory",
            )

        # Validate all files are within vault
        self._validate_files_in_vault(file_info_list, vault_config)

        stats = self._process_files_with_stats(
            file_info_list,
            vault_id=vault_id,
            vault_config=vault_config,
            dry_run=options.dry_run,
            progress_callback=options.progress_callback,
        )

        # Collect filesystem paths for deletion detection (relative paths)
        filesystem_paths = {
            self._compute_relative_path(fi.path, vault_config.container_path)
            for fi in file_info_list
        }

        # Delete orphaned documents unless no_delete flag is set
        deleted_count = 0
        deletion_errors = 0
        if not options.no_delete and vault_id is not None:
            deleted_count, deletion_errors = self._delete_orphaned_documents(
                filesystem_paths,
                vault_id=vault_id,
                dry_run=options.dry_run,
            )
        else:
            _msg = "Deletion phase skipped (no_delete=True or dry_run)"
            log.info(_msg)

        elapsed_time = time.time() - start_time

        # Build message based on whether deletion was skipped
        if options.no_delete:
            message = (
                f"Ingested {total} files: {stats['new']} new, "
                f"{stats['updated']} updated, {stats['unchanged']} unchanged, "
                f"{stats['errors']} errors, deletion skipped"
            )
        else:
            message = (
                f"Ingested {total} files: {stats['new']} new, "
                f"{stats['updated']} updated, {stats['unchanged']} unchanged, "
                f"{stats['errors']} errors, {deleted_count} deleted"
            )

        result = IngestionResult(
            total=total,
            new=stats["new"],
            updated=stats["updated"],
            unchanged=stats["unchanged"],
            errors=stats["errors"] + deletion_errors,
            deleted=deleted_count,
            processing_time_seconds=elapsed_time,
            message=message,
        )

        _msg = f"ingest_vault completed: {result.message}"
        log.info(_msg)

        return result

    def _ingest_single_file(
        self,
        file_info: FileInfo,
        *,
        vault_id: uuid.UUID | None,
        vault_config: VaultConfig,
        dry_run: bool = False,
    ) -> str:
        """Ingest a single file.

        Args:
            file_info: File information including path, content, checksum.
            vault_id: Vault ID (UUID) or None if dry_run.
            vault_config: Vault configuration.
            dry_run: If True, don't write to database.

        Returns:
            Status string: 'new', 'updated', 'unchanged', or raises exception.

        Raises:
            Exception: If processing fails.

        Notes:
            Performs database operations within a transaction per file.
            Parses frontmatter and tasks from file content.
            Generates embeddings if embedding_provider is configured.
            Stores relative path instead of absolute path.

        """
        _msg = f"_ingest_single_file starting for: {file_info.path}"
        log.debug(_msg)

        # Parse frontmatter and content
        kind, tags, metadata, content = parse_frontmatter(file_info.content)

        # Parse tasks
        parsed_tasks = parse_tasks_from_content(content)

        # Compute relative path
        relative_path = self._compute_relative_path(
            file_info.path,
            vault_config.container_path,
        )

        if dry_run:
            _msg = "Dry run mode - simulating new document"
            log.debug(_msg)
            return "new"

        if vault_id is None:
            _msg = "No vault ID available (should not happen outside dry_run)"
            raise RuntimeError(_msg)

        # Process in a single transaction for this file
        with self.db_manager.get_session() as session:
            # Check if document exists (by vault_id and relative path)
            existing = (
                session.query(Document)
                .filter_by(vault_id=vault_id, file_path=relative_path)
                .first()
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
                document = self._create_document(
                    file_info,
                    parsed_data,
                    vault_id=vault_id,
                    relative_path=relative_path,
                )
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
        *,
        vault_id: uuid.UUID,
        relative_path: str,
    ) -> Document:
        """Create a new Document instance.

        Args:
            file_info: File information.
            parsed_data: Tuple of (kind, tags, metadata, content).
            vault_id: UUID of the vault.
            relative_path: Relative path from vault root.

        Returns:
            New Document instance.

        Notes:
            Generates embeddings using the configured embedding provider.
            Network access may occur when calling external embedding APIs.

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
            except EmbeddingError as e:
                _msg = f"Failed to generate embedding: {e}"
                log.warning(_msg)
                embedding = None

        # Extract file name from relative path
        file_name = Path(relative_path).name

        document = Document(
            vault_id=vault_id,
            file_path=relative_path,
            file_name=file_name,
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

    def _delete_orphaned_documents(
        self,
        filesystem_paths: set[str],
        *,
        vault_id: uuid.UUID,
        dry_run: bool = False,
    ) -> tuple[int, int]:
        """Delete documents from database that no longer exist on filesystem.

        Args:
            filesystem_paths: Set of relative file paths that exist on the filesystem.
            vault_id: UUID of the vault to filter by.
            dry_run: If True, don't actually delete (just count).

        Returns:
            Tuple of (deleted_count, error_count).

        Notes:
            Performs database operations for document deletion.
            Tasks are automatically deleted via ON DELETE CASCADE.
            Each deletion is logged at INFO level.
            Errors are logged but do not stop processing.

        """
        _msg = "_delete_orphaned_documents starting"
        log.debug(_msg)

        # Query documents for this vault and identify orphans
        with self.db_manager.get_session() as session:
            vault_documents = session.query(Document).filter_by(vault_id=vault_id).all()
            orphaned_documents = [
                doc for doc in vault_documents if doc.file_path not in filesystem_paths
            ]

        if not orphaned_documents:
            _msg = "No orphaned documents found"
            log.debug(_msg)
            return 0, 0

        total_orphaned = len(orphaned_documents)
        _msg = f"Found {total_orphaned} orphaned documents to delete"
        log.info(_msg)

        if dry_run:
            _msg = f"Dry run mode - would delete {total_orphaned} orphaned documents"
            log.info(_msg)
            return total_orphaned, 0

        # Process deletions in batches
        return self._process_deletion_batches(orphaned_documents)

    def _process_deletion_batches(
        self,
        orphaned_documents: list[Document],
    ) -> tuple[int, int]:
        """Process deletion of orphaned documents in batches.

        Args:
            orphaned_documents: List of documents to delete.

        Returns:
            Tuple of (deleted_count, error_count).

        """
        deleted_count = 0
        error_count = 0
        batch_size = 100

        with self.db_manager.get_session() as session:
            for i in range(0, len(orphaned_documents), batch_size):
                batch = orphaned_documents[i : i + batch_size]
                batch_deleted, batch_errors = self._delete_batch(session, batch)
                deleted_count += batch_deleted
                error_count += batch_errors

        _msg = f"Deleted {deleted_count} orphaned documents ({error_count} errors)"
        log.info(_msg)

        _msg = "_delete_orphaned_documents returning"
        log.debug(_msg)

        return deleted_count, error_count

    def _delete_batch(
        self,
        session: Session,
        batch: list[Document],
    ) -> tuple[int, int]:
        """Delete a batch of documents.

        Args:
            session: Database session.
            batch: List of documents to delete in this batch.

        Returns:
            Tuple of (deleted_count, error_count) for this batch.

        """
        deleted_count = 0

        for document in batch:
            try:
                _msg = f"Deleting orphaned document: {document.file_path}"
                log.info(_msg)
                session.delete(document)
                deleted_count += 1
            except (OSError, RuntimeError) as e:
                _msg = f"Failed to delete document {document.file_path}: {e}"
                log.error(_msg)

        # Commit the batch
        try:
            session.commit()
            _msg = f"Deleted batch of {deleted_count} orphaned documents"
            log.debug(_msg)
        except (OSError, RuntimeError) as e:
            _msg = f"Failed to commit deletion batch: {e}"
            log.error(_msg)
            session.rollback()
            # All documents in batch failed
            return 0, len(batch)

        return deleted_count, len(batch) - deleted_count
