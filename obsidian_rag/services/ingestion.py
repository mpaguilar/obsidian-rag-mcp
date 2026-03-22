"""Ingestion service for processing markdown documents."""

import logging
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from sqlalchemy.orm import Session

from obsidian_rag.chunking import should_chunk_document
from obsidian_rag.config import VaultConfig
from obsidian_rag.database.engine import DatabaseManager
from obsidian_rag.database.models import Document, DocumentChunk, Task, Vault
from obsidian_rag.llm.base import EmbeddingError, EmbeddingProvider
from obsidian_rag.parsing.frontmatter import parse_frontmatter
from obsidian_rag.parsing.scanner import (
    FileInfo,
    process_files_in_batches,
    scan_markdown_files,
)
from obsidian_rag.parsing.tasks import parse_tasks_from_content
from obsidian_rag.services.ingestion_chunks import create_chunks_with_embeddings
from obsidian_rag.services.ingestion_cleanup import delete_orphaned_documents

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
        chunks_created: Number of document chunks created for large documents.
        empty_documents: Number of empty documents (no content, skipped for embeddings).
        total_chunks: Total number of chunks across all documents.
        avg_chunk_tokens: Average token count per chunk.
        task_chunk_count: Number of chunks containing tasks.
        content_chunk_count: Number of regular content chunks.
        processing_time_seconds: Time taken to process all files.
        message: Human-readable summary message.

    """

    total: int
    new: int
    updated: int
    unchanged: int
    errors: int
    deleted: int
    chunks_created: int
    empty_documents: int
    processing_time_seconds: float
    message: str
    # New chunk statistics fields
    total_chunks: int = 0
    avg_chunk_tokens: int = 0
    task_chunk_count: int = 0
    content_chunk_count: int = 0

    def to_dict(self) -> dict[str, object]:
        """Convert result to dictionary."""
        return {
            "total": self.total,
            "new": self.new,
            "updated": self.updated,
            "unchanged": self.unchanged,
            "errors": self.errors,
            "deleted": self.deleted,
            "chunks_created": self.chunks_created,
            "empty_documents": self.empty_documents,
            "total_chunks": self.total_chunks,
            "avg_chunk_tokens": self.avg_chunk_tokens,
            "task_chunk_count": self.task_chunk_count,
            "content_chunk_count": self.content_chunk_count,
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
            Dictionary with statistics including chunks_created and empty_documents.

        """
        stats = {
            "new": 0,
            "updated": 0,
            "unchanged": 0,
            "errors": 0,
            "chunks_created": 0,
            "empty_documents": 0,
        }
        total = len(file_info_list)

        for idx, file_info in enumerate(file_info_list):
            try:
                result, chunks_created, is_empty = self._ingest_single_file(
                    file_info,
                    vault_id=vault_id,
                    vault_config=vault_config,
                    dry_run=dry_run,
                )
                stats[result] += 1
                stats["chunks_created"] += chunks_created
                if is_empty:
                    stats["empty_documents"] += 1
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
                chunks_created=0,
                empty_documents=0,
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
                f"{stats['errors']} errors, {stats['chunks_created']} chunks created, "
                f"{stats['empty_documents']} empty, deletion skipped"
            )
        else:
            message = (
                f"Ingested {total} files: {stats['new']} new, "
                f"{stats['updated']} updated, {stats['unchanged']} unchanged, "
                f"{stats['errors']} errors, {deleted_count} deleted, "
                f"{stats['chunks_created']} chunks created, "
                f"{stats['empty_documents']} empty"
            )

        result = IngestionResult(
            total=total,
            new=stats["new"],
            updated=stats["updated"],
            unchanged=stats["unchanged"],
            errors=stats["errors"] + deletion_errors,
            deleted=deleted_count,
            chunks_created=stats["chunks_created"],
            empty_documents=stats["empty_documents"],
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
    ) -> tuple[str, int, bool]:
        """Ingest a single file.

        Args:
            file_info: File information including path, content, checksum.
            vault_id: Vault ID (UUID) or None if dry_run.
            vault_config: Vault configuration.
            dry_run: If True, don't write to database.

        Returns:
            Tuple of (status, chunks_created, is_empty) where:
            - status: 'new', 'updated', 'unchanged', or raises exception
            - chunks_created: Number of chunks created for large documents
            - is_empty: True if document has no content

        Raises:
            Exception: If processing fails.

        Notes:
            Performs database operations within a transaction per file.
            Parses frontmatter and tasks from file content.
            Generates embeddings if embedding_provider is configured.
            Stores relative path instead of absolute path.
            Large documents are split into chunks with individual embeddings.

        """
        _msg = f"_ingest_single_file starting for: {file_info.path}"
        log.debug(_msg)

        # Parse frontmatter and content
        tags, metadata, content = parse_frontmatter(file_info.content)

        # Parse tasks
        parsed_tasks = parse_tasks_from_content(content)

        # Compute relative path
        relative_path = self._compute_relative_path(
            file_info.path,
            vault_config.container_path,
        )

        # Check if document is empty
        is_empty = not content or not content.strip()

        if dry_run:
            _msg = "Dry run mode - simulating new document"
            log.debug(_msg)
            return ("new", 0, is_empty)

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
                    return ("unchanged", 0, is_empty)

                # Update existing document
                _msg = "Updating existing document"
                log.debug(_msg)
                parsed_data = (tags, metadata, content)
                chunks_created = self._update_document(
                    session, existing, file_info, parsed_data
                )
                self._update_tasks(session, existing, parsed_tasks)
                result = "updated"
            else:
                # Create new document
                _msg = "Creating new document"
                log.debug(_msg)
                parsed_data = (tags, metadata, content)

                # Determine chunking requirements BEFORE creating document
                chunk_size = self.settings.chunking.chunk_size
                model_name = self.settings.chunking.tokenizer_model
                should_chunk = should_chunk_document(content, chunk_size, model_name)

                document, _ = self._create_document(
                    _session=session,
                    file_info=file_info,
                    parsed_data=parsed_data,
                    vault_id=vault_id,
                    relative_path=relative_path,
                )
                session.add(document)
                session.flush()  # Get document ID
                self._create_tasks(session, document, parsed_tasks)

                # BUG-001 FIX: Create chunks for new documents that need chunking
                chunks_created = self._create_chunks_for_new_document(
                    session=session,
                    document=document,
                    content=content,
                    should_chunk=should_chunk,
                    is_empty=is_empty,
                )
                result = "new"

        _msg = f"_ingest_single_file returning: {result}, chunks={chunks_created}"
        log.debug(_msg)
        return (result, chunks_created, is_empty)

    def _create_chunks_for_new_document(
        self,
        session: Session,
        document: Document,
        content: str,
        *,
        should_chunk: bool,
        is_empty: bool,
    ) -> int:
        """Create chunks for a new document if needed.

        Args:
            session: Database session.
            document: The newly created document.
            content: Document content.
            should_chunk: Whether the document should be chunked.
            is_empty: Whether the document is empty.

        Returns:
            Number of chunks created (0 if not chunked).

        """
        if not should_chunk or is_empty:
            return 0

        _msg = "Creating chunks for new document"
        log.info(_msg)

        chunk_size = self.settings.chunking.chunk_size
        chunk_overlap = self.settings.chunking.chunk_overlap
        model_name = self.settings.chunking.tokenizer_model

        return create_chunks_with_embeddings(
            session,
            document.id,
            content,
            self.embedding_provider,
            chunk_size,
            chunk_overlap,
            model_name,
        )

    def _is_document_empty(self, content: str) -> bool:
        """Check if document content is empty.

        Args:
            content: The document content.

        Returns:
            True if content is empty or whitespace-only.

        """
        return not content or not content.strip()

    def _generate_embedding(self, text: str) -> list[float] | None:
        """Generate embedding for text using the configured provider.

        Args:
            text: The text to generate embedding for.

        Returns:
            Embedding vector or None if generation fails.

        Notes:
            Network access may occur when calling external embedding APIs.

        """
        if not self.embedding_provider:
            return None

        try:
            return self.embedding_provider.generate_embedding(text)
        except EmbeddingError as e:
            _msg = f"Failed to generate embedding: {e}"
            log.warning(_msg)
            return None

    def _delete_existing_chunks(
        self,
        session: Session,
        document: Document,
    ) -> None:
        """Delete existing chunks for a document.

        Args:
            session: Database session.
            document: Document whose chunks should be deleted.

        """
        if document.chunks:
            _msg = "Deleting old chunks"
            log.debug(_msg)
            session.query(DocumentChunk).filter_by(document_id=document.id).delete()
            document.chunks = []

    def _create_document(
        self,
        _session: Session,
        file_info: FileInfo,
        parsed_data: tuple[list[str] | None, dict[str, Any], str],
        *,
        vault_id: uuid.UUID,
        relative_path: str,
    ) -> tuple[Document, int]:
        """Create a new Document instance.

        Args:
            _session: Database session (unused, present for API consistency).
            file_info: File information.
            parsed_data: Tuple of (tags, metadata, content).
            vault_id: UUID of the vault.
            relative_path: Relative path from vault root.

        Returns:
            Tuple of (Document, chunks_created).

        Notes:
            Generates embeddings using the configured embedding provider.
            Large documents are split into chunks with individual embeddings.
            Empty documents have no embedding (None).
            Network access may occur when calling external embedding APIs.

        """
        _msg = "_create_document starting"
        log.debug(_msg)

        tags, metadata, content = parsed_data
        chunks_created = 0

        # Get chunking settings
        chunk_size = self.settings.chunking.chunk_size
        model_name = self.settings.chunking.tokenizer_model

        # Determine document handling based on content
        is_empty = self._is_document_empty(content)
        model_name = self.settings.chunking.tokenizer_model
        chunk_size = self.settings.chunking.chunk_size
        should_chunk = should_chunk_document(content, chunk_size, model_name)

        if is_empty:
            _msg = "Empty document - skipping embedding generation"
            log.debug(_msg)
            embedding = None
        elif should_chunk:
            _msg = "Large document detected - creating chunks"
            log.info(_msg)
            # Create chunks after document is flushed and has an ID
            chunks_created = 0  # Will be created after flush
            embedding = None
        else:
            _msg = "Generating embedding for document"
            log.debug(_msg)
            embedding = self._generate_embedding(content)

        # Create document
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
            tags=tags,
            frontmatter_json=metadata,
        )

        _msg = f"_create_document returning with {chunks_created} chunks"
        log.debug(_msg)

        return (document, chunks_created)

    def _update_document(
        self,
        session: Session,
        document: Document,
        file_info: FileInfo,
        parsed_data: tuple[list[str] | None, dict[str, Any], str],
    ) -> int:
        """Update an existing Document instance.

        Args:
            session: Database session.
            document: Existing document to update.
            file_info: File information.
            parsed_data: Tuple of (tags, metadata, content).

        Returns:
            Number of chunks created (0 if not chunked).

        Notes:
            Deletes old chunks and recreates them if document is chunked.
            Regenerates embedding for non-chunked documents.

        """
        _msg = "_update_document starting"
        log.debug(_msg)

        tags, metadata, content = parsed_data
        chunks_created = 0

        # Update document fields
        document.content = content
        document.checksum_md5 = file_info.checksum
        document.modified_at_fs = file_info.modified_at
        document.tags = tags
        document.frontmatter_json = metadata

        # Get chunking settings
        chunk_size = self.settings.chunking.chunk_size
        chunk_overlap = self.settings.chunking.chunk_overlap
        model_name = self.settings.chunking.tokenizer_model

        # Determine document handling
        is_empty = self._is_document_empty(content)
        should_chunk = should_chunk_document(content, chunk_size, model_name)

        # Delete old chunks
        self._delete_existing_chunks(session, document)

        if is_empty:
            _msg = "Empty document - clearing embedding"
            log.debug(_msg)
            document.content_vector = None
        elif should_chunk:
            _msg = "Large document detected - recreating chunks"
            log.info(_msg)
            chunks_created = create_chunks_with_embeddings(
                session,
                document.id,
                content,
                self.embedding_provider,
                chunk_size,
                chunk_overlap,
                model_name,
            )
            document.content_vector = None
        else:
            _msg = "Regenerating embedding for updated document"
            log.debug(_msg)
            document.content_vector = self._generate_embedding(content)

        _msg = f"_update_document completed with {chunks_created} chunks"
        log.debug(_msg)

        return chunks_created

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
            Delegates to ingestion_cleanup module to keep file size under limit.
            Performs database operations for document deletion.
            Tasks are automatically deleted via ON DELETE CASCADE.
        """
        return delete_orphaned_documents(
            self,
            filesystem_paths,
            vault_id=vault_id,
            dry_run=dry_run,
        )
