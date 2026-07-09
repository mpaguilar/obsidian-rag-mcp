"""Data models for the ingestion service."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from obsidian_rag.parsing.scanner import FileInfo

if TYPE_CHECKING:
    from obsidian_rag.config import VaultConfig


@dataclass
class IngestVaultOptions:
    """Options for ingest_vault method.

    Attributes:
        vault: Vault configuration or vault name.
        dry_run: If True, don't write to database.
        progress_callback: Optional callback for progress updates.
        file_infos: Optional pre-scanned file info objects.
        no_delete: If True, skip deletion of orphaned documents.
        force: If True, re-ingest all documents regardless of checksums.

    """

    vault: "VaultConfig | str"
    dry_run: bool = False
    progress_callback: Callable[[int, int, int, int], None] | None = None
    file_infos: list[FileInfo] | None = None
    no_delete: bool = False
    force: bool = False


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
        skipped: True only on the synthetic no-op-skip result built when a force
            re-ingest holds the vault lock; False for all real ingestion results.
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
    skipped: bool = False

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
            "skipped": self.skipped,
        }
