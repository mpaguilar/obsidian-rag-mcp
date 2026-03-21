"""Document cleanup operations for the ingestion service.

This module contains deletion-related operations that were extracted from
ingestion.py to keep file sizes under the 1000 line limit.
"""

import logging
import uuid
from typing import TYPE_CHECKING

from sqlalchemy.orm import Session

from obsidian_rag.database.models import Document

if TYPE_CHECKING:
    from obsidian_rag.services.ingestion import IngestionService

log = logging.getLogger(__name__)


def delete_orphaned_documents(
    service: "IngestionService",
    filesystem_paths: set[str],
    *,
    vault_id: uuid.UUID,
    dry_run: bool = False,
) -> tuple[int, int]:
    """Delete documents from database that no longer exist on filesystem.

    Args:
        service: The ingestion service instance.
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
    _msg = "delete_orphaned_documents starting"
    log.debug(_msg)

    # Query documents for this vault and identify orphans
    # Extract only id and file_path to avoid DetachedInstanceError
    with service.db_manager.get_session() as session:
        vault_documents = session.query(Document).filter_by(vault_id=vault_id).all()
        orphaned_doc_info = [
            (doc.id, doc.file_path)
            for doc in vault_documents
            if doc.file_path not in filesystem_paths
        ]

    if not orphaned_doc_info:
        _msg = "No orphaned documents found"
        log.debug(_msg)
        return 0, 0

    total_orphaned = len(orphaned_doc_info)
    _msg = f"Found {total_orphaned} orphaned documents to delete"
    log.info(_msg)

    if dry_run:
        _msg = f"Dry run mode - would delete {total_orphaned} orphaned documents"
        log.info(_msg)
        return total_orphaned, 0

    # Process deletions in batches
    return _process_deletion_batches(service, orphaned_doc_info)


def _process_deletion_batches(
    service: "IngestionService",
    orphaned_doc_info: list[tuple[uuid.UUID, str]],
) -> tuple[int, int]:
    """Process deletion of orphaned documents in batches.

    Args:
        service: The ingestion service instance.
        orphaned_doc_info: List of (document_id, file_path) tuples to delete.

    Returns:
        Tuple of (deleted_count, error_count).

    """
    deleted_count = 0
    error_count = 0
    batch_size = 100

    with service.db_manager.get_session() as session:
        for i in range(0, len(orphaned_doc_info), batch_size):
            batch = orphaned_doc_info[i : i + batch_size]
            batch_deleted, batch_errors = _delete_batch(session, batch)
            deleted_count += batch_deleted
            error_count += batch_errors

    _msg = f"Deleted {deleted_count} orphaned documents ({error_count} errors)"
    log.info(_msg)

    _msg = "delete_orphaned_documents returning"
    log.debug(_msg)

    return deleted_count, error_count


def _delete_batch(
    session: Session,
    batch: list[tuple[uuid.UUID, str]],
) -> tuple[int, int]:
    """Delete a batch of documents.

    Args:
        session: Database session.
        batch: List of (document_id, file_path) tuples to delete.

    Returns:
        Tuple of (deleted_count, error_count) for this batch.

    """
    deleted_count = 0
    error_count = 0

    for doc_id, file_path in batch:
        try:
            _msg = f"Deleting orphaned document: {file_path}"
            log.info(_msg)
            document = session.get(Document, doc_id)
            if document:
                session.delete(document)
                deleted_count += 1
            else:
                _msg = f"Document {file_path} not found in session"
                log.warning(_msg)
                error_count += 1
        except (OSError, RuntimeError) as e:
            _msg = f"Failed to delete document {file_path}: {e}"
            log.error(_msg)
            error_count += 1

    # Commit the batch
    try:
        session.commit()
        _msg = f"Deleted batch of {deleted_count} orphaned documents"
        log.debug(_msg)
    except (OSError, RuntimeError) as e:
        _msg = f"Failed to commit deletion batch: {e}"
        log.error(_msg)
        session.rollback()
        # All documents in batch failed (including those we thought we deleted)
        return 0, len(batch)

    return deleted_count, error_count
