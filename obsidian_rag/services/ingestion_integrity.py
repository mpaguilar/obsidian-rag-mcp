"""IntegrityError recovery for document ingestion."""

import logging
import uuid
from typing import TYPE_CHECKING, Any

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from obsidian_rag.database.models import Document

if TYPE_CHECKING:
    from obsidian_rag.parsing.scanner import FileInfo
    from obsidian_rag.parsing.tasks import ParsedTask
    from obsidian_rag.services.ingestion import IngestionService

log = logging.getLogger(__name__)


def ingest_new_document(
    service: "IngestionService",
    session: Session,
    *,
    file_info: "FileInfo",
    tags: list[str] | None,
    metadata: dict[str, Any],
    content: str,
    parsed_tasks: list[tuple[int, "ParsedTask"]],
    vault_id: uuid.UUID,
    relative_path: str,
    should_chunk: bool,
    is_empty: bool,
) -> tuple[str, int]:
    """Create a new document, recovering from IntegrityError by updating.

    Args:
        service: The ingestion service handling creation.
        session: Database session.
        file_info: File information.
        tags: Document tags.
        metadata: Document metadata.
        content: Document content.
        parsed_tasks: Parsed task list.
        vault_id: Vault UUID.
        relative_path: Relative file path.
        should_chunk: Whether the document should be chunked.
        is_empty: Whether the document is empty.

    Returns:
        Tuple of (result, chunks_created).

    """
    _msg = "Creating new document"
    log.debug(_msg)
    parsed_data = (tags, metadata, content)

    try:
        document, _ = service._create_document(
            _session=session,
            file_info=file_info,
            parsed_data=parsed_data,
            vault_id=vault_id,
            relative_path=relative_path,
        )
        session.add(document)
        session.flush()  # Get document ID
        service._create_tasks(session, document, parsed_tasks)

        chunks_created = service._create_chunks_for_new_document(
            session=session,
            document=document,
            content=content,
            should_chunk=should_chunk,
            is_empty=is_empty,
        )
        return ("new", chunks_created)
    except IntegrityError as err:
        # REQ-001: Recover from UniqueViolation by updating instead
        result, chunks_created = handle_integrity_error(
            service,
            session,
            err,
            vault_id=vault_id,
            relative_path=relative_path,
            file_info=file_info,
            parsed_data=parsed_data,
            parsed_tasks=parsed_tasks,
            is_empty=is_empty,
        )
        return (result, chunks_created)


def handle_integrity_error(
    service: "IngestionService",
    session: Session,
    err: IntegrityError,
    *,
    vault_id: uuid.UUID,
    relative_path: str,
    file_info: "FileInfo",
    parsed_data: tuple[list[str] | None, dict[str, Any], str],
    parsed_tasks: list[tuple[int, "ParsedTask"]],
    is_empty: bool,
) -> tuple[str, int]:
    """Recover from IntegrityError by updating the existing document.

    Args:
        service: The ingestion service handling the recovery.
        session: Database session (in invalid state, needs rollback).
        err: The IntegrityError that was raised.
        vault_id: Vault UUID for re-query.
        relative_path: File path for re-query.
        file_info: File information for update.
        parsed_data: Tuple of (tags, metadata, content).
        parsed_tasks: Parsed task list.
        is_empty: Whether the document is empty.

    Returns:
        Tuple of (status, chunks_created) where status is "updated".

    Raises:
        IntegrityError: Re-raised if not a UniqueViolation on
            uq_document_vault_path, or if re-query returns None.

    """
    _msg = "handle_integrity_error starting"
    log.debug(_msg)

    # Keep API consistent with the ingest path; value not needed for recovery
    _ = is_empty

    # Check if this is the expected UniqueViolation
    if "uq_document_vault_path" not in str(err):
        _msg = "IntegrityError is not uq_document_vault_path - re-raising"
        log.warning(_msg)
        raise err

    # Rollback to restore session to valid state
    session.rollback()

    _msg = (
        f"IntegrityError on uq_document_vault_path recovered - "
        f"updating document: vault_id={vault_id}, file_path={relative_path}"
    )
    log.warning(_msg)

    # Re-query for the existing document
    existing = (
        session.query(Document)
        .filter_by(vault_id=vault_id, file_path=relative_path)
        .first()
    )

    if existing is None:
        _msg = "Re-query returned None after IntegrityError - re-raising"
        log.warning(_msg)
        raise err

    # Update the existing document (same path as the "existing" branch)
    chunks_created = service._update_document(session, existing, file_info, parsed_data)
    service._update_tasks(session, existing, parsed_tasks)

    _msg = f"handle_integrity_error returning: updated, chunks={chunks_created}"
    log.debug(_msg)
    return ("updated", chunks_created)
