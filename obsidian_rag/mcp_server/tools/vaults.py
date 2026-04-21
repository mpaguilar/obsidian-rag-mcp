"""MCP tools for vault operations."""

import logging
import uuid
from typing import TYPE_CHECKING

from sqlalchemy import func

from obsidian_rag.database.models import Document, DocumentChunk, Task, Vault
from obsidian_rag.mcp_server.models import (
    VaultListResponse,
    VaultResponse,
    _validate_limit,
    _validate_offset,
    create_vault_response,
)
from obsidian_rag.mcp_server.tools.vaults_params import VaultUpdateParams

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

log = logging.getLogger(__name__)


def list_vaults(
    session: "Session",
    limit: int = 20,
    offset: int = 0,
) -> VaultListResponse:
    """List all vaults with document counts.

    Args:
        session: Database session.
        limit: Maximum number of results (default: 20, max: 100).
        offset: Number of results to skip (default: 0).

    Returns:
        VaultListResponse with vaults and pagination info.

    """
    _msg = "list_vaults starting"
    log.debug(_msg)

    # Validate inputs
    limit = _validate_limit(limit)
    offset = _validate_offset(offset)

    # Build query with document counts
    # Use subquery to count documents per vault
    document_counts = (
        session.query(
            Document.vault_id,
            func.count(Document.id).label("doc_count"),
        )
        .group_by(Document.vault_id)
        .subquery()
    )

    # Main query joining vaults with document counts
    query = (
        session.query(
            Vault,
            func.coalesce(document_counts.c.doc_count, 0).label("document_count"),
        )
        .outerjoin(
            document_counts,
            Vault.id == document_counts.c.vault_id,
        )
        .order_by(Vault.name)
    )

    # Get total count
    total_count = query.count()

    # Get paginated results
    results = query.offset(offset).limit(limit).all()

    # Calculate pagination
    has_more = (offset + limit) < total_count
    next_offset = offset + limit if has_more else None

    # Build response
    vault_responses = []
    for vault, doc_count in results:
        vault_responses.append(
            VaultResponse(
                id=vault.id,
                name=vault.name,
                description=vault.description,
                container_path=vault.container_path,
                host_path=vault.host_path,
                document_count=doc_count,
                created_at=vault.created_at,
            ),
        )

    _msg = f"list_vaults returning {len(vault_responses)} vaults"
    log.debug(_msg)

    return VaultListResponse(
        results=vault_responses,
        total_count=total_count,
        has_more=has_more,
        next_offset=next_offset,
    )


def _lookup_vault_by_name(session: "Session", name: str) -> Vault | None:
    """Lookup a vault by name.

    Args:
        session: Database session.
        name: Vault name to lookup.

    Returns:
        Vault instance if found, None otherwise.

    """
    _msg = "_lookup_vault_by_name starting"
    log.debug(_msg)

    vault = session.query(Vault).filter(Vault.name == name).first()

    _msg = "_lookup_vault_by_name returning"
    log.debug(_msg)
    return vault


def _get_available_vault_names(session: "Session") -> list[str]:
    """Get list of all available vault names.

    Args:
        session: Database session.

    Returns:
        List of vault names.

    """
    _msg = "_get_available_vault_names starting"
    log.debug(_msg)

    vaults = session.query(Vault).all()
    names = [v.name for v in vaults]

    _msg = "_get_available_vault_names returning"
    log.debug(_msg)
    return names


def _count_vault_documents(session: "Session", vault_id: uuid.UUID) -> int:
    """Count documents in a vault.

    Args:
        session: Database session.
        vault_id: UUID of the vault.

    Returns:
        Number of documents in the vault.

    """
    _msg = "_count_vault_documents starting"
    log.debug(_msg)

    count = (
        session.query(func.count(Document.id))
        .filter(Document.vault_id == vault_id)
        .scalar()
    )

    _msg = "_count_vault_documents returning"
    log.debug(_msg)
    return count or 0


def get_vault(
    session: "Session",
    *,
    name: str | None = None,
    vault_id: str | None = None,
) -> VaultResponse:
    """Get a single vault by name or ID.

    Args:
        session: Database session.
        name: Vault name to lookup (preferred if both provided).
        vault_id: Vault UUID string to lookup.

    Returns:
        VaultResponse with vault details and document count.

    Raises:
        ValueError: If neither name nor vault_id is provided, or if vault not found,
            or if vault_id is not a valid UUID.

    """
    _msg = "get_vault starting"
    log.debug(_msg)

    # Validate that at least one lookup criteria is provided
    if name is None and vault_id is None:
        _error_msg = "Must provide name or vault_id"
        log.error(_error_msg)
        raise ValueError(_error_msg)

    vault: Vault | None = None

    # Prefer name lookup if provided
    if name is not None:
        vault = _lookup_vault_by_name(session, name=name)
        lookup_key = name
    else:
        # vault_id is not None here (checked above)
        assert vault_id is not None  # Type narrowing for mypy
        try:
            vault_uuid = uuid.UUID(vault_id)
        except (ValueError, TypeError) as err:
            _error_msg = f"Invalid vault_id UUID format: '{vault_id}'"
            log.error(_error_msg)
            raise ValueError(_error_msg) from err

        vault = session.query(Vault).filter(Vault.id == vault_uuid).first()
        lookup_key = vault_id

    # Handle vault not found
    if vault is None:
        available = _get_available_vault_names(session)
        available_str = ", ".join(available) if available else "none"
        _error_msg = f"Vault '{lookup_key}' not found. Available: {available_str}"
        log.error(_error_msg)
        raise ValueError(_error_msg)

    # Count documents in the vault
    doc_count = _count_vault_documents(session, vault_id=vault.id)

    # Create response
    result = VaultResponse(
        id=vault.id,
        name=vault.name,
        description=vault.description,
        container_path=vault.container_path,
        host_path=vault.host_path,
        document_count=doc_count,
        created_at=vault.created_at,
    )

    _msg = "get_vault returning"
    log.debug(_msg)
    return result


def _has_vault_changed(vault: Vault, params: VaultUpdateParams) -> bool:
    """Check if any vault fields need updating.

    Args:
        vault: The current vault instance.
        params: Update parameters to compare against.

    Returns:
        True if any field differs from current values.

    """
    _msg = "_has_vault_changed starting"
    log.debug(_msg)

    if params.description is not None and params.description != vault.description:
        _msg = "_has_vault_changed returning"
        log.debug(_msg)
        return True

    if (
        params.container_path is not None
        and params.container_path != vault.container_path
    ):
        _msg = "_has_vault_changed returning"
        log.debug(_msg)
        return True

    if params.host_path is not None and params.host_path != vault.host_path:
        _msg = "_has_vault_changed returning"
        log.debug(_msg)
        return True

    _msg = "_has_vault_changed returning"
    log.debug(_msg)
    return False


def _check_container_path_update(
    params: VaultUpdateParams, vault: Vault
) -> dict[str, object] | None:
    """Validate container_path update requirements.

    Args:
        params: Update parameters containing potential container_path change.
        vault: The current vault instance.

    Returns:
        Error dict if container_path change attempted without force,
        None if valid or no container_path change needed.

    """
    _msg = "_check_container_path_update starting"
    log.debug(_msg)

    # Only check if container_path is actually changing
    if (
        params.container_path is not None
        and params.container_path != vault.container_path
    ):
        if not params.force:
            _error_msg = "Changing container_path will delete all documents, tasks, and chunks for this vault and require re-ingestion. Set force=True to confirm."
            _msg = "_check_container_path_update returning"
            log.debug(_msg)
            return {
                "success": False,
                "error": _error_msg,
            }

    _msg = "_check_container_path_update returning"
    log.debug(_msg)
    return None


def _delete_vault_documents(session: "Session", vault_id: uuid.UUID) -> int:
    """Delete all documents for a vault.

    Uses SQLAlchemy cascade to delete related tasks and chunks.

    Args:
        session: Database session.
        vault_id: UUID of the vault whose documents to delete.

    Returns:
        Number of documents deleted.

    Notes:
        Uses the cascade="all, delete-orphan" relationship defined in
        Vault.documents to also delete related tasks and document_chunks.

    """
    _msg = "_delete_vault_documents starting"
    log.debug(_msg)

    count = (
        session.query(Document)
        .filter(Document.vault_id == vault_id)
        .delete(synchronize_session=False)
    )

    _result_msg = f"_delete_vault_documents returning (deleted {count} documents)"
    log.debug(_result_msg)
    return count


def _is_container_path_changing(params: VaultUpdateParams, vault: Vault) -> bool:
    """Check if container_path is being changed.

    Args:
        params: Update parameters containing potential container_path change.
        vault: The current vault instance.

    Returns:
        True if container_path is being changed to a different value.

    """
    _msg = "_is_container_path_changing starting"
    log.debug(_msg)

    if params.container_path is None:
        _msg = "_is_container_path_changing returning (no change specified)"
        log.debug(_msg)
        return False

    result = params.container_path != vault.container_path
    _msg = "_is_container_path_changing returning"
    log.debug(_msg)
    return result


def _apply_vault_updates(
    vault: Vault,
    params: VaultUpdateParams,
    session: "Session",
) -> None:
    """Apply vault field updates based on parameters.

    Handles container_path change with force flag (deletes documents).
    Updates description and host_path as partial updates.

    Args:
        vault: The vault instance to update.
        params: Update parameters.
        session: Database session for operations.

    Notes:
        If container_path is changing and force=True, logs warning and
        deletes all documents for the vault before updating the path.

    """
    _msg = "_apply_vault_updates starting"
    log.debug(_msg)

    # Handle container_path change with force
    if _is_container_path_changing(params, vault) and params.force:
        _warning_msg = f"Deleting all documents for vault '{params.name}' due to container_path change"
        log.warning(_warning_msg)
        _delete_vault_documents(session, vault_id=vault.id)
        # params.container_path is not None when _is_container_path_changing is True
        assert params.container_path is not None  # Type narrowing for mypy
        vault.container_path = params.container_path

    # Update other fields (partial update)
    if params.description is not None:
        vault.description = params.description

    if params.host_path is not None:
        vault.host_path = params.host_path

    _msg = "_apply_vault_updates returning"
    log.debug(_msg)


def _handle_flush_with_integrity_check(
    session: "Session",
    container_path: str | None,
) -> dict[str, object] | None:
    """Attempt to flush session and handle integrity errors.

    Args:
        session: Database session.
        container_path: The container_path value for error messages.

    Returns:
        Error dict if integrity error occurs, None on success.

    Raises:
        Exception: Re-raises non-integrity errors.

    """
    _msg = "_handle_flush_with_integrity_check starting"
    log.debug(_msg)

    try:
        session.flush()
    except Exception as err:
        # Check if it's an integrity error (duplicate container_path)
        if "unique" in str(err).lower() or "integrity" in str(err).lower():
            _error_msg = f"Duplicate container_path value: {container_path}"
            log.error(_error_msg)
            _msg = "_handle_flush_with_integrity_check returning (integrity error)"
            log.debug(_msg)
            return {
                "success": False,
                "error": _error_msg,
            }
        raise

    _msg = "_handle_flush_with_integrity_check returning (success)"
    log.debug(_msg)
    return None


def update_vault(
    session: "Session",
    params: VaultUpdateParams,
) -> VaultResponse | dict[str, object]:
    """Update a vault's properties.

    The name field in params is used for lookup only and cannot be changed.
    Changing container_path requires force=True as it deletes all documents.

    Args:
        session: Database session.
        params: Update parameters including name for lookup.

    Returns:
        VaultResponse on success, or error dict on validation failure.

    Raises:
        ValueError: If vault with given name is not found.

    Notes:
        The container_path change is destructive - it deletes all documents,
        tasks, and chunks for the vault. The user must set force=True to confirm.
        IntegrityError from duplicate container_path is caught and returned
        as an error dict.

    """
    _msg = "update_vault starting"
    log.debug(_msg)

    # 1. Look up vault by name
    vault = _lookup_vault_by_name(session, name=params.name)
    if vault is None:
        _error_msg = f"Vault '{params.name}' not found"
        log.error(_error_msg)
        raise ValueError(_error_msg)

    # 2. Check if any fields to update
    if not _has_vault_changed(vault, params):
        _msg = "update_vault returning (no changes)"
        log.debug(_msg)
        doc_count = _count_vault_documents(session, vault_id=vault.id)
        return create_vault_response(vault, doc_count)

    # 3. Check container_path change requirements
    container_error = _check_container_path_update(params, vault)
    if container_error is not None:
        _msg = "update_vault returning (container_path change without force)"
        log.debug(_msg)
        return container_error

    # 4. Apply updates (handles container_path deletion if needed)
    _apply_vault_updates(vault, params, session)

    # 5. Try to flush, catch IntegrityError for duplicates
    integrity_error = _handle_flush_with_integrity_check(
        session,
        params.container_path or vault.container_path,
    )
    if integrity_error is not None:
        _msg = "update_vault returning (integrity error)"
        log.debug(_msg)
        return integrity_error

    # 6. Get new document count and return response
    new_doc_count = _count_vault_documents(session, vault_id=vault.id)

    _msg = "update_vault returning (success)"
    log.debug(_msg)
    return create_vault_response(vault, new_doc_count)


def _count_vault_cascade_targets(
    session: "Session",
    vault_id: uuid.UUID,
) -> tuple[int, int, int]:
    """Count documents, tasks, and chunks for a vault before deletion.

    Args:
        session: Database session.
        vault_id: UUID of the vault to count cascade targets for.

    Returns:
        Tuple of (document_count, task_count, chunk_count).

    Notes:
        Tasks and chunks are counted via join with documents since
        they have foreign key relationships to documents.

    """
    _msg = "_count_vault_cascade_targets starting"
    log.debug(_msg)

    # Count documents
    doc_count = (
        session.query(func.count(Document.id))
        .filter(Document.vault_id == vault_id)
        .scalar()
        or 0
    )

    # Count tasks (via document join)
    task_count = (
        session.query(func.count(Task.id))
        .join(Document, Task.document_id == Document.id)
        .filter(Document.vault_id == vault_id)
        .scalar()
        or 0
    )

    # Count chunks (via document join)
    chunk_count = (
        session.query(func.count(DocumentChunk.id))
        .join(Document, DocumentChunk.document_id == Document.id)
        .filter(Document.vault_id == vault_id)
        .scalar()
        or 0
    )

    _result_msg = f"_count_vault_cascade_targets returning (docs={doc_count}, tasks={task_count}, chunks={chunk_count})"
    log.debug(_result_msg)
    return (doc_count, task_count, chunk_count)


def delete_vault(
    session: "Session",
    *,
    name: str,
    confirm: bool,
) -> dict[str, object]:
    """Delete a vault and all associated documents, tasks, and chunks.

    This operation is irreversible and cascade-deletes all associated
    data. Requires explicit confirmation via confirm=True parameter.

    Args:
        session: Database session.
        name: Vault name to delete.
        confirm: Must be True to proceed with deletion. If False, returns
            an error dict explaining the requirement.

    Returns:
        Success dict with deletion counts if confirmed:
        {
            "success": True,
            "name": vault name,
            "id": vault UUID string,
            "documents_deleted": int,
            "tasks_deleted": int,
            "chunks_deleted": int,
            "warning": config warning message
        }
        Error dict if not confirmed:
        {
            "success": False,
            "error": confirmation requirement message
        }

    Raises:
        ValueError: If vault with given name is not found.

    Notes:
        The vault configuration entry in the config file is NOT deleted
        by this operation. The user must manually remove it from the
        config file. If not removed, the next ingestion will recreate
        the vault in the database.

    """
    _msg = "delete_vault starting"
    log.debug(_msg)

    # 1. Check confirmation
    if not confirm:
        _error_msg = "confirm=True is required to delete a vault. This action is irreversible and will cascade-delete all associated documents, tasks, and chunks."
        _msg = "delete_vault returning (confirm not True)"
        log.debug(_msg)
        return {
            "success": False,
            "error": _error_msg,
        }

    # 2. Look up vault by name
    vault = _lookup_vault_by_name(session, name=name)
    if vault is None:
        available = _get_available_vault_names(session)
        available_str = ", ".join(available) if available else "none"
        _error_msg = f"Vault '{name}' not found. Available: {available_str}"
        log.error(_error_msg)
        raise ValueError(_error_msg)

    # 3. Count cascade targets before deletion
    doc_count, task_count, chunk_count = _count_vault_cascade_targets(
        session,
        vault_id=vault.id,
    )

    # 4. Log warning before deletion
    _warning_msg = f"Deleting vault '{name}' with {doc_count} documents, {task_count} tasks, {chunk_count} chunks"
    log.warning(_warning_msg)

    # 5. Delete the vault (SQLAlchemy cascade handles the rest)
    session.delete(vault)
    session.flush()

    # 6. Return success response
    _msg = "delete_vault returning (success)"
    log.debug(_msg)
    return {
        "success": True,
        "name": vault.name,
        "id": str(vault.id),
        "documents_deleted": doc_count,
        "tasks_deleted": task_count,
        "chunks_deleted": chunk_count,
        "warning": "Vault config entry still exists. Next ingestion will recreate this vault.",
    }
