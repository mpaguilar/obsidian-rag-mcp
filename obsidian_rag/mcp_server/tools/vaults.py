"""MCP tools for vault operations."""

import logging
from typing import TYPE_CHECKING

from sqlalchemy import func

from obsidian_rag.database.models import Document, Vault
from obsidian_rag.mcp_server.models import (
    VaultListResponse,
    VaultResponse,
    _validate_limit,
    _validate_offset,
)

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
                host_path=vault.host_path,
                document_count=doc_count,
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
