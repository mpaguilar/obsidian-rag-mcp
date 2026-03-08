"""Tool handlers for MCP server."""

import logging
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Literal, TypedDict, cast

from obsidian_rag.config import Settings
from obsidian_rag.database.engine import DatabaseManager
from obsidian_rag.llm.base import EmbeddingProvider
from obsidian_rag.mcp_server.models import PropertyFilter, TagFilter
from obsidian_rag.mcp_server.tools.documents import (
    get_all_tags as get_all_tags_tool,
)
from obsidian_rag.mcp_server.tools.documents import (
    get_documents_by_tag as get_documents_by_tag_tool,
)
from obsidian_rag.mcp_server.tools.tasks import (
    get_completed_tasks as get_completed_tasks_tool,
)
from obsidian_rag.mcp_server.tools.tasks import (
    get_incomplete_tasks as get_incomplete_tasks_tool,
)
from obsidian_rag.mcp_server.tools.tasks import (
    get_tasks_by_tag as get_tasks_by_tag_tool,
)
from obsidian_rag.mcp_server.tools.tasks import (
    get_tasks_due_this_week as get_tasks_due_this_week_tool,
)
from obsidian_rag.mcp_server.tools.vaults import (
    list_vaults as list_vaults_tool,
)
from obsidian_rag.services.ingestion import IngestionService, IngestVaultOptions

log = logging.getLogger(__name__)


class DocumentTagParams(TypedDict, total=False):
    """Parameters for get_documents_by_tag tool."""

    include_tags: list[str]
    exclude_tags: list[str]
    match_mode: str
    vault_name: str | None
    limit: int
    offset: int


@dataclass
class QueryFilterParams:
    """Parameters for query_documents filter configuration."""

    include_properties: list[dict] | None
    exclude_properties: list[dict] | None
    include_tags: list[str] | None
    exclude_tags: list[str] | None


@dataclass
class IngestHandlerParams:
    """Parameters for _ingest_handler."""

    settings: Settings
    db_manager: DatabaseManager
    embedding_provider: EmbeddingProvider | None
    vault_name: str
    path_override: str | None
    no_delete: bool = False


def _get_incomplete_tasks_handler(
    db_manager: DatabaseManager,
    limit: int,
    offset: int,
    *,
    include_cancelled: bool,
) -> dict[str, object]:
    """Handle get_incomplete_tasks tool call."""
    _msg = "_get_incomplete_tasks_handler starting"
    log.debug(_msg)
    with db_manager.get_session() as session:
        result = get_incomplete_tasks_tool(
            session=session,
            limit=limit,
            offset=offset,
            include_cancelled=include_cancelled,
        )
        _msg = "_get_incomplete_tasks_handler returning"
        log.debug(_msg)
        return result.model_dump()


def _get_tasks_due_this_week_handler(
    db_manager: DatabaseManager,
    limit: int,
    offset: int,
    *,
    include_completed: bool,
) -> dict[str, object]:
    """Handle get_tasks_due_this_week tool call."""
    _msg = "_get_tasks_due_this_week_handler starting"
    log.debug(_msg)
    with db_manager.get_session() as session:
        result = get_tasks_due_this_week_tool(
            session=session,
            limit=limit,
            offset=offset,
            include_completed=include_completed,
        )
        _msg = "_get_tasks_due_this_week_handler returning"
        log.debug(_msg)
        return result.model_dump()


def _get_tasks_by_tag_handler(
    db_manager: DatabaseManager,
    tag: str,
    limit: int,
    offset: int,
) -> dict[str, object]:
    """Handle get_tasks_by_tag tool call."""
    _msg = "_get_tasks_by_tag_handler starting"
    log.debug(_msg)
    with db_manager.get_session() as session:
        result = get_tasks_by_tag_tool(
            session=session,
            tag=tag,
            limit=limit,
            offset=offset,
        )
        _msg = "_get_tasks_by_tag_handler returning"
        log.debug(_msg)
        return result.model_dump()


def _get_completed_tasks_handler(
    db_manager: DatabaseManager,
    limit: int,
    offset: int,
    completed_since: str | None,
) -> dict[str, object]:
    """Handle get_completed_tasks tool call."""
    _msg = "_get_completed_tasks_handler starting"
    log.debug(_msg)
    since_date = None
    if completed_since:
        try:
            since_date = date.fromisoformat(completed_since)
        except ValueError:
            _msg = f"Invalid date format: {completed_since}"
            log.warning(_msg)

    with db_manager.get_session() as session:
        result = get_completed_tasks_tool(
            session=session,
            limit=limit,
            offset=offset,
            completed_since=since_date,
        )
        _msg = "_get_completed_tasks_handler returning"
        log.debug(_msg)
        return result.model_dump()


def _get_documents_by_tag_handler(
    db_manager: DatabaseManager,
    params: DocumentTagParams,
) -> dict[str, object]:
    """Handle get_documents_by_tag tool call.

    Args:
        db_manager: Database manager for sessions.
        params: Dictionary with include_tags, exclude_tags, match_mode, vault_name, limit, offset.

    Returns:
        Document list response as dictionary.

    """
    _msg = "_get_documents_by_tag_handler starting"
    log.debug(_msg)

    # Create TagFilter from params with proper type casting
    match_mode_value = params.get("match_mode", "all")
    match_mode_casted = cast("Literal['all', 'any']", match_mode_value)
    tag_filter = TagFilter(
        include_tags=params.get("include_tags", []),
        exclude_tags=params.get("exclude_tags", []),
        match_mode=match_mode_casted,
    )

    with db_manager.get_session() as session:
        result = get_documents_by_tag_tool(
            session=session,
            tag_filter=tag_filter,
            vault_name=params.get("vault_name"),
            limit=params.get("limit", 20),
            offset=params.get("offset", 0),
        )
        _msg = "_get_documents_by_tag_handler returning"
        log.debug(_msg)
        return result.model_dump()


def _get_all_tags_handler(
    db_manager: DatabaseManager,
    pattern: str | None,
    limit: int,
    offset: int,
) -> dict[str, object]:
    """Handle get_all_tags tool call."""
    _msg = "_get_all_tags_handler starting"
    log.debug(_msg)
    with db_manager.get_session() as session:
        result = get_all_tags_tool(
            session=session,
            pattern=pattern,
            limit=limit,
            offset=offset,
        )
        _msg = "_get_all_tags_handler returning"
        log.debug(_msg)
        return result.model_dump()


def _list_vaults_handler(
    db_manager: DatabaseManager,
    limit: int,
    offset: int,
) -> dict[str, object]:
    """Handle list_vaults tool call."""
    _msg = "_list_vaults_handler starting"
    log.debug(_msg)
    with db_manager.get_session() as session:
        result = list_vaults_tool(
            session=session,
            limit=limit,
            offset=offset,
        )
        _msg = "_list_vaults_handler returning"
        log.debug(_msg)
        return result.model_dump()


def _convert_property_filters(
    properties: list[dict] | None,
) -> list[PropertyFilter] | None:
    """Convert list of dict property filters to PropertyFilter objects.

    Args:
        properties: List of property filter dicts.

    Returns:
        List of PropertyFilter objects or None.

    """
    _msg = "_convert_property_filters starting"
    log.debug(_msg)
    if not properties:
        _msg = "_convert_property_filters returning"
        log.debug(_msg)
        return None
    result = [PropertyFilter(**prop) for prop in properties]
    _msg = "_convert_property_filters returning"
    log.debug(_msg)
    return result


def _create_tag_filter(
    include_tags: list[str] | None,
    exclude_tags: list[str] | None,
    match_mode: str,
) -> TagFilter | None:
    """Build TagFilter from parameters.

    Args:
        include_tags: List of tags documents must have.
        exclude_tags: List of tags documents must NOT have.
        match_mode: Whether to match "all" or "any" of include_tags.

    Returns:
        TagFilter or None if no tags specified.

    """
    _msg = "_create_tag_filter starting"
    log.debug(_msg)
    if not include_tags and not exclude_tags:
        _msg = "_create_tag_filter returning"
        log.debug(_msg)
        return None

    valid_match_mode = match_mode if match_mode in ("all", "any") else "all"
    result = TagFilter(
        include_tags=include_tags or [],
        exclude_tags=exclude_tags or [],
        match_mode=cast("Literal['all', 'any']", valid_match_mode),
    )
    _msg = "_create_tag_filter returning"
    log.debug(_msg)
    return result


def _validate_ingest_path(ingest_path: str) -> Path:
    """Validate the ingest path."""
    _msg = "_validate_ingest_path starting"
    log.debug(_msg)

    if ".." in ingest_path:
        _msg = "Path cannot contain parent directory references (..)"
        log.error(_msg)
        raise ValueError(_msg)

    path = Path(ingest_path)

    if not path.exists():
        _msg = f"Data directory '{ingest_path}' does not exist. Please ensure the volume is mounted."
        log.error(_msg)
        raise ValueError(_msg)

    if not path.is_dir():
        _msg = f"Path '{ingest_path}' exists but is not a directory"
        log.error(_msg)
        raise ValueError(_msg)

    _msg = "_validate_ingest_path returning"
    log.debug(_msg)
    return path


def _ingest_handler(params: IngestHandlerParams) -> dict[str, object]:
    """Handle ingest tool call.

    Args:
        params: Ingest handler parameters.

    Returns:
        Dictionary with ingestion results including deleted count.

    Raises:
        ValueError: If vault_name is not found in configuration.

    """
    _msg = "ingest handler starting"
    log.debug(_msg)

    # Validate vault exists
    vault_config = params.settings.get_vault(params.vault_name)
    if vault_config is None:
        available = params.settings.get_vault_names()
        _msg = (
            f"Vault '{params.vault_name}' not found in configuration. "
            f"Available vaults: {', '.join(available)}"
        )
        raise ValueError(_msg)

    # Use path override if provided, otherwise use vault container_path
    ingest_path = (
        params.path_override if params.path_override else vault_config.container_path
    )
    path = _validate_ingest_path(ingest_path)

    ingestion_service = IngestionService(
        db_manager=params.db_manager,
        embedding_provider=params.embedding_provider,
        settings=params.settings,
    )

    options = IngestVaultOptions(
        vault=params.vault_name,
        dry_run=False,
        no_delete=params.no_delete,
    )
    result = ingestion_service.ingest_vault(path, options)

    _msg = f"ingest handler completed: {result.message}"
    log.info(_msg)

    return result.to_dict()
