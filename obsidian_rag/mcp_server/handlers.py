"""Tool handlers for MCP server."""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal, TypedDict, cast

from pydantic import BeforeValidator

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
from obsidian_rag.mcp_server.tools.tasks import get_tasks as get_tasks_tool
from obsidian_rag.mcp_server.tools.tasks_dates import parse_iso_date
from obsidian_rag.mcp_server.tools.tasks_params import (
    GetTasksFilterParams,
    GetTasksRequest,
)
from obsidian_rag.mcp_server.tools.vaults import (
    delete_vault,
    get_vault,
    update_vault,
)
from obsidian_rag.mcp_server.tools.vaults import (
    list_vaults as list_vaults_tool,
)
from obsidian_rag.mcp_server.tools.vaults_params import VaultUpdateParams
from obsidian_rag.services.ingestion import IngestionService, IngestVaultOptions

log = logging.getLogger(__name__)


def parse_json_str(value: Any) -> Any:
    """Parse JSON string to dict, or pass through other values unchanged.

    This helper function is used with Pydantic's BeforeValidator to handle
    clients that pass JSON-encoded strings instead of JSON objects.

    Args:
        value: The input value, which may be a JSON string, dict, or None.

    Returns:
        Parsed dict if input was a JSON string, original value otherwise.
        Empty or whitespace-only strings are treated as None.

    Raises:
        json.JSONDecodeError: If the input is a string but not valid JSON.

    Notes:
        This function logs at DEBUG level when parsing JSON strings to help
        diagnose client-side double-encoding issues.

    """
    _msg = "parse_json_str starting"
    log.debug(_msg)

    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            _msg = "parse_json_str returning None for empty string"
            log.debug(_msg)
            return None
        try:
            parsed = json.loads(stripped)
            _msg = "parse_json_str parsed JSON string"
            log.debug(_msg)
            return parsed
        except json.JSONDecodeError:
            _msg = "parse_json_str failed to parse JSON string"
            log.debug(_msg)
            raise

    _msg = "parse_json_str returning value unchanged"
    log.debug(_msg)
    return value


class DocumentTagParams(TypedDict, total=False):
    """Parameters for get_documents_by_tag tool."""

    include_tags: list[str]
    exclude_tags: list[str]
    match_mode: str
    vault_name: str | None
    limit: int
    offset: int
    include_content: bool


@dataclass
class QueryFilterParams:
    """Parameters for query_documents filter configuration.

    Attributes:
        include_properties: List of property filters to include.
        exclude_properties: List of property filters to exclude.
        include_tags: List of tags documents must have.
        exclude_tags: List of tags documents must NOT have.
        match_mode: Whether documents must have ALL or ANY of the include_tags.
            Use "all" for AND logic (default), "any" for OR logic.

    """

    include_properties: list[dict] | None = None
    exclude_properties: list[dict] | None = None
    include_tags: list[str] | None = None
    exclude_tags: list[str] | None = None
    match_mode: Literal["all", "any"] = "all"


@dataclass
class IngestHandlerParams:
    """Parameters for _ingest_handler."""

    settings: Settings
    db_manager: DatabaseManager
    embedding_provider: EmbeddingProvider | None
    vault_name: str
    path_override: str | None
    no_delete: bool | None = None
    force: bool = False


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
            include_content=params.get("include_content", True),
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
    filters: QueryFilterParams | None,
) -> TagFilter | None:
    """Build TagFilter from QueryFilterParams.

    Args:
        filters: Query filter parameters containing include_tags, exclude_tags,
            and match_mode.

    Returns:
        TagFilter or None if no tags specified.

    """
    _msg = "_create_tag_filter starting"
    log.debug(_msg)
    if filters is None:
        _msg = "_create_tag_filter returning"
        log.debug(_msg)
        return None

    include_tags = filters.include_tags
    exclude_tags = filters.exclude_tags

    if not include_tags and not exclude_tags:
        _msg = "_create_tag_filter returning"
        log.debug(_msg)
        return None

    valid_match_mode = (
        filters.match_mode if filters.match_mode in ("all", "any") else "all"
    )
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


def _is_incremental_path(path: str | None, container_path: str) -> bool:
    """Determine whether a path is incremental — inside container_path but not equal.

    Args:
        path: Candidate path to check (may be None).
        container_path: The vault's configured container_path.

    Returns:
        True if path resolves to a strict subdirectory of container_path;
        False if path is None, equals container_path, or is outside it.

    Notes:
        Uses Path.resolve() for symlink and trailing-slash normalization,
        mirroring the _validate_path_matches_vault() pattern in cli_ingest.py.

    """
    _msg = "_is_incremental_path starting"
    log.debug(_msg)
    if path is None:
        _msg = "_is_incremental_path returning: False (path is None)"
        log.debug(_msg)
        return False
    resolved_path = Path(path).resolve()
    resolved_container = Path(container_path).resolve()
    if resolved_path == resolved_container:
        _msg = "_is_incremental_path returning: False (equals container_path)"
        log.debug(_msg)
        return False
    try:
        resolved_path.relative_to(resolved_container)
    except ValueError:
        _msg = "_is_incremental_path returning: False (outside container_path)"
        log.debug(_msg)
        return False
    _msg = "_is_incremental_path returning: True (incremental)"
    log.debug(_msg)
    return True


def _resolve_no_delete(
    path_override: str | None,
    container_path: str,
    *,
    no_delete: bool | None,
) -> bool:
    """Resolve no_delete, auto-forcing True for incremental paths when unspecified.

    Args:
        path_override: Optional MCP path override.
        container_path: Vault's configured container_path.
        no_delete: Client-specified value (None = not specified).

    Returns:
        Concrete bool to pass to IngestVaultOptions.

    """
    _msg = "_resolve_no_delete starting"
    log.debug(_msg)
    if no_delete is not None:
        _msg = f"_resolve_no_delete returning: {no_delete} (explicit)"
        log.debug(_msg)
        return no_delete
    if _is_incremental_path(path_override, container_path):
        _msg = (
            f"no_delete auto-enabled: incremental path '{path_override}' "
            "is a subdirectory of vault container_path"
        )
        log.info(_msg)
        _msg = "_resolve_no_delete returning: True (auto-forced)"
        log.debug(_msg)
        return True
    _msg = "_resolve_no_delete returning: False (default)"
    log.debug(_msg)
    return False


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

    no_delete = _resolve_no_delete(
        params.path_override,
        vault_config.container_path,
        no_delete=params.no_delete,
    )
    options = IngestVaultOptions(
        vault=params.vault_name,
        dry_run=False,
        no_delete=no_delete,
        force=params.force,
    )
    result = ingestion_service.ingest_vault(path, options)

    _msg = f"ingest handler completed: {result.message}"
    log.info(_msg)

    return result.to_dict()


@dataclass
class TaskDateFilterStrings:
    """Date filter string parameters for get_tasks handler.

    Attributes:
        due_after: ISO date string for due date lower bound.
        due_before: ISO date string for due date upper bound.
        scheduled_after: ISO date string for scheduled date lower bound.
        scheduled_before: ISO date string for scheduled date upper bound.
        completion_after: ISO date string for completion date lower bound.
        completion_before: ISO date string for completion date upper bound.
        match_mode: How to combine date filters - "all" for AND logic (default),
            "any" for OR logic across all date conditions.

    """

    due_after: str | None = None
    due_before: str | None = None
    scheduled_after: str | None = None
    scheduled_before: str | None = None
    completion_after: str | None = None
    completion_before: str | None = None
    match_mode: Literal["all", "any"] = "all"


@dataclass
class TagFilterStrings:
    """Tag filter string parameters for get_tasks handler.

    Tags should NOT include the '#' prefix. Use plain tag names like
    "personal/expenses" instead of "#personal/expenses".

    Attributes:
        include_tags: List of tags that tasks must have.
            Use match_mode to control AND vs OR logic.
        exclude_tags: List of tags that tasks must NOT have.
            Always uses OR logic (any excluded tag disqualifies).
        match_mode: How to combine include_tags - "all" for AND logic (default),
            "any" for OR logic (task matches if it has ANY of the include_tags).

    """

    include_tags: list[str] | None = None
    exclude_tags: list[str] | None = None
    match_mode: Literal["all", "any"] = "all"


def _get_tasks_handler(
    db_manager: DatabaseManager,
    request: "GetTasksRequest",
) -> dict[str, object]:
    """Handle get_tasks tool call with comprehensive filtering.

    Args:
        db_manager: Database manager for session management.
        request: GetTasksRequest containing all filter parameters.

    Returns:
        Dictionary with task list response.

    Raises:
        ValueError: If tag filter validation fails (conflicting tags).

    Notes:
        Parses ISO date strings and builds GetTasksFilterParams.
        Invalid date strings are logged and treated as None.
        Tag validation prevents conflicting tags in include/exclude lists.

    """
    _msg = "_get_tasks_handler starting"
    log.debug(_msg)

    # Use default filters if none provided
    date_filters = request.date_filters or TaskDateFilterStrings()
    tag_filters = request.tag_filters or TagFilterStrings()

    # Parse all date parameters
    due_after_date = parse_iso_date(date_filters.due_after)
    due_before_date = parse_iso_date(date_filters.due_before)
    scheduled_after_date = parse_iso_date(date_filters.scheduled_after)
    scheduled_before_date = parse_iso_date(date_filters.scheduled_before)
    completion_after_date = parse_iso_date(date_filters.completion_after)
    completion_before_date = parse_iso_date(date_filters.completion_before)

    # Build filter parameters with all tag filtering options
    filters = GetTasksFilterParams(
        status=request.status,
        due_after=due_after_date,
        due_before=due_before_date,
        scheduled_after=scheduled_after_date,
        scheduled_before=scheduled_before_date,
        completion_after=completion_after_date,
        completion_before=completion_before_date,
        include_tags=tag_filters.include_tags,
        exclude_tags=tag_filters.exclude_tags,
        tag_match_mode=tag_filters.match_mode,
        priority=request.priority,
        date_match_mode=date_filters.match_mode,
        include_content=request.include_content,
        limit=request.limit,
        offset=request.offset,
    )

    with db_manager.get_session() as session:
        result = get_tasks_tool(session=session, filters=filters)
        _msg = "_get_tasks_handler returning"
        log.debug(_msg)
        return result.model_dump()


def _get_vault_handler(
    db_manager: DatabaseManager,
    *,
    name: str | None = None,
    vault_id: str | None = None,
) -> dict[str, object]:
    """Handle get_vault tool call.

    Args:
        db_manager: Database manager for sessions.
        name: Vault name to lookup (preferred if both provided).
        vault_id: Vault UUID string to lookup.

    Returns:
        VaultResponse as dictionary on success, or error dict on failure.

    Notes:
        Catches ValueError from get_vault tool and returns error dict.

    """
    _msg = "_get_vault_handler starting"
    log.debug(_msg)

    try:
        with db_manager.get_session() as session:
            result = get_vault(
                session=session,
                name=name,
                vault_id=vault_id,
            )
            _msg = "_get_vault_handler returning"
            log.debug(_msg)
            return result.model_dump()
    except ValueError as err:
        _msg = f"_get_vault_handler error: {err}"
        log.error(_msg)
        return {"success": False, "error": str(err)}


def _update_vault_handler(
    db_manager: DatabaseManager,
    params: VaultUpdateParams,
) -> dict[str, object]:
    """Handle update_vault tool call.

    Args:
        db_manager: Database manager for sessions.
        params: Vault update parameters.

    Returns:
        VaultResponse as dictionary on success, or error dict on failure.

    Notes:
        Catches ValueError from update_vault tool and returns error dict.
        Returns error dict directly from tool when force is required.

    """
    _msg = "_update_vault_handler starting"
    log.debug(_msg)

    try:
        with db_manager.get_session() as session:
            result = update_vault(
                session=session,
                params=params,
            )
            # Check if result is an error dict (when force is required)
            if isinstance(result, dict):
                _msg = "_update_vault_handler returning (error dict)"
                log.debug(_msg)
                return result
            # Otherwise it's a VaultResponse
            _msg = "_update_vault_handler returning"
            log.debug(_msg)
            return result.model_dump()
    except ValueError as err:
        _msg = f"_update_vault_handler error: {err}"
        log.error(_msg)
        return {"success": False, "error": str(err)}


def _delete_vault_handler(
    db_manager: DatabaseManager,
    *,
    name: str,
    confirm: bool,
) -> dict[str, object]:
    """Handle delete_vault tool call.

    Args:
        db_manager: Database manager for sessions.
        name: Vault name to delete.
        confirm: Must be True to proceed with deletion.

    Returns:
        Success dict with deletion counts if confirmed, or error dict.

    Notes:
        Catches ValueError from delete_vault tool and returns error dict.
        Returns error dict directly when confirm is not True.

    """
    _msg = "_delete_vault_handler starting"
    log.debug(_msg)

    try:
        with db_manager.get_session() as session:
            result = delete_vault(
                session=session,
                name=name,
                confirm=confirm,
            )
            # Result is always a dict (success or error)
            _msg = "_delete_vault_handler returning"
            log.debug(_msg)
            return result
    except ValueError as err:
        _msg = f"_delete_vault_handler error: {err}"
        log.error(_msg)
        return {"success": False, "error": str(err)}


@dataclass
class GetDocumentHandlerParams:
    """Input parameters for the get_document handler.

    Attributes:
        db_manager: Database manager for session management.
        vault_name: Vault name (required when using file_path).
        file_path: Relative file path from vault root.
        document_id: Document UUID string.
        include_content: Whether to include document content in the response.
    """

    db_manager: DatabaseManager
    vault_name: str | None = None
    file_path: str | None = None
    document_id: str | None = None
    include_content: bool = True


@dataclass
class ListDocumentsHandlerParams:
    """Input parameters for the list_documents handler.

    Attributes:
        db_manager: Database manager for session management.
        file_name: Document file name to search for.
        vault_name: Optional vault name to scope results.
        limit: Maximum number of results.
        offset: Number of results to skip.
        include_content: Whether to include document content in each result.
    """

    db_manager: DatabaseManager
    file_name: str | None = None
    vault_name: str | None = None
    limit: int = 20
    offset: int = 0
    include_content: bool = True


def _get_document_handler(params: GetDocumentHandlerParams) -> dict[str, object]:
    """Handle get_document tool call.

    Args:
        params: GetDocumentHandlerParams with db_manager and lookup fields.

    Returns:
        Document response as dictionary on success, or error dict on failure.

    Notes:
        Catches ValueError from get_document tool and returns error dict.

    """
    _msg = "_get_document_handler starting"
    log.debug(_msg)

    try:
        with params.db_manager.get_session() as session:
            from obsidian_rag.mcp_server.tools.documents import (
                get_document as get_document_impl,
            )

            result = get_document_impl(
                session=session,
                vault_name=params.vault_name,
                file_path=params.file_path,
                document_id=params.document_id,
                include_content=params.include_content,
            )
            _msg = "_get_document_handler returning"
            log.debug(_msg)
            return result.model_dump()
    except ValueError as err:
        _msg = f"_get_document_handler error: {err}"
        log.error(_msg)
        return {"success": False, "error": str(err)}


def _list_documents_handler(params: ListDocumentsHandlerParams) -> dict[str, object]:
    """Handle list_documents tool call.

    Args:
        params: ListDocumentsHandlerParams with db_manager and filter fields.

    Returns:
        Document list response as dictionary on success, or error dict on failure.

    Notes:
        Catches ValueError from list_documents tool and returns error dict.

    """
    _msg = "_list_documents_handler starting"
    log.debug(_msg)

    try:
        with params.db_manager.get_session() as session:
            from obsidian_rag.mcp_server.tools.documents import (
                list_documents as list_documents_impl,
            )

            result = list_documents_impl(
                session=session,
                file_name=params.file_name,
                vault_name=params.vault_name,
                limit=params.limit,
                offset=params.offset,
                include_content=params.include_content,
            )
            _msg = "_list_documents_handler returning"
            log.debug(_msg)
            return result.model_dump()
    except ValueError as err:
        _msg = f"_list_documents_handler error: {err}"
        log.error(_msg)
        return {"success": False, "error": str(err)}


# Annotated types for MCP tool parameters that may be passed as JSON strings
# These are defined at the end of the module to avoid forward reference issues
AnnotatedQueryFilter = Annotated[
    QueryFilterParams | None,
    BeforeValidator(parse_json_str),
]
"""QueryFilterParams that can be passed as JSON string or dict.

This type wraps QueryFilterParams with a BeforeValidator that automatically
parses JSON strings before Pydantic validation. This makes the server robust
to clients that double-encode their parameters.

Examples:
    - Dict input: {"include_tags": ["work"], "match_mode": "any"}
    - JSON string input: '{"include_tags": ["work"], "match_mode": "any"}'
    - None input: None (returns None)
    - Empty string input: "" (returns None)

"""
