"""MCP tool implementations and registry.

This module contains the MCPToolRegistry class and all tool implementation
functions that were previously in server.py. This separation keeps server.py
under the 1000 line limit per CONVENTIONS.md.
"""

import logging
from collections.abc import Callable
from typing import TypedDict

from obsidian_rag.config import EndpointConfig, Settings
from obsidian_rag.database.engine import DatabaseManager
from obsidian_rag.llm.base import EmbeddingProvider
from obsidian_rag.llm.providers import ProviderFactory
from obsidian_rag.mcp_server.handlers import (
    QueryFilterParams,
    _convert_property_filters,
    _create_tag_filter,
    _delete_vault_handler,
    _get_all_tags_handler,
    _get_tasks_handler,
    _get_vault_handler,
    _list_vaults_handler,
    _update_vault_handler,
)
from obsidian_rag.mcp_server.tools.documents_params import PaginationParams
from obsidian_rag.mcp_server.tools.tasks_params import GetTasksRequest
from obsidian_rag.mcp_server.tools.vaults_params import VaultUpdateParams

log = logging.getLogger(__name__)

# Module-level tool registry (initialized at server startup)
_tool_registry: "MCPToolRegistry | None" = None


class DocumentTagParamsImport(TypedDict, total=False):
    """Parameters for get_documents_by_tag tool."""

    include_tags: list[str]
    exclude_tags: list[str]
    match_mode: str
    vault_name: str | None
    limit: int
    offset: int


class MCPToolRegistry:
    """Holds dependencies for MCP tools.

    This class holds the dependencies (db_manager, embedding_provider, settings)
    as instance attributes. Tools access it through the module-level variable.

    Attributes:
        db_manager: Database manager for session management.
        embedding_provider: Optional embedding provider for semantic search.
        settings: Application settings.

    """

    def __init__(
        self,
        db_manager: DatabaseManager,
        embedding_provider: EmbeddingProvider | None,
        settings: Settings,
    ) -> None:
        """Initialize the tool registry with dependencies.

        Args:
            db_manager: Database manager for session management.
            embedding_provider: Optional embedding provider for semantic search.
            settings: Application settings.

        """
        self.db_manager = db_manager
        self.embedding_provider = embedding_provider
        self.settings = settings


def _get_registry() -> MCPToolRegistry:
    """Get the global tool registry.

    Returns:
        The global MCPToolRegistry instance.

    Raises:
        RuntimeError: If registry has not been initialized.

    """
    if _tool_registry is None:
        _msg = "Tool registry not initialized"
        raise RuntimeError(_msg)
    return _tool_registry


def _set_registry(registry: MCPToolRegistry | None) -> None:
    """Set the global tool registry.

    Args:
        registry: The MCPToolRegistry instance to set, or None to clear.

    Notes:
        This function is used by create_mcp_server to initialize the registry
        before registering tools. Tests can also use this to inject mocks.

    """
    global _tool_registry
    _tool_registry = registry


def query_documents_tool(
    db_manager: DatabaseManager,
    embedding_provider: EmbeddingProvider | None,
    query: str,
    filters: QueryFilterParams | None = None,
    pagination: "PaginationParams | None" = None,
    *,
    use_chunks: bool = False,
    rerank: bool = False,
) -> dict[str, object]:
    """Tool implementation for semantic search over document content using chunk-based search.

    Large documents are split into chunks for better embedding quality. The search
    queries all chunks and returns the best matching chunk per document, with
    document relevance determined by the highest chunk similarity score.

    Args:
        db_manager: Database manager for session management.
        embedding_provider: Embedding provider for generating query embeddings.
        query: Search query text.
        filters: QueryFilterParams with include_properties, exclude_properties,
            include_tags, exclude_tags, and match_mode.
        pagination: Pagination parameters (default: limit=20, offset=0).
        use_chunks: If True, search at chunk level instead of document level.
            Returns the best matching chunk per document for more precise
            semantic matching in large documents.
        rerank: If True, apply flashrank re-ranking to chunk results.
            Only applies when use_chunks is True.

    Returns:
        Document list response with pagination and similarity scores.
        When use_chunks is True, the content field contains the matching
        chunk text and matching_chunk field indicates chunk search was used.

    Raises:
        RuntimeError: If embedding provider is not available.
        ValueError: If filter validation fails.

    Notes:
        This is a module-level function for testability.
        The @mcp.tool() decorator is applied in the registration function.
        Requires network access to embedding provider API.

    """
    from obsidian_rag.mcp_server.tools.documents import (
        query_documents as query_documents_impl,
    )
    from obsidian_rag.mcp_server.tools.documents_params import (
        PropertyFilterParams,
    )

    _msg = f"Tool query_documents called with query: {query[:50]}..."
    log.info(_msg)

    if not embedding_provider:
        _msg = "Embedding provider not configured"
        log.error(_msg)
        raise RuntimeError(_msg)

    query_embedding = embedding_provider.generate_embedding(query)

    # Ensure filters is a QueryFilterParams dataclass
    query_filters = filters or QueryFilterParams(
        include_properties=None,
        exclude_properties=None,
        include_tags=None,
        exclude_tags=None,
        match_mode="all",
    )

    prop_filters_include = _convert_property_filters(
        query_filters.include_properties,
    )
    prop_filters_exclude = _convert_property_filters(
        query_filters.exclude_properties,
    )
    tag_filter = _create_tag_filter(query_filters)

    # Bundle property filters into PropertyFilterParams
    property_filter_params = PropertyFilterParams(
        include_filters=prop_filters_include,
        exclude_filters=prop_filters_exclude,
    )
    pagination_params = pagination or PaginationParams(limit=20, offset=0)

    with db_manager.get_session() as session:
        result = query_documents_impl(
            session=session,
            query_embedding=query_embedding,
            filter_params=property_filter_params,
            tag_filter=tag_filter,
            pagination=pagination_params,
            use_chunks=use_chunks,
            rerank=rerank,
            query_text=query,
        )
        return result.model_dump()


def _create_openai_provider(config: EndpointConfig) -> EmbeddingProvider:
    """Create OpenAI embedding provider.

    Args:
        config: Endpoint configuration.

    Returns:
        OpenAI embedding provider instance.

    """
    return ProviderFactory.create_embedding_provider(
        provider_name="openai",
        config={
            "api_key": config.api_key,
            "model": config.model,
            "base_url": config.base_url,
        },
    )


def _create_openrouter_provider(config: EndpointConfig) -> EmbeddingProvider:
    """Create OpenRouter embedding provider.

    Args:
        config: Endpoint configuration.

    Returns:
        OpenRouter embedding provider instance.

    """
    return ProviderFactory.create_embedding_provider(
        provider_name="openrouter",
        config={
            "api_key": config.api_key,
            "model": config.model,
            "base_url": config.base_url,
        },
    )


def _create_huggingface_provider(config: EndpointConfig) -> EmbeddingProvider:
    """Create HuggingFace embedding provider.

    Args:
        config: Endpoint configuration.

    Returns:
        HuggingFace embedding provider instance.

    """
    return ProviderFactory.create_embedding_provider(
        provider_name="huggingface",
        config={"model": config.model},
    )


def _get_provider_creator(
    provider_name: str,
) -> "Callable[[EndpointConfig], EmbeddingProvider]":
    """Get provider creation function for the given provider name.

    Args:
        provider_name: Name of the embedding provider.

    Returns:
        Function that creates the embedding provider.

    Raises:
        ValueError: If provider name is unknown.

    """
    from collections.abc import Callable

    creators: dict[str, Callable[[EndpointConfig], EmbeddingProvider]] = {
        "openai": _create_openai_provider,
        "openrouter": _create_openrouter_provider,
        "huggingface": _create_huggingface_provider,
    }

    if provider_name not in creators:
        _msg = f"Unknown embedding provider: {provider_name}"
        raise ValueError(_msg)

    return creators[provider_name]


def _create_embedding_provider(
    settings: Settings,
) -> EmbeddingProvider | None:
    """Create embedding provider for semantic search.

    Args:
        settings: Application settings.

    Returns:
        Embedding provider instance or None if no embedding config exists
            or if provider creation fails (e.g., missing API key).

    Raises:
        ValueError: If provider name is unknown.
        ImportError: If required dependency is not installed.

    """
    _msg = "_create_embedding_provider starting"
    log.debug(_msg)

    embedding_config = settings.endpoints.get("embedding")
    if not embedding_config:
        _msg = "No embedding configuration found, returning None"
        log.debug(_msg)
        return None

    try:
        creator = _get_provider_creator(embedding_config.provider)
        provider = creator(embedding_config)
    except (ValueError, ImportError) as e:
        _msg = f"Failed to create embedding provider '{embedding_config.provider}': {e}"
        log.warning(_msg)
        return None

    _msg = "_create_embedding_provider returning"
    log.debug(_msg)
    return provider


def get_all_tags_tool(
    db_manager: DatabaseManager,
    pattern: str | None = None,
    limit: int = 20,
    offset: int = 0,
) -> dict[str, object]:
    """Tool implementation for querying all unique document tags.

    Args:
        db_manager: Database manager for session management.
        pattern: Glob pattern for filtering tags (optional).
            Supports * (any chars), ? (single char), [abc] (char class).
        limit: Maximum number of results (default: 20, max: 100).
        offset: Number of results to skip (default: 0).

    Returns:
        Dictionary with tag list response and pagination info.

    Notes:
        This is a module-level function for testability.
        The @mcp.tool() decorator is applied in the registration function.

    """
    _msg = "Tool get_all_tags called"
    log.info(_msg)
    return _get_all_tags_handler(db_manager, pattern, limit, offset)


def list_vaults_tool(
    db_manager: DatabaseManager,
    limit: int = 20,
    offset: int = 0,
) -> dict[str, object]:
    """Tool implementation for listing all configured vaults.

    Args:
        db_manager: Database manager for session management.
        limit: Maximum number of results (default 20, max 100).
        offset: Number of results to skip.

    Returns:
        Dictionary with vault list response including metadata and document counts.

    Notes:
        This is a module-level function for testability.
        The @mcp.tool() decorator is applied in the registration function.

    """
    _msg = "Tool list_vaults called"
    log.info(_msg)
    return _list_vaults_handler(db_manager, limit, offset)


def get_tasks_tool(
    db_manager: DatabaseManager,
    request: "GetTasksRequest",
) -> dict[str, object]:
    """Tool implementation for querying tasks with comprehensive filtering.

    This tool provides flexible task filtering by status, date ranges, tags,
    and priority. All filters are optional and combined with AND logic by default.

    Valid Status Values:
        - "not_completed": Tasks that are not yet completed
        - "completed": Tasks that have been completed
        - "in_progress": Tasks currently being worked on
        - "cancelled": Tasks that have been cancelled

    Valid Priority Values:
        - "highest": Critical priority tasks
        - "high": High priority tasks
        - "normal": Normal priority tasks (default)
        - "low": Low priority tasks
        - "lowest": Lowest priority tasks

    Tag Filtering:
        Tags should NOT include the '#' prefix. Use plain tag names like
        "personal/expenses" or "business/iConnections" instead of
        "#personal/expenses" or "#business/iConnections".

        Tag filters are specified in the request.tag_filters object:

        include_tags: Tasks must have these tags (controlled by match_mode).
            - match_mode="all" (default): Task must have ALL include tags
            - match_mode="any": Task must have ANY of the include tags

        exclude_tags: Tasks must NOT have any of these tags (always OR logic).

        Examples:
            - tag_filters={"include_tags": ["work", "urgent"], "match_mode": "all"}:
              Returns tasks with BOTH "work" AND "urgent" tags
            - tag_filters={"include_tags": ["work", "personal"], "match_mode": "any"}:
              Returns tasks with EITHER "work" OR "personal" tag
            - tag_filters={"exclude_tags": ["blocked"]}: Returns tasks WITHOUT "blocked" tag
            - tag_filters={"include_tags": ["work"], "exclude_tags": ["blocked"]}:
              Returns tasks with "work" but NOT "blocked"

        Validation:
            - Tags cannot appear in both include_tags and exclude_tags
            - Case-insensitive matching ("Work" matches "work")

    Date Filtering:
        Date filters are specified in the request.date_filters object:

        Available date fields:
            - due_after: Filter tasks due on or after this date
            - due_before: Filter tasks due on or before this date
            - scheduled_after: Filter tasks scheduled on or after this date
            - scheduled_before: Filter tasks scheduled on or before this date
            - completion_after: Filter tasks completed on or after this date
            - completion_before: Filter tasks completed on or before this date

        match_mode: How to combine date filters
            - "all" (default): AND logic across all date conditions
            - "any": OR logic across all date conditions

    Filter Logic Summary:
        - Multiple status values: OR logic (task matches ANY status)
        - Multiple priority values: OR logic (task matches ANY priority)
        - tag_filters.include_tags with match_mode="all": AND logic
        - tag_filters.include_tags with match_mode="any": OR logic
        - tag_filters.exclude_tags: OR logic (any match excludes task)
        - Date filters: Configurable via date_filters.match_mode
            - "all" (default): AND logic across all date conditions
            - "any": OR logic across all date conditions
        - Different filter types (status, tags, priority, dates): AND logic

    Args:
        db_manager: Database manager for session management.
        request: GetTasksRequest containing all filter parameters.

    Returns:
        Dictionary with task list response including:
        - results: List of matching tasks
        - total_count: Total number of matching tasks
        - has_more: Whether more results are available
        - next_offset: Offset for next page (if has_more is True)

    Raises:
        ValueError: If tag filter validation fails (conflicting tags in
            include_tags and exclude_tags).

    Notes:
        This is a module-level function for testability.
        The @mcp.tool() decorator is applied in the registration function.
        Date comparisons are inclusive (>= for after, <= for before).

    """
    _msg = "Tool get_tasks called"
    log.info(_msg)

    return _get_tasks_handler(db_manager, request)


def get_vault_tool(
    db_manager: DatabaseManager,
    *,
    name: str | None = None,
    vault_id: str | None = None,
) -> dict[str, object]:
    """Tool implementation for getting a single vault by name or ID.

    Retrieves vault details including document count. Either name or vault_id
    must be provided, with name taking precedence if both are given.

    Args:
        db_manager: Database manager for session management.
        name: Vault name to lookup (preferred if both provided).
        vault_id: Vault UUID string to lookup.

    Returns:
        Vault response as dictionary on success, or error dict on failure:
        - Success: {"id": ..., "name": ..., "description": ...}
        - Error: {"success": False, "error": "..."}

    Notes:
        This is a module-level function for testability.
        The @mcp.tool() decorator is applied in the registration function.

    """
    _msg = "Tool get_vault called"
    log.info(_msg)
    return _get_vault_handler(db_manager, name=name, vault_id=vault_id)


def update_vault_tool(
    db_manager: DatabaseManager,
    params: VaultUpdateParams,
) -> dict[str, object]:
    """Tool implementation for updating a vault's properties.

    The name field in params is used for lookup only and cannot be changed.
    Changing container_path requires force=True as it deletes all documents.

    Args:
        db_manager: Database manager for session management.
        params: Vault update parameters including name for lookup.

    Returns:
        Vault response as dictionary on success, or error dict on failure:
        - Success: {"id": ..., "name": ..., "description": ...}
        - Error: {"success": False, "error": "..."}

    Notes:
        This is a module-level function for testability.
        The @mcp.tool() decorator is applied in the registration function.
        Changing container_path is destructive - it deletes all documents,
        tasks, and chunks for the vault.

    """
    _msg = "Tool update_vault called"
    log.info(_msg)
    return _update_vault_handler(db_manager, params)


def delete_vault_tool(
    db_manager: DatabaseManager,
    *,
    name: str,
    confirm: bool,
) -> dict[str, object]:
    """Tool implementation for deleting a vault and all associated data.

    This operation is irreversible and cascade-deletes all associated documents,
    tasks, and chunks. Requires explicit confirmation via confirm=True parameter.

    Args:
        db_manager: Database manager for session management.
        name: Vault name to delete.
        confirm: Must be True to proceed with deletion.

    Returns:
        Success dict with deletion counts if confirmed:
        {"success": True, "name": ..., "documents_deleted": ..., ...}
        Error dict if not confirmed or vault not found:
        {"success": False, "error": "..."}

    Notes:
        This is a module-level function for testability.
        The @mcp.tool() decorator is applied in the registration function.
        The vault configuration entry in the config file is NOT deleted.

    """
    _msg = "Tool delete_vault called"
    log.info(_msg)
    return _delete_vault_handler(db_manager, name=name, confirm=confirm)
