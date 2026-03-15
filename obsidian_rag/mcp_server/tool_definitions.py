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
    TaskDateFilterStrings,
    _convert_property_filters,
    _create_tag_filter,
    _get_all_tags_handler,
    _get_tasks_handler,
    _list_vaults_handler,
)

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


def query_documents_tool(  # noqa: PLR0913
    db_manager: DatabaseManager,
    embedding_provider: EmbeddingProvider | None,
    query: str,
    filters: QueryFilterParams | None = None,
    limit: int = 20,
    offset: int = 0,
) -> dict[str, object]:
    """Tool implementation for semantic search over document content.

    Args:
        db_manager: Database manager for session management.
        embedding_provider: Embedding provider for generating query embeddings.
        query: Search query text.
        filters: QueryFilterParams with include_properties, exclude_properties,
            include_tags, exclude_tags, and match_mode.
        limit: Maximum number of results (default: 20, max: 100).
        offset: Number of results to skip (default: 0).

    Returns:
        Document list response with pagination and similarity scores.

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
        PaginationParams,
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
    pagination = PaginationParams(limit=limit, offset=offset)

    with db_manager.get_session() as session:
        result = query_documents_impl(
            session=session,
            query_embedding=query_embedding,
            filter_params=property_filter_params,
            tag_filter=tag_filter,
            pagination=pagination,
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


def get_tasks_tool(  # noqa: PLR0913
    db_manager: DatabaseManager,
    status: list[str] | None = None,
    date_filters: "TaskDateFilterStrings | None" = None,
    tags: list[str] | None = None,
    priority: list[str] | None = None,
    *,
    include_completed: bool = True,
    include_cancelled: bool = False,
    limit: int = 20,
    offset: int = 0,
) -> dict[str, object]:
    """Tool implementation for querying tasks with comprehensive filtering.

    Args:
        db_manager: Database manager for session management.
        status: List of statuses to filter by.
        date_filters: Date filter parameters with ISO date strings and match mode.
            Use date_match_mode="all" (default) for AND logic across all date filters,
            or "any" for OR logic (task matches if ANY date condition is satisfied).
        tags: List of tags to filter by.
        priority: List of priorities to filter by.
        include_completed: Whether to include completed tasks.
        include_cancelled: Whether to include cancelled tasks.
        limit: Maximum number of results.
        offset: Number of results to skip.

    Returns:
        Dictionary with task list response.

    Notes:
        This is a module-level function for testability.
        The @mcp.tool() decorator is applied in the registration function.

    """
    _msg = "Tool get_tasks called"
    log.info(_msg)
    return _get_tasks_handler(
        db_manager,
        status=status,
        date_filters=date_filters,
        tags=tags,
        priority=priority,
        include_completed=include_completed,
        include_cancelled=include_cancelled,
        limit=limit,
        offset=offset,
    )
