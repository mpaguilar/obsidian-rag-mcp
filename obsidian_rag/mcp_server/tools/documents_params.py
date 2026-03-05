"""Query parameter dataclasses for document tools.

This module contains dataclasses for bundling query parameters
to comply with the 5 argument limit per function (PLR0913).
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

    from obsidian_rag.mcp_server.models import PropertyFilter, TagFilter


@dataclass
class PaginationParams:
    """Pagination parameters for query results.

    Attributes:
        limit: Maximum number of results.
        offset: Number of results to skip.

    """

    limit: int
    offset: int


@dataclass
class PropertyFilterParams:
    """Property filter parameters for document queries.

    Attributes:
        include_filters: Property filters to include (AND logic).
        exclude_filters: Property filters to exclude (OR logic).

    """

    include_filters: Optional[list["PropertyFilter"]]
    exclude_filters: Optional[list["PropertyFilter"]]


@dataclass
class TagFilterParams:
    """Tag filter parameters for document queries.

    Attributes:
        tag_filter: Tag filter with include/exclude lists.

    """

    tag_filter: Optional["TagFilter"]


@dataclass
class QueryFilterParams:
    """Combined filter parameters for document queries.

    Attributes:
        property_filters: Property filter parameters.
        tag_params: Tag filter parameters.

    """

    property_filters: PropertyFilterParams
    tag_params: TagFilterParams


@dataclass
class DocumentQueryParams:
    """Complete query parameters for document queries.

    Attributes:
        session: Database session.
        query_embedding: Vector embedding for semantic search.
        filter_params: Combined filter parameters.
        pagination: Pagination parameters.

    """

    session: "Session"
    query_embedding: list[float]
    filter_params: QueryFilterParams
    pagination: PaginationParams


@dataclass
class PropertyQueryParams:
    """Parameters for property-based document queries.

    Attributes:
        session: Database session.
        property_filters: Property filter parameters.
        tag_params: Tag filter parameters.
        vault_root: Optional vault root filter.
        pagination: Pagination parameters.

    """

    session: "Session"
    property_filters: PropertyFilterParams
    tag_params: TagFilterParams
    vault_root: Optional[str]
    pagination: PaginationParams
