"""Mock helper utilities for database testing.

This module provides helper functions for configuring mock database
sessions with realistic return values for PostgreSQL testing.
"""

from typing import Any
from unittest.mock import MagicMock


def configure_mock_query_chain(
    session: MagicMock,
    return_value: Any,
    count_value: int = 0,
) -> None:
    """Configure mock session query chain to return specific values.

    Args:
        session: The mock database session to configure.
        return_value: Value to return from .all() call (list of objects).
        count_value: Value to return from .count() call (int).

    Example:
        >>> configure_mock_query_chain(
        ...     db_session,
        ...     return_value=sample_documents,
        ...     count_value=len(sample_documents),
        ... )

    """
    query_mock = MagicMock()
    query_mock.all.return_value = return_value
    query_mock.count.return_value = count_value
    query_mock.filter.return_value = query_mock
    query_mock.order_by.return_value = query_mock
    query_mock.limit.return_value = query_mock
    query_mock.offset.return_value = query_mock
    query_mock.join.return_value = query_mock
    query_mock.first.return_value = return_value[0] if return_value else None

    session.query.return_value = query_mock


def configure_mock_with_documents(
    session: MagicMock,
    documents: list[Any],
) -> None:
    """Configure mock to return documents for Document queries.

    Args:
        session: The mock database session.
        documents: List of Document objects to return.

    """
    configure_mock_query_chain(
        session,
        return_value=documents,
        count_value=len(documents),
    )


def configure_mock_with_tasks(
    session: MagicMock,
    tasks: list[tuple[Any, Any]],
) -> None:
    """Configure mock to return tasks for Task queries.

    Args:
        session: The mock database session.
        tasks: List of (Task, Document) tuples to return.

    """
    configure_mock_query_chain(
        session,
        return_value=tasks,
        count_value=len(tasks),
    )
