"""Test fixtures and configuration."""

import pytest
from unittest.mock import MagicMock


@pytest.fixture
def mock_db_session():
    """Create a mock database session for PostgreSQL testing."""
    session = MagicMock()
    session.bind.dialect.name = "postgresql"

    # Setup query chain for common operations
    query_mock = MagicMock()
    session.query.return_value = query_mock
    query_mock.filter.return_value = query_mock
    query_mock.order_by.return_value = query_mock
    query_mock.limit.return_value = query_mock
    query_mock.offset.return_value = query_mock
    query_mock.join.return_value = query_mock
    query_mock.all.return_value = []
    query_mock.first.return_value = None
    query_mock.count.return_value = 0

    return session


@pytest.fixture
def db_session(mock_db_session):
    """Alias for mock_db_session for backward compatibility."""
    return mock_db_session


@pytest.fixture
def mock_db_manager():
    """Create a mock database manager."""
    manager = MagicMock()
    manager.get_session.return_value.__enter__ = MagicMock(return_value=MagicMock())
    manager.get_session.return_value.__exit__ = MagicMock(return_value=False)
    return manager
