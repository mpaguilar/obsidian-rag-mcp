"""Test fixtures and configuration."""

import logging
import sys
from unittest.mock import MagicMock

import pytest

log = logging.getLogger(__name__)

# Mock litellm before it's imported (Python 3.14 compatibility)
# This must happen before any obsidian_rag imports
_LITELL_MOCK = MagicMock()
_LITELL_MOCK.embedding.return_value = {"data": [{"embedding": [0.1] * 1536}]}
_LITELL_MOCK.completion.return_value = {"choices": [{"message": {"content": "test"}}]}
sys.modules["litellm"] = _LITELL_MOCK


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


@pytest.fixture
def rollback_session():
    """Create a database session with automatic rollback for test isolation.

    Notes:
        This fixture provides a session that is automatically rolled back
        after each test, preventing test pollution. If no real db_engine
        is available, it yields a mock session for compatibility.
    """
    _msg = "rollback_session fixture starting"
    log.debug(_msg)

    engine = MagicMock()
    conn = engine.connect()
    trans = conn.begin()
    session = MagicMock()
    session.bind = engine

    yield session

    session.close()
    trans.rollback()
    conn.close()

    _msg = "rollback_session fixture returning"
    log.debug(_msg)
