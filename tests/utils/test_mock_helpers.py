"""Tests for mock helper utilities."""

from unittest.mock import MagicMock

import pytest

from tests.utils.mock_helpers import (
    configure_mock_query_chain,
    configure_mock_with_documents,
    configure_mock_with_tasks,
)


def test_configure_mock_query_chain():
    """Test that mock query chain is properly configured."""
    session = MagicMock()
    sample_data = [{"id": 1}, {"id": 2}]

    configure_mock_query_chain(session, return_value=sample_data, count_value=2)

    # Verify the chain works
    result = session.query().filter().order_by().all()
    assert result == sample_data

    count = session.query().filter().count()
    assert count == 2


def test_configure_mock_query_chain_first():
    """Test that first() returns first item."""
    session = MagicMock()
    sample_data = [{"id": 1}, {"id": 2}]

    configure_mock_query_chain(session, return_value=sample_data, count_value=2)

    first = session.query().filter().first()
    assert first == {"id": 1}


def test_configure_mock_query_chain_empty():
    """Test that empty return value works correctly."""
    session = MagicMock()

    configure_mock_query_chain(session, return_value=[], count_value=0)

    result = session.query().filter().all()
    assert result == []

    first = session.query().filter().first()
    assert first is None


def test_configure_mock_with_documents():
    """Test configuring mock with documents."""
    session = MagicMock()
    docs = [{"id": 1}, {"id": 2}]

    configure_mock_with_documents(session, docs)

    result = session.query().all()
    assert result == docs
    assert session.query().count() == 2


def test_configure_mock_with_tasks():
    """Test configuring mock with tasks."""
    session = MagicMock()
    tasks = [("task1", "doc1"), ("task2", "doc2")]

    configure_mock_with_tasks(session, tasks)

    result = session.query().all()
    assert result == tasks
    assert session.query().count() == 2
