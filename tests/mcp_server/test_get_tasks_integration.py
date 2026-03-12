"""Integration tests for get_tasks MCP tool."""

import uuid
from datetime import date, timedelta

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from obsidian_rag.database.models import Base, Document, Task, TaskStatus, Vault
from obsidian_rag.mcp_server.handlers import _get_tasks_handler
from obsidian_rag.mcp_server.tools.tasks import get_tasks
from obsidian_rag.mcp_server.tools.tasks_params import GetTasksFilterParams


@pytest.fixture
def db_engine():
    """Create a test database engine using SQLite."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    yield engine
    Base.metadata.drop_all(engine)


@pytest.fixture
def db_session(db_engine):
    """Create a test database session."""
    SessionLocal = sessionmaker(bind=db_engine)
    session = SessionLocal()
    yield session
    session.close()


@pytest.fixture
def sample_data(db_session):
    """Create sample vault, document, and tasks for testing."""
    from datetime import datetime

    vault = Vault(
        id=uuid.uuid4(),
        name="test_vault",
        container_path="/test",
        host_path="/test",
    )
    db_session.add(vault)
    db_session.commit()

    doc = Document(
        id=uuid.uuid4(),
        vault_id=vault.id,
        file_path="/test/doc.md",
        file_name="doc.md",
        content="# Test",
        checksum_md5="abc123",
        created_at_fs=datetime.now(),
        modified_at_fs=datetime.now(),
        frontmatter_json={},
        tags=["work", "urgent"],
    )
    db_session.add(doc)
    db_session.commit()

    today = date.today()

    # Create diverse tasks
    tasks = [
        Task(
            id=uuid.uuid4(),
            document_id=doc.id,
            line_number=1,
            raw_text="- [ ] High priority due soon",
            status=TaskStatus.NOT_COMPLETED.value,
            description="High priority due soon",
            due=today + timedelta(days=2),
            priority="high",
            tags=["work"],
        ),
        Task(
            id=uuid.uuid4(),
            document_id=doc.id,
            line_number=2,
            raw_text="- [ ] Normal priority due later",
            status=TaskStatus.NOT_COMPLETED.value,
            description="Normal priority due later",
            due=today + timedelta(days=10),
            priority="normal",
            tags=["work"],
        ),
        Task(
            id=uuid.uuid4(),
            document_id=doc.id,
            line_number=3,
            raw_text="- [x] Completed yesterday",
            status=TaskStatus.COMPLETED.value,
            description="Completed yesterday",
            completion=today - timedelta(days=1),
            priority="normal",
        ),
        Task(
            id=uuid.uuid4(),
            document_id=doc.id,
            line_number=4,
            raw_text="- [-] Cancelled task",
            status=TaskStatus.CANCELLED.value,
            description="Cancelled task",
            priority="low",
        ),
        Task(
            id=uuid.uuid4(),
            document_id=doc.id,
            line_number=5,
            raw_text="- [/] In progress",
            status=TaskStatus.IN_PROGRESS.value,
            description="In progress",
            due=today + timedelta(days=5),
            priority="high",
            tags=["urgent"],
        ),
    ]

    for task in tasks:
        db_session.add(task)
    db_session.commit()

    return vault, doc, tasks


class TestGetTasksIntegration:
    """Integration tests for get_tasks tool."""

    def test_get_tasks_no_filters(self, db_session, sample_data):
        """Test get_tasks with no filters returns all non-cancelled tasks."""
        vault, doc, tasks = sample_data

        filters = GetTasksFilterParams()
        result = get_tasks(db_session, filters)

        # Default is include_cancelled=False, so we get 4 tasks (all except cancelled)
        assert result.total_count == 4
        assert len(result.results) == 4

    def test_get_tasks_status_filter(self, db_session, sample_data):
        """Test filtering by multiple statuses."""
        vault, doc, tasks = sample_data

        filters = GetTasksFilterParams(status=["not_completed", "in_progress"])
        result = get_tasks(db_session, filters)

        assert result.total_count == 3  # 2 not_completed + 1 in_progress

    def test_get_tasks_due_date_range(self, db_session, sample_data):
        """Test filtering by due date range."""
        vault, doc, tasks = sample_data
        today = date.today()

        filters = GetTasksFilterParams(
            due_after=today,
            due_before=today + timedelta(days=7),
        )
        result = get_tasks(db_session, filters)

        # Should find tasks due within next 7 days
        assert result.total_count >= 1

    def test_get_tasks_priority_filter(self, db_session, sample_data):
        """Test filtering by priority."""
        vault, doc, tasks = sample_data

        filters = GetTasksFilterParams(
            priority=["high"],
            include_completed=False,
        )
        result = get_tasks(db_session, filters)

        # Should find high priority incomplete tasks
        assert result.total_count >= 1
        for task in result.results:
            assert task.priority == "high"

    def test_get_tasks_exclude_completed(self, db_session, sample_data):
        """Test excluding completed tasks."""
        vault, doc, tasks = sample_data

        # Include cancelled to get all non-completed tasks
        filters = GetTasksFilterParams(
            include_completed=False,
            include_cancelled=True,
        )
        result = get_tasks(db_session, filters)

        # Should get 4 tasks: 2 not_completed + 1 in_progress + 1 cancelled
        assert result.total_count == 4
        for task in result.results:
            assert task.status != "completed"

    def test_get_tasks_include_cancelled(self, db_session, sample_data):
        """Test including cancelled tasks."""
        vault, doc, tasks = sample_data

        filters = GetTasksFilterParams(
            include_completed=False,
            include_cancelled=True,
        )
        result = get_tasks(db_session, filters)

        # Should include cancelled
        cancelled_count = sum(1 for t in result.results if t.status == "cancelled")
        assert cancelled_count == 1

    def test_get_tasks_tag_filter(self, db_session, sample_data):
        """Test filtering by tags."""
        vault, doc, tasks = sample_data

        filters = GetTasksFilterParams(tags=["urgent"])
        result = get_tasks(db_session, filters)

        # Should find task with urgent tag
        assert result.total_count >= 1

    def test_get_tasks_combined_filters(self, db_session, sample_data):
        """Test combining multiple filters."""
        vault, doc, tasks = sample_data
        today = date.today()

        filters = GetTasksFilterParams(
            status=["not_completed", "in_progress"],
            due_after=today,
            due_before=today + timedelta(days=7),
            priority=["high"],
            include_completed=False,
        )
        result = get_tasks(db_session, filters)

        # All returned tasks should match all criteria
        for task in result.results:
            assert task.status in ["not_completed", "in_progress"]
            assert task.priority == "high"

    def test_get_tasks_pagination(self, db_session, sample_data):
        """Test pagination with limit and offset."""
        vault, doc, tasks = sample_data

        # Get first 2 results
        filters = GetTasksFilterParams(limit=2, offset=0)
        result1 = get_tasks(db_session, filters)

        assert len(result1.results) == 2
        assert result1.has_more is True
        assert result1.next_offset == 2

        # Get next 2 results
        filters = GetTasksFilterParams(limit=2, offset=2)
        result2 = get_tasks(db_session, filters)

        assert len(result2.results) == 2

    def test_get_tasks_handler_integration(self, db_session, sample_data):
        """Test _get_tasks_handler with database session."""
        from unittest.mock import MagicMock

        from obsidian_rag.mcp_server.handlers import TaskDateFilterStrings

        vault, doc, tasks = sample_data
        today = date.today()

        # Create a mock db_manager that returns our test session
        db_manager = MagicMock()
        db_manager.get_session.return_value.__enter__ = MagicMock(
            return_value=db_session
        )
        db_manager.get_session.return_value.__exit__ = MagicMock(return_value=False)

        date_filters = TaskDateFilterStrings(
            due_after=today.isoformat(),
            due_before=(today + timedelta(days=30)).isoformat(),
        )

        result = _get_tasks_handler(
            db_manager=db_manager,
            status=["not_completed"],
            date_filters=date_filters,
            include_completed=False,
            include_cancelled=True,
            limit=10,
        )

        assert "results" in result
        assert "total_count" in result
        total_count = result["total_count"]
        assert isinstance(total_count, int) and total_count >= 1


class TestGetTasksEdgeCases:
    """Edge case tests for get_tasks tool."""

    def test_get_tasks_empty_database(self, db_session):
        """Test with no tasks in database."""
        filters = GetTasksFilterParams()
        result = get_tasks(db_session, filters)

        assert result.total_count == 0
        assert result.results == []
        assert result.has_more is False

    def test_get_tasks_no_matching_filters(self, db_session, sample_data):
        """Test when filters match no tasks."""
        vault, doc, tasks = sample_data

        filters = GetTasksFilterParams(
            status=["completed"],
            tags=["nonexistent_tag"],
        )
        result = get_tasks(db_session, filters)

        assert result.total_count == 0
        assert result.results == []

    def test_get_tasks_invalid_date_range(self, db_session, sample_data):
        """Test with impossible date range (after > before)."""
        vault, doc, tasks = sample_data
        today = date.today()

        # This is valid - just returns no results
        filters = GetTasksFilterParams(
            due_after=today + timedelta(days=30),
            due_before=today,
        )
        result = get_tasks(db_session, filters)

        # Should return empty (no tasks can satisfy this)
        assert result.total_count == 0
