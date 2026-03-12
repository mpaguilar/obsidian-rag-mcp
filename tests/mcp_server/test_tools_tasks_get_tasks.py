"""Tests for get_tasks function."""

import uuid
from datetime import date, timedelta

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from obsidian_rag.database.models import Base, Document, Task, TaskStatus, Vault
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
def sample_document(db_session):
    """Create a sample document for testing."""
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
        tags=["test"],
    )
    db_session.add(doc)
    db_session.commit()
    return doc


class TestGetTasks:
    """Tests for get_tasks function."""

    def test_empty_result(self, db_session):
        """Test with no tasks in database."""
        filters = GetTasksFilterParams()
        result = get_tasks(db_session, filters)

        assert result.results == []
        assert result.total_count == 0
        assert result.has_more is False

    def test_no_filters_returns_all(self, db_session, sample_document):
        """Test that no filters returns all tasks."""
        task = Task(
            id=uuid.uuid4(),
            document_id=sample_document.id,
            line_number=1,
            raw_text="- [ ] Test task",
            status=TaskStatus.NOT_COMPLETED.value,
            description="Test task",
        )
        db_session.add(task)
        db_session.commit()

        filters = GetTasksFilterParams()
        result = get_tasks(db_session, filters)

        assert result.total_count == 1
        assert len(result.results) == 1

    def test_status_filter(self, db_session, sample_document):
        """Test filtering by status."""
        # Create tasks with different statuses
        task1 = Task(
            id=uuid.uuid4(),
            document_id=sample_document.id,
            line_number=1,
            raw_text="- [ ] Incomplete",
            status=TaskStatus.NOT_COMPLETED.value,
            description="Incomplete task",
        )
        task2 = Task(
            id=uuid.uuid4(),
            document_id=sample_document.id,
            line_number=2,
            raw_text="- [x] Completed",
            status=TaskStatus.COMPLETED.value,
            description="Completed task",
        )
        db_session.add_all([task1, task2])
        db_session.commit()

        filters = GetTasksFilterParams(status=["not_completed"])
        result = get_tasks(db_session, filters)

        assert result.total_count == 1
        assert result.results[0].status == "not_completed"

    def test_due_date_range_filter(self, db_session, sample_document):
        """Test filtering by due date range."""
        today = date.today()

        # Task due within range
        task1 = Task(
            id=uuid.uuid4(),
            document_id=sample_document.id,
            line_number=1,
            raw_text="- [ ] Due in range",
            status=TaskStatus.NOT_COMPLETED.value,
            description="Due in range",
            due=today + timedelta(days=5),
        )
        # Task due outside range
        task2 = Task(
            id=uuid.uuid4(),
            document_id=sample_document.id,
            line_number=2,
            raw_text="- [ ] Due outside range",
            status=TaskStatus.NOT_COMPLETED.value,
            description="Due outside range",
            due=today + timedelta(days=20),
        )
        db_session.add_all([task1, task2])
        db_session.commit()

        filters = GetTasksFilterParams(
            due_after=today,
            due_before=today + timedelta(days=10),
        )
        result = get_tasks(db_session, filters)

        assert result.total_count == 1
        assert result.results[0].description == "Due in range"

    def test_scheduled_date_filter(self, db_session, sample_document):
        """Test filtering by scheduled date."""
        today = date.today()

        task = Task(
            id=uuid.uuid4(),
            document_id=sample_document.id,
            line_number=1,
            raw_text="- [ ] Scheduled task",
            status=TaskStatus.NOT_COMPLETED.value,
            description="Scheduled task",
            scheduled=today + timedelta(days=3),
        )
        db_session.add(task)
        db_session.commit()

        filters = GetTasksFilterParams(
            scheduled_after=today,
            scheduled_before=today + timedelta(days=7),
        )
        result = get_tasks(db_session, filters)

        assert result.total_count == 1

    def test_completion_date_filter(self, db_session, sample_document):
        """Test filtering by completion date."""
        today = date.today()

        task = Task(
            id=uuid.uuid4(),
            document_id=sample_document.id,
            line_number=1,
            raw_text="- [x] Completed task",
            status=TaskStatus.COMPLETED.value,
            description="Completed task",
            completion=today - timedelta(days=2),
        )
        db_session.add(task)
        db_session.commit()

        filters = GetTasksFilterParams(
            completion_after=today - timedelta(days=7),
            completion_before=today,
        )
        result = get_tasks(db_session, filters)

        assert result.total_count == 1

    def test_include_completed_false(self, db_session, sample_document):
        """Test excluding completed tasks."""
        task1 = Task(
            id=uuid.uuid4(),
            document_id=sample_document.id,
            line_number=1,
            raw_text="- [ ] Incomplete",
            status=TaskStatus.NOT_COMPLETED.value,
            description="Incomplete",
        )
        task2 = Task(
            id=uuid.uuid4(),
            document_id=sample_document.id,
            line_number=2,
            raw_text="- [x] Completed",
            status=TaskStatus.COMPLETED.value,
            description="Completed",
        )
        db_session.add_all([task1, task2])
        db_session.commit()

        filters = GetTasksFilterParams(include_completed=False)
        result = get_tasks(db_session, filters)

        assert result.total_count == 1
        assert result.results[0].description == "Incomplete"

    def test_include_cancelled_true(self, db_session, sample_document):
        """Test including cancelled tasks."""
        task = Task(
            id=uuid.uuid4(),
            document_id=sample_document.id,
            line_number=1,
            raw_text="- [-] Cancelled",
            status=TaskStatus.CANCELLED.value,
            description="Cancelled",
        )
        db_session.add(task)
        db_session.commit()

        filters = GetTasksFilterParams(include_cancelled=True)
        result = get_tasks(db_session, filters)

        assert result.total_count == 1
        assert result.results[0].status == "cancelled"

    def test_priority_filter(self, db_session, sample_document):
        """Test filtering by priority."""
        task = Task(
            id=uuid.uuid4(),
            document_id=sample_document.id,
            line_number=1,
            raw_text="- [ ] High priority",
            status=TaskStatus.NOT_COMPLETED.value,
            description="High priority",
            priority="high",
        )
        db_session.add(task)
        db_session.commit()

        filters = GetTasksFilterParams(priority=["high", "highest"])
        result = get_tasks(db_session, filters)

        assert result.total_count == 1
        assert result.results[0].priority == "high"

    def test_pagination(self, db_session, sample_document):
        """Test pagination with limit and offset."""
        for i in range(5):
            task = Task(
                id=uuid.uuid4(),
                document_id=sample_document.id,
                line_number=i,
                raw_text=f"- [ ] Task {i}",
                status=TaskStatus.NOT_COMPLETED.value,
                description=f"Task {i}",
            )
            db_session.add(task)
        db_session.commit()

        filters = GetTasksFilterParams(limit=2, offset=0)
        result = get_tasks(db_session, filters)

        assert result.total_count == 5
        assert len(result.results) == 2
        assert result.has_more is True
        assert result.next_offset == 2

    def test_multiple_filters_combined(self, db_session, sample_document):
        """Test that multiple filters are combined with AND logic."""
        today = date.today()

        # Task that matches all filters
        task1 = Task(
            id=uuid.uuid4(),
            document_id=sample_document.id,
            line_number=1,
            raw_text="- [ ] Match all",
            status=TaskStatus.NOT_COMPLETED.value,
            description="Match all",
            due=today + timedelta(days=3),
            priority="high",
        )
        # Task that matches only some filters
        task2 = Task(
            id=uuid.uuid4(),
            document_id=sample_document.id,
            line_number=2,
            raw_text="- [ ] Match some",
            status=TaskStatus.NOT_COMPLETED.value,
            description="Match some",
            due=today + timedelta(days=20),  # Outside date range
            priority="high",
        )
        db_session.add_all([task1, task2])
        db_session.commit()

        filters = GetTasksFilterParams(
            status=["not_completed"],
            due_after=today,
            due_before=today + timedelta(days=10),
            priority=["high"],
        )
        result = get_tasks(db_session, filters)

        assert result.total_count == 1
        assert result.results[0].description == "Match all"
