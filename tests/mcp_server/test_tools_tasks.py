"""Unit tests for MCP task tools."""

import uuid
from datetime import date, timedelta
from unittest.mock import MagicMock

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from obsidian_rag.database.models import (
    Base,
    Document,
    Task,
    TaskStatus,
    Vault,
)
from obsidian_rag.mcp_server.tools.tasks import (
    get_completed_tasks,
    get_incomplete_tasks,
    get_tasks_by_tag,
    get_tasks_due_this_week,
)


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

    # Create vault first
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
        kind="article",
        tags=["project", "test"],
    )
    db_session.add(doc)
    db_session.commit()
    return doc


class TestGetIncompleteTasks:
    """Tests for get_incomplete_tasks function."""

    def test_empty_result(self, db_session):
        """Test with no tasks in database."""
        result = get_incomplete_tasks(db_session)

        assert result.results == []
        assert result.total_count == 0
        assert result.has_more is False
        assert result.next_offset is None

    def test_excludes_completed_tasks(self, db_session, sample_document):
        """Test that completed tasks are excluded."""
        task = Task(
            id=uuid.uuid4(),
            document_id=sample_document.id,
            line_number=1,
            raw_text="- [x] Completed task",
            status=TaskStatus.COMPLETED.value,
            description="Completed task",
        )
        db_session.add(task)
        db_session.commit()

        result = get_incomplete_tasks(db_session)

        assert result.total_count == 0
        assert len(result.results) == 0

    def test_includes_not_completed(self, db_session, sample_document):
        """Test that not_completed tasks are included."""
        task = Task(
            id=uuid.uuid4(),
            document_id=sample_document.id,
            line_number=1,
            raw_text="- [ ] Incomplete task",
            status=TaskStatus.NOT_COMPLETED.value,
            description="Incomplete task",
        )
        db_session.add(task)
        db_session.commit()

        result = get_incomplete_tasks(db_session)

        assert result.total_count == 1
        assert len(result.results) == 1
        assert result.results[0].status == "not_completed"

    def test_includes_in_progress(self, db_session, sample_document):
        """Test that in_progress tasks are included."""
        task = Task(
            id=uuid.uuid4(),
            document_id=sample_document.id,
            line_number=1,
            raw_text="- [/] In progress task",
            status=TaskStatus.IN_PROGRESS.value,
            description="In progress task",
        )
        db_session.add(task)
        db_session.commit()

        result = get_incomplete_tasks(db_session)

        assert result.total_count == 1
        assert result.results[0].status == "in_progress"

    def test_exclude_cancelled_by_default(self, db_session, sample_document):
        """Test that cancelled tasks are excluded by default."""
        task = Task(
            id=uuid.uuid4(),
            document_id=sample_document.id,
            line_number=1,
            raw_text="- [-] Cancelled task",
            status=TaskStatus.CANCELLED.value,
            description="Cancelled task",
        )
        db_session.add(task)
        db_session.commit()

        result = get_incomplete_tasks(db_session, include_cancelled=False)

        assert result.total_count == 0

    def test_include_cancelled_when_requested(self, db_session, sample_document):
        """Test that cancelled tasks are included when requested."""
        task = Task(
            id=uuid.uuid4(),
            document_id=sample_document.id,
            line_number=1,
            raw_text="- [-] Cancelled task",
            status=TaskStatus.CANCELLED.value,
            description="Cancelled task",
        )
        db_session.add(task)
        db_session.commit()

        result = get_incomplete_tasks(db_session, include_cancelled=True)

        assert result.total_count == 1
        assert result.results[0].status == "cancelled"

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

        result = get_incomplete_tasks(db_session, limit=2, offset=0)

        assert result.total_count == 5
        assert len(result.results) == 2
        assert result.has_more is True
        assert result.next_offset == 2

    def test_limit_validation(self, db_session, sample_document):
        """Test that limit is validated and clamped."""
        task = Task(
            id=uuid.uuid4(),
            document_id=sample_document.id,
            line_number=1,
            raw_text="- [ ] Task",
            status=TaskStatus.NOT_COMPLETED.value,
            description="Task",
        )
        db_session.add(task)
        db_session.commit()

        # Test limit above maximum
        result = get_incomplete_tasks(db_session, limit=200)
        assert len(result.results) == 1  # Should still work, just clamped


class TestGetTasksDueThisWeek:
    """Tests for get_tasks_due_this_week function."""

    def test_empty_result(self, db_session):
        """Test with no tasks in database."""
        result = get_tasks_due_this_week(db_session)

        assert result.results == []
        assert result.total_count == 0

    def test_includes_tasks_due_this_week(self, db_session, sample_document):
        """Test tasks due within next 7 days are included."""
        today = date.today()
        task = Task(
            id=uuid.uuid4(),
            document_id=sample_document.id,
            line_number=1,
            raw_text="- [ ] Due soon",
            status=TaskStatus.NOT_COMPLETED.value,
            description="Due soon",
            due=today + timedelta(days=3),
        )
        db_session.add(task)
        db_session.commit()

        result = get_tasks_due_this_week(db_session)

        assert result.total_count == 1
        assert len(result.results) == 1

    def test_excludes_tasks_past_due_date(self, db_session, sample_document):
        """Test tasks past the 7-day window are excluded."""
        today = date.today()
        task = Task(
            id=uuid.uuid4(),
            document_id=sample_document.id,
            line_number=1,
            raw_text="- [ ] Due later",
            status=TaskStatus.NOT_COMPLETED.value,
            description="Due later",
            due=today + timedelta(days=10),
        )
        db_session.add(task)
        db_session.commit()

        result = get_tasks_due_this_week(db_session)

        assert result.total_count == 0

    def test_excludes_tasks_without_due_date(self, db_session, sample_document):
        """Test tasks without due dates are excluded."""
        task = Task(
            id=uuid.uuid4(),
            document_id=sample_document.id,
            line_number=1,
            raw_text="- [ ] No due date",
            status=TaskStatus.NOT_COMPLETED.value,
            description="No due date",
            due=None,
        )
        db_session.add(task)
        db_session.commit()

        result = get_tasks_due_this_week(db_session)

        assert result.total_count == 0

    def test_include_completed_option(self, db_session, sample_document):
        """Test completed tasks can be included or excluded."""
        today = date.today()

        # Create completed task
        completed_task = Task(
            id=uuid.uuid4(),
            document_id=sample_document.id,
            line_number=1,
            raw_text="- [x] Completed",
            status=TaskStatus.COMPLETED.value,
            description="Completed",
            due=today + timedelta(days=2),
        )
        db_session.add(completed_task)

        # Create incomplete task
        incomplete_task = Task(
            id=uuid.uuid4(),
            document_id=sample_document.id,
            line_number=2,
            raw_text="- [ ] Incomplete",
            status=TaskStatus.NOT_COMPLETED.value,
            description="Incomplete",
            due=today + timedelta(days=2),
        )
        db_session.add(incomplete_task)
        db_session.commit()

        # With include_completed=True (default)
        result_with = get_tasks_due_this_week(db_session, include_completed=True)
        assert result_with.total_count == 2

        # With include_completed=False
        result_without = get_tasks_due_this_week(db_session, include_completed=False)
        assert result_without.total_count == 1


class TestGetTasksByTag:
    """Tests for get_tasks_by_tag function."""

    def test_empty_result(self, db_session):
        """Test with no matching tasks."""
        result = get_tasks_by_tag(db_session, tag="nonexistent")

        assert result.results == []
        assert result.total_count == 0

    def test_matches_task_tags(self, db_session, sample_document):
        """Test matching task-level tags."""
        task = Task(
            id=uuid.uuid4(),
            document_id=sample_document.id,
            line_number=1,
            raw_text="- [ ] Task with tag #project",
            status=TaskStatus.NOT_COMPLETED.value,
            description="Task with tag",
            tags=["project"],
        )
        db_session.add(task)
        db_session.commit()

        result = get_tasks_by_tag(db_session, tag="project")

        assert result.total_count == 1
        assert len(result.results) == 1

    def test_case_insensitive_matching(self, db_session, sample_document):
        """Test case-insensitive tag matching."""
        task = Task(
            id=uuid.uuid4(),
            document_id=sample_document.id,
            line_number=1,
            raw_text="- [ ] Task with tag #Project",
            status=TaskStatus.NOT_COMPLETED.value,
            description="Task with tag",
            tags=["Project"],
        )
        db_session.add(task)
        db_session.commit()

        result = get_tasks_by_tag(db_session, tag="PROJECT")

        assert result.total_count == 1

    def test_matches_document_tags(self, db_session, sample_document):
        """Test matching document-level tags."""
        # sample_document has tags ["project", "test"]
        task = Task(
            id=uuid.uuid4(),
            document_id=sample_document.id,
            line_number=1,
            raw_text="- [ ] Task without tag",
            status=TaskStatus.NOT_COMPLETED.value,
            description="Task without tag",
            tags=[],
        )
        db_session.add(task)
        db_session.commit()

        result = get_tasks_by_tag(db_session, tag="test")

        assert result.total_count == 1


class TestGetCompletedTasks:
    """Tests for get_completed_tasks function."""

    def test_empty_result(self, db_session):
        """Test with no completed tasks."""
        result = get_completed_tasks(db_session)

        assert result.results == []
        assert result.total_count == 0

    def test_includes_completed_tasks(self, db_session, sample_document):
        """Test completed tasks are included."""
        from datetime import date

        task = Task(
            id=uuid.uuid4(),
            document_id=sample_document.id,
            line_number=1,
            raw_text="- [x] Completed task",
            status=TaskStatus.COMPLETED.value,
            description="Completed task",
            completion=date(2025, 3, 1),
        )
        db_session.add(task)
        db_session.commit()

        result = get_completed_tasks(db_session)

        assert result.total_count == 1
        assert len(result.results) == 1
        assert result.results[0].status == "completed"

    def test_excludes_incomplete_tasks(self, db_session, sample_document):
        """Test incomplete tasks are excluded."""
        task = Task(
            id=uuid.uuid4(),
            document_id=sample_document.id,
            line_number=1,
            raw_text="- [ ] Incomplete task",
            status=TaskStatus.NOT_COMPLETED.value,
            description="Incomplete task",
        )
        db_session.add(task)
        db_session.commit()

        result = get_completed_tasks(db_session)

        assert result.total_count == 0

    def test_date_filter(self, db_session, sample_document):
        """Test filtering by completion date."""
        from datetime import date

        # Old completed task
        old_task = Task(
            id=uuid.uuid4(),
            document_id=sample_document.id,
            line_number=1,
            raw_text="- [x] Old task",
            status=TaskStatus.COMPLETED.value,
            description="Old task",
            completion=date(2020, 1, 1),
        )
        db_session.add(old_task)

        # Recent completed task
        recent_task = Task(
            id=uuid.uuid4(),
            document_id=sample_document.id,
            line_number=2,
            raw_text="- [x] Recent task",
            status=TaskStatus.COMPLETED.value,
            description="Recent task",
            completion=date(2025, 3, 1),
        )
        db_session.add(recent_task)
        db_session.commit()

        result = get_completed_tasks(db_session, completed_since=date(2024, 1, 1))

        assert result.total_count == 1
        assert result.results[0].description == "Recent task"
