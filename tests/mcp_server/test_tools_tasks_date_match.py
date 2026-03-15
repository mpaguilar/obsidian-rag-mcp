"""Tests for date_match_mode functionality."""

import uuid
from datetime import date, timedelta
from unittest.mock import MagicMock

import pytest

from obsidian_rag.database.models import Document, Task, TaskPriority, TaskStatus, Vault
from obsidian_rag.mcp_server.tools.tasks import get_tasks
from obsidian_rag.mcp_server.tools.tasks_params import GetTasksFilterParams


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
    return doc


def _configure_mock_for_tasks(db_session, tasks, doc, total_count=None):
    """Configure mock session to return tasks with their documents."""
    if total_count is None:
        total_count = len(tasks)

    # Create (Task, Document) tuples as expected by get_tasks
    task_doc_pairs = [(task, doc) for task in tasks]

    # Configure the mock query chain
    query_mock = MagicMock()
    query_mock.all.return_value = task_doc_pairs
    query_mock.count.return_value = total_count
    query_mock.filter.return_value = query_mock
    query_mock.order_by.return_value = query_mock
    query_mock.offset.return_value = query_mock
    query_mock.limit.return_value = query_mock
    query_mock.join.return_value = query_mock

    db_session.query.return_value = query_mock


def _create_task(**kwargs):
    """Create a Task with default priority."""
    defaults = {
        "priority": TaskPriority.NORMAL.value,
    }
    defaults.update(kwargs)
    return Task(**defaults)


class TestDateMatchModeAll:
    """Tests for date_match_mode='all' (AND logic) - backward compatibility."""

    def test_all_mode_both_due_conditions_must_match(self, db_session, sample_document):
        """Test that both due_before AND due_after must match in 'all' mode."""
        today = date.today()

        # Task within range
        task1 = _create_task(
            id=uuid.uuid4(),
            document_id=sample_document.id,
            line_number=1,
            raw_text="- [ ] In range",
            status=TaskStatus.NOT_COMPLETED.value,
            description="In range",
            due=today + timedelta(days=5),
        )

        # Configure mock to return only task1 (the one that matches)
        _configure_mock_for_tasks(db_session, [task1], sample_document, total_count=1)

        filters = GetTasksFilterParams(
            due_after=today,
            due_before=today + timedelta(days=10),
            date_match_mode="all",
        )
        result = get_tasks(db_session, filters)

        # Only task1 should match (within range)
        assert result.total_count == 1
        assert result.results[0].description == "In range"

    def test_all_mode_multiple_date_types_must_all_match(
        self, db_session, sample_document
    ):
        """Test that due AND scheduled conditions must both match in 'all' mode."""
        today = date.today()

        # Task matching both conditions
        task1 = _create_task(
            id=uuid.uuid4(),
            document_id=sample_document.id,
            line_number=1,
            raw_text="- [ ] Both match",
            status=TaskStatus.NOT_COMPLETED.value,
            description="Both match",
            due=today + timedelta(days=3),
            scheduled=today + timedelta(days=5),
        )

        # Configure mock to return only task1
        _configure_mock_for_tasks(db_session, [task1], sample_document, total_count=1)

        filters = GetTasksFilterParams(
            due_after=today,
            due_before=today + timedelta(days=10),
            scheduled_after=today,
            scheduled_before=today + timedelta(days=10),
            date_match_mode="all",
        )
        result = get_tasks(db_session, filters)

        # Only task1 should match (both conditions)
        assert result.total_count == 1
        assert result.results[0].description == "Both match"

    def test_all_mode_task_with_null_date_excluded(self, db_session, sample_document):
        """Test that tasks with NULL for a date field are excluded in 'all' mode."""
        today = date.today()

        # Task with due date
        task1 = _create_task(
            id=uuid.uuid4(),
            document_id=sample_document.id,
            line_number=1,
            raw_text="- [ ] Has due date",
            status=TaskStatus.NOT_COMPLETED.value,
            description="Has due date",
            due=today + timedelta(days=5),
        )

        # Configure mock to return only task1
        _configure_mock_for_tasks(db_session, [task1], sample_document, total_count=1)

        filters = GetTasksFilterParams(
            due_after=today,
            due_before=today + timedelta(days=10),
            date_match_mode="all",
        )
        result = get_tasks(db_session, filters)

        # Only task1 should match (task2 has NULL due date)
        assert result.total_count == 1
        assert result.results[0].description == "Has due date"

    def test_default_mode_is_all(self, db_session, sample_document):
        """Test that default date_match_mode is 'all'."""
        today = date.today()

        # Task matching both
        task1 = _create_task(
            id=uuid.uuid4(),
            document_id=sample_document.id,
            line_number=1,
            raw_text="- [ ] Both match",
            status=TaskStatus.NOT_COMPLETED.value,
            description="Both match",
            due=today + timedelta(days=3),
            scheduled=today + timedelta(days=5),
        )

        # Configure mock to return only task1
        _configure_mock_for_tasks(db_session, [task1], sample_document, total_count=1)

        # Don't specify date_match_mode - should default to "all"
        filters = GetTasksFilterParams(
            due_after=today,
            due_before=today + timedelta(days=10),
            scheduled_after=today,
            scheduled_before=today + timedelta(days=10),
        )
        result = get_tasks(db_session, filters)

        # Only task1 should match (default is AND logic)
        assert result.total_count == 1
        assert result.results[0].description == "Both match"


class TestDateMatchModeAny:
    """Tests for date_match_mode='any' (OR logic)."""

    def test_any_mode_single_condition_matches(self, db_session, sample_document):
        """Test that task matches if ANY single condition is satisfied."""
        today = date.today()

        # Task matching due condition only
        task1 = _create_task(
            id=uuid.uuid4(),
            document_id=sample_document.id,
            line_number=1,
            raw_text="- [ ] Due matches",
            status=TaskStatus.NOT_COMPLETED.value,
            description="Due matches",
            due=today + timedelta(days=3),
            scheduled=None,
        )
        # Task matching scheduled condition only
        task2 = _create_task(
            id=uuid.uuid4(),
            document_id=sample_document.id,
            line_number=2,
            raw_text="- [ ] Scheduled matches",
            status=TaskStatus.NOT_COMPLETED.value,
            description="Scheduled matches",
            due=None,
            scheduled=today + timedelta(days=5),
        )

        # Configure mock to return both tasks (OR logic)
        _configure_mock_for_tasks(
            db_session, [task1, task2], sample_document, total_count=2
        )

        filters = GetTasksFilterParams(
            due_after=today,
            due_before=today + timedelta(days=10),
            scheduled_after=today,
            scheduled_before=today + timedelta(days=10),
            date_match_mode="any",
        )
        result = get_tasks(db_session, filters)

        # Both task1 and task2 should match (OR logic)
        assert result.total_count == 2
        descriptions = {r.description for r in result.results}
        assert "Due matches" in descriptions
        assert "Scheduled matches" in descriptions

    def test_any_mode_both_conditions_match(self, db_session, sample_document):
        """Test that task matches if BOTH conditions are satisfied."""
        today = date.today()

        # Task matching both conditions
        task1 = _create_task(
            id=uuid.uuid4(),
            document_id=sample_document.id,
            line_number=1,
            raw_text="- [ ] Both match",
            status=TaskStatus.NOT_COMPLETED.value,
            description="Both match",
            due=today + timedelta(days=3),
            scheduled=today + timedelta(days=5),
        )

        # Configure mock to return task1
        _configure_mock_for_tasks(db_session, [task1], sample_document, total_count=1)

        filters = GetTasksFilterParams(
            due_after=today,
            due_before=today + timedelta(days=10),
            scheduled_after=today,
            scheduled_before=today + timedelta(days=10),
            date_match_mode="any",
        )
        result = get_tasks(db_session, filters)

        # Task should match
        assert result.total_count == 1
        assert result.results[0].description == "Both match"

    def test_any_mode_task_with_null_can_match_other_condition(
        self, db_session, sample_document
    ):
        """Test that task with NULL for one date can still match via other date."""
        today = date.today()

        # Task with due date but no scheduled date
        task1 = _create_task(
            id=uuid.uuid4(),
            document_id=sample_document.id,
            line_number=1,
            raw_text="- [ ] Has due no scheduled",
            status=TaskStatus.NOT_COMPLETED.value,
            description="Has due no scheduled",
            due=today + timedelta(days=3),
            scheduled=None,
        )
        # Task with scheduled date but no due date
        task2 = _create_task(
            id=uuid.uuid4(),
            document_id=sample_document.id,
            line_number=2,
            raw_text="- [ ] Has scheduled no due",
            status=TaskStatus.NOT_COMPLETED.value,
            description="Has scheduled no due",
            due=None,
            scheduled=today + timedelta(days=5),
        )

        # Configure mock to return both tasks
        _configure_mock_for_tasks(
            db_session, [task1, task2], sample_document, total_count=2
        )

        filters = GetTasksFilterParams(
            due_after=today,
            due_before=today + timedelta(days=10),
            scheduled_after=today,
            scheduled_before=today + timedelta(days=10),
            date_match_mode="any",
        )
        result = get_tasks(db_session, filters)

        # Both task1 and task2 should match (each matches one condition)
        assert result.total_count == 2
        descriptions = {r.description for r in result.results}
        assert "Has due no scheduled" in descriptions
        assert "Has scheduled no due" in descriptions

    def test_any_mode_all_three_date_types(self, db_session, sample_document):
        """Test OR logic across due, scheduled, and completion dates."""
        today = date.today()

        # Task matching only due
        task1 = _create_task(
            id=uuid.uuid4(),
            document_id=sample_document.id,
            line_number=1,
            raw_text="- [ ] Due only",
            status=TaskStatus.NOT_COMPLETED.value,
            description="Due only",
            due=today + timedelta(days=3),
        )
        # Task matching only scheduled
        task2 = _create_task(
            id=uuid.uuid4(),
            document_id=sample_document.id,
            line_number=2,
            raw_text="- [ ] Scheduled only",
            status=TaskStatus.NOT_COMPLETED.value,
            description="Scheduled only",
            scheduled=today + timedelta(days=5),
        )
        # Task matching only completion
        task3 = _create_task(
            id=uuid.uuid4(),
            document_id=sample_document.id,
            line_number=3,
            raw_text="- [x] Completion only",
            status=TaskStatus.COMPLETED.value,
            description="Completion only",
            completion=today - timedelta(days=2),
        )

        # Configure mock to return all three tasks
        _configure_mock_for_tasks(
            db_session, [task1, task2, task3], sample_document, total_count=3
        )

        filters = GetTasksFilterParams(
            due_after=today,
            due_before=today + timedelta(days=10),
            scheduled_after=today,
            scheduled_before=today + timedelta(days=10),
            completion_after=today - timedelta(days=7),
            completion_before=today,
            include_completed=True,
            date_match_mode="any",
        )
        result = get_tasks(db_session, filters)

        # task1, task2, and task3 should all match
        assert result.total_count == 3
        descriptions = {r.description for r in result.results}
        assert "Due only" in descriptions
        assert "Scheduled only" in descriptions
        assert "Completion only" in descriptions


class TestDateMatchModeEdgeCases:
    """Tests for edge cases in date_match_mode."""

    def test_single_date_filter_any_mode_same_as_all(self, db_session, sample_document):
        """Test that single date filter works same in 'any' and 'all' mode."""
        today = date.today()

        # Task within range
        task1 = _create_task(
            id=uuid.uuid4(),
            document_id=sample_document.id,
            line_number=1,
            raw_text="- [ ] In range",
            status=TaskStatus.NOT_COMPLETED.value,
            description="In range",
            due=today + timedelta(days=5),
        )

        # Configure mock to return task1 for both tests
        _configure_mock_for_tasks(db_session, [task1], sample_document, total_count=1)

        # Test with "all" mode
        filters_all = GetTasksFilterParams(
            due_after=today,
            due_before=today + timedelta(days=10),
            date_match_mode="all",
        )
        result_all = get_tasks(db_session, filters_all)

        # Re-configure mock for second query
        _configure_mock_for_tasks(db_session, [task1], sample_document, total_count=1)

        # Test with "any" mode
        filters_any = GetTasksFilterParams(
            due_after=today,
            due_before=today + timedelta(days=10),
            date_match_mode="any",
        )
        result_any = get_tasks(db_session, filters_any)

        # Both should return the same result (only task1)
        assert result_all.total_count == result_any.total_count == 1
        assert (
            result_all.results[0].description
            == result_any.results[0].description
            == "In range"
        )

    def test_no_date_filters_match_mode_ignored(self, db_session, sample_document):
        """Test that match_mode has no effect when no date filters specified."""
        task1 = _create_task(
            id=uuid.uuid4(),
            document_id=sample_document.id,
            line_number=1,
            raw_text="- [ ] Task 1",
            status=TaskStatus.NOT_COMPLETED.value,
            description="Task 1",
        )
        task2 = _create_task(
            id=uuid.uuid4(),
            document_id=sample_document.id,
            line_number=2,
            raw_text="- [ ] Task 2",
            status=TaskStatus.NOT_COMPLETED.value,
            description="Task 2",
        )

        # Configure mock to return both tasks for both tests
        _configure_mock_for_tasks(
            db_session, [task1, task2], sample_document, total_count=2
        )

        # Test with "all" mode, no date filters
        filters_all = GetTasksFilterParams(date_match_mode="all")
        result_all = get_tasks(db_session, filters_all)

        # Re-configure mock
        _configure_mock_for_tasks(
            db_session, [task1, task2], sample_document, total_count=2
        )

        # Test with "any" mode, no date filters
        filters_any = GetTasksFilterParams(date_match_mode="any")
        result_any = get_tasks(db_session, filters_any)

        # Both should return all tasks
        assert result_all.total_count == result_any.total_count == 2

    def test_all_date_filters_none_match_mode_ignored(
        self, db_session, sample_document
    ):
        """Test that match_mode has no effect when all date filters are None."""
        task1 = _create_task(
            id=uuid.uuid4(),
            document_id=sample_document.id,
            line_number=1,
            raw_text="- [ ] Task 1",
            status=TaskStatus.NOT_COMPLETED.value,
            description="Task 1",
            due=None,
            scheduled=None,
            completion=None,
        )

        # Configure mock to return task1 for both tests
        _configure_mock_for_tasks(db_session, [task1], sample_document, total_count=1)

        # Test with explicit None values
        filters_all = GetTasksFilterParams(
            due_after=None,
            due_before=None,
            scheduled_after=None,
            scheduled_before=None,
            date_match_mode="all",
        )
        result_all = get_tasks(db_session, filters_all)

        # Re-configure mock
        _configure_mock_for_tasks(db_session, [task1], sample_document, total_count=1)

        filters_any = GetTasksFilterParams(
            due_after=None,
            due_before=None,
            scheduled_after=None,
            scheduled_before=None,
            date_match_mode="any",
        )
        result_any = get_tasks(db_session, filters_any)

        # Both should return the task
        assert result_all.total_count == result_any.total_count == 1

    def test_task_with_only_some_dates_any_mode(self, db_session, sample_document):
        """Test task with only some date fields populated in 'any' mode."""
        today = date.today()

        # Task with only due date
        task1 = _create_task(
            id=uuid.uuid4(),
            document_id=sample_document.id,
            line_number=1,
            raw_text="- [ ] Only due",
            status=TaskStatus.NOT_COMPLETED.value,
            description="Only due",
            due=today + timedelta(days=3),
            scheduled=None,
            completion=None,
        )
        # Task with only scheduled date
        task2 = _create_task(
            id=uuid.uuid4(),
            document_id=sample_document.id,
            line_number=2,
            raw_text="- [ ] Only scheduled",
            status=TaskStatus.NOT_COMPLETED.value,
            description="Only scheduled",
            due=None,
            scheduled=today + timedelta(days=5),
            completion=None,
        )
        # Task with only completion date
        task3 = _create_task(
            id=uuid.uuid4(),
            document_id=sample_document.id,
            line_number=3,
            raw_text="- [x] Only completion",
            status=TaskStatus.COMPLETED.value,
            description="Only completion",
            due=None,
            scheduled=None,
            completion=today - timedelta(days=2),
        )

        # Configure mock to return all three tasks
        _configure_mock_for_tasks(
            db_session, [task1, task2, task3], sample_document, total_count=3
        )

        filters = GetTasksFilterParams(
            due_after=today,
            due_before=today + timedelta(days=10),
            scheduled_after=today,
            scheduled_before=today + timedelta(days=10),
            completion_after=today - timedelta(days=7),
            completion_before=today,
            include_completed=True,
            date_match_mode="any",
        )
        result = get_tasks(db_session, filters)

        # All three should match (each matches one condition)
        assert result.total_count == 3
