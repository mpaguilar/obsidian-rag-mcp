"""Tests for database models."""

import uuid
from datetime import datetime

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from obsidian_rag.database.models import (
    Base,
    Document,
    Task,
    TaskPriority,
    TaskStatus,
)


@pytest.fixture
def db_session():
    """Create a test database session."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()
    Base.metadata.drop_all(engine)


class TestDocument:
    """Test cases for Document model."""

    def test_create_document(self, db_session):
        """Test creating a document."""
        doc = Document(
            file_path="/test/file.md",
            file_name="file.md",
            content="Test content",
            checksum_md5="abc123",
            created_at_fs=datetime.now(),
            modified_at_fs=datetime.now(),
        )
        db_session.add(doc)
        db_session.commit()

        assert doc.id is not None
        assert doc.file_path == "/test/file.md"
        assert doc.checksum_md5 == "abc123"

    def test_document_unique_file_path(self, db_session):
        """Test that file_path must be unique."""
        doc1 = Document(
            file_path="/test/file.md",
            file_name="file.md",
            content="Content 1",
            checksum_md5="abc123",
            created_at_fs=datetime.now(),
            modified_at_fs=datetime.now(),
        )
        db_session.add(doc1)
        db_session.commit()

        doc2 = Document(
            file_path="/test/file.md",  # Same path
            file_name="file.md",
            content="Content 2",
            checksum_md5="def456",
            created_at_fs=datetime.now(),
            modified_at_fs=datetime.now(),
        )
        db_session.add(doc2)

        with pytest.raises(Exception):  # Integrity error
            db_session.commit()

    def test_document_with_tags(self, db_session):
        """Test document with tags array."""
        doc = Document(
            file_path="/test/file.md",
            file_name="file.md",
            content="Test content",
            checksum_md5="abc123",
            created_at_fs=datetime.now(),
            modified_at_fs=datetime.now(),
            tags=["tag1", "tag2", "tag3"],
        )
        db_session.add(doc)
        db_session.commit()

        # Query back
        result = db_session.query(Document).first()
        assert result.tags == ["tag1", "tag2", "tag3"]

    def test_document_with_metadata(self, db_session):
        """Test document with frontmatter metadata."""
        doc = Document(
            file_path="/test/file.md",
            file_name="file.md",
            content="Test content",
            checksum_md5="abc123",
            created_at_fs=datetime.now(),
            modified_at_fs=datetime.now(),
            kind="article",
            frontmatter_json={"author": "Test", "version": 1.0},
        )
        db_session.add(doc)
        db_session.commit()

        result = db_session.query(Document).first()
        assert result.kind == "article"
        assert result.frontmatter_json == {"author": "Test", "version": 1.0}


class TestTask:
    """Test cases for Task model."""

    def test_create_task(self, db_session):
        """Test creating a task associated with a document."""
        doc = Document(
            file_path="/test/file.md",
            file_name="file.md",
            content="Test content",
            checksum_md5="abc123",
            created_at_fs=datetime.now(),
            modified_at_fs=datetime.now(),
        )
        db_session.add(doc)
        db_session.flush()

        task = Task(
            document_id=doc.id,
            line_number=5,
            raw_text="- [ ] A test task",
            status=TaskStatus.NOT_COMPLETED.value,
            description="A test task",
            priority=TaskPriority.NORMAL.value,
        )
        db_session.add(task)
        db_session.commit()

        assert task.id is not None
        assert task.document_id == doc.id
        assert task.status == TaskStatus.NOT_COMPLETED.value

    def test_task_all_status_values(self, db_session):
        """Test task with all status values."""
        doc = Document(
            file_path="/test/file.md",
            file_name="file.md",
            content="Test",
            checksum_md5="abc123",
            created_at_fs=datetime.now(),
            modified_at_fs=datetime.now(),
        )
        db_session.add(doc)
        db_session.flush()

        statuses = [
            TaskStatus.NOT_COMPLETED.value,
            TaskStatus.COMPLETED.value,
            TaskStatus.IN_PROGRESS.value,
            TaskStatus.CANCELLED.value,
        ]

        for i, status in enumerate(statuses):
            task = Task(
                document_id=doc.id,
                line_number=i,
                raw_text=f"- [ ] Task {i}",
                status=status,
                description=f"Task {i}",
            )
            db_session.add(task)

        db_session.commit()

        results = db_session.query(Task).all()
        assert len(results) == 4
        assert {t.status for t in results} == set(statuses)

    def test_task_cascade_delete(self, db_session):
        """Test that tasks are deleted when document is deleted."""
        doc = Document(
            file_path="/test/file.md",
            file_name="file.md",
            content="Test",
            checksum_md5="abc123",
            created_at_fs=datetime.now(),
            modified_at_fs=datetime.now(),
        )
        db_session.add(doc)
        db_session.flush()

        task = Task(
            document_id=doc.id,
            line_number=1,
            raw_text="- [ ] Task",
            status=TaskStatus.NOT_COMPLETED.value,
            description="Task",
        )
        db_session.add(task)
        db_session.commit()

        # Delete document
        db_session.delete(doc)
        db_session.commit()

        # Task should be deleted
        remaining_tasks = db_session.query(Task).all()
        assert len(remaining_tasks) == 0

    def test_task_with_all_fields(self, db_session):
        """Test task with all optional fields set."""
        doc = Document(
            file_path="/test/file.md",
            file_name="file.md",
            content="Test",
            checksum_md5="abc123",
            created_at_fs=datetime.now(),
            modified_at_fs=datetime.now(),
        )
        db_session.add(doc)
        db_session.flush()

        from datetime import date

        task = Task(
            document_id=doc.id,
            line_number=1,
            raw_text="- [ ] Task [due:: 2024-03-15]",
            status=TaskStatus.NOT_COMPLETED.value,
            description="Task",
            tags=["important", "work"],
            repeat="FREQ=DAILY",
            scheduled=date(2024, 3, 10),
            due=date(2024, 3, 15),
            completion=None,
            priority=TaskPriority.HIGH.value,
            custom_metadata={"project": "test"},
        )
        db_session.add(task)
        db_session.commit()

        result = db_session.query(Task).first()
        assert result.tags == ["important", "work"]
        assert result.repeat == "FREQ=DAILY"
        assert result.due.year == 2024
        assert result.priority == TaskPriority.HIGH.value
        assert result.custom_metadata == {"project": "test"}
