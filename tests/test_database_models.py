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
    Vault,
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
        # Create vault first
        vault = Vault(
            name="Test Vault",
            container_path="/test",
            host_path="/test",
        )
        db_session.add(vault)
        db_session.flush()

        doc = Document(
            vault_id=vault.id,
            file_path="file.md",
            file_name="file.md",
            content="Test content",
            checksum_md5="abc123",
            created_at_fs=datetime.now(),
            modified_at_fs=datetime.now(),
        )
        db_session.add(doc)
        db_session.commit()

        assert doc.id is not None
        assert doc.file_path == "file.md"
        assert doc.checksum_md5 == "abc123"

    def test_document_unique_file_path(self, db_session):
        """Test that file_path must be unique per vault."""
        # Create vault first
        vault = Vault(
            name="Test Vault",
            container_path="/test",
            host_path="/test",
        )
        db_session.add(vault)
        db_session.flush()

        doc1 = Document(
            vault_id=vault.id,
            file_path="file.md",
            file_name="file.md",
            content="Content 1",
            checksum_md5="abc123",
            created_at_fs=datetime.now(),
            modified_at_fs=datetime.now(),
        )
        db_session.add(doc1)
        db_session.commit()

        doc2 = Document(
            vault_id=vault.id,
            file_path="file.md",  # Same path in same vault
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
        # Create vault first
        vault = Vault(
            name="Test Vault",
            container_path="/test",
            host_path="/test",
        )
        db_session.add(vault)
        db_session.flush()

        doc = Document(
            vault_id=vault.id,
            file_path="file.md",
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
        # Create vault first
        vault = Vault(
            name="Test Vault",
            container_path="/test",
            host_path="/test",
        )
        db_session.add(vault)
        db_session.flush()

        doc = Document(
            vault_id=vault.id,
            file_path="file.md",
            file_name="file.md",
            content="Test content",
            checksum_md5="abc123",
            created_at_fs=datetime.now(),
            modified_at_fs=datetime.now(),
            frontmatter_json={"kind": "article", "author": "Test", "version": 1.0},
        )
        db_session.add(doc)
        db_session.commit()

        result = db_session.query(Document).first()
        assert result.frontmatter_json.get("kind") == "article"
        assert result.frontmatter_json.get("author") == "Test"
        assert result.frontmatter_json.get("version") == 1.0


class TestTask:
    """Test cases for Task model."""

    def test_create_task(self, db_session):
        """Test creating a task associated with a document."""
        # Create vault first
        vault = Vault(
            name="Test Vault",
            container_path="/test",
            host_path="/test",
        )
        db_session.add(vault)
        db_session.flush()

        doc = Document(
            vault_id=vault.id,
            file_path="file.md",
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
        # Create vault first
        vault = Vault(
            name="Test Vault",
            container_path="/test",
            host_path="/test",
        )
        db_session.add(vault)
        db_session.flush()

        doc = Document(
            vault_id=vault.id,
            file_path="file.md",
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
        # Create vault first
        vault = Vault(
            name="Test Vault",
            container_path="/test",
            host_path="/test",
        )
        db_session.add(vault)
        db_session.flush()

        doc = Document(
            vault_id=vault.id,
            file_path="file.md",
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
        # Create vault first
        vault = Vault(
            name="Test Vault",
            container_path="/test",
            host_path="/test",
        )
        db_session.add(vault)
        db_session.flush()

        doc = Document(
            vault_id=vault.id,
            file_path="file.md",
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


class TestArrayType:
    """Test cases for ArrayType TypeDecorator."""

    def test_array_type_loads_postgresql_dialect(self):
        """Test ArrayType loads PG_ARRAY for PostgreSQL dialect."""
        from sqlalchemy.dialects.postgresql import dialect as pg_dialect

        from obsidian_rag.database.models import ArrayType

        array_type = ArrayType()
        impl = array_type.load_dialect_impl(pg_dialect())

        # Should use PostgreSQL ARRAY type (class name may vary by driver)
        assert "ARRAY" in impl.__class__.__name__

    def test_array_type_loads_non_postgresql_dialect(self):
        """Test ArrayType falls back to JSON for non-PostgreSQL dialect."""
        from sqlalchemy.dialects.sqlite import dialect as sqlite_dialect

        from obsidian_rag.database.models import ArrayType

        array_type = ArrayType()
        impl = array_type.load_dialect_impl(sqlite_dialect())

        # Should fallback to JSON for SQLite (class name may vary by dialect)
        class_name = impl.__class__.__name__
        assert "JSON" in class_name or "Json" in class_name


class TestPgvectorExtension:
    """Test cases for pgvector extension creation."""

    def test_pgvector_extension_not_created_for_sqlite(self, caplog):
        """Test that pgvector extension is not created for SQLite."""
        from sqlalchemy import create_engine, text

        from obsidian_rag.database.models import Base

        # Create SQLite engine
        engine = create_engine("sqlite:///:memory:")

        # The before_create event should not raise for SQLite
        # and should not try to create the extension
        Base.metadata.create_all(engine)

        # Verify tables were created
        with engine.connect() as conn:
            result = conn.execute(
                text("SELECT name FROM sqlite_master WHERE type='table'")
            )
            tables = [row[0] for row in result]
            assert "documents" in tables
            assert "tasks" in tables

        engine.dispose()

    def test_document_repr(self):
        """Test Document __repr__ method."""
        from obsidian_rag.database.models import Document

        doc = Document(
            file_path="/test/file.md",
            file_name="file.md",
            content="Test",
            checksum_md5="abc123",
            created_at_fs=datetime.now(),
            modified_at_fs=datetime.now(),
        )
        repr_str = repr(doc)

        assert "Document" in repr_str
        assert "/test/file.md" in repr_str

    def test_task_repr(self):
        """Test Task __repr__ method."""
        from obsidian_rag.database.models import Task

        task = Task(
            document_id=uuid.uuid4(),
            line_number=1,
            raw_text="- [ ] Test task",
            status="not_completed",
            description="Test task description",
        )
        repr_str = repr(task)

        assert "Task" in repr_str
        assert "not_completed" in repr_str
