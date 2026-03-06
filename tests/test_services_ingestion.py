"""Tests for the IngestionService."""

from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

from obsidian_rag.database.models import Document, Task
from obsidian_rag.llm.base import EmbeddingError
from obsidian_rag.services.ingestion import IngestionResult, IngestionService

if TYPE_CHECKING:
    from obsidian_rag.config import Settings
    from obsidian_rag.parsing.scanner import FileInfo


@pytest.fixture
def mock_settings() -> "Settings":
    """Create mock settings for testing."""
    settings = MagicMock()
    settings.ingestion.batch_size = 100
    settings.ingestion.progress_interval = 10
    return settings


@pytest.fixture
def mock_db_manager() -> MagicMock:
    """Create mock database manager for testing."""
    return MagicMock()


@pytest.fixture
def mock_embedding_provider() -> MagicMock:
    """Create mock embedding provider for testing."""
    provider = MagicMock()
    provider.generate_embedding.return_value = [0.1, 0.2, 0.3]
    return provider


@pytest.fixture
def ingestion_service(
    mock_db_manager: MagicMock,
    mock_embedding_provider: MagicMock,
    mock_settings: "Settings",
) -> IngestionService:
    """Create IngestionService instance for testing."""
    return IngestionService(
        db_manager=mock_db_manager,
        embedding_provider=mock_embedding_provider,
        settings=mock_settings,
    )


class TestIngestionResult:
    """Test IngestionResult dataclass."""

    def test_ingestion_result_creation(self) -> None:
        """Test IngestionResult can be created with all fields."""
        result = IngestionResult(
            total=10,
            new=3,
            updated=2,
            unchanged=4,
            errors=1,
            processing_time_seconds=5.5,
            message="Test message",
        )

        assert result.total == 10
        assert result.new == 3
        assert result.updated == 2
        assert result.unchanged == 4
        assert result.errors == 1
        assert result.processing_time_seconds == 5.5
        assert result.message == "Test message"

    def test_to_dict(self) -> None:
        """Test to_dict method converts result to dictionary."""
        result = IngestionResult(
            total=10,
            new=3,
            updated=2,
            unchanged=4,
            errors=1,
            processing_time_seconds=5.5,
            message="Test message",
        )

        data = result.to_dict()

        assert data["total"] == 10
        assert data["new"] == 3
        assert data["updated"] == 2
        assert data["unchanged"] == 4
        assert data["errors"] == 1
        assert data["processing_time_seconds"] == 5.5
        assert data["message"] == "Test message"


class TestIngestionServiceInit:
    """Test IngestionService initialization."""

    def test_init_with_embedding_provider(
        self,
        mock_db_manager: MagicMock,
        mock_embedding_provider: MagicMock,
        mock_settings: "Settings",
    ) -> None:
        """Test initialization with embedding provider."""
        service = IngestionService(
            db_manager=mock_db_manager,
            embedding_provider=mock_embedding_provider,
            settings=mock_settings,
        )

        assert service.db_manager is mock_db_manager
        assert service.embedding_provider is mock_embedding_provider
        assert service.settings is mock_settings

    def test_init_without_embedding_provider(
        self,
        mock_db_manager: MagicMock,
        mock_settings: "Settings",
    ) -> None:
        """Test initialization without embedding provider."""
        service = IngestionService(
            db_manager=mock_db_manager,
            embedding_provider=None,
            settings=mock_settings,
        )

        assert service.db_manager is mock_db_manager
        assert service.embedding_provider is None
        assert service.settings is mock_settings


class TestIngestVault:
    """Test ingest_vault method."""

    def test_ingest_vault_empty_directory(
        self,
        ingestion_service: IngestionService,
        tmp_path: Path,
    ) -> None:
        """Test ingesting empty directory returns zero counts."""
        with patch("obsidian_rag.services.ingestion.scan_markdown_files") as mock_scan:
            mock_scan.return_value = []

            result = ingestion_service.ingest_vault(tmp_path)

            assert result.total == 0
            assert result.new == 0
            assert result.updated == 0
            assert result.unchanged == 0
            assert result.errors == 0
            assert "No markdown files found" in result.message

    def test_ingest_vault_with_provided_file_infos(
        self,
        ingestion_service: IngestionService,
        tmp_path: Path,
    ) -> None:
        """Test ingesting with pre-provided file_infos."""
        mock_session = MagicMock()
        mock_session_context = MagicMock()
        mock_session_context.__enter__ = MagicMock(return_value=mock_session)
        mock_session_context.__exit__ = MagicMock(return_value=None)
        ingestion_service.db_manager.get_session.return_value = mock_session_context

        mock_session.query.return_value.filter_by.return_value.first.return_value = None

        file_info = MagicMock()
        file_info.path = tmp_path / "test.md"
        file_info.name = "test.md"
        file_info.content = "# Test\n\n- [ ] Task 1"
        file_info.checksum = "abc123"
        file_info.created_at = datetime.now(timezone.utc)
        file_info.modified_at = datetime.now(timezone.utc)

        with patch(
            "obsidian_rag.services.ingestion.parse_frontmatter"
        ) as mock_parse_fm:
            mock_parse_fm.return_value = (None, None, {}, "# Test\n\n- [ ] Task 1")

            with patch(
                "obsidian_rag.services.ingestion.parse_tasks_from_content"
            ) as mock_parse_tasks:
                mock_parse_tasks.return_value = []

                result = ingestion_service.ingest_vault(
                    vault_path=tmp_path,
                    file_infos=[file_info],
                )

                assert result.total == 1
                assert result.new == 1
                assert result.updated == 0
                assert result.unchanged == 0
                assert result.errors == 0

    def test_ingest_vault_with_new_document(
        self,
        ingestion_service: IngestionService,
        tmp_path: Path,
    ) -> None:
        """Test ingesting new document."""
        mock_session = MagicMock()
        mock_session_context = MagicMock()
        mock_session_context.__enter__ = MagicMock(return_value=mock_session)
        mock_session_context.__exit__ = MagicMock(return_value=None)
        ingestion_service.db_manager.get_session.return_value = mock_session_context

        mock_session.query.return_value.filter_by.return_value.first.return_value = None

        test_file = tmp_path / "test.md"
        test_file.write_text("# Test Document\n\n- [ ] A task")

        with patch("obsidian_rag.services.ingestion.scan_markdown_files") as mock_scan:
            mock_file_info = MagicMock()
            mock_file_info.path = test_file
            mock_file_info.name = "test.md"
            mock_file_info.content = "# Test Document\n\n- [ ] A task"
            mock_file_info.checksum = "abc123"
            mock_file_info.created_at = datetime.now(timezone.utc)
            mock_file_info.modified_at = datetime.now(timezone.utc)

            mock_scan.return_value = [test_file]

            with patch(
                "obsidian_rag.services.ingestion.process_files_in_batches"
            ) as mock_process:
                mock_process.return_value = [mock_file_info]

                with patch(
                    "obsidian_rag.services.ingestion.parse_frontmatter"
                ) as mock_parse_fm:
                    mock_parse_fm.return_value = (
                        None,
                        None,
                        {},
                        "# Test Document\n\n- [ ] A task",
                    )

                    with patch(
                        "obsidian_rag.services.ingestion.parse_tasks_from_content"
                    ) as mock_parse_tasks:
                        mock_parse_tasks.return_value = []

                        result = ingestion_service.ingest_vault(tmp_path)

                        assert result.total == 1
                        assert result.new == 1
                        assert result.updated == 0
                        assert result.unchanged == 0
                        assert result.errors == 0

    def test_ingest_vault_with_unchanged_document(
        self,
        ingestion_service: IngestionService,
        tmp_path: Path,
    ) -> None:
        """Test ingesting unchanged document."""
        mock_session = MagicMock()
        mock_session_context = MagicMock()
        mock_session_context.__enter__ = MagicMock(return_value=mock_session)
        mock_session_context.__exit__ = MagicMock(return_value=None)
        ingestion_service.db_manager.get_session.return_value = mock_session_context

        existing_doc = MagicMock()
        existing_doc.checksum_md5 = "abc123"

        mock_session.query.return_value.filter_by.return_value.first.return_value = (
            existing_doc
        )

        test_file = tmp_path / "test.md"
        test_file.write_text("# Test Document")

        with patch("obsidian_rag.services.ingestion.scan_markdown_files") as mock_scan:
            mock_file_info = MagicMock()
            mock_file_info.path = test_file
            mock_file_info.name = "test.md"
            mock_file_info.content = "# Test Document"
            mock_file_info.checksum = "abc123"  # Same checksum

            mock_scan.return_value = [test_file]

            with patch(
                "obsidian_rag.services.ingestion.process_files_in_batches"
            ) as mock_process:
                mock_process.return_value = [mock_file_info]

                with patch(
                    "obsidian_rag.services.ingestion.parse_frontmatter"
                ) as mock_parse_fm:
                    mock_parse_fm.return_value = (None, None, {}, "# Test Document")

                    result = ingestion_service.ingest_vault(tmp_path)

                    assert result.total == 1
                    assert result.new == 0
                    assert result.updated == 0
                    assert result.unchanged == 1
                    assert result.errors == 0

    def test_ingest_vault_with_updated_document(
        self,
        ingestion_service: IngestionService,
        tmp_path: Path,
    ) -> None:
        """Test ingesting updated document."""
        mock_session = MagicMock()
        mock_session_context = MagicMock()
        mock_session_context.__enter__ = MagicMock(return_value=mock_session)
        mock_session_context.__exit__ = MagicMock(return_value=None)
        ingestion_service.db_manager.get_session.return_value = mock_session_context

        existing_doc = MagicMock()
        existing_doc.checksum_md5 = "old_checksum"
        existing_doc.id = "doc-id"

        mock_session.query.return_value.filter_by.return_value.first.return_value = (
            existing_doc
        )

        test_file = tmp_path / "test.md"
        test_file.write_text("# Updated Document")

        with patch("obsidian_rag.services.ingestion.scan_markdown_files") as mock_scan:
            mock_file_info = MagicMock()
            mock_file_info.path = test_file
            mock_file_info.name = "test.md"
            mock_file_info.content = "# Updated Document"
            mock_file_info.checksum = "new_checksum"  # Different checksum
            mock_file_info.modified_at = datetime.now(timezone.utc)

            mock_scan.return_value = [test_file]

            with patch(
                "obsidian_rag.services.ingestion.process_files_in_batches"
            ) as mock_process:
                mock_process.return_value = [mock_file_info]

                with patch(
                    "obsidian_rag.services.ingestion.parse_frontmatter"
                ) as mock_parse_fm:
                    mock_parse_fm.return_value = (None, None, {}, "# Updated Document")

                    with patch(
                        "obsidian_rag.services.ingestion.parse_tasks_from_content"
                    ) as mock_parse_tasks:
                        mock_parse_tasks.return_value = []

                        result = ingestion_service.ingest_vault(tmp_path)

                        assert result.total == 1
                        assert result.new == 0
                        assert result.updated == 1
                        assert result.unchanged == 0
                        assert result.errors == 0

    def test_ingest_vault_with_file_error(
        self,
        ingestion_service: IngestionService,
        tmp_path: Path,
    ) -> None:
        """Test that file errors are counted but don't stop processing."""
        test_file = tmp_path / "test.md"
        test_file.write_text("# Test")

        with patch("obsidian_rag.services.ingestion.scan_markdown_files") as mock_scan:
            mock_file_info = MagicMock()
            mock_file_info.path = test_file
            mock_file_info.name = "test.md"
            mock_file_info.content = "# Test"

            mock_scan.return_value = [test_file]

            with patch(
                "obsidian_rag.services.ingestion.process_files_in_batches"
            ) as mock_process:
                mock_process.return_value = [mock_file_info]

                with patch(
                    "obsidian_rag.services.ingestion.parse_frontmatter"
                ) as mock_parse_fm:
                    mock_parse_fm.side_effect = Exception("Parse error")

                    result = ingestion_service.ingest_vault(tmp_path)

                    assert result.total == 1
                    assert result.new == 0
                    assert result.updated == 0
                    assert result.unchanged == 0
                    assert result.errors == 1

    def test_ingest_vault_dry_run(
        self,
        ingestion_service: IngestionService,
        tmp_path: Path,
    ) -> None:
        """Test dry run mode doesn't write to database."""
        test_file = tmp_path / "test.md"
        test_file.write_text("# Test Document")

        with patch("obsidian_rag.services.ingestion.scan_markdown_files") as mock_scan:
            mock_file_info = MagicMock()
            mock_file_info.path = test_file
            mock_file_info.name = "test.md"
            mock_file_info.content = "# Test Document"

            mock_scan.return_value = [test_file]

            with patch(
                "obsidian_rag.services.ingestion.process_files_in_batches"
            ) as mock_process:
                mock_process.return_value = [mock_file_info]

                with patch(
                    "obsidian_rag.services.ingestion.parse_frontmatter"
                ) as mock_parse_fm:
                    mock_parse_fm.return_value = (None, None, {}, "# Test Document")

                    result = ingestion_service.ingest_vault(
                        tmp_path,
                        dry_run=True,
                    )

                    assert result.total == 1
                    assert result.new == 1
                    assert result.updated == 0
                    assert result.unchanged == 0
                    assert result.errors == 0

                    # Verify db_manager.get_session was never called
                    ingestion_service.db_manager.get_session.assert_not_called()

    def test_ingest_vault_with_progress_callback(
        self,
        ingestion_service: IngestionService,
        tmp_path: Path,
    ) -> None:
        """Test progress callback is called during ingestion."""
        mock_session = MagicMock()
        mock_session_context = MagicMock()
        mock_session_context.__enter__ = MagicMock(return_value=mock_session)
        mock_session_context.__exit__ = MagicMock(return_value=None)
        ingestion_service.db_manager.get_session.return_value = mock_session_context

        mock_session.query.return_value.filter_by.return_value.first.return_value = None

        test_file = tmp_path / "test.md"
        test_file.write_text("# Test")

        progress_calls = []

        def progress_callback(
            current: int, total: int, successes: int, errors: int
        ) -> None:
            progress_calls.append((current, total, successes, errors))

        with patch("obsidian_rag.services.ingestion.scan_markdown_files") as mock_scan:
            mock_file_info = MagicMock()
            mock_file_info.path = test_file
            mock_file_info.name = "test.md"
            mock_file_info.content = "# Test"
            mock_file_info.checksum = "abc123"

            mock_scan.return_value = [test_file]

            with patch(
                "obsidian_rag.services.ingestion.process_files_in_batches"
            ) as mock_process:
                mock_process.return_value = [mock_file_info]

                with patch(
                    "obsidian_rag.services.ingestion.parse_frontmatter"
                ) as mock_parse_fm:
                    mock_parse_fm.return_value = (None, None, {}, "# Test")

                    with patch(
                        "obsidian_rag.services.ingestion.parse_tasks_from_content"
                    ) as mock_parse_tasks:
                        mock_parse_tasks.return_value = []

                        ingestion_service.ingest_vault(
                            tmp_path,
                            progress_callback=progress_callback,
                        )

                        assert len(progress_calls) > 0
                        # Last call should have current=1, total=1
                        assert progress_calls[-1][0] == 1
                        assert progress_calls[-1][1] == 1


class TestIngestSingleFile:
    """Test _ingest_single_file method."""

    def test_create_document_with_embedding(
        self,
        ingestion_service: IngestionService,
        tmp_path: Path,
    ) -> None:
        """Test document creation with embedding generation."""
        mock_session = MagicMock()
        mock_session_context = MagicMock()
        mock_session_context.__enter__ = MagicMock(return_value=mock_session)
        mock_session_context.__exit__ = MagicMock(return_value=None)
        ingestion_service.db_manager.get_session.return_value = mock_session_context

        mock_session.query.return_value.filter_by.return_value.first.return_value = None

        mock_file_info = MagicMock()
        mock_file_info.path = tmp_path / "test.md"
        mock_file_info.name = "test.md"
        mock_file_info.content = "Test content"
        mock_file_info.checksum = "abc123"
        mock_file_info.created_at = datetime.now(timezone.utc)
        mock_file_info.modified_at = datetime.now(timezone.utc)

        with patch(
            "obsidian_rag.services.ingestion.parse_frontmatter"
        ) as mock_parse_fm:
            mock_parse_fm.return_value = (None, None, {}, "Test content")

            with patch(
                "obsidian_rag.services.ingestion.parse_tasks_from_content"
            ) as mock_parse_tasks:
                mock_parse_tasks.return_value = []

                result = ingestion_service._ingest_single_file(mock_file_info)

                assert result == "new"
                ingestion_service.embedding_provider.generate_embedding.assert_called_once_with(
                    "Test content"
                )

    def test_create_document_without_embedding_provider(
        self,
        mock_db_manager: MagicMock,
        mock_settings: "Settings",
        tmp_path: Path,
    ) -> None:
        """Test document creation without embedding provider."""
        service = IngestionService(
            db_manager=mock_db_manager,
            embedding_provider=None,
            settings=mock_settings,
        )

        mock_session = MagicMock()
        mock_session_context = MagicMock()
        mock_session_context.__enter__ = MagicMock(return_value=mock_session)
        mock_session_context.__exit__ = MagicMock(return_value=None)
        service.db_manager.get_session.return_value = mock_session_context

        mock_session.query.return_value.filter_by.return_value.first.return_value = None

        mock_file_info = MagicMock()
        mock_file_info.path = tmp_path / "test.md"
        mock_file_info.name = "test.md"
        mock_file_info.content = "Test content"
        mock_file_info.checksum = "abc123"
        mock_file_info.created_at = datetime.now(timezone.utc)
        mock_file_info.modified_at = datetime.now(timezone.utc)

        with patch(
            "obsidian_rag.services.ingestion.parse_frontmatter"
        ) as mock_parse_fm:
            mock_parse_fm.return_value = (None, None, {}, "Test content")

            with patch(
                "obsidian_rag.services.ingestion.parse_tasks_from_content"
            ) as mock_parse_tasks:
                mock_parse_tasks.return_value = []

                result = service._ingest_single_file(mock_file_info)

                assert result == "new"

    def test_embedding_generation_failure(
        self,
        ingestion_service: IngestionService,
        tmp_path: Path,
    ) -> None:
        """Test that embedding failure doesn't block document creation."""
        mock_session = MagicMock()
        mock_session_context = MagicMock()
        mock_session_context.__enter__ = MagicMock(return_value=mock_session)
        mock_session_context.__exit__ = MagicMock(return_value=None)
        ingestion_service.db_manager.get_session.return_value = mock_session_context

        mock_session.query.return_value.filter_by.return_value.first.return_value = None

        ingestion_service.embedding_provider.generate_embedding.side_effect = (
            EmbeddingError("Embedding failed")
        )

        mock_file_info = MagicMock()
        mock_file_info.path = tmp_path / "test.md"
        mock_file_info.name = "test.md"
        mock_file_info.content = "Test content"
        mock_file_info.checksum = "abc123"
        mock_file_info.created_at = datetime.now(timezone.utc)
        mock_file_info.modified_at = datetime.now(timezone.utc)

        with patch(
            "obsidian_rag.services.ingestion.parse_frontmatter"
        ) as mock_parse_fm:
            mock_parse_fm.return_value = (None, None, {}, "Test content")

            with patch(
                "obsidian_rag.services.ingestion.parse_tasks_from_content"
            ) as mock_parse_tasks:
                mock_parse_tasks.return_value = []

                result = ingestion_service._ingest_single_file(mock_file_info)

                assert result == "new"


class TestTaskOperations:
    """Test task creation and update operations."""

    def test_create_tasks(
        self,
        ingestion_service: IngestionService,
    ) -> None:
        """Test task creation for a document."""
        mock_session = MagicMock()
        mock_document = MagicMock()
        mock_document.id = "doc-id"

        mock_task = MagicMock()
        mock_task.raw_text = "- [ ] Test task"
        mock_task.status = "not_completed"
        mock_task.description = "Test task"
        mock_task.tags = []
        mock_task.repeat = None
        mock_task.scheduled = None
        mock_task.due = None
        mock_task.completion = None
        mock_task.priority = "normal"
        mock_task.custom_metadata = {}

        parsed_tasks = [(1, mock_task)]

        ingestion_service._create_tasks(mock_session, mock_document, parsed_tasks)

        mock_session.add.assert_called_once()
        added_task = mock_session.add.call_args[0][0]
        assert isinstance(added_task, Task)
        assert added_task.document_id == "doc-id"
        assert added_task.line_number == 1
        assert added_task.description == "Test task"

    def test_update_tasks_deletes_old_and_creates_new(
        self,
        ingestion_service: IngestionService,
    ) -> None:
        """Test task update deletes old tasks and creates new ones."""
        mock_session = MagicMock()
        mock_document = MagicMock()
        mock_document.id = "doc-id"

        mock_task = MagicMock()
        mock_task.raw_text = "- [ ] New task"
        mock_task.status = "not_completed"
        mock_task.description = "New task"
        mock_task.tags = []
        mock_task.repeat = None
        mock_task.scheduled = None
        mock_task.due = None
        mock_task.completion = None
        mock_task.priority = "normal"
        mock_task.custom_metadata = {}

        parsed_tasks = [(1, mock_task)]

        ingestion_service._update_tasks(mock_session, mock_document, parsed_tasks)

        # Verify old tasks are deleted
        mock_session.query.return_value.filter_by.return_value.delete.assert_called_once()

        # Verify new task is created
        mock_session.add.assert_called_once()
