"""Tests for task operations and document cleanup in IngestionService."""

import uuid
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

from obsidian_rag.config import VaultConfig
from obsidian_rag.database.models import Document, Task
from obsidian_rag.services.ingestion import IngestionService, IngestVaultOptions
from obsidian_rag.services.ingestion_cleanup import _delete_batch

if TYPE_CHECKING:
    from obsidian_rag.config import Settings


@pytest.fixture(autouse=True)
def _patch_try_acquire_ingest_lock() -> None:
    """Patch try_acquire_ingest_lock so vault-level tests don't need real DB setup."""
    with patch(
        "obsidian_rag.services.ingestion.try_acquire_ingest_lock",
        return_value=(True, None),
    ):
        yield


@pytest.fixture
def mock_settings() -> "Settings":
    """Create mock settings for testing."""
    settings = MagicMock()
    settings.ingestion.batch_size = 100
    settings.ingestion.progress_interval = 10
    settings.ingestion.max_chunk_chars = 24000
    settings.ingestion.chunk_overlap_chars = 800
    # Add chunking settings for token-based chunking
    settings.chunking.chunk_size = 512
    settings.chunking.chunk_overlap = 50
    settings.chunking.model_name = "gpt2"
    settings.chunking.cache_dir = None
    settings.vaults = {
        "test-vault": VaultConfig(
            container_path="/test/vault",
            host_path="/test/vault",
        ),
    }
    settings.get_vault.return_value = settings.vaults["test-vault"]
    settings.get_vault_names.return_value = ["test-vault"]
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
def mock_vault_config() -> VaultConfig:
    """Create mock vault config for testing."""
    return VaultConfig(
        container_path="/test/vault",
        host_path="/test/vault",
    )


@pytest.fixture
def mock_vault_record() -> MagicMock:
    """Create mock vault record for testing."""
    vault = MagicMock()
    vault.id = uuid.uuid4()
    vault.name = "test-vault"
    vault.container_path = "/test/vault"
    return vault


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
        mock_task.inline_fields = {}

        parsed_tasks = [(1, mock_task)]

        ingestion_service._create_tasks(mock_session, mock_document, parsed_tasks)  # type: ignore[arg-type]

        mock_session.add.assert_called_once()
        added_task = mock_session.add.call_args[0][0]
        assert isinstance(added_task, Task)
        assert added_task.document_id == "doc-id"
        assert added_task.line_number == 1
        assert added_task.description == "Test task"

    def test_create_tasks_inline_fields_contains_well_known_fields(
        self,
        ingestion_service: IngestionService,
    ) -> None:
        """Test Task.inline_fields is populated from parsed task inline fields."""
        mock_session = MagicMock()
        mock_document = MagicMock()
        mock_document.id = "doc-id"
        mock_document.tags = None

        mock_task = MagicMock()
        mock_task.raw_text = "- [ ] Test task [project:: alpha] [effort:: high]"
        mock_task.status = "not_completed"
        mock_task.description = "Test task"
        mock_task.tags = []
        mock_task.repeat = None
        mock_task.scheduled = None
        mock_task.due = None
        mock_task.completion = None
        mock_task.priority = "normal"
        mock_task.inline_fields = {"project": "alpha", "effort": "high"}

        ingestion_service._create_tasks(mock_session, mock_document, [(1, mock_task)])  # type: ignore[arg-type]

        created_task = mock_session.add.call_args[0][0]
        assert created_task.inline_fields == {"project": "alpha", "effort": "high"}

    def test_create_tasks_inline_fields_with_due_and_vendor(
        self,
        ingestion_service: IngestionService,
    ) -> None:
        """Test well-known fields (due) and custom fields (vendor) both land in inline_fields."""
        mock_session = MagicMock()
        mock_document = MagicMock()
        mock_document.id = "doc-id"
        mock_document.tags = None

        mock_task = MagicMock()
        mock_task.raw_text = "- [ ] Test task [due:: 2026-03-20] [vendor:: Amazon]"
        mock_task.status = "not_completed"
        mock_task.description = "Test task"
        mock_task.tags = []
        mock_task.repeat = None
        mock_task.scheduled = None
        mock_task.due = "2026-03-20"
        mock_task.completion = None
        mock_task.priority = "normal"
        mock_task.inline_fields = {"due": "2026-03-20", "vendor": "Amazon"}

        ingestion_service._create_tasks(mock_session, mock_document, [(1, mock_task)])  # type: ignore[arg-type]

        created_task = mock_session.add.call_args[0][0]
        assert created_task.inline_fields == {"due": "2026-03-20", "vendor": "Amazon"}

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
        mock_task.inline_fields = {}

        parsed_tasks = [(1, mock_task)]

        ingestion_service._update_tasks(mock_session, mock_document, parsed_tasks)  # type: ignore[arg-type]

        # Verify old tasks are deleted
        mock_session.query.return_value.filter_by.return_value.delete.assert_called_once()

        # Verify new task is created
        mock_session.add.assert_called_once()

    def test_create_tasks_merges_document_tags(
        self,
        ingestion_service: IngestionService,
    ) -> None:
        """Verify created Task merges document and task tags."""
        mock_session = MagicMock()
        mock_document = MagicMock()
        mock_document.id = "doc-id"
        mock_document.tags = ["Work", "Urgent"]

        mock_task = MagicMock()
        mock_task.raw_text = "- [ ] do thing #personal"
        mock_task.status = "not_completed"
        mock_task.description = "do thing"
        mock_task.tags = ["personal"]
        mock_task.repeat = None
        mock_task.scheduled = None
        mock_task.due = None
        mock_task.completion = None
        mock_task.priority = "normal"
        mock_task.inline_fields = {}

        ingestion_service._create_tasks(mock_session, mock_document, [(1, mock_task)])  # type: ignore[arg-type]

        created_task = mock_session.add.call_args[0][0]
        assert created_task.tags == ["work", "urgent", "personal"]

    def test_create_tasks_with_none_document_tags(
        self,
        ingestion_service: IngestionService,
    ) -> None:
        """doc tags None, task tags present -> task tags only."""
        mock_session = MagicMock()
        mock_document = MagicMock()
        mock_document.id = "doc-id"
        mock_document.tags = None

        mock_task = MagicMock()
        mock_task.raw_text = "- [ ] do thing"
        mock_task.status = "not_completed"
        mock_task.description = "do thing"
        mock_task.tags = ["personal"]
        mock_task.repeat = None
        mock_task.scheduled = None
        mock_task.due = None
        mock_task.completion = None
        mock_task.priority = "normal"
        mock_task.inline_fields = {}

        ingestion_service._create_tasks(mock_session, mock_document, [(1, mock_task)])  # type: ignore[arg-type]

        created_task = mock_session.add.call_args[0][0]
        assert created_task.tags == ["personal"]

    def test_create_tasks_with_empty_document_tags(
        self,
        ingestion_service: IngestionService,
    ) -> None:
        """doc tags empty, task tags present -> task tags only."""
        mock_session = MagicMock()
        mock_document = MagicMock()
        mock_document.id = "doc-id"
        mock_document.tags = []

        mock_task = MagicMock()
        mock_task.raw_text = "- [ ] do thing"
        mock_task.status = "not_completed"
        mock_task.description = "do thing"
        mock_task.tags = ["personal"]
        mock_task.repeat = None
        mock_task.scheduled = None
        mock_task.due = None
        mock_task.completion = None
        mock_task.priority = "normal"
        mock_task.inline_fields = {}

        ingestion_service._create_tasks(mock_session, mock_document, [(1, mock_task)])  # type: ignore[arg-type]

        created_task = mock_session.add.call_args[0][0]
        assert created_task.tags == ["personal"]

    def test_create_tasks_document_tags_only(
        self,
        ingestion_service: IngestionService,
    ) -> None:
        """doc tags present, task tags None -> doc tags only lowercased."""
        mock_session = MagicMock()
        mock_document = MagicMock()
        mock_document.id = "doc-id"
        mock_document.tags = ["Work"]

        mock_task = MagicMock()
        mock_task.raw_text = "- [ ] do thing"
        mock_task.status = "not_completed"
        mock_task.description = "do thing"
        mock_task.tags = None
        mock_task.repeat = None
        mock_task.scheduled = None
        mock_task.due = None
        mock_task.completion = None
        mock_task.priority = "normal"
        mock_task.inline_fields = {}

        ingestion_service._create_tasks(mock_session, mock_document, [(1, mock_task)])  # type: ignore[arg-type]

        created_task = mock_session.add.call_args[0][0]
        assert created_task.tags == ["work"]

    def test_create_tasks_case_insensitive_merge(
        self,
        ingestion_service: IngestionService,
    ) -> None:
        """Duplicate tags across doc/task are deduplicated."""
        mock_session = MagicMock()
        mock_document = MagicMock()
        mock_document.id = "doc-id"
        mock_document.tags = ["Work"]

        mock_task = MagicMock()
        mock_task.raw_text = "- [ ] do thing #work"
        mock_task.status = "not_completed"
        mock_task.description = "do thing"
        mock_task.tags = ["work"]
        mock_task.repeat = None
        mock_task.scheduled = None
        mock_task.due = None
        mock_task.completion = None
        mock_task.priority = "normal"
        mock_task.inline_fields = {}

        ingestion_service._create_tasks(mock_session, mock_document, [(1, mock_task)])  # type: ignore[arg-type]

        created_task = mock_session.add.call_args[0][0]
        assert created_task.tags == ["work"]

    def test_update_tasks_merges_document_tags(
        self,
        ingestion_service: IngestionService,
    ) -> None:
        """Verify _update_tasks() also uses merged tags since it calls _create_tasks()."""
        mock_session = MagicMock()
        mock_document = MagicMock()
        mock_document.id = "doc-id"
        mock_document.tags = ["Updated"]

        mock_existing_task = MagicMock()
        mock_existing_task.id = uuid.uuid4()

        mock_new_task = MagicMock()
        mock_new_task.raw_text = "- [ ] new task"
        mock_new_task.status = "not_completed"
        mock_new_task.description = "new task"
        mock_new_task.tags = ["inline"]
        mock_new_task.repeat = None
        mock_new_task.scheduled = None
        mock_new_task.due = None
        mock_new_task.completion = None
        mock_new_task.priority = "normal"
        mock_new_task.inline_fields = {}

        mock_session.query.return_value.filter_by.return_value.all.return_value = [
            mock_existing_task
        ]
        mock_session.query.return_value.filter.return_value.first.return_value = None

        ingestion_service._update_tasks(
            mock_session, mock_document, [(1, mock_new_task)]
        )  # type: ignore[arg-type]

        # Verify the new task was created with merged tags
        created_task = mock_session.add.call_args[0][0]
        assert created_task.tags == ["updated", "inline"]


class TestDeleteOrphanedDocuments:
    """Test document deletion during ingestion."""

    def test_delete_orphaned_documents_success(
        self,
        ingestion_service: IngestionService,
        mock_vault_record: MagicMock,
    ) -> None:
        """Test orphaned documents are deleted from database."""
        mock_session = MagicMock()
        mock_session_context = MagicMock()
        mock_session_context.__enter__ = MagicMock(return_value=mock_session)
        mock_session_context.__exit__ = MagicMock(return_value=None)
        ingestion_service.db_manager.get_session.return_value = mock_session_context  # type: ignore[attr-defined]

        # Create mock documents - one orphaned, one not
        mock_doc1 = MagicMock()
        mock_doc1.id = uuid.uuid4()
        mock_doc1.file_path = "orphaned.md"
        mock_doc2 = MagicMock()
        mock_doc2.id = uuid.uuid4()
        mock_doc2.file_path = "existing.md"

        mock_session.query.return_value.filter_by.return_value.all.return_value = [
            mock_doc1,
            mock_doc2,
        ]

        # Mock session.get() to return the document for deletion
        mock_session.get.return_value = mock_doc1

        # Filesystem only has existing.md
        filesystem_paths = {"existing.md"}

        deleted_count, error_count = ingestion_service._delete_orphaned_documents(
            filesystem_paths,
            vault_id=mock_vault_record.id,
        )

        assert deleted_count == 1
        assert error_count == 0
        mock_session.get.assert_called_once_with(Document, mock_doc1.id)
        mock_session.delete.assert_called_once_with(mock_doc1)
        mock_session.commit.assert_called_once()

    def test_delete_orphaned_documents_empty_db(
        self,
        ingestion_service: IngestionService,
        mock_vault_record: MagicMock,
    ) -> None:
        """Test deletion with empty database returns zero counts."""
        mock_session = MagicMock()
        mock_session_context = MagicMock()
        mock_session_context.__enter__ = MagicMock(return_value=mock_session)
        mock_session_context.__exit__ = MagicMock(return_value=None)
        ingestion_service.db_manager.get_session.return_value = mock_session_context  # type: ignore[attr-defined]

        # Empty database
        mock_session.query.return_value.filter_by.return_value.all.return_value = []

        filesystem_paths = {"file.md"}

        deleted_count, error_count = ingestion_service._delete_orphaned_documents(
            filesystem_paths,
            vault_id=mock_vault_record.id,
        )

        assert deleted_count == 0
        assert error_count == 0
        mock_session.delete.assert_not_called()

    def test_delete_orphaned_documents_no_orphans(
        self,
        ingestion_service: IngestionService,
        mock_vault_record: MagicMock,
    ) -> None:
        """Test when all DB documents exist in filesystem."""
        mock_session = MagicMock()
        mock_session_context = MagicMock()
        mock_session_context.__enter__ = MagicMock(return_value=mock_session)
        mock_session_context.__exit__ = MagicMock(return_value=None)
        ingestion_service.db_manager.get_session.return_value = mock_session_context  # type: ignore[attr-defined]

        mock_doc = MagicMock()
        mock_doc.file_path = "existing.md"

        mock_session.query.return_value.filter_by.return_value.all.return_value = [
            mock_doc
        ]

        # Filesystem has the same file
        filesystem_paths = {"existing.md"}

        deleted_count, error_count = ingestion_service._delete_orphaned_documents(
            filesystem_paths,
            vault_id=mock_vault_record.id,
        )

        assert deleted_count == 0
        assert error_count == 0
        mock_session.delete.assert_not_called()

    def test_delete_orphaned_documents_dry_run(
        self,
        ingestion_service: IngestionService,
        mock_vault_record: MagicMock,
    ) -> None:
        """Test dry run mode doesn't actually delete."""
        mock_session = MagicMock()
        mock_session_context = MagicMock()
        mock_session_context.__enter__ = MagicMock(return_value=mock_session)
        mock_session_context.__exit__ = MagicMock(return_value=None)
        ingestion_service.db_manager.get_session.return_value = mock_session_context  # type: ignore[attr-defined]

        mock_doc = MagicMock()
        mock_doc.file_path = "orphaned.md"

        mock_session.query.return_value.filter_by.return_value.all.return_value = [
            mock_doc
        ]

        filesystem_paths: set[str] = set()  # Empty filesystem

        deleted_count, error_count = ingestion_service._delete_orphaned_documents(
            filesystem_paths,
            vault_id=mock_vault_record.id,
            dry_run=True,
        )

        assert deleted_count == 1  # Reports what would be deleted
        assert error_count == 0
        mock_session.delete.assert_not_called()  # But doesn't actually delete
        mock_session.commit.assert_not_called()

    def test_delete_orphaned_documents_batch_processing(
        self,
        ingestion_service: IngestionService,
        mock_vault_record: MagicMock,
    ) -> None:
        """Test documents are deleted in batches."""
        mock_session = MagicMock()
        mock_session_context = MagicMock()
        mock_session_context.__enter__ = MagicMock(return_value=mock_session)
        mock_session_context.__exit__ = MagicMock(return_value=None)
        ingestion_service.db_manager.get_session.return_value = mock_session_context  # type: ignore[attr-defined]

        # Create many orphaned documents
        orphaned_docs = []
        for i in range(150):
            mock_doc = MagicMock()
            mock_doc.file_path = f"orphaned{i}.md"
            orphaned_docs.append(mock_doc)

        mock_session.query.return_value.filter_by.return_value.all.return_value = (
            orphaned_docs
        )

        filesystem_paths: set[str] = set()  # Empty filesystem

        deleted_count, error_count = ingestion_service._delete_orphaned_documents(
            filesystem_paths,
            vault_id=mock_vault_record.id,
            dry_run=False,
        )

        assert deleted_count == 150
        assert error_count == 0
        # Should be called twice (two batches of 100 and 50)
        assert mock_session.commit.call_count == 2

    def test_ingest_vault_reports_deleted_count(
        self,
        ingestion_service: IngestionService,
        tmp_path: Path,
        mock_vault_record: MagicMock,
    ) -> None:
        """Test that ingest_vault reports deleted count in result."""
        mock_session = MagicMock()
        mock_session_context = MagicMock()
        mock_session_context.__enter__ = MagicMock(return_value=mock_session)
        mock_session_context.__exit__ = MagicMock(return_value=None)
        ingestion_service.db_manager.get_session.return_value = mock_session_context  # type: ignore[attr-defined]

        # One document in DB (will be orphaned)
        mock_doc = MagicMock()
        mock_doc.file_path = "old.md"
        mock_session.query.return_value.filter_by.return_value.all.return_value = [
            mock_doc
        ]
        mock_session.query.return_value.filter_by.return_value.first.return_value = None

        # Create vault config using tmp_path as container_path
        vault_config = VaultConfig(
            container_path=str(tmp_path),
            host_path=str(tmp_path),
        )

        # Create one file in filesystem
        test_file = tmp_path / "new.md"
        test_file.write_text("# New Document")

        with patch("obsidian_rag.services.ingestion.scan_markdown_files") as mock_scan:
            mock_file_info = MagicMock()
            mock_file_info.path = test_file
            mock_file_info.name = "new.md"
            mock_file_info.content = "# New Document"

            mock_scan.return_value = [test_file]

            with patch(
                "obsidian_rag.services.ingestion.process_files_in_batches"
            ) as mock_process:
                mock_process.return_value = [mock_file_info]

                with patch(
                    "obsidian_rag.services.ingestion.parse_frontmatter"
                ) as mock_parse_fm:
                    mock_parse_fm.return_value = (None, {}, "# New Document")

                    with patch.object(
                        ingestion_service,
                        "_get_or_create_vault",
                        return_value=mock_vault_record,
                    ):
                        options = IngestVaultOptions(vault=vault_config)
                        result = ingestion_service.ingest_vault(tmp_path, options)

                        assert result.deleted == 1
                        assert "1 deleted" in result.message

    def test_ingest_vault_with_no_delete_flag(
        self,
        ingestion_service: IngestionService,
        tmp_path: Path,
        mock_vault_record: MagicMock,
    ) -> None:
        """Test that no_delete flag skips deletion phase."""
        mock_session = MagicMock()
        mock_session_context = MagicMock()
        mock_session_context.__enter__ = MagicMock(return_value=mock_session)
        mock_session_context.__exit__ = MagicMock(return_value=None)
        ingestion_service.db_manager.get_session.return_value = mock_session_context  # type: ignore[attr-defined]

        # Document exists in DB (would be orphaned)
        mock_session.query.return_value.filter_by.return_value.first.return_value = None

        # Create vault config using tmp_path as container_path
        vault_config = VaultConfig(
            container_path=str(tmp_path),
            host_path=str(tmp_path),
        )

        # Create file in filesystem
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
                    mock_parse_fm.return_value = (None, {}, "# Test Document")

                    with patch.object(
                        ingestion_service,
                        "_get_or_create_vault",
                        return_value=mock_vault_record,
                    ):
                        options = IngestVaultOptions(
                            vault=vault_config,
                            no_delete=True,
                        )
                        result = ingestion_service.ingest_vault(tmp_path, options)

                        assert result.deleted == 0
                        assert "deletion skipped" in result.message

    def test_delete_batch_with_commit_failure(
        self,
        ingestion_service: IngestionService,
    ) -> None:
        """Test batch deletion when commit fails."""
        mock_session = MagicMock()

        # Create mock document info tuples (id, file_path)
        doc1_id = uuid.uuid4()
        doc2_id = uuid.uuid4()
        batch = [(doc1_id, "doc1.md"), (doc2_id, "doc2.md")]

        # Mock session.get() to return mock documents
        mock_doc1 = MagicMock()
        mock_doc2 = MagicMock()
        mock_session.get.side_effect = [mock_doc1, mock_doc2]

        # Simulate commit failure
        mock_session.commit.side_effect = RuntimeError("Database error")

        deleted_count, error_count = _delete_batch(
            mock_session,
            batch,
        )

        # When commit fails, all documents in batch are marked as failed
        assert deleted_count == 0
        assert error_count == 2
        mock_session.commit.assert_called_once()
