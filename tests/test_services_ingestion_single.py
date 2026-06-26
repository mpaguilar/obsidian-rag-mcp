"""Tests for single-file ingestion in IngestionService."""

import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

from obsidian_rag.config import VaultConfig
from obsidian_rag.database.models import Task
from obsidian_rag.llm.base import EmbeddingError
from obsidian_rag.services.ingestion import IngestionService

if TYPE_CHECKING:
    from obsidian_rag.config import Settings


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


class TestIngestSingleFile:
    """Test _ingest_single_file method."""

    def test_ingest_single_file_merges_document_tags(
        self,
        ingestion_service: IngestionService,
        tmp_path: Path,
    ) -> None:
        """Test end-to-end tag merging during new document ingestion."""
        mock_session = MagicMock()
        mock_session_context = MagicMock()
        mock_session_context.__enter__ = MagicMock(return_value=mock_session)
        mock_session_context.__exit__ = MagicMock(return_value=None)
        ingestion_service.db_manager.get_session.return_value = mock_session_context  # type: ignore[attr-defined]

        mock_session.query.return_value.filter_by.return_value.first.return_value = None

        vault_config = VaultConfig(
            container_path=str(tmp_path),
            host_path=str(tmp_path),
        )

        file_info = MagicMock()
        file_info.path = tmp_path / "test.md"
        file_info.name = "test.md"
        file_info.content = "---\ntags: [Work, Urgent]\n---\n- [ ] do thing #personal"
        file_info.checksum = "abc123"
        file_info.created_at = datetime.now(timezone.utc)
        file_info.modified_at = datetime.now(timezone.utc)

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
        mock_task.custom_metadata = {}

        with patch(
            "obsidian_rag.services.ingestion.parse_frontmatter"
        ) as mock_parse_fm:
            mock_parse_fm.return_value = (
                ["Work", "Urgent"],
                {},
                "- [ ] do thing #personal",
            )

            with patch(
                "obsidian_rag.services.ingestion.parse_tasks_from_content"
            ) as mock_parse_tasks:
                mock_parse_tasks.return_value = [(1, mock_task)]

                with patch(
                    "obsidian_rag.services.ingestion.should_chunk_document"
                ) as mock_should_chunk:
                    mock_should_chunk.return_value = False

                    ingestion_service._ingest_single_file(
                        file_info,
                        vault_id=uuid.uuid4(),
                        vault_config=vault_config,
                    )

        calls = mock_session.add.call_args_list
        task_calls = [c for c in calls if isinstance(c[0][0], Task)]
        assert len(task_calls) > 0
        created_task = task_calls[0][0][0]
        assert created_task.tags == ["work", "urgent", "personal"]

    def test_ingest_single_file_update_merges_tags(
        self,
        ingestion_service: IngestionService,
        tmp_path: Path,
    ) -> None:
        """Mock existing document with old tags being updated to new document-level tags."""
        mock_session = MagicMock()
        mock_session_context = MagicMock()
        mock_session_context.__enter__ = MagicMock(return_value=mock_session)
        mock_session_context.__exit__ = MagicMock(return_value=None)
        ingestion_service.db_manager.get_session.return_value = mock_session_context  # type: ignore[attr-defined]

        mock_existing_doc = MagicMock()
        mock_existing_doc.id = uuid.uuid4()
        mock_existing_doc.checksum_md5 = "old_checksum"
        mock_existing_doc.tags = ["old"]

        mock_session.query.return_value.filter_by.return_value.first.return_value = (
            mock_existing_doc
        )

        vault_config = VaultConfig(
            container_path=str(tmp_path),
            host_path=str(tmp_path),
        )

        file_info = MagicMock()
        file_info.path = tmp_path / "test.md"
        file_info.name = "test.md"
        file_info.content = "---\ntags: [Updated]\n---\n- [ ] do thing #inline"
        file_info.checksum = "new_checksum"
        file_info.created_at = datetime.now(timezone.utc)
        file_info.modified_at = datetime.now(timezone.utc)

        mock_task = MagicMock()
        mock_task.raw_text = "- [ ] do thing #inline"
        mock_task.status = "not_completed"
        mock_task.description = "do thing"
        mock_task.tags = ["inline"]
        mock_task.repeat = None
        mock_task.scheduled = None
        mock_task.due = None
        mock_task.completion = None
        mock_task.priority = "normal"
        mock_task.custom_metadata = {}

        with patch(
            "obsidian_rag.services.ingestion.parse_frontmatter"
        ) as mock_parse_fm:
            mock_parse_fm.return_value = (
                ["Updated"],
                {},
                "- [ ] do thing #inline",
            )

            with patch(
                "obsidian_rag.services.ingestion.parse_tasks_from_content"
            ) as mock_parse_tasks:
                mock_parse_tasks.return_value = [(1, mock_task)]

                with patch(
                    "obsidian_rag.services.ingestion.should_chunk_document"
                ) as mock_should_chunk:
                    mock_should_chunk.return_value = False

                    result, chunks_created, is_empty = (
                        ingestion_service._ingest_single_file(
                            file_info,
                            vault_id=uuid.uuid4(),
                            vault_config=vault_config,
                        )
                    )

                    assert result == "updated"

        calls = mock_session.add.call_args_list
        task_calls = [c for c in calls if isinstance(c[0][0], Task)]
        assert len(task_calls) > 0
        created_task = task_calls[0][0][0]
        assert created_task.tags == ["updated", "inline"]

    def test_ingest_single_file_no_document_tags(
        self,
        ingestion_service: IngestionService,
        tmp_path: Path,
    ) -> None:
        """Mock parse_frontmatter returning (None, {}, 'content') and task with ['personal']."""
        mock_session = MagicMock()
        mock_session_context = MagicMock()
        mock_session_context.__enter__ = MagicMock(return_value=mock_session)
        mock_session_context.__exit__ = MagicMock(return_value=None)
        ingestion_service.db_manager.get_session.return_value = mock_session_context  # type: ignore[attr-defined]

        mock_session.query.return_value.filter_by.return_value.first.return_value = None

        vault_config = VaultConfig(
            container_path=str(tmp_path),
            host_path=str(tmp_path),
        )

        file_info = MagicMock()
        file_info.path = tmp_path / "test.md"
        file_info.name = "test.md"
        file_info.content = "- [ ] do thing #personal"
        file_info.checksum = "abc123"
        file_info.created_at = datetime.now(timezone.utc)
        file_info.modified_at = datetime.now(timezone.utc)

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
        mock_task.custom_metadata = {}

        with patch(
            "obsidian_rag.services.ingestion.parse_frontmatter"
        ) as mock_parse_fm:
            mock_parse_fm.return_value = (None, {}, "- [ ] do thing #personal")

            with patch(
                "obsidian_rag.services.ingestion.parse_tasks_from_content"
            ) as mock_parse_tasks:
                mock_parse_tasks.return_value = [(1, mock_task)]

                with patch(
                    "obsidian_rag.services.ingestion.should_chunk_document"
                ) as mock_should_chunk:
                    mock_should_chunk.return_value = False

                    ingestion_service._ingest_single_file(
                        file_info,
                        vault_id=uuid.uuid4(),
                        vault_config=vault_config,
                    )

        calls = mock_session.add.call_args_list
        task_calls = [c for c in calls if isinstance(c[0][0], Task)]
        assert len(task_calls) > 0
        created_task = task_calls[0][0][0]
        assert created_task.tags == ["personal"]

    def test_ingest_single_file_no_task_tags(
        self,
        ingestion_service: IngestionService,
        tmp_path: Path,
    ) -> None:
        """Mock parse_frontmatter returning (['Work'], {}, 'content') and task with no inline tags."""
        mock_session = MagicMock()
        mock_session_context = MagicMock()
        mock_session_context.__enter__ = MagicMock(return_value=mock_session)
        mock_session_context.__exit__ = MagicMock(return_value=None)
        ingestion_service.db_manager.get_session.return_value = mock_session_context  # type: ignore[attr-defined]

        mock_session.query.return_value.filter_by.return_value.first.return_value = None

        vault_config = VaultConfig(
            container_path=str(tmp_path),
            host_path=str(tmp_path),
        )

        file_info = MagicMock()
        file_info.path = tmp_path / "test.md"
        file_info.name = "test.md"
        file_info.content = "---\ntags: [Work]\n---\n- [ ] do thing"
        file_info.checksum = "abc123"
        file_info.created_at = datetime.now(timezone.utc)
        file_info.modified_at = datetime.now(timezone.utc)

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
        mock_task.custom_metadata = {}

        with patch(
            "obsidian_rag.services.ingestion.parse_frontmatter"
        ) as mock_parse_fm:
            mock_parse_fm.return_value = (
                ["Work"],
                {},
                "- [ ] do thing",
            )

            with patch(
                "obsidian_rag.services.ingestion.parse_tasks_from_content"
            ) as mock_parse_tasks:
                mock_parse_tasks.return_value = [(1, mock_task)]

                with patch(
                    "obsidian_rag.services.ingestion.should_chunk_document"
                ) as mock_should_chunk:
                    mock_should_chunk.return_value = False

                    ingestion_service._ingest_single_file(
                        file_info,
                        vault_id=uuid.uuid4(),
                        vault_config=vault_config,
                    )

        calls = mock_session.add.call_args_list
        task_calls = [c for c in calls if isinstance(c[0][0], Task)]
        assert len(task_calls) > 0
        created_task = task_calls[0][0][0]
        assert created_task.tags == ["work"]

    @patch("obsidian_rag.services.ingestion.should_chunk_document")
    def test_create_document_with_embedding(
        self,
        mock_should_chunk: MagicMock,
        ingestion_service: IngestionService,
        tmp_path: Path,
        mock_vault_record: MagicMock,
    ) -> None:
        """Test document creation with embedding generation."""
        mock_should_chunk.return_value = False

        mock_session = MagicMock()
        mock_session_context = MagicMock()
        mock_session_context.__enter__ = MagicMock(return_value=mock_session)
        mock_session_context.__exit__ = MagicMock(return_value=None)
        ingestion_service.db_manager.get_session.return_value = mock_session_context  # type: ignore[attr-defined]

        mock_session.query.return_value.filter_by.return_value.first.return_value = None

        # Create vault config using tmp_path as container_path
        vault_config = VaultConfig(
            container_path=str(tmp_path),
            host_path=str(tmp_path),
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
            mock_parse_fm.return_value = (None, {}, "Test content")

            with patch(
                "obsidian_rag.services.ingestion.parse_tasks_from_content"
            ) as mock_parse_tasks:
                mock_parse_tasks.return_value = []

                result, chunks_created, is_empty = (
                    ingestion_service._ingest_single_file(
                        mock_file_info,
                        vault_id=mock_vault_record.id,
                        vault_config=vault_config,
                    )
                )

                assert result == "new"
                assert chunks_created == 0
                assert is_empty is False
                ingestion_service.embedding_provider.generate_embedding.assert_called_once_with(  # type: ignore[union-attr]
                    "Test content"
                )

    @patch("obsidian_rag.services.ingestion.should_chunk_document")
    def test_create_document_without_embedding_provider(
        self,
        mock_should_chunk: MagicMock,
        mock_db_manager: MagicMock,
        mock_settings: "Settings",
        tmp_path: Path,
        mock_vault_record: MagicMock,
    ) -> None:
        """Test document creation without embedding provider."""
        mock_should_chunk.return_value = False

        service = IngestionService(
            db_manager=mock_db_manager,
            embedding_provider=None,
            settings=mock_settings,
        )

        mock_session = MagicMock()
        mock_session_context = MagicMock()
        mock_session_context.__enter__ = MagicMock(return_value=mock_session)
        mock_session_context.__exit__ = MagicMock(return_value=None)
        service.db_manager.get_session.return_value = mock_session_context  # type: ignore[attr-defined]

        mock_session.query.return_value.filter_by.return_value.first.return_value = None

        # Create vault config using tmp_path as container_path
        vault_config = VaultConfig(
            container_path=str(tmp_path),
            host_path=str(tmp_path),
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
            mock_parse_fm.return_value = (None, {}, "Test content")

            with patch(
                "obsidian_rag.services.ingestion.parse_tasks_from_content"
            ) as mock_parse_tasks:
                mock_parse_tasks.return_value = []

                result, chunks_created, is_empty = service._ingest_single_file(
                    mock_file_info,
                    vault_id=mock_vault_record.id,
                    vault_config=vault_config,
                )

                assert result == "new"
                assert chunks_created == 0
                assert is_empty is False

    @patch("obsidian_rag.services.ingestion.should_chunk_document")
    def test_embedding_generation_failure(
        self,
        mock_should_chunk: MagicMock,
        ingestion_service: IngestionService,
        tmp_path: Path,
        mock_vault_record: MagicMock,
    ) -> None:
        """Test that embedding failure doesn't block document creation."""
        mock_should_chunk.return_value = False

        mock_session = MagicMock()
        mock_session_context = MagicMock()
        mock_session_context.__enter__ = MagicMock(return_value=mock_session)
        mock_session_context.__exit__ = MagicMock(return_value=None)
        ingestion_service.db_manager.get_session.return_value = mock_session_context  # type: ignore[attr-defined]

        mock_session.query.return_value.filter_by.return_value.first.return_value = None

        ingestion_service.embedding_provider.generate_embedding.side_effect = (  # type: ignore[union-attr]
            EmbeddingError("Embedding failed")
        )

        # Create vault config using tmp_path as container_path
        vault_config = VaultConfig(
            container_path=str(tmp_path),
            host_path=str(tmp_path),
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
            mock_parse_fm.return_value = (None, {}, "Test content")

            with patch(
                "obsidian_rag.services.ingestion.parse_tasks_from_content"
            ) as mock_parse_tasks:
                mock_parse_tasks.return_value = []

                result, chunks_created, is_empty = (
                    ingestion_service._ingest_single_file(
                        mock_file_info,
                        vault_id=mock_vault_record.id,
                        vault_config=vault_config,
                    )
                )

                assert result == "new"
                assert chunks_created == 0  # Small document, no chunking
                assert is_empty is False


class TestIngestSingleFileExceptions:
    """Test _ingest_single_file exception handling (lines 547-548)."""

    def test_ingest_single_file_no_vault_record(
        self,
        ingestion_service: IngestionService,
        mock_vault_config: VaultConfig,
    ) -> None:
        """Test RuntimeError when vault_record is None outside dry_run (lines 547-548)."""
        mock_file_info = MagicMock()
        mock_file_info.path = Path("/test/vault/note.md")
        mock_file_info.checksum_md5 = "abc123"
        mock_file_info.content = "# Test content"

        with patch("obsidian_rag.services.ingestion.parse_frontmatter") as mock_parse:
            mock_parse.return_value = (["tag"], {"kind": "note"}, "content")

            with pytest.raises(RuntimeError) as exc_info:
                ingestion_service._ingest_single_file(
                    mock_file_info,
                    vault_id=None,
                    vault_config=mock_vault_config,
                    dry_run=False,
                )

        assert "No vault ID available" in str(exc_info.value)
