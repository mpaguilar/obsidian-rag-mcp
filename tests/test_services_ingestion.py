"""Tests for the IngestionService."""

import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TYPE_CHECKING, cast
from unittest.mock import MagicMock, patch

import pytest

from obsidian_rag.config import VaultConfig
from obsidian_rag.database.models import Document, Task, Vault
from obsidian_rag.llm.base import EmbeddingError
from obsidian_rag.services.ingestion import (
    IngestionResult,
    IngestionService,
    IngestVaultOptions,
)
from obsidian_rag.services.ingestion_cleanup import _delete_batch

if TYPE_CHECKING:
    from obsidian_rag.config import Settings
    from obsidian_rag.parsing.scanner import FileInfo


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
            deleted=2,
            chunks_created=5,
            empty_documents=1,
            processing_time_seconds=5.5,
            message="Test message",
        )

        assert result.total == 10
        assert result.new == 3
        assert result.updated == 2
        assert result.unchanged == 4
        assert result.errors == 1
        assert result.deleted == 2
        assert result.chunks_created == 5
        assert result.empty_documents == 1
        assert result.processing_time_seconds == 5.5
        assert result.message == "Test message"

    def test_ingestion_result_chunk_statistics_fields(self) -> None:
        """Test IngestionResult with new chunk statistics fields."""
        result = IngestionResult(
            total=10,
            new=3,
            updated=2,
            unchanged=4,
            errors=1,
            deleted=2,
            chunks_created=5,
            empty_documents=1,
            processing_time_seconds=5.5,
            message="Test message",
            total_chunks=15,
            avg_chunk_tokens=250,
            task_chunk_count=5,
            content_chunk_count=10,
        )

        assert result.total_chunks == 15
        assert result.avg_chunk_tokens == 250
        assert result.task_chunk_count == 5
        assert result.content_chunk_count == 10

    def test_ingestion_result_chunk_statistics_defaults(self) -> None:
        """Test IngestionResult chunk statistics have default values."""
        result = IngestionResult(
            total=10,
            new=3,
            updated=2,
            unchanged=4,
            errors=1,
            deleted=2,
            chunks_created=5,
            empty_documents=1,
            processing_time_seconds=5.5,
            message="Test message",
        )

        # Should have default values of 0
        assert result.total_chunks == 0
        assert result.avg_chunk_tokens == 0
        assert result.task_chunk_count == 0
        assert result.content_chunk_count == 0

    def test_to_dict_includes_chunk_statistics(self) -> None:
        """Test to_dict includes chunk statistics fields."""
        result = IngestionResult(
            total=10,
            new=3,
            updated=2,
            unchanged=4,
            errors=1,
            deleted=2,
            chunks_created=5,
            empty_documents=1,
            processing_time_seconds=5.5,
            message="Test message",
            total_chunks=15,
            avg_chunk_tokens=250,
            task_chunk_count=5,
            content_chunk_count=10,
        )

        data = result.to_dict()

        assert data["total_chunks"] == 15
        assert data["avg_chunk_tokens"] == 250
        assert data["task_chunk_count"] == 5
        assert data["content_chunk_count"] == 10

    def test_to_dict_chunk_statistics_defaults(self) -> None:
        """Test to_dict includes default chunk statistics values."""
        result = IngestionResult(
            total=10,
            new=3,
            updated=2,
            unchanged=4,
            errors=1,
            deleted=2,
            chunks_created=5,
            empty_documents=1,
            processing_time_seconds=5.5,
            message="Test message",
        )

        data = result.to_dict()

        # Should have default values in dict
        assert data["total_chunks"] == 0
        assert data["avg_chunk_tokens"] == 0
        assert data["task_chunk_count"] == 0
        assert data["content_chunk_count"] == 0

    def test_to_dict(self) -> None:
        """Test to_dict method converts result to dictionary."""
        result = IngestionResult(
            total=10,
            new=3,
            updated=2,
            unchanged=4,
            errors=1,
            deleted=2,
            chunks_created=5,
            empty_documents=1,
            processing_time_seconds=5.5,
            message="Test message",
        )

        data = result.to_dict()

        assert data["total"] == 10
        assert data["new"] == 3
        assert data["updated"] == 2
        assert data["unchanged"] == 4
        assert data["errors"] == 1
        assert data["deleted"] == 2
        assert data["chunks_created"] == 5
        assert data["empty_documents"] == 1
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


class TestResolveVaultConfig:
    """Test _resolve_vault_config method."""

    def test_resolve_vault_config_with_config_object(
        self,
        ingestion_service: IngestionService,
        mock_vault_config: VaultConfig,
    ) -> None:
        """Test resolving VaultConfig object returns itself."""
        result = ingestion_service._resolve_vault_config(mock_vault_config)
        assert result is mock_vault_config

    def test_resolve_vault_config_with_name(
        self,
        ingestion_service: IngestionService,
        mock_settings: "Settings",
    ) -> None:
        """Test resolving vault by name."""
        result = ingestion_service._resolve_vault_config("test-vault")
        assert result == mock_settings.vaults["test-vault"]

    def test_resolve_vault_config_not_found(
        self,
        ingestion_service: IngestionService,
    ) -> None:
        """Test resolving non-existent vault raises error."""
        # Configure mock to return None for non-existent vault
        mock_settings = cast(MagicMock, ingestion_service.settings)
        mock_settings.get_vault.return_value = None
        mock_settings.get_vault_names.return_value = ["test-vault"]

        with pytest.raises(ValueError, match="Vault 'nonexistent' not found"):
            ingestion_service._resolve_vault_config("nonexistent")


class TestComputeRelativePath:
    """Test _compute_relative_path method."""

    def test_compute_relative_path(
        self,
        ingestion_service: IngestionService,
    ) -> None:
        """Test computing relative path from file to vault root."""
        file_path = Path("/test/vault/folder/note.md")
        container_path = "/test/vault"

        result = ingestion_service._compute_relative_path(file_path, container_path)

        assert result == "folder/note.md"

    def test_compute_relative_path_root_file(
        self,
        ingestion_service: IngestionService,
    ) -> None:
        """Test computing relative path for file in vault root."""
        file_path = Path("/test/vault/note.md")
        container_path = "/test/vault"

        result = ingestion_service._compute_relative_path(file_path, container_path)

        assert result == "note.md"


class TestValidateFilesInVault:
    """Test _validate_files_in_vault method."""

    def test_validate_files_in_vault_success(
        self,
        ingestion_service: IngestionService,
        mock_vault_config: VaultConfig,
    ) -> None:
        """Test validation passes for files within vault."""
        mock_file_info = MagicMock()
        mock_file_info.path = Path("/test/vault/note.md")

        # Should not raise
        ingestion_service._validate_files_in_vault(
            [mock_file_info],
            mock_vault_config,
        )

    def test_validate_files_in_vault_outside_vault(
        self,
        ingestion_service: IngestionService,
        mock_vault_config: VaultConfig,
    ) -> None:
        """Test validation fails for files outside vault."""
        mock_file_info = MagicMock()
        mock_file_info.path = Path("/other/path/note.md")

        with pytest.raises(ValueError, match="outside vault container path"):
            ingestion_service._validate_files_in_vault(
                [mock_file_info],
                mock_vault_config,
            )

    def test_validate_files_in_vault_path_traversal(
        self,
        ingestion_service: IngestionService,
        mock_vault_config: VaultConfig,
    ) -> None:
        """Test validation fails for path traversal attempt."""
        mock_file_info = MagicMock()
        mock_file_info.path = Path("/test/vault/../etc/passwd")

        with pytest.raises(ValueError, match="Path traversal detected"):
            ingestion_service._validate_files_in_vault(
                [mock_file_info],
                mock_vault_config,
            )


class TestIngestVault:
    """Test ingest_vault method."""

    def test_ingest_vault_empty_directory(
        self,
        ingestion_service: IngestionService,
        tmp_path: Path,
        mock_vault_config: VaultConfig,
    ) -> None:
        """Test ingesting empty directory returns zero counts."""
        with patch("obsidian_rag.services.ingestion.scan_markdown_files") as mock_scan:
            mock_scan.return_value = []

            options = IngestVaultOptions(vault=mock_vault_config)
            result = ingestion_service.ingest_vault(tmp_path, options)

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
        mock_vault_record: MagicMock,
    ) -> None:
        """Test ingesting with pre-provided file_infos."""
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

        # Create test file inside tmp_path
        test_file = tmp_path / "test.md"
        test_file.write_text("# Test\n\n- [ ] Task 1")

        file_info = MagicMock()
        file_info.path = test_file
        file_info.name = "test.md"
        file_info.content = "# Test\n\n- [ ] Task 1"
        file_info.checksum = "abc123"
        file_info.created_at = datetime.now(timezone.utc)
        file_info.modified_at = datetime.now(timezone.utc)

        with patch(
            "obsidian_rag.services.ingestion.parse_frontmatter"
        ) as mock_parse_fm:
            mock_parse_fm.return_value = (None, {}, "# Test\n\n- [ ] Task 1")

            with patch(
                "obsidian_rag.services.ingestion.parse_tasks_from_content"
            ) as mock_parse_tasks:
                mock_parse_tasks.return_value = []

                with patch(
                    "obsidian_rag.services.ingestion.should_chunk_document"
                ) as mock_should_chunk:
                    mock_should_chunk.return_value = False

                    with patch.object(
                        ingestion_service,
                        "_get_or_create_vault",
                        return_value=mock_vault_record,
                    ):
                        options = IngestVaultOptions(
                            vault=vault_config,
                            file_infos=[file_info],
                        )
                        result = ingestion_service.ingest_vault(tmp_path, options)

                        assert result.total == 1
                        assert result.new == 1
                        assert result.updated == 0
                        assert result.unchanged == 0
                        assert result.errors == 0

    @patch("obsidian_rag.services.ingestion.should_chunk_document")
    def test_ingest_vault_with_new_document(
        self,
        mock_should_chunk: MagicMock,
        ingestion_service: IngestionService,
        tmp_path: Path,
        mock_vault_record: MagicMock,
    ) -> None:
        """Test ingesting a new document."""
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
                        {},
                        "# Test Document\n\n- [ ] A task",
                    )

                    with patch(
                        "obsidian_rag.services.ingestion.parse_tasks_from_content"
                    ) as mock_parse_tasks:
                        mock_parse_tasks.return_value = []

                        with patch.object(
                            ingestion_service,
                            "_get_or_create_vault",
                            return_value=mock_vault_record,
                        ):
                            options = IngestVaultOptions(vault=vault_config)
                            result = ingestion_service.ingest_vault(tmp_path, options)

                            assert result.total == 1
                            assert result.new == 1
                            assert result.updated == 0
                            assert result.unchanged == 0
                            assert result.errors == 0

    def test_ingest_vault_with_unchanged_document(
        self,
        ingestion_service: IngestionService,
        tmp_path: Path,
        mock_vault_record: MagicMock,
    ) -> None:
        """Test ingesting unchanged document."""
        mock_session = MagicMock()
        mock_session_context = MagicMock()
        mock_session_context.__enter__ = MagicMock(return_value=mock_session)
        mock_session_context.__exit__ = MagicMock(return_value=None)
        ingestion_service.db_manager.get_session.return_value = mock_session_context  # type: ignore[attr-defined]

        existing_doc = MagicMock()
        existing_doc.checksum_md5 = "abc123"

        mock_session.query.return_value.filter_by.return_value.first.return_value = (
            existing_doc
        )

        # Create vault config using tmp_path as container_path
        vault_config = VaultConfig(
            container_path=str(tmp_path),
            host_path=str(tmp_path),
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
                    mock_parse_fm.return_value = (None, {}, "# Test Document")

                    with patch.object(
                        ingestion_service,
                        "_get_or_create_vault",
                        return_value=mock_vault_record,
                    ):
                        options = IngestVaultOptions(vault=vault_config)
                        result = ingestion_service.ingest_vault(tmp_path, options)

                        assert result.total == 1
                        assert result.new == 0
                        assert result.updated == 0
                        assert result.unchanged == 1
                        assert result.errors == 0

    @patch("obsidian_rag.services.ingestion.should_chunk_document")
    def test_ingest_vault_with_updated_document(
        self,
        mock_should_chunk: MagicMock,
        ingestion_service: IngestionService,
        tmp_path: Path,
        mock_vault_record: MagicMock,
    ) -> None:
        """Test ingesting updated document."""
        mock_should_chunk.return_value = False

        mock_session = MagicMock()
        mock_session_context = MagicMock()
        mock_session_context.__enter__ = MagicMock(return_value=mock_session)
        mock_session_context.__exit__ = MagicMock(return_value=None)
        ingestion_service.db_manager.get_session.return_value = mock_session_context  # type: ignore[attr-defined]

        existing_doc = MagicMock()
        existing_doc.checksum_md5 = "old_checksum"
        existing_doc.id = "doc-id"

        mock_session.query.return_value.filter_by.return_value.first.return_value = (
            existing_doc
        )

        # Create vault config using tmp_path as container_path
        vault_config = VaultConfig(
            container_path=str(tmp_path),
            host_path=str(tmp_path),
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
                    mock_parse_fm.return_value = (None, {}, "# Updated Document")

                    with patch(
                        "obsidian_rag.services.ingestion.parse_tasks_from_content"
                    ) as mock_parse_tasks:
                        mock_parse_tasks.return_value = []

                        with patch.object(
                            ingestion_service,
                            "_get_or_create_vault",
                            return_value=mock_vault_record,
                        ):
                            options = IngestVaultOptions(vault=vault_config)
                            result = ingestion_service.ingest_vault(tmp_path, options)

                            assert result.total == 1
                            assert result.new == 0
                            assert result.updated == 1
                            assert result.unchanged == 0
                            assert result.errors == 0

    def test_ingest_vault_with_file_error(
        self,
        ingestion_service: IngestionService,
        tmp_path: Path,
        mock_vault_record: MagicMock,
    ) -> None:
        """Test that file errors are counted but don't stop processing."""
        # Create vault config using tmp_path as container_path
        vault_config = VaultConfig(
            container_path=str(tmp_path),
            host_path=str(tmp_path),
        )

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

                    with patch.object(
                        ingestion_service,
                        "_get_or_create_vault",
                        return_value=mock_vault_record,
                    ):
                        options = IngestVaultOptions(vault=vault_config)
                        result = ingestion_service.ingest_vault(tmp_path, options)

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
        # Create vault config using tmp_path as container_path
        vault_config = VaultConfig(
            container_path=str(tmp_path),
            host_path=str(tmp_path),
        )

        test_file = tmp_path / "test.md"
        test_file.write_text("# Test Document")

        mock_session = MagicMock()
        mock_session_context = MagicMock()
        mock_session_context.__enter__ = MagicMock(return_value=mock_session)
        mock_session_context.__exit__ = MagicMock(return_value=None)
        ingestion_service.db_manager.get_session.return_value = mock_session_context  # type: ignore[attr-defined]

        # Mock empty database (no orphaned documents)
        mock_session.query.return_value.all.return_value = []

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
                        return_value=None,  # dry_run returns None
                    ):
                        options = IngestVaultOptions(
                            vault=vault_config,
                            dry_run=True,
                        )
                        result = ingestion_service.ingest_vault(tmp_path, options)

                        assert result.total == 1
                        assert result.new == 1
                        assert result.updated == 0
                        assert result.unchanged == 0
                        assert result.errors == 0
                        assert result.deleted == 0

                        # Verify session.commit was never called (no writes)
                        mock_session.commit.assert_not_called()

    def test_ingest_vault_with_progress_callback(
        self,
        ingestion_service: IngestionService,
        tmp_path: Path,
        mock_vault_record: MagicMock,
    ) -> None:
        """Test progress callback is called during ingestion."""
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
                    mock_parse_fm.return_value = (None, {}, "# Test")

                    with patch(
                        "obsidian_rag.services.ingestion.parse_tasks_from_content"
                    ) as mock_parse_tasks:
                        mock_parse_tasks.return_value = []

                        with patch.object(
                            ingestion_service,
                            "_get_or_create_vault",
                            return_value=mock_vault_record,
                        ):
                            options = IngestVaultOptions(
                                vault=vault_config,
                                progress_callback=progress_callback,
                            )
                            ingestion_service.ingest_vault(tmp_path, options)

                            assert len(progress_calls) > 0
                            # Last call should have current=1, total=1
                            assert progress_calls[-1][0] == 1
                            assert progress_calls[-1][1] == 1


class TestIngestSingleFile:
    """Test _ingest_single_file method."""

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

        ingestion_service._create_tasks(mock_session, mock_document, parsed_tasks)  # type: ignore[arg-type]

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

        ingestion_service._update_tasks(mock_session, mock_document, parsed_tasks)  # type: ignore[arg-type]

        # Verify old tasks are deleted
        mock_session.query.return_value.filter_by.return_value.delete.assert_called_once()

        # Verify new task is created
        mock_session.add.assert_called_once()


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


class TestGetOrCreateVault:
    """Test _get_or_create_vault method."""

    def test_get_or_create_vault_dry_run(
        self,
        ingestion_service: IngestionService,
        mock_vault_config: VaultConfig,
    ) -> None:
        """Test dry run returns None."""
        result = ingestion_service._get_or_create_vault(
            mock_vault_config,
            dry_run=True,
        )
        assert result is None

    def test_get_or_create_vault_existing(
        self,
        ingestion_service: IngestionService,
        mock_vault_config: VaultConfig,
        mock_vault_record: MagicMock,
    ) -> None:
        """Test getting existing vault record."""
        mock_session = MagicMock()
        mock_session_context = MagicMock()
        mock_session_context.__enter__ = MagicMock(return_value=mock_session)
        mock_session_context.__exit__ = MagicMock(return_value=None)
        ingestion_service.db_manager.get_session.return_value = mock_session_context  # type: ignore[attr-defined]

        mock_session.query.return_value.filter_by.return_value.first.return_value = (
            mock_vault_record
        )

        result = ingestion_service._get_or_create_vault(mock_vault_config)

        assert result == mock_vault_record.id
        mock_session.add.assert_not_called()

    def test_get_or_create_vault_new(
        self,
        ingestion_service: IngestionService,
        mock_vault_config: VaultConfig,
    ) -> None:
        """Test creating new vault record."""
        mock_session = MagicMock()
        mock_session_context = MagicMock()
        mock_session_context.__enter__ = MagicMock(return_value=mock_session)
        mock_session_context.__exit__ = MagicMock(return_value=None)
        ingestion_service.db_manager.get_session.return_value = mock_session_context  # type: ignore[attr-defined]

        mock_session.query.return_value.filter_by.return_value.first.return_value = None

        # Mock the Vault object that gets added to session
        mock_vault = MagicMock()
        test_uuid = uuid.uuid4()
        mock_vault.id = test_uuid
        mock_session.add.side_effect = lambda x: setattr(x, "id", test_uuid)

        result = ingestion_service._get_or_create_vault(mock_vault_config)

        assert result is not None
        assert isinstance(result, uuid.UUID)
        assert result == test_uuid
        mock_session.add.assert_called_once()
        mock_session.commit.assert_called_once()

    def test_get_or_create_vault_name_not_in_settings(
        self,
        ingestion_service: IngestionService,
    ) -> None:
        """Test creating vault when name not found in settings (uses 'Unknown')."""
        from obsidian_rag.config import VaultConfig

        # Create a vault config with a path that won't match any settings
        unmatched_config = VaultConfig(
            container_path="/unmatched/path",
            host_path="/unmatched/path",
        )

        mock_session = MagicMock()
        mock_session_context = MagicMock()
        mock_session_context.__enter__ = MagicMock(return_value=mock_session)
        mock_session_context.__exit__ = MagicMock(return_value=None)
        ingestion_service.db_manager.get_session.return_value = mock_session_context  # type: ignore[attr-defined]

        mock_session.query.return_value.filter_by.return_value.first.return_value = None

        # Mock empty settings.vaults
        mock_settings = cast(MagicMock, ingestion_service.settings)
        mock_settings.vaults = {}

        # Mock the Vault object that gets added to session
        mock_vault = MagicMock()
        test_uuid = uuid.uuid4()
        mock_vault.id = test_uuid
        mock_session.add.side_effect = lambda x: setattr(x, "id", test_uuid)

        result = ingestion_service._get_or_create_vault(unmatched_config)

        assert result is not None
        assert isinstance(result, uuid.UUID)
        assert result == test_uuid
        mock_session.add.assert_called_once()
        # Verify the vault was created with name "Unknown"
        added_vault = mock_session.add.call_args[0][0]
        assert added_vault.name == "Unknown"

    def test_get_or_create_vault_multiple_settings_loop(
        self,
        ingestion_service: IngestionService,
    ) -> None:
        """Test vault lookup loops through multiple settings entries."""
        from obsidian_rag.config import VaultConfig

        # Create a vault config that matches the second entry
        target_config = VaultConfig(
            container_path="/target/path",
            host_path="/target/path",
        )

        mock_session = MagicMock()
        mock_session_context = MagicMock()
        mock_session_context.__enter__ = MagicMock(return_value=mock_session)
        mock_session_context.__exit__ = MagicMock(return_value=None)
        ingestion_service.db_manager.get_session.return_value = mock_session_context  # type: ignore[attr-defined]

        mock_session.query.return_value.filter_by.return_value.first.return_value = None

        # Mock settings with multiple vaults - target is second
        mock_settings = cast(MagicMock, ingestion_service.settings)
        mock_settings.vaults = {
            "First Vault": VaultConfig(
                container_path="/first/path",
                host_path="/first/path",
            ),
            "Target Vault": target_config,
        }

        # Mock the Vault object that gets added to session
        mock_vault = MagicMock()
        test_uuid = uuid.uuid4()
        mock_vault.id = test_uuid
        mock_session.add.side_effect = lambda x: setattr(x, "id", test_uuid)

        result = ingestion_service._get_or_create_vault(target_config)

        assert result is not None
        assert isinstance(result, uuid.UUID)
        # Verify the vault was created with the correct name from settings
        added_vault = mock_session.add.call_args[0][0]
        assert added_vault.name == "Target Vault"


class TestComputeRelativePathExceptions:
    """Test _compute_relative_path exception handling (lines 244-246)."""

    def test_compute_relative_path_outside_vault(
        self,
        ingestion_service: IngestionService,
    ) -> None:
        """Test relative path when file is outside vault container."""
        # File path that is not under the container path
        file_path = Path("/outside/vault/note.md")
        container_path = "/test/vault"

        result = ingestion_service._compute_relative_path(file_path, container_path)

        # Should return absolute path with forward slashes
        assert result == "/outside/vault/note.md"


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


class TestDeleteBatchExceptions:
    """Test _delete_batch exception handling (lines 857-859)."""

    def test_delete_batch_document_delete_failure(
        self,
        ingestion_service: IngestionService,
    ) -> None:
        """Test handling of individual document deletion failure (lines 857-859)."""
        mock_session = MagicMock()

        # Create mock document info tuples (id, file_path)
        doc1_id = uuid.uuid4()
        doc2_id = uuid.uuid4()
        batch = [(doc1_id, "doc1.md"), (doc2_id, "doc2.md")]

        # Create mock documents where one fails to delete
        mock_doc1 = MagicMock()
        mock_doc2 = MagicMock()

        # Mock session.get() to return documents
        mock_session.get.side_effect = [mock_doc1, mock_doc2]

        # First delete succeeds, second fails
        mock_session.delete.side_effect = [None, RuntimeError("Delete failed")]

        deleted_count, error_count = _delete_batch(
            mock_session,
            batch,
        )

        # One deleted successfully, one failed
        assert deleted_count == 1
        assert error_count == 1
        assert mock_session.delete.call_count == 2

    def test_delete_batch_document_not_found(
        self,
        ingestion_service: IngestionService,
    ) -> None:
        """Test handling when document is not found in session during deletion."""
        mock_session = MagicMock()

        # Create mock document info tuple (id, file_path)
        doc_id = uuid.uuid4()
        batch = [(doc_id, "missing.md")]

        # Mock session.get() to return None (document not found)
        mock_session.get.return_value = None

        deleted_count, error_count = _delete_batch(
            mock_session,
            batch,
        )

        # Document not found, counted as error
        assert deleted_count == 0
        assert error_count == 1
        mock_session.get.assert_called_once_with(Document, doc_id)
        mock_session.delete.assert_not_called()


class TestChunkingOperations:
    """Test document chunking operations (lines 687-706, 720, 770-779, 802, 857-867)."""

    @patch("obsidian_rag.services.ingestion_chunks.create_chunks_with_embeddings")
    def test_create_chunks_with_embeddings(
        self,
        mock_create_chunks: MagicMock,
        ingestion_service: IngestionService,
    ) -> None:
        """Test create_chunks_with_embeddings function is called correctly (lines 836-845)."""
        # Create content large enough to trigger chunking
        content = "Word " * 5000  # Large content that will be chunked
        document_id = uuid.uuid4()

        # Mock the function to return 5 chunks created
        mock_create_chunks.return_value = 5

        # Create a mock session
        mock_session = MagicMock()

        # Call the standalone function directly
        from obsidian_rag.services.ingestion_chunks import create_chunks_with_embeddings

        chunks_created = create_chunks_with_embeddings(
            db_session=mock_session,
            document_id=document_id,
            content=content,
            embedding_provider=ingestion_service.embedding_provider,
            chunk_size=512,
            chunk_overlap=50,
            model_name="gpt2",
        )

        # Should create multiple chunks
        assert chunks_created == 5
        mock_create_chunks.assert_called_once()

    def test_delete_existing_chunks_with_chunks(
        self,
        ingestion_service: IngestionService,
    ) -> None:
        """Test _delete_existing_chunks when document has chunks (line 720)."""
        mock_session = MagicMock()
        mock_document = MagicMock()
        mock_document.id = uuid.uuid4()
        # Document has existing chunks
        mock_document.chunks = [MagicMock(), MagicMock()]

        ingestion_service._delete_existing_chunks(mock_session, mock_document)

        # Should query and delete chunks
        mock_session.query.return_value.filter_by.return_value.delete.assert_called_once()
        # Should clear document.chunks
        assert mock_document.chunks == []

    def test_delete_existing_chunks_no_chunks(
        self,
        ingestion_service: IngestionService,
    ) -> None:
        """Test _delete_existing_chunks when document has no chunks (line 720->exit)."""
        mock_session = MagicMock()
        mock_document = MagicMock()
        mock_document.id = uuid.uuid4()
        # Document has no chunks
        mock_document.chunks = []

        ingestion_service._delete_existing_chunks(mock_session, mock_document)

        # Should not attempt to delete anything
        mock_session.query.assert_not_called()

    @patch("obsidian_rag.services.ingestion.should_chunk_document")
    def test_create_document_empty_content(
        self,
        mock_should_chunk: MagicMock,
        mock_db_manager: MagicMock,
        mock_embedding_provider: MagicMock,
        mock_settings: "Settings",
        tmp_path: Path,
    ) -> None:
        """Test _create_document with empty content (lines 770-772)."""
        mock_should_chunk.return_value = False

        service = IngestionService(
            db_manager=mock_db_manager,
            embedding_provider=mock_embedding_provider,
            settings=mock_settings,
        )

        file_info = MagicMock()
        file_info.path = tmp_path / "empty.md"
        file_info.checksum = "abc123"
        file_info.created_at = datetime.now(timezone.utc)
        file_info.modified_at = datetime.now(timezone.utc)

        # Empty content (only whitespace)
        parsed_data: tuple[list[str] | None, dict[str, Any], str] = (
            None,
            {},
            "   \n\n   ",
        )
        vault_id = uuid.uuid4()
        relative_path = "empty.md"

        document, chunks_created = service._create_document(
            MagicMock(),  # _session unused
            file_info,
            parsed_data,
            vault_id=vault_id,
            relative_path=relative_path,
        )

        # Should have no embedding for empty document
        assert document.content_vector is None
        assert chunks_created == 0
        assert document.chunks == []

    @patch("obsidian_rag.services.ingestion.create_chunks_with_embeddings")
    @patch("obsidian_rag.services.ingestion.should_chunk_document")
    def test_create_document_large_content_chunking(
        self,
        mock_should_chunk: MagicMock,
        mock_create_chunks: MagicMock,
        mock_db_manager: MagicMock,
        mock_embedding_provider: MagicMock,
        mock_settings: "Settings",
        tmp_path: Path,
    ) -> None:
        """Test _create_document with large content triggers chunking (lines 774-779, 802)."""
        mock_should_chunk.return_value = True
        mock_create_chunks.return_value = 5

        service = IngestionService(
            db_manager=mock_db_manager,
            embedding_provider=mock_embedding_provider,
            settings=mock_settings,
        )

        file_info = MagicMock()
        file_info.path = tmp_path / "large.md"
        file_info.checksum = "abc123"
        file_info.created_at = datetime.now(timezone.utc)
        file_info.modified_at = datetime.now(timezone.utc)

        # Large content that triggers chunking
        content = "Word " * 5000
        parsed_data: tuple[list[str] | None, dict[str, Any], str] = (None, {}, content)
        vault_id = uuid.uuid4()
        relative_path = "large.md"

        document, chunks_created = service._create_document(
            MagicMock(),  # _session unused
            file_info,
            parsed_data,
            vault_id=vault_id,
            relative_path=relative_path,
        )

        # Should mark for chunking (chunks created later), no document-level embedding
        assert (
            chunks_created == 0
        )  # Chunks created after flush, not in _create_document
        assert document.content_vector is None

    @patch("obsidian_rag.services.ingestion.should_chunk_document")
    def test_update_document_empty_content(
        self,
        mock_should_chunk: MagicMock,
        mock_db_manager: MagicMock,
        mock_embedding_provider: MagicMock,
        mock_settings: "Settings",
    ) -> None:
        """Test _update_document with empty content (lines 857-859)."""
        mock_should_chunk.return_value = False

        service = IngestionService(
            db_manager=mock_db_manager,
            embedding_provider=mock_embedding_provider,
            settings=mock_settings,
        )

        mock_session = MagicMock()
        mock_document = MagicMock()
        mock_document.chunks = []  # No existing chunks

        file_info = MagicMock()
        file_info.checksum = "new_checksum"
        file_info.modified_at = datetime.now(timezone.utc)

        # Empty content
        parsed_data: tuple[list[str] | None, dict[str, Any], str] = (
            None,
            {},
            "   \n\n   ",
        )

        chunks_created = service._update_document(
            mock_session,
            mock_document,
            file_info,
            parsed_data,
        )

        # Should clear embedding for empty document
        assert mock_document.content_vector is None
        assert chunks_created == 0

    @patch("obsidian_rag.services.ingestion.create_chunks_with_embeddings")
    @patch("obsidian_rag.services.ingestion.should_chunk_document")
    def test_update_document_large_content_chunking(
        self,
        mock_should_chunk: MagicMock,
        mock_create_chunks: MagicMock,
        mock_db_manager: MagicMock,
        mock_embedding_provider: MagicMock,
        mock_settings: "Settings",
    ) -> None:
        """Test _update_document with large content triggers chunking (lines 861-867)."""
        mock_should_chunk.return_value = True
        mock_create_chunks.return_value = 5

        service = IngestionService(
            db_manager=mock_db_manager,
            embedding_provider=mock_embedding_provider,
            settings=mock_settings,
        )

        mock_session = MagicMock()
        mock_document = MagicMock()
        mock_document.chunks = []  # No existing chunks initially

        file_info = MagicMock()
        file_info.checksum = "new_checksum"
        file_info.modified_at = datetime.now(timezone.utc)

        # Large content that triggers chunking
        content = "Word " * 5000
        parsed_data: tuple[list[str] | None, dict[str, Any], str] = (None, {}, content)

        chunks_created = service._update_document(
            mock_session,
            mock_document,
            file_info,
            parsed_data,
        )

        # Should create chunks, no document-level embedding
        assert chunks_created == 5
        assert mock_document.content_vector is None


class TestEmptyDocumentTracking:
    """Test empty document tracking in _process_files_with_stats (line 391)."""

    def test_process_files_with_stats_tracks_empty_documents(
        self,
        ingestion_service: IngestionService,
        tmp_path: Path,
        mock_vault_record: MagicMock,
    ) -> None:
        """Test that empty documents are tracked in stats (line 391)."""
        # Create vault config
        vault_config = VaultConfig(
            container_path=str(tmp_path),
            host_path=str(tmp_path),
        )

        # Create file info for empty document
        mock_file_info = MagicMock()
        mock_file_info.path = tmp_path / "empty.md"
        mock_file_info.name = "empty.md"

        # Mock _ingest_single_file to return is_empty=True
        with patch.object(
            ingestion_service,
            "_ingest_single_file",
            return_value=("new", 0, True),  # result, chunks_created, is_empty=True
        ):
            file_info_list = [mock_file_info]
            stats = ingestion_service._process_files_with_stats(
                file_info_list,  # type: ignore[arg-type]
                vault_id=mock_vault_record.id,
                vault_config=vault_config,
                dry_run=False,
                progress_callback=None,
            )

            # Should track empty document
            assert stats["empty_documents"] == 1
            assert stats["new"] == 1
