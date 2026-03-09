"""Tests for the IngestionService."""

import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, cast
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

if TYPE_CHECKING:
    from obsidian_rag.config import Settings
    from obsidian_rag.parsing.scanner import FileInfo


@pytest.fixture
def mock_settings() -> "Settings":
    """Create mock settings for testing."""
    settings = MagicMock()
    settings.ingestion.batch_size = 100
    settings.ingestion.progress_interval = 10
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
            processing_time_seconds=5.5,
            message="Test message",
        )

        assert result.total == 10
        assert result.new == 3
        assert result.updated == 2
        assert result.unchanged == 4
        assert result.errors == 1
        assert result.deleted == 2
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
            deleted=2,
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

    def test_ingest_vault_with_new_document(
        self,
        ingestion_service: IngestionService,
        tmp_path: Path,
        mock_vault_record: MagicMock,
    ) -> None:
        """Test ingesting new document."""
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

    def test_ingest_vault_with_updated_document(
        self,
        ingestion_service: IngestionService,
        tmp_path: Path,
        mock_vault_record: MagicMock,
    ) -> None:
        """Test ingesting updated document."""
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

    def test_create_document_with_embedding(
        self,
        ingestion_service: IngestionService,
        tmp_path: Path,
        mock_vault_record: MagicMock,
    ) -> None:
        """Test document creation with embedding generation."""
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

                result = ingestion_service._ingest_single_file(
                    mock_file_info,
                    vault_id=mock_vault_record.id,
                    vault_config=vault_config,
                )

                assert result == "new"
                ingestion_service.embedding_provider.generate_embedding.assert_called_once_with(  # type: ignore[union-attr]
                    "Test content"
                )

    def test_create_document_without_embedding_provider(
        self,
        mock_db_manager: MagicMock,
        mock_settings: "Settings",
        tmp_path: Path,
        mock_vault_record: MagicMock,
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

                result = service._ingest_single_file(
                    mock_file_info,
                    vault_id=mock_vault_record.id,
                    vault_config=vault_config,
                )

                assert result == "new"

    def test_embedding_generation_failure(
        self,
        ingestion_service: IngestionService,
        tmp_path: Path,
        mock_vault_record: MagicMock,
    ) -> None:
        """Test that embedding failure doesn't block document creation."""
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

                result = ingestion_service._ingest_single_file(
                    mock_file_info,
                    vault_id=mock_vault_record.id,
                    vault_config=vault_config,
                )

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
        mock_doc1.file_path = "orphaned.md"
        mock_doc2 = MagicMock()
        mock_doc2.file_path = "existing.md"

        mock_session.query.return_value.filter_by.return_value.all.return_value = [
            mock_doc1,
            mock_doc2,
        ]

        # Filesystem only has existing.md
        filesystem_paths = {"existing.md"}

        deleted_count, error_count = ingestion_service._delete_orphaned_documents(
            filesystem_paths,
            vault_id=mock_vault_record.id,
        )

        assert deleted_count == 1
        assert error_count == 0
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

        # Create mock documents
        mock_doc1 = MagicMock()
        mock_doc1.file_path = "doc1.md"
        mock_doc2 = MagicMock()
        mock_doc2.file_path = "doc2.md"
        batch = [mock_doc1, mock_doc2]

        # Simulate commit failure
        mock_session.commit.side_effect = RuntimeError("Database error")

        deleted_count, error_count = ingestion_service._delete_batch(
            mock_session,
            batch,  # type: ignore[arg-type]
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

        # Create mock documents where one fails to delete
        mock_doc1 = MagicMock()
        mock_doc1.file_path = "doc1.md"
        mock_doc2 = MagicMock()
        mock_doc2.file_path = "doc2.md"

        # First delete succeeds, second fails
        mock_session.delete.side_effect = [None, RuntimeError("Delete failed")]

        batch = [mock_doc1, mock_doc2]

        deleted_count, error_count = ingestion_service._delete_batch(
            mock_session,
            batch,  # type: ignore[arg-type]
        )

        # One deleted successfully, one failed
        assert deleted_count == 1
        assert error_count == 1
        assert mock_session.delete.call_count == 2
