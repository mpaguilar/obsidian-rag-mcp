"""Tests for core IngestionService functionality."""

import uuid
from pathlib import Path
from typing import TYPE_CHECKING, cast
from unittest.mock import MagicMock

import pytest

from obsidian_rag.config import VaultConfig
from obsidian_rag.database.models import Document
from obsidian_rag.services.ingestion import (
    IngestionResult,
    IngestionService,
    IngestVaultOptions,
)
from obsidian_rag.services.ingestion_cleanup import _delete_batch
from obsidian_rag.services.tag_merging import _merge_tags

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


class TestIngestVaultOptions:
    """Test IngestVaultOptions dataclass."""

    def test_ingest_vault_options_has_force_field(self) -> None:
        """Test IngestVaultOptions accepts force=True."""
        options = IngestVaultOptions(
            vault=VaultConfig(container_path="/test", host_path="/test"),
            force=True,
        )
        assert options.force is True

    def test_ingest_vault_options_force_defaults_to_false(self) -> None:
        """Test IngestVaultOptions force defaults to False."""
        options = IngestVaultOptions(
            vault=VaultConfig(container_path="/test", host_path="/test"),
        )
        assert options.force is False


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


def test_merge_tags_both_none():
    """both None -> returns None"""
    assert _merge_tags(None, None) is None


def test_merge_tags_both_empty():
    """both [] -> returns None"""
    assert _merge_tags([], []) is None


def test_merge_tags_no_overlap_both_lowercase():
    """["a"], ["b"] -> ["a", "b"]"""
    assert _merge_tags(["a"], ["b"]) == ["a", "b"]


def test_merge_tags_full_overlap():
    """["work"], ["work"] -> ["work"]"""
    assert _merge_tags(["work"], ["work"]) == ["work"]


def test_merge_tags_partial_overlap():
    """["work", "urgent"], ["work", "personal"] -> ["work", "urgent", "personal"]"""
    assert _merge_tags(["work", "urgent"], ["work", "personal"]) == [
        "work",
        "urgent",
        "personal",
    ]


def test_merge_tags_doc_only_task_none():
    """["Work"], None -> ["work"]"""
    assert _merge_tags(["Work"], None) == ["work"]


def test_merge_tags_task_only_doc_none():
    """None, ["personal"] -> ["personal"]"""
    assert _merge_tags(None, ["personal"]) == ["personal"]


def test_merge_tags_doc_only_task_empty():
    """["Work"], [] -> ["work"]"""
    assert _merge_tags(["Work"], []) == ["work"]


def test_merge_tags_case_insensitive_dedup():
    """["Work"], ["work"] -> ["work"]"""
    assert _merge_tags(["Work"], ["work"]) == ["work"]


def test_merge_tags_mixed_case_multiple():
    """["Work", "Urgent"], ["work", "urgent"] -> ["work", "urgent"]"""
    assert _merge_tags(["Work", "Urgent"], ["work", "urgent"]) == [
        "work",
        "urgent",
    ]


def test_merge_tags_doc_uppercase_result_lowercase():
    """["IMPORTANT"] -> ["important"]"""
    assert _merge_tags(["IMPORTANT"], None) == ["important"]


def test_merge_tags_empty_string_filtered():
    """["", "work"], [""] -> ["work"]"""
    assert _merge_tags(["", "work"], [""]) == ["work"]


def test_merge_tags_strips_hash_prefix():
    """None, ["#personal"] -> ["personal"]"""
    assert _merge_tags(None, ["#personal"]) == ["personal"]


def test_merge_tags_task_hash_only_returns_none():
    """None, ["#"] -> None (empty after lstrip)."""
    assert _merge_tags(None, ["#"]) is None


def test_merge_tags_doc_preserved_task_hash_only():
    """["work"], ["#"] -> ["work"] (task tag filtered out)."""
    assert _merge_tags(["work"], ["#"]) == ["work"]


def test_merge_tags_frontmatter_with_body_tags_no_hash() -> None:
    """frontmatter tags + body inline tags (already # stripped) merge lowercased."""
    assert _merge_tags(["Work", "Personal"], ["meeting", "project"]) == [
        "work",
        "personal",
        "meeting",
        "project",
    ]


def test_merge_tags_body_tags_case_insensitive_dedup_with_frontmatter() -> None:
    """body tag already lowercased dedups with matching frontmatter tag."""
    assert _merge_tags(["work"], ["work"]) == ["work"]


def test_merge_tags_body_tags_none_frontmatter_only() -> None:
    """frontmatter-only when body tags arg is None."""
    assert _merge_tags(["frontmatter"], None) == ["frontmatter"]


def test_merge_tags_none_body_only() -> None:
    """body-only when frontmatter arg is None."""
    assert _merge_tags(None, ["bodyside"]) == ["bodyside"]
