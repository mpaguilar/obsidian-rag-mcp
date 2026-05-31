"""Integration tests for force re-ingestion in IngestionService."""

import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

from obsidian_rag.config import VaultConfig
from obsidian_rag.parsing.scanner import FileInfo
from obsidian_rag.services.ingestion import IngestionService, IngestVaultOptions

if TYPE_CHECKING:
    from obsidian_rag.config import Settings


@pytest.fixture
def mock_settings() -> "Settings":
    """Create mock settings for testing."""
    settings = MagicMock()
    settings.ingestion.batch_size = 100
    settings.ingestion.progress_interval = 10
    settings.chunking.chunk_size = 512
    settings.chunking.chunk_overlap = 50
    settings.chunking.tokenizer_model = "gpt2"
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


def test_force_reingest_unchanged_document_returns_updated(
    ingestion_service: IngestionService,
) -> None:
    """Test force=True re-ingests unchanged documents as updated."""
    file_info = FileInfo(
        path=Path("/test/vault/note.md"),
        name="note.md",
        content="content",
        checksum="abc123",
        created_at=datetime.now(timezone.utc),
        modified_at=datetime.now(timezone.utc),
    )

    existing = MagicMock()
    existing.checksum_md5 = "abc123"  # Same checksum
    existing.id = uuid.uuid4()

    mock_session = MagicMock()
    mock_session.query.return_value.filter_by.return_value.first.return_value = existing
    mock_session_context = MagicMock()
    mock_session_context.__enter__ = MagicMock(return_value=mock_session)
    mock_session_context.__exit__ = MagicMock(return_value=None)
    ingestion_service.db_manager.get_session.return_value = mock_session_context  # type: ignore[attr-defined]

    with patch.object(ingestion_service, "_update_document", return_value=0) as mock_update:
        with patch.object(ingestion_service, "_update_tasks"):
            result = ingestion_service._ingest_single_file(
                file_info,
                vault_id=uuid.uuid4(),
                vault_config=VaultConfig(
                    container_path="/test/vault", host_path="/test/vault"
                ),
                force=True,
            )

    assert result[0] == "updated"
    mock_update.assert_called_once()


def test_force_reingest_updates_ingested_at_timestamp(
    ingestion_service: IngestionService,
) -> None:
    """Test force=True triggers document update which updates ingested_at."""
    file_info = FileInfo(
        path=Path("/test/vault/note.md"),
        name="note.md",
        content="content",
        checksum="abc123",
        created_at=datetime.now(timezone.utc),
        modified_at=datetime.now(timezone.utc),
    )

    existing = MagicMock()
    existing.checksum_md5 = "abc123"
    existing.id = uuid.uuid4()

    mock_session = MagicMock()
    mock_session.query.return_value.filter_by.return_value.first.return_value = existing
    mock_session_context = MagicMock()
    mock_session_context.__enter__ = MagicMock(return_value=mock_session)
    mock_session_context.__exit__ = MagicMock(return_value=None)
    ingestion_service.db_manager.get_session.return_value = mock_session_context  # type: ignore[attr-defined]

    with patch.object(ingestion_service, "_update_document", return_value=0) as mock_update:
        with patch.object(ingestion_service, "_update_tasks"):
            ingestion_service._ingest_single_file(
                file_info,
                vault_id=uuid.uuid4(),
                vault_config=VaultConfig(
                    container_path="/test/vault", host_path="/test/vault"
                ),
                force=True,
            )

    mock_update.assert_called_once()


def test_force_reingest_regenerates_embedding(
    ingestion_service: IngestionService,
) -> None:
    """Test force=True causes _update_document to regenerate embedding."""
    file_info = FileInfo(
        path=Path("/test/vault/note.md"),
        name="note.md",
        content="content",
        checksum="abc123",
        created_at=datetime.now(timezone.utc),
        modified_at=datetime.now(timezone.utc),
    )

    existing = MagicMock()
    existing.checksum_md5 = "abc123"
    existing.id = uuid.uuid4()

    mock_session = MagicMock()
    mock_session.query.return_value.filter_by.return_value.first.return_value = existing
    mock_session_context = MagicMock()
    mock_session_context.__enter__ = MagicMock(return_value=mock_session)
    mock_session_context.__exit__ = MagicMock(return_value=None)
    ingestion_service.db_manager.get_session.return_value = mock_session_context  # type: ignore[attr-defined]

    with patch.object(ingestion_service, "_update_document", return_value=0) as mock_update:
        with patch.object(ingestion_service, "_update_tasks"):
            ingestion_service._ingest_single_file(
                file_info,
                vault_id=uuid.uuid4(),
                vault_config=VaultConfig(
                    container_path="/test/vault", host_path="/test/vault"
                ),
                force=True,
            )

    mock_update.assert_called_once()


def test_force_false_preserves_unchanged_behavior(
    ingestion_service: IngestionService,
) -> None:
    """Test force=False still returns unchanged for matching checksums."""
    file_info = FileInfo(
        path=Path("/test/vault/note.md"),
        name="note.md",
        content="content",
        checksum="abc123",
        created_at=datetime.now(timezone.utc),
        modified_at=datetime.now(timezone.utc),
    )

    existing = MagicMock()
    existing.checksum_md5 = "abc123"

    mock_session = MagicMock()
    mock_session.query.return_value.filter_by.return_value.first.return_value = existing
    mock_session_context = MagicMock()
    mock_session_context.__enter__ = MagicMock(return_value=mock_session)
    mock_session_context.__exit__ = MagicMock(return_value=None)
    ingestion_service.db_manager.get_session.return_value = mock_session_context  # type: ignore[attr-defined]

    result = ingestion_service._ingest_single_file(
        file_info,
        vault_id=uuid.uuid4(),
        vault_config=VaultConfig(
            container_path="/test/vault", host_path="/test/vault"
        ),
        force=False,
    )

    assert result[0] == "unchanged"


def test_force_does_not_affect_new_documents(
    ingestion_service: IngestionService,
) -> None:
    """Test force has no effect on new documents."""
    file_info = FileInfo(
        path=Path("/test/vault/note.md"),
        name="note.md",
        content="content",
        checksum="abc123",
        created_at=datetime.now(timezone.utc),
        modified_at=datetime.now(timezone.utc),
    )

    mock_session = MagicMock()
    mock_session.query.return_value.filter_by.return_value.first.return_value = None
    mock_session_context = MagicMock()
    mock_session_context.__enter__ = MagicMock(return_value=mock_session)
    mock_session_context.__exit__ = MagicMock(return_value=None)
    ingestion_service.db_manager.get_session.return_value = mock_session_context  # type: ignore[attr-defined]

    with patch.object(ingestion_service, "_create_document") as mock_create:
        mock_create.return_value = (MagicMock(), 0)
        with patch.object(ingestion_service, "_create_tasks"):
            with patch.object(
                ingestion_service, "_create_chunks_for_new_document", return_value=0
            ):
                with patch(
                    "obsidian_rag.services.ingestion.should_chunk_document"
                ) as mock_should_chunk:
                    mock_should_chunk.return_value = False
                    result = ingestion_service._ingest_single_file(
                        file_info,
                        vault_id=uuid.uuid4(),
                        vault_config=VaultConfig(
                            container_path="/test/vault", host_path="/test/vault"
                        ),
                        force=True,
                    )

    assert result[0] == "new"


def test_force_does_not_affect_dry_run(
    ingestion_service: IngestionService,
) -> None:
    """Test dry-run returns new regardless of force."""
    file_info = FileInfo(
        path=Path("/test/vault/note.md"),
        name="note.md",
        content="content",
        checksum="abc123",
        created_at=datetime.now(timezone.utc),
        modified_at=datetime.now(timezone.utc),
    )

    result = ingestion_service._ingest_single_file(
        file_info,
        vault_id=uuid.uuid4(),
        vault_config=VaultConfig(
            container_path="/test/vault", host_path="/test/vault"
        ),
        dry_run=True,
        force=True,
    )

    assert result[0] == "new"


def test_ingest_vault_with_force_processes_all_files(
    ingestion_service: IngestionService,
) -> None:
    """Test ingest_vault with force=True processes all files as updated."""
    file_infos = [
        FileInfo(
            path=Path("/test/vault/note.md"),
            name="note.md",
            content="content",
            checksum="abc123",
            created_at=datetime.now(timezone.utc),
            modified_at=datetime.now(timezone.utc),
        ),
    ]

    vault_config = VaultConfig(
        container_path="/test/vault", host_path="/test/vault"
    )
    vault_id = uuid.uuid4()

    existing = MagicMock()
    existing.checksum_md5 = "abc123"
    existing.id = uuid.uuid4()

    mock_session = MagicMock()
    mock_session.query.return_value.filter_by.return_value.first.return_value = existing
    mock_session_context = MagicMock()
    mock_session_context.__enter__ = MagicMock(return_value=mock_session)
    mock_session_context.__exit__ = MagicMock(return_value=None)
    ingestion_service.db_manager.get_session.return_value = mock_session_context  # type: ignore[attr-defined]

    options = IngestVaultOptions(
        vault=vault_config,
        file_infos=file_infos,
        force=True,
    )

    with patch.object(ingestion_service, "_get_or_create_vault", return_value=vault_id):
        with patch.object(ingestion_service, "_validate_files_in_vault"):
            with patch.object(
                ingestion_service, "_delete_orphaned_documents", return_value=(0, 0)
            ):
                with patch.object(
                    ingestion_service, "_update_document", return_value=0
                ) as mock_update:
                    with patch.object(ingestion_service, "_update_tasks"):
                        result = ingestion_service.ingest_vault(
                            Path("/test/vault"), options
                        )

    assert result.updated == 1
    assert result.unchanged == 0
    mock_update.assert_called_once()


def test_ingest_vault_without_force_skips_unchanged(
    ingestion_service: IngestionService,
) -> None:
    """Test ingest_vault with force=False skips unchanged files."""
    file_infos = [
        FileInfo(
            path=Path("/test/vault/note.md"),
            name="note.md",
            content="content",
            checksum="abc123",
            created_at=datetime.now(timezone.utc),
            modified_at=datetime.now(timezone.utc),
        ),
    ]

    vault_config = VaultConfig(
        container_path="/test/vault", host_path="/test/vault"
    )
    vault_id = uuid.uuid4()

    existing = MagicMock()
    existing.checksum_md5 = "abc123"

    mock_session = MagicMock()
    mock_session.query.return_value.filter_by.return_value.first.return_value = existing
    mock_session_context = MagicMock()
    mock_session_context.__enter__ = MagicMock(return_value=mock_session)
    mock_session_context.__exit__ = MagicMock(return_value=None)
    ingestion_service.db_manager.get_session.return_value = mock_session_context  # type: ignore[attr-defined]

    options = IngestVaultOptions(
        vault=vault_config,
        file_infos=file_infos,
        force=False,
    )

    with patch.object(ingestion_service, "_get_or_create_vault", return_value=vault_id):
        with patch.object(ingestion_service, "_validate_files_in_vault"):
            with patch.object(
                ingestion_service, "_delete_orphaned_documents", return_value=(0, 0)
            ):
                result = ingestion_service.ingest_vault(Path("/test/vault"), options)

    assert result.unchanged == 1
    assert result.updated == 0


def test_force_with_different_checksum_still_updates(
    ingestion_service: IngestionService,
) -> None:
    """Test force=True with different checksum still updates document."""
    file_info = FileInfo(
        path=Path("/test/vault/note.md"),
        name="note.md",
        content="content",
        checksum="new_checksum",
        created_at=datetime.now(timezone.utc),
        modified_at=datetime.now(timezone.utc),
    )

    existing = MagicMock()
    existing.checksum_md5 = "old_checksum"
    existing.id = uuid.uuid4()

    mock_session = MagicMock()
    mock_session.query.return_value.filter_by.return_value.first.return_value = existing
    mock_session_context = MagicMock()
    mock_session_context.__enter__ = MagicMock(return_value=mock_session)
    mock_session_context.__exit__ = MagicMock(return_value=None)
    ingestion_service.db_manager.get_session.return_value = mock_session_context  # type: ignore[attr-defined]

    with patch.object(ingestion_service, "_update_document", return_value=0) as mock_update:
        with patch.object(ingestion_service, "_update_tasks"):
            result = ingestion_service._ingest_single_file(
                file_info,
                vault_id=uuid.uuid4(),
                vault_config=VaultConfig(
                    container_path="/test/vault", host_path="/test/vault"
                ),
                force=True,
            )

    assert result[0] == "updated"
    mock_update.assert_called_once()


def test_force_with_no_delete(
    ingestion_service: IngestionService,
) -> None:
    """Test force=True and no_delete=True work together."""
    file_infos = [
        FileInfo(
            path=Path("/test/vault/note.md"),
            name="note.md",
            content="content",
            checksum="abc123",
            created_at=datetime.now(timezone.utc),
            modified_at=datetime.now(timezone.utc),
        ),
    ]

    vault_config = VaultConfig(
        container_path="/test/vault", host_path="/test/vault"
    )
    vault_id = uuid.uuid4()

    existing = MagicMock()
    existing.checksum_md5 = "abc123"
    existing.id = uuid.uuid4()

    mock_session = MagicMock()
    mock_session.query.return_value.filter_by.return_value.first.return_value = existing
    mock_session_context = MagicMock()
    mock_session_context.__enter__ = MagicMock(return_value=mock_session)
    mock_session_context.__exit__ = MagicMock(return_value=None)
    ingestion_service.db_manager.get_session.return_value = mock_session_context  # type: ignore[attr-defined]

    options = IngestVaultOptions(
        vault=vault_config,
        file_infos=file_infos,
        force=True,
        no_delete=True,
    )

    with patch.object(ingestion_service, "_get_or_create_vault", return_value=vault_id):
        with patch.object(ingestion_service, "_validate_files_in_vault"):
            with patch.object(
                ingestion_service, "_delete_orphaned_documents", return_value=(0, 0)
            ):
                with patch.object(
                    ingestion_service, "_update_document", return_value=0
                ) as mock_update:
                    with patch.object(ingestion_service, "_update_tasks"):
                        result = ingestion_service.ingest_vault(
                            Path("/test/vault"), options
                        )

    assert result.updated == 1
    assert result.unchanged == 0
    mock_update.assert_called_once()
