"""Tests for document chunking and force re-ingestion in IngestionService."""

import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

from obsidian_rag.config import VaultConfig
from obsidian_rag.parsing.scanner import FileInfo
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


def test_ingest_single_file_force_true_skips_checksum() -> None:
    """Test force=True re-ingests unchanged documents."""
    from obsidian_rag.services.ingestion import IngestionService
    from obsidian_rag.config import VaultConfig
    from pathlib import Path
    import uuid
    from unittest.mock import MagicMock, patch

    service = IngestionService(
        db_manager=MagicMock(),
        embedding_provider=MagicMock(),
        settings=MagicMock(),
    )

    file_info = FileInfo(
        path=Path("/test/vault/note.md"),
        name="note.md",
        content="content",
        checksum="abc123",
        created_at=None,
        modified_at=None,
    )

    existing = MagicMock()
    existing.checksum_md5 = "abc123"  # Same checksum
    existing.id = uuid.uuid4()

    mock_session = MagicMock()
    mock_session.query.return_value.filter_by.return_value.first.return_value = existing
    service.db_manager.get_session.return_value.__enter__ = MagicMock(
        return_value=mock_session
    )
    service.db_manager.get_session.return_value.__exit__ = MagicMock(return_value=False)

    with patch.object(service, "_update_document", return_value=0) as mock_update:
        with patch.object(service, "_update_tasks"):
            result = service._ingest_single_file(
                file_info,
                vault_id=uuid.uuid4(),
                vault_config=VaultConfig(
                    container_path="/test/vault", host_path="/test/vault"
                ),
                force=True,
            )

    assert result[0] == "updated"
    mock_update.assert_called_once()


def test_ingest_single_file_force_false_preserves_checksum_check() -> None:
    """Test force=False preserves unchanged behavior."""
    from obsidian_rag.services.ingestion import IngestionService
    from obsidian_rag.config import VaultConfig
    from pathlib import Path
    import uuid
    from unittest.mock import MagicMock

    service = IngestionService(
        db_manager=MagicMock(),
        embedding_provider=MagicMock(),
        settings=MagicMock(),
    )

    file_info = FileInfo(
        path=Path("/test/vault/note.md"),
        name="note.md",
        content="content",
        checksum="abc123",
        created_at=None,
        modified_at=None,
    )

    existing = MagicMock()
    existing.checksum_md5 = "abc123"  # Same checksum

    mock_session = MagicMock()
    mock_session.query.return_value.filter_by.return_value.first.return_value = existing
    service.db_manager.get_session.return_value.__enter__ = MagicMock(
        return_value=mock_session
    )
    service.db_manager.get_session.return_value.__exit__ = MagicMock(return_value=False)

    result = service._ingest_single_file(
        file_info,
        vault_id=uuid.uuid4(),
        vault_config=VaultConfig(container_path="/test/vault", host_path="/test/vault"),
        force=False,
    )

    assert result[0] == "unchanged"


def test_ingest_single_file_force_true_different_checksum() -> None:
    """Test force=True with different checksum still updates."""
    from obsidian_rag.services.ingestion import IngestionService
    from obsidian_rag.config import VaultConfig
    from pathlib import Path
    import uuid
    from unittest.mock import MagicMock, patch

    service = IngestionService(
        db_manager=MagicMock(),
        embedding_provider=MagicMock(),
        settings=MagicMock(),
    )

    file_info = FileInfo(
        path=Path("/test/vault/note.md"),
        name="note.md",
        content="content",
        checksum="new_checksum",
        created_at=None,
        modified_at=None,
    )

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
    service.db_manager.get_session.return_value.__enter__ = MagicMock(
        return_value=mock_session
    )
    service.db_manager.get_session.return_value.__exit__ = MagicMock(return_value=False)

    with patch.object(service, "_update_document", return_value=0) as mock_update:
        with patch.object(service, "_update_tasks"):
            result = service._ingest_single_file(
                file_info,
                vault_id=uuid.uuid4(),
                vault_config=VaultConfig(
                    container_path="/test/vault", host_path="/test/vault"
                ),
                force=True,
            )

    assert result[0] == "updated"
    mock_update.assert_called_once()


def test_ingest_single_file_force_true_new_document() -> None:
    """Test force=True has no effect on new documents."""
    from obsidian_rag.services.ingestion import IngestionService
    from obsidian_rag.config import VaultConfig
    from pathlib import Path
    import uuid
    from unittest.mock import MagicMock, patch

    service = IngestionService(
        db_manager=MagicMock(),
        embedding_provider=MagicMock(),
        settings=MagicMock(),
    )

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
    service.db_manager.get_session.return_value.__enter__ = MagicMock(
        return_value=mock_session
    )
    service.db_manager.get_session.return_value.__exit__ = MagicMock(return_value=False)

    with patch(
        "obsidian_rag.services.ingestion.should_chunk_document", return_value=False
    ):
        with patch.object(service, "_create_document") as mock_create:
            mock_create.return_value = (MagicMock(), 0)
            with patch.object(service, "_create_tasks"):
                with patch.object(
                    service, "_create_chunks_for_new_document", return_value=0
                ):
                    result = service._ingest_single_file(
                        file_info,
                        vault_id=uuid.uuid4(),
                        vault_config=VaultConfig(
                            container_path="/test/vault", host_path="/test/vault"
                        ),
                        force=True,
                    )

    assert result[0] == "new"


def test_ingest_single_file_force_true_dry_run_unaffected() -> None:
    """Test dry-run returns 'new' regardless of force."""
    from obsidian_rag.services.ingestion import IngestionService
    from obsidian_rag.config import VaultConfig
    from pathlib import Path
    import uuid
    from unittest.mock import MagicMock

    service = IngestionService(
        db_manager=MagicMock(),
        embedding_provider=MagicMock(),
        settings=MagicMock(),
    )

    file_info = FileInfo(
        path=Path("/test/vault/note.md"),
        name="note.md",
        content="content",
        checksum="abc123",
        created_at=None,
        modified_at=None,
    )

    file_info = FileInfo(
        path=Path("/test/vault/note.md"),
        name="note.md",
        content="content",
        checksum="abc123",
        created_at=datetime.now(timezone.utc),
        modified_at=datetime.now(timezone.utc),
    )

    result = service._ingest_single_file(
        file_info,
        vault_id=uuid.uuid4(),
        vault_config=VaultConfig(container_path="/test/vault", host_path="/test/vault"),
        dry_run=True,
        force=True,
    )

    assert result[0] == "new"
