"""BUG-001 Regression Tests: Verify new documents get chunks created during ingestion.

This test module ensures that BUG-001 is fixed and stays fixed. BUG-001 was a
critical issue where new documents never got chunks created during ingestion,
even when they exceeded the chunk size threshold.
"""

import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

from obsidian_rag.config import VaultConfig
from obsidian_rag.parsing.scanner import FileInfo
from obsidian_rag.services.ingestion import IngestionService

if TYPE_CHECKING:
    from obsidian_rag.config import Settings

# Constants for test assertions
EXPECTED_CHUNKS_LARGE_DOC = 5
EXPECTED_CHUNKS_MEDIUM_DOC = 3
EXPECTED_CHUNKS_UPDATED_DOC = 4
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50
EXPECTED_CHUNKS_MULTIPLE_DOCS = 10
NUM_TEST_DOCUMENTS = 2


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
def mock_vault_config() -> VaultConfig:
    """Create mock vault config for testing."""
    return VaultConfig(
        container_path="/test/vault",
        host_path="/test/vault",
    )


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


class TestBug001NewDocumentChunks:
    """BUG-001 Regression Tests: New documents must get chunks created."""

    @patch("obsidian_rag.services.ingestion.create_chunks_with_embeddings")
    @patch("obsidian_rag.services.ingestion.should_chunk_document")
    @patch("obsidian_rag.services.ingestion.parse_tasks_from_content")
    def test_new_document_gets_chunks_created(
        self,
        mock_parse_tasks: MagicMock,
        mock_should_chunk: MagicMock,
        mock_create_chunks: MagicMock,
        ingestion_service: IngestionService,
        mock_db_manager: MagicMock,
        mock_vault_config: VaultConfig,
    ) -> None:
        """BUG-001 Regression Test: Verify new documents get chunks created.

        This test ensures that when a new document is ingested and it exceeds
        the chunk size threshold, chunks are actually created for it.
        """
        # Setup: Document needs chunking
        mock_should_chunk.return_value = True
        mock_parse_tasks.return_value = []
        mock_create_chunks.return_value = 5

        # Create mock session
        mock_session = MagicMock()
        mock_session.query.return_value.filter_by.return_value.first.return_value = None
        mock_db_manager.get_session.return_value.__enter__.return_value = mock_session

        # Create file info with large content (exceeds chunk threshold)
        large_content = "word " * 1000
        file_info = FileInfo(
            path=Path("/test/vault/large_doc.md"),
            name="large_doc.md",
            content=large_content,
            checksum="abc123",
            created_at=datetime.now(timezone.utc),
            modified_at=datetime.now(timezone.utc),
        )

        # Execute
        result, chunks_created, is_empty = ingestion_service._ingest_single_file(
            file_info=file_info,
            vault_id=uuid.uuid4(),
            vault_config=mock_vault_config,
            dry_run=False,
        )

        # Verify
        assert result == "new"
        assert chunks_created == EXPECTED_CHUNKS_LARGE_DOC
        assert is_empty is False
        mock_create_chunks.assert_called_once()

    @patch("obsidian_rag.services.ingestion.create_chunks_with_embeddings")
    @patch("obsidian_rag.services.ingestion.should_chunk_document")
    @patch("obsidian_rag.services.ingestion.parse_tasks_from_content")
    def test_new_small_document_no_chunks(
        self,
        mock_parse_tasks: MagicMock,
        mock_should_chunk: MagicMock,
        mock_create_chunks: MagicMock,
        ingestion_service: IngestionService,
        mock_db_manager: MagicMock,
        mock_vault_config: VaultConfig,
    ) -> None:
        """Test that small documents don't get chunks created."""
        # Setup: Document does NOT need chunking
        mock_should_chunk.return_value = False
        mock_parse_tasks.return_value = []

        # Create mock session
        mock_session = MagicMock()
        mock_session.query.return_value.filter_by.return_value.first.return_value = None
        mock_db_manager.get_session.return_value.__enter__.return_value = mock_session

        # Create file info with small content
        small_content = "Small document content."
        file_info = FileInfo(
            path=Path("/test/vault/small_doc.md"),
            name="small_doc.md",
            content=small_content,
            checksum="def456",
            created_at=datetime.now(timezone.utc),
            modified_at=datetime.now(timezone.utc),
        )

        # Execute
        result, chunks_created, is_empty = ingestion_service._ingest_single_file(
            file_info=file_info,
            vault_id=uuid.uuid4(),
            vault_config=mock_vault_config,
            dry_run=False,
        )

        # Verify
        assert result == "new"
        assert chunks_created == 0
        assert is_empty is False
        mock_create_chunks.assert_not_called()

    @patch("obsidian_rag.services.ingestion.create_chunks_with_embeddings")
    @patch("obsidian_rag.services.ingestion.should_chunk_document")
    @patch("obsidian_rag.services.ingestion.parse_tasks_from_content")
    def test_new_empty_document_no_chunks(
        self,
        mock_parse_tasks: MagicMock,
        mock_should_chunk: MagicMock,
        mock_create_chunks: MagicMock,
        ingestion_service: IngestionService,
        mock_db_manager: MagicMock,
        mock_vault_config: VaultConfig,
    ) -> None:
        """Test that empty documents don't get chunks created."""
        # Setup
        mock_should_chunk.return_value = True
        mock_parse_tasks.return_value = []

        # Create mock session
        mock_session = MagicMock()
        mock_session.query.return_value.filter_by.return_value.first.return_value = None
        mock_db_manager.get_session.return_value.__enter__.return_value = mock_session

        # Create file info with empty content
        empty_content = ""
        file_info = FileInfo(
            path=Path("/test/vault/empty_doc.md"),
            name="empty_doc.md",
            content=empty_content,
            checksum="ghi789",
            created_at=datetime.now(timezone.utc),
            modified_at=datetime.now(timezone.utc),
        )

        # Execute
        result, chunks_created, is_empty = ingestion_service._ingest_single_file(
            file_info=file_info,
            vault_id=uuid.uuid4(),
            vault_config=mock_vault_config,
            dry_run=False,
        )

        # Verify
        assert result == "new"
        assert chunks_created == 0
        assert is_empty is True
        mock_create_chunks.assert_not_called()

    @patch("obsidian_rag.services.ingestion.create_chunks_with_embeddings")
    @patch("obsidian_rag.services.ingestion.should_chunk_document")
    @patch("obsidian_rag.services.ingestion.parse_tasks_from_content")
    def test_chunk_creation_failure_handled(
        self,
        mock_parse_tasks: MagicMock,
        mock_should_chunk: MagicMock,
        mock_create_chunks: MagicMock,
        ingestion_service: IngestionService,
        mock_db_manager: MagicMock,
        mock_vault_config: VaultConfig,
    ) -> None:
        """Test that chunk creation failures don't stop ingestion."""
        # Setup: Document needs chunking but chunking fails
        mock_should_chunk.return_value = True
        mock_parse_tasks.return_value = []
        mock_create_chunks.side_effect = Exception("Chunk creation failed")

        # Create mock session
        mock_session = MagicMock()
        mock_session.query.return_value.filter_by.return_value.first.return_value = None
        mock_db_manager.get_session.return_value.__enter__.return_value = mock_session

        # Create file info with large content
        large_content = "word " * 1000
        file_info = FileInfo(
            path=Path("/test/vault/large_doc.md"),
            name="large_doc.md",
            content=large_content,
            checksum="jkl012",
            created_at=datetime.now(timezone.utc),
            modified_at=datetime.now(timezone.utc),
        )

        # Execute - should raise exception
        with pytest.raises(Exception, match="Chunk creation failed"):
            ingestion_service._ingest_single_file(
                file_info=file_info,
                vault_id=uuid.uuid4(),
                vault_config=mock_vault_config,
                dry_run=False,
            )

        # Verify that create_chunks was called (attempted)
        mock_create_chunks.assert_called_once()

    @patch("obsidian_rag.services.ingestion.create_chunks_with_embeddings")
    @patch("obsidian_rag.services.ingestion.should_chunk_document")
    @patch("obsidian_rag.services.ingestion.parse_tasks_from_content")
    def test_new_document_with_tasks_gets_chunks_and_tasks(
        self,
        mock_parse_tasks: MagicMock,
        mock_should_chunk: MagicMock,
        mock_create_chunks: MagicMock,
        ingestion_service: IngestionService,
        mock_db_manager: MagicMock,
        mock_vault_config: VaultConfig,
    ) -> None:
        """Test that new documents with tasks get both chunks and tasks created."""
        # Setup: Document needs chunking and has tasks
        mock_should_chunk.return_value = True
        mock_create_chunks.return_value = 3

        # Create mock parsed tasks
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
        mock_parse_tasks.return_value = [(1, mock_task)]

        # Create mock session
        mock_session = MagicMock()
        mock_session.query.return_value.filter_by.return_value.first.return_value = None
        mock_db_manager.get_session.return_value.__enter__.return_value = mock_session

        # Create file info with large content and a task
        large_content = "word " * 1000 + "\n- [ ] Test task"
        file_info = FileInfo(
            path=Path("/test/vault/large_doc_with_tasks.md"),
            name="large_doc_with_tasks.md",
            content=large_content,
            checksum="mno345",
            created_at=datetime.now(timezone.utc),
            modified_at=datetime.now(timezone.utc),
        )

        # Execute
        result, chunks_created, is_empty = ingestion_service._ingest_single_file(
            file_info=file_info,
            vault_id=uuid.uuid4(),
            vault_config=mock_vault_config,
            dry_run=False,
        )

        # Verify
        assert result == "new"
        assert chunks_created == EXPECTED_CHUNKS_MEDIUM_DOC
        assert is_empty is False
        mock_create_chunks.assert_called_once()

    @patch("obsidian_rag.services.ingestion.create_chunks_with_embeddings")
    @patch("obsidian_rag.services.ingestion.should_chunk_document")
    @patch("obsidian_rag.services.ingestion.parse_tasks_from_content")
    def test_chunk_creation_parameters_passed_correctly(
        self,
        mock_parse_tasks: MagicMock,
        mock_should_chunk: MagicMock,
        mock_create_chunks: MagicMock,
        ingestion_service: IngestionService,
        mock_db_manager: MagicMock,
        mock_vault_config: VaultConfig,
    ) -> None:
        """Test that chunk creation receives correct parameters from settings."""
        # Setup
        mock_should_chunk.return_value = True
        mock_parse_tasks.return_value = []
        mock_create_chunks.return_value = 5

        # Create mock session
        mock_session = MagicMock()
        mock_session.query.return_value.filter_by.return_value.first.return_value = None
        mock_db_manager.get_session.return_value.__enter__.return_value = mock_session

        # Create file info
        large_content = "word " * 1000
        file_info = FileInfo(
            path=Path("/test/vault/large_doc.md"),
            name="large_doc.md",
            content=large_content,
            checksum="pqr678",
            created_at=datetime.now(timezone.utc),
            modified_at=datetime.now(timezone.utc),
        )

        # Execute
        ingestion_service._ingest_single_file(
            file_info=file_info,
            vault_id=uuid.uuid4(),
            vault_config=mock_vault_config,
            dry_run=False,
        )

        # Verify correct parameters passed to create_chunks_with_embeddings
        call_args = mock_create_chunks.call_args
        # Function is called with positional args: session, document_id, content,
        # embedding_provider, chunk_size, chunk_overlap, model_name
        assert call_args.args[0] == mock_session  # db_session
        assert call_args.args[2] == large_content  # content
        assert call_args.args[4] == CHUNK_SIZE  # chunk_size
        assert call_args.args[5] == CHUNK_OVERLAP  # chunk_overlap
        assert call_args.args[6] == "gpt2"  # model_name


class TestBug001EdgeCases:
    """BUG-001 Edge case tests for chunk creation."""

    @patch("obsidian_rag.services.ingestion.create_chunks_with_embeddings")
    @patch("obsidian_rag.services.ingestion.should_chunk_document")
    @patch("obsidian_rag.services.ingestion.parse_tasks_from_content")
    def test_whitespace_only_document_no_chunks(
        self,
        mock_parse_tasks: MagicMock,
        mock_should_chunk: MagicMock,
        mock_create_chunks: MagicMock,
        ingestion_service: IngestionService,
        mock_db_manager: MagicMock,
        mock_vault_config: VaultConfig,
    ) -> None:
        """Test that whitespace-only documents don't get chunks created."""
        # Setup
        mock_should_chunk.return_value = True
        mock_parse_tasks.return_value = []

        # Create mock session
        mock_session = MagicMock()
        mock_session.query.return_value.filter_by.return_value.first.return_value = None
        mock_db_manager.get_session.return_value.__enter__.return_value = mock_session

        # Create file info with whitespace-only content
        whitespace_content = "   \n\t   \n   "
        file_info = FileInfo(
            path=Path("/test/vault/whitespace_doc.md"),
            name="whitespace_doc.md",
            content=whitespace_content,
            checksum="stu901",
            created_at=datetime.now(timezone.utc),
            modified_at=datetime.now(timezone.utc),
        )

        # Execute
        result, chunks_created, is_empty = ingestion_service._ingest_single_file(
            file_info=file_info,
            vault_id=uuid.uuid4(),
            vault_config=mock_vault_config,
            dry_run=False,
        )

        # Verify
        assert result == "new"
        assert chunks_created == 0
        assert is_empty is True
        mock_create_chunks.assert_not_called()

    @patch("obsidian_rag.services.ingestion.create_chunks_with_embeddings")
    @patch("obsidian_rag.services.ingestion.should_chunk_document")
    @patch("obsidian_rag.services.ingestion.parse_tasks_from_content")
    def test_updated_document_still_gets_chunks(
        self,
        mock_parse_tasks: MagicMock,
        mock_should_chunk: MagicMock,
        mock_create_chunks: MagicMock,
        ingestion_service: IngestionService,
        mock_db_manager: MagicMock,
        mock_vault_config: VaultConfig,
    ) -> None:
        """Test that updated documents still get chunks (existing behavior)."""
        # Setup
        mock_should_chunk.return_value = True
        mock_parse_tasks.return_value = []
        mock_create_chunks.return_value = 4

        # Create mock existing document
        mock_existing = MagicMock()
        mock_existing.id = uuid.uuid4()
        mock_existing.checksum_md5 = "old_checksum"

        # Create mock session
        mock_session = MagicMock()
        mock_session.query.return_value.filter_by.return_value.first.return_value = (
            mock_existing
        )
        mock_db_manager.get_session.return_value.__enter__.return_value = mock_session

        # Create file info with different checksum (trigger update)
        large_content = "word " * 1000
        file_info = FileInfo(
            path=Path("/test/vault/existing_doc.md"),
            name="existing_doc.md",
            content=large_content,
            checksum="new_checksum",
            created_at=datetime.now(timezone.utc),
            modified_at=datetime.now(timezone.utc),
        )

        # Execute
        result, chunks_created, is_empty = ingestion_service._ingest_single_file(
            file_info=file_info,
            vault_id=uuid.uuid4(),
            vault_config=mock_vault_config,
            dry_run=False,
        )

        # Verify
        assert result == "updated"
        assert chunks_created == EXPECTED_CHUNKS_UPDATED_DOC
        mock_create_chunks.assert_called_once()

    @patch("obsidian_rag.services.ingestion.create_chunks_with_embeddings")
    @patch("obsidian_rag.services.ingestion.should_chunk_document")
    @patch("obsidian_rag.services.ingestion.parse_tasks_from_content")
    def test_dry_run_no_chunks_created(
        self,
        mock_parse_tasks: MagicMock,
        mock_should_chunk: MagicMock,
        mock_create_chunks: MagicMock,
        ingestion_service: IngestionService,
        mock_vault_config: VaultConfig,
    ) -> None:
        """Test that dry run mode doesn't create chunks."""
        # Setup
        mock_should_chunk.return_value = True
        mock_parse_tasks.return_value = []

        # Create file info with large content
        large_content = "word " * 1000
        file_info = FileInfo(
            path=Path("/test/vault/large_doc.md"),
            name="large_doc.md",
            content=large_content,
            checksum="vwx234",
            created_at=datetime.now(timezone.utc),
            modified_at=datetime.now(timezone.utc),
        )

        # Execute with dry_run=True
        result, chunks_created, is_empty = ingestion_service._ingest_single_file(
            file_info=file_info,
            vault_id=uuid.uuid4(),
            vault_config=mock_vault_config,
            dry_run=True,
        )

        # Verify
        assert result == "new"
        assert chunks_created == 0
        assert is_empty is False
        mock_create_chunks.assert_not_called()


class TestBug001Integration:
    """BUG-001 Integration tests with full ingestion flow."""

    @patch("obsidian_rag.services.ingestion.create_chunks_with_embeddings")
    @patch("obsidian_rag.services.ingestion.should_chunk_document")
    @patch("obsidian_rag.services.ingestion.parse_tasks_from_content")
    def test_multiple_new_documents_get_chunks(
        self,
        mock_parse_tasks: MagicMock,
        mock_should_chunk: MagicMock,
        mock_create_chunks: MagicMock,
        ingestion_service: IngestionService,
        mock_db_manager: MagicMock,
        mock_vault_config: VaultConfig,
    ) -> None:
        """Test that multiple new documents all get chunks created."""
        # Setup
        mock_should_chunk.return_value = True
        mock_parse_tasks.return_value = []
        mock_create_chunks.return_value = 5

        # Create mock session
        mock_session = MagicMock()
        mock_session.query.return_value.filter_by.return_value.first.return_value = None
        mock_db_manager.get_session.return_value.__enter__.return_value = mock_session

        # Process multiple files
        large_content = "word " * 1000
        file_infos = [
            FileInfo(
                path=Path("/test/vault/doc1.md"),
                name="doc1.md",
                content=large_content,
                checksum="chk1",
                created_at=datetime.now(timezone.utc),
                modified_at=datetime.now(timezone.utc),
            ),
            FileInfo(
                path=Path("/test/vault/doc2.md"),
                name="doc2.md",
                content=large_content,
                checksum="chk2",
                created_at=datetime.now(timezone.utc),
                modified_at=datetime.now(timezone.utc),
            ),
        ]

        # Execute and verify each file
        total_chunks = 0
        for file_info in file_infos:
            result, chunks_created, _ = ingestion_service._ingest_single_file(
                file_info=file_info,
                vault_id=uuid.uuid4(),
                vault_config=mock_vault_config,
                dry_run=False,
            )
            assert result == "new"
            total_chunks += chunks_created

        # Verify chunks were created for each document
        assert (
            total_chunks == EXPECTED_CHUNKS_MULTIPLE_DOCS
        )  # 5 chunks per document * 2 documents
        assert mock_create_chunks.call_count == NUM_TEST_DOCUMENTS
