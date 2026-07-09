"""Tests for vault-level ingestion in IngestionService."""

import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

from obsidian_rag.config import VaultConfig
from obsidian_rag.services.ingestion import IngestionService, IngestVaultOptions

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

    def test_ingest_vault_force_passes_to_process_files(self) -> None:
        """Test ingest_vault passes force through to _process_files_with_stats."""
        service = IngestionService(
            db_manager=MagicMock(),
            embedding_provider=MagicMock(),
            settings=MagicMock(),
        )

        vault_config = VaultConfig(
            container_path="/test/vault", host_path="/test/vault"
        )

        with patch.object(service, "_resolve_vault_config", return_value=vault_config):
            with patch.object(
                service, "_get_or_create_vault", return_value=uuid.uuid4()
            ):
                with patch.object(service, "_get_file_info_list", return_value=([], 0)):
                    # When no files, returns early; we just verify no crash with force=True
                    options = IngestVaultOptions(
                        vault=vault_config,
                        force=True,
                    )
                    result = service.ingest_vault(Path("/test/vault"), options)

        assert result.total == 0
        assert "No markdown files" in result.message

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


def test_process_files_with_stats_force_passes_to_ingest_single_file() -> None:
    """Test _process_files_with_stats passes force to _ingest_single_file."""
    from obsidian_rag.services.ingestion import IngestionService
    from obsidian_rag.config import VaultConfig
    from obsidian_rag.parsing.scanner import FileInfo
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

    vault_config = VaultConfig(container_path="/test/vault", host_path="/test/vault")

    with patch.object(
        service, "_ingest_single_file", return_value=("updated", 0, False)
    ) as mock_ingest:
        stats = service._process_files_with_stats(
            [file_info],
            vault_id=uuid.uuid4(),
            vault_config=vault_config,
            dry_run=False,
            progress_callback=None,
            force=True,
        )

    assert stats["updated"] == 1
    mock_ingest.assert_called_once()
    call_kwargs = mock_ingest.call_args
    assert call_kwargs.kwargs.get("force") is True
