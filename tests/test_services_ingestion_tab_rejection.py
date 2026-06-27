"""Tests for tab-indented frontmatter rejection during ingestion."""

import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from obsidian_rag.config import VaultConfig
from obsidian_rag.parsing.frontmatter import FrontMatterParsingError
from obsidian_rag.parsing.scanner import FileInfo
from obsidian_rag.services.ingestion import IngestionService, IngestVaultOptions

_TAB_FRONTMATTER = "---\n\ttitle: Bad\n---\n\nbody"
_QUOTED_TAB_FRONTMATTER = '---\ntitle: "hello\tworld"\n---\n\nbody'


def _make_settings() -> MagicMock:
    """Create a mock settings object with required ingestion attributes."""
    settings = MagicMock()
    settings.ingestion.batch_size = 100
    settings.ingestion.progress_interval = 10
    settings.ingestion.max_chunk_chars = 24000
    settings.ingestion.chunk_overlap_chars = 800
    settings.chunking.chunk_size = 512
    settings.chunking.chunk_overlap = 50
    settings.chunking.model_name = "gpt2"
    settings.chunking.cache_dir = None
    return settings


def _make_service(settings: MagicMock | None = None) -> IngestionService:
    """Create an IngestionService with mock dependencies."""
    if settings is None:
        settings = _make_settings()

    db_manager = MagicMock()
    embedding_provider = MagicMock()
    embedding_provider.generate_embedding.return_value = [0.1, 0.2, 0.3]
    return IngestionService(
        db_manager=db_manager,
        embedding_provider=embedding_provider,
        settings=settings,
    )


def _make_file_info(
    path: Path,
    content: str,
    checksum: str = "abc123",
) -> FileInfo:
    """Create a FileInfo instance for tests."""
    return FileInfo(
        path=path,
        name=path.name,
        content=content,
        checksum=checksum,
        created_at=datetime.now(timezone.utc),
        modified_at=datetime.now(timezone.utc),
    )


def test_ingest_single_file_rejects_tab_indented_frontmatter(tmp_path: Path) -> None:
    """_ingest_single_file raises FrontMatterParsingError for tab frontmatter."""
    service = _make_service()
    file_path = tmp_path / "note.md"
    file_info = _make_file_info(file_path, _TAB_FRONTMATTER)
    vault_config = VaultConfig(
        container_path=str(tmp_path),
        host_path=str(tmp_path),
    )

    with pytest.raises(FrontMatterParsingError):
        service._ingest_single_file(  # noqa: SLF001
            file_info,
            vault_id=uuid.uuid4(),
            vault_config=vault_config,
        )


def test_process_files_with_stats_counts_tab_errors(tmp_path: Path) -> None:
    """_process_files_with_stats catches the error and stats['errors'] >= 1."""
    service = _make_service()
    file_path = tmp_path / "note.md"
    file_info = _make_file_info(file_path, _TAB_FRONTMATTER)
    vault_config = VaultConfig(
        container_path=str(tmp_path),
        host_path=str(tmp_path),
    )

    stats = service._process_files_with_stats(  # noqa: SLF001
        [file_info],
        vault_id=uuid.uuid4(),
        vault_config=vault_config,
        dry_run=False,
        progress_callback=None,
    )

    assert stats["errors"] >= 1


def test_process_files_with_stats_logs_tab_error(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The tab frontmatter error is logged with the file path."""
    caplog.set_level(logging.ERROR)

    service = _make_service()
    file_path = tmp_path / "note.md"
    file_info = _make_file_info(file_path, _TAB_FRONTMATTER)
    vault_config = VaultConfig(
        container_path=str(tmp_path),
        host_path=str(tmp_path),
    )

    service._process_files_with_stats(  # noqa: SLF001
        [file_info],
        vault_id=uuid.uuid4(),
        vault_config=vault_config,
        dry_run=False,
        progress_callback=None,
    )

    log_text = caplog.text
    assert str(file_path) in log_text
    assert "Tab characters found" in log_text


def test_ingest_vault_includes_tab_errors_in_result(tmp_path: Path) -> None:
    """IngestionResult.errors includes tab-rejected files."""
    service = _make_service()
    file_path = tmp_path / "note.md"
    file_info = _make_file_info(file_path, _TAB_FRONTMATTER)
    vault_config = VaultConfig(
        container_path=str(tmp_path),
        host_path=str(tmp_path),
    )

    with (
        patch("obsidian_rag.services.ingestion.scan_markdown_files") as mock_scan,
        patch(
            "obsidian_rag.services.ingestion.process_files_in_batches"
        ) as mock_process,
    ):
        mock_scan.return_value = [file_path]
        mock_process.return_value = [file_info]

        with patch.object(
            service,
            "_get_or_create_vault",
            return_value=uuid.uuid4(),
        ):
            options = IngestVaultOptions(
                vault=vault_config,
                no_delete=True,
            )
            result = service.ingest_vault(tmp_path, options)

    assert result.total == 1
    assert result.errors == 1


def test_force_reingestion_still_rejects_tab_frontmatter(tmp_path: Path) -> None:
    """force=True does not bypass the tab frontmatter check."""
    service = _make_service()
    file_path = tmp_path / "note.md"
    file_info = _make_file_info(file_path, _TAB_FRONTMATTER)
    vault_config = VaultConfig(
        container_path=str(tmp_path),
        host_path=str(tmp_path),
    )

    stats = service._process_files_with_stats(  # noqa: SLF001
        [file_info],
        vault_id=uuid.uuid4(),
        vault_config=vault_config,
        dry_run=False,
        progress_callback=None,
        force=True,
    )

    assert stats["errors"] >= 1


def test_ingest_accepts_tabs_in_frontmatter_values(tmp_path: Path) -> None:
    """Tabs inside quoted frontmatter values are accepted."""
    service = _make_service()
    file_path = tmp_path / "note.md"
    file_info = _make_file_info(file_path, _QUOTED_TAB_FRONTMATTER)
    vault_config = VaultConfig(
        container_path=str(tmp_path),
        host_path=str(tmp_path),
    )

    mock_session = MagicMock()
    mock_session_context = MagicMock()
    mock_session_context.__enter__ = MagicMock(return_value=mock_session)
    mock_session_context.__exit__ = MagicMock(return_value=None)
    service.db_manager.get_session.return_value = mock_session_context  # type: ignore[attr-defined]
    mock_session.query.return_value.filter_by.return_value.first.return_value = None

    with patch(
        "obsidian_rag.services.ingestion.parse_tasks_from_content"
    ) as mock_parse_tasks:
        mock_parse_tasks.return_value = []

        with patch(
            "obsidian_rag.services.ingestion.should_chunk_document"
        ) as mock_should_chunk:
            mock_should_chunk.return_value = False

            result, chunks_created, is_empty = service._ingest_single_file(  # noqa: SLF001
                file_info,
                vault_id=uuid.uuid4(),
                vault_config=vault_config,
            )

    assert result == "new"
    assert chunks_created == 0
    assert is_empty is False
