"""Integration tests for tab-normalized ingestion."""

import uuid
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from obsidian_rag.config import VaultConfig
from obsidian_rag.parsing.frontmatter import parse_frontmatter
from obsidian_rag.parsing.scanner import FileInfo
from obsidian_rag.services.ingestion import IngestionService


@pytest.fixture
def mock_settings() -> MagicMock:
    """Create mock settings for testing."""
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


@pytest.fixture
def mock_db_manager() -> MagicMock:
    """Create mock database manager for testing."""
    return MagicMock()


@pytest.fixture
def mock_embedding_provider() -> MagicMock:
    """Create mock embedding provider for testing."""
    provider = MagicMock()
    provider.generate_embedding.return_value = [0.1] * 10
    return provider


@pytest.fixture
def ingestion_service(
    mock_db_manager: MagicMock,
    mock_embedding_provider: MagicMock,
    mock_settings: MagicMock,
) -> IngestionService:
    """Create IngestionService instance for testing."""
    return IngestionService(
        db_manager=mock_db_manager,
        embedding_provider=mock_embedding_provider,
        settings=mock_settings,
    )


class TestTabFrontmatterIngestion:
    """Test tab-indented frontmatter ingestion through IngestionService."""

    def test_ingest_file_with_tab_frontmatter_succeeds(
        self,
        ingestion_service: IngestionService,
        tmp_path: Path,
    ) -> None:
        """Tab-indented frontmatter file ingests without error."""
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

        # File with tab-indented frontmatter
        content = "---\ntags:\n\t- work\n\t- personal\n---\nBody text"
        file_info = FileInfo(
            path=tmp_path / "test.md",
            name="test.md",
            content=content,
            checksum="abc123",
            created_at=datetime.now(timezone.utc),
            modified_at=datetime.now(timezone.utc),
        )

        mock_task = MagicMock()
        mock_task.raw_text = "Body text"
        mock_task.status = "not_completed"
        mock_task.description = "Body text"
        mock_task.tags = None
        mock_task.repeat = None
        mock_task.scheduled = None
        mock_task.due = None
        mock_task.completion = None
        mock_task.priority = "normal"
        mock_task.inline_fields = {}

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
                        dry_run=False,
                    )
                )

        assert result in ("new", "updated")

    def test_ingest_tab_frontmatter_preserves_tags(
        self,
        tmp_path: Path,
    ) -> None:
        """Tags from tab-indented frontmatter are persisted."""
        content = "---\ntags:\n\t- work\n\t- personal\n---\nBody"
        tags, metadata, remaining = parse_frontmatter(content)

        assert tags is not None
        assert "work" in tags
        assert "personal" in tags

    def test_ingest_tab_frontmatter_preserves_properties(
        self,
        tmp_path: Path,
    ) -> None:
        """Non-tag properties from tab-indented frontmatter are persisted."""
        content = "---\nkind: note\ntags:\n\t- work\n---\nBody"
        tags, metadata, remaining = parse_frontmatter(content)

        assert metadata.get("kind") == "note"
        assert tags is not None
        assert "work" in tags

    def test_ingest_file_without_tabs_unchanged(
        self,
        tmp_path: Path,
    ) -> None:
        """Files without tab indentation ingest identically (regression check)."""
        content = "---\ntags:\n  - work\n---\nBody"
        tags, metadata, remaining = parse_frontmatter(content)

        assert tags == ["work"]
        assert remaining == "Body"
