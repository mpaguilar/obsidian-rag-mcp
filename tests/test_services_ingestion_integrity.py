"""Tests for IntegrityError recovery in document ingestion."""

import logging
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy.exc import IntegrityError

from obsidian_rag.config import VaultConfig
from obsidian_rag.parsing.scanner import FileInfo
from obsidian_rag.services.ingestion import IngestionService
from obsidian_rag.services.ingestion_integrity import handle_integrity_error

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


class _FakeDbError(Exception):
    """Fake database exception with a constraint name in its message."""

    def __init__(self, constraint_name: str) -> None:
        """Initialize with the constraint name.

        Args:
            constraint_name: Name of the constraint that was violated.

        """
        super().__init__(
            f"duplicate key value violates unique constraint \"{constraint_name}\""
        )


def _create_integrity_error(constraint_name: str) -> IntegrityError:
    """Create an IntegrityError with the specified constraint name in its message.

    Args:
        constraint_name: Name of the constraint to include in the error message.

    Returns:
        SQLAlchemy IntegrityError instance.

    """
    orig = _FakeDbError(constraint_name)
    return IntegrityError("INSERT INTO documents ...", None, orig)


def _create_file_info(tmp_path: Path) -> FileInfo:
    """Create a FileInfo for testing.

    Args:
        tmp_path: Temporary path to use as the file location.

    Returns:
        FileInfo instance ready for ingestion tests.

    """
    test_file = tmp_path / "test.md"
    test_file.write_text("content")
    return FileInfo(
        path=test_file,
        name="test.md",
        content="content",
        checksum="abc123",
        created_at=datetime.now(UTC),
        modified_at=datetime.now(UTC),
    )


def test_handle_integrity_error_recovers_to_update(
    ingestion_service: IngestionService,
) -> None:
    """IntegrityError on flush rolls back, re-queries, and updates existing doc."""
    expected_chunks = 7
    mock_session = MagicMock()
    existing_doc = MagicMock()
    mock_session.query.return_value.filter_by.return_value.first.return_value = (
        existing_doc
    )

    err = _create_integrity_error("uq_document_vault_path")

    with patch.object(
        ingestion_service, "_update_document", return_value=expected_chunks
    ):
        with patch.object(ingestion_service, "_update_tasks"):
            result, chunks = handle_integrity_error(
                ingestion_service,
                mock_session,
                err,
                vault_id=uuid.uuid4(),
                relative_path="folder/note.md",
                file_info=MagicMock(),
                parsed_data=(None, {}, "content"),
                parsed_tasks=[],
                is_empty=False,
            )

    assert result == "updated"
    assert chunks == expected_chunks


def test_handle_integrity_error_non_unique_violation_reraises(
    ingestion_service: IngestionService,
) -> None:
    """IntegrityError on a different constraint is re-raised."""
    mock_session = MagicMock()
    err = _create_integrity_error("some_other_constraint")

    with pytest.raises(IntegrityError):
        handle_integrity_error(
            ingestion_service,
            mock_session,
            err,
            vault_id=uuid.uuid4(),
            relative_path="folder/note.md",
            file_info=MagicMock(),
            parsed_data=(None, {}, "content"),
            parsed_tasks=[],
            is_empty=False,
        )


def test_handle_integrity_error_requery_none_reraises(
    ingestion_service: IngestionService,
) -> None:
    """IntegrityError recovery re-raises if re-query returns None."""
    mock_session = MagicMock()
    mock_session.query.return_value.filter_by.return_value.first.return_value = None

    err = _create_integrity_error("uq_document_vault_path")

    with pytest.raises(IntegrityError):
        handle_integrity_error(
            ingestion_service,
            mock_session,
            err,
            vault_id=uuid.uuid4(),
            relative_path="folder/note.md",
            file_info=MagicMock(),
            parsed_data=(None, {}, "content"),
            parsed_tasks=[],
            is_empty=False,
        )


def test_handle_integrity_error_calls_rollback(
    ingestion_service: IngestionService,
) -> None:
    """session.rollback() is called before re-query during recovery."""
    mock_session = MagicMock()
    existing_doc = MagicMock()
    mock_session.query.return_value.filter_by.return_value.first.return_value = (
        existing_doc
    )

    err = _create_integrity_error("uq_document_vault_path")

    with patch.object(ingestion_service, "_update_document", return_value=0):
        with patch.object(ingestion_service, "_update_tasks"):
            handle_integrity_error(
                ingestion_service,
                mock_session,
                err,
                vault_id=uuid.uuid4(),
                relative_path="folder/note.md",
                file_info=MagicMock(),
                parsed_data=(None, {}, "content"),
                parsed_tasks=[],
                is_empty=False,
            )

    mock_session.rollback.assert_called_once()


def test_handle_integrity_error_logs_warning(
    ingestion_service: IngestionService,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Recovery logs a WARNING containing vault_id and file_path."""
    mock_session = MagicMock()
    existing_doc = MagicMock()
    mock_session.query.return_value.filter_by.return_value.first.return_value = (
        existing_doc
    )

    err = _create_integrity_error("uq_document_vault_path")
    vault_id = uuid.uuid4()
    relative_path = "folder/note.md"

    with patch.object(ingestion_service, "_update_document", return_value=0):
        with patch.object(ingestion_service, "_update_tasks"):
            with caplog.at_level(logging.WARNING):
                handle_integrity_error(
                    ingestion_service,
                    mock_session,
                    err,
                    vault_id=vault_id,
                    relative_path=relative_path,
                    file_info=MagicMock(),
                    parsed_data=(None, {}, "content"),
                    parsed_tasks=[],
                    is_empty=False,
                )

    assert "uq_document_vault_path recovered" in caplog.text
    assert str(vault_id) in caplog.text
    assert relative_path in caplog.text


def test_ingest_single_file_integrity_error_full_flow(
    ingestion_service: IngestionService,
    tmp_path: Path,
) -> None:
    """End-to-end _ingest_single_file recovers from IntegrityError and returns updated."""
    vault_config = VaultConfig(
        container_path=str(tmp_path),
        host_path=str(tmp_path),
    )
    file_info = _create_file_info(tmp_path)
    vault_id = uuid.uuid4()

    mock_session = MagicMock()
    mock_session_context = MagicMock()
    mock_session_context.__enter__ = MagicMock(return_value=mock_session)
    mock_session_context.__exit__ = MagicMock(return_value=None)
    ingestion_service.db_manager.get_session.return_value = mock_session_context  # type: ignore[attr-defined]

    expected_chunks = 5
    existing_doc = MagicMock()
    existing_doc.id = uuid.uuid4()

    # First lookup returns None (new doc), recovery lookup returns existing doc
    mock_session.query.return_value.filter_by.return_value.first.side_effect = [
        None,
        existing_doc,
    ]

    err = _create_integrity_error("uq_document_vault_path")
    mock_session.flush.side_effect = err

    with patch(
        "obsidian_rag.services.ingestion.parse_frontmatter",
        return_value=(None, {}, "content"),
    ):
        with patch(
            "obsidian_rag.services.ingestion.parse_tasks_from_content",
            return_value=[],
        ):
            with patch(
                "obsidian_rag.services.ingestion.should_chunk_document",
                return_value=False,
            ):
                with patch.object(
                    ingestion_service,
                    "_create_document",
                    return_value=(MagicMock(), 0),
                ):
                    with patch.object(
                        ingestion_service,
                        "_update_document",
                        return_value=expected_chunks,
                    ) as mock_update:
                        with patch.object(ingestion_service, "_update_tasks"):
                            result, chunks_created, _is_empty = (
                                ingestion_service._ingest_single_file(
                                    file_info,
                                    vault_id=vault_id,
                                    vault_config=vault_config,
                                )
                            )

    assert result == "updated"
    assert chunks_created == expected_chunks
    mock_update.assert_called_once_with(
        mock_session, existing_doc, file_info, (None, {}, "content")
    )


def test_ingest_single_file_integrity_error_non_unique_reraises(
    ingestion_service: IngestionService,
    tmp_path: Path,
) -> None:
    """End-to-end _ingest_single_file propagates non-uq IntegrityError as an error."""
    vault_config = VaultConfig(
        container_path=str(tmp_path),
        host_path=str(tmp_path),
    )
    file_info = _create_file_info(tmp_path)
    vault_id = uuid.uuid4()

    mock_session = MagicMock()
    mock_session_context = MagicMock()
    mock_session_context.__enter__ = MagicMock(return_value=mock_session)
    mock_session_context.__exit__ = MagicMock(return_value=None)
    ingestion_service.db_manager.get_session.return_value = mock_session_context  # type: ignore[attr-defined]

    mock_session.query.return_value.filter_by.return_value.first.return_value = None

    err = _create_integrity_error("some_other_constraint")
    mock_session.flush.side_effect = err

    with patch(
        "obsidian_rag.services.ingestion.parse_frontmatter",
        return_value=(None, {}, "content"),
    ):
        with patch(
            "obsidian_rag.services.ingestion.parse_tasks_from_content",
            return_value=[],
        ):
            with patch(
                "obsidian_rag.services.ingestion.should_chunk_document",
                return_value=False,
            ):
                with patch.object(
                    ingestion_service,
                    "_create_document",
                    return_value=(MagicMock(), 0),
                ):
                    with pytest.raises(IntegrityError):
                        ingestion_service._ingest_single_file(
                            file_info,
                            vault_id=vault_id,
                            vault_config=vault_config,
                        )


def test_process_files_with_stats_integrity_recovery_counts_updated(
    ingestion_service: IngestionService,
    tmp_path: Path,
) -> None:
    """After IntegrityError recovery, _process_files_with_stats counts file as updated."""
    vault_config = VaultConfig(
        container_path=str(tmp_path),
        host_path=str(tmp_path),
    )
    file_info = _create_file_info(tmp_path)
    vault_id = uuid.uuid4()

    mock_session = MagicMock()
    mock_session_context = MagicMock()
    mock_session_context.__enter__ = MagicMock(return_value=mock_session)
    mock_session_context.__exit__ = MagicMock(return_value=None)
    ingestion_service.db_manager.get_session.return_value = mock_session_context  # type: ignore[attr-defined]

    existing_doc = MagicMock()
    existing_doc.id = uuid.uuid4()

    mock_session.query.return_value.filter_by.return_value.first.side_effect = [
        None,
        existing_doc,
    ]

    err = _create_integrity_error("uq_document_vault_path")
    mock_session.flush.side_effect = err

    with patch(
        "obsidian_rag.services.ingestion.parse_frontmatter",
        return_value=(None, {}, "content"),
    ):
        with patch(
            "obsidian_rag.services.ingestion.parse_tasks_from_content",
            return_value=[],
        ):
            with patch(
                "obsidian_rag.services.ingestion.should_chunk_document",
                return_value=False,
            ):
                with patch.object(
                    ingestion_service,
                    "_create_document",
                    return_value=(MagicMock(), 0),
                ):
                    with patch.object(
                        ingestion_service,
                        "_update_document",
                        return_value=3,
                    ):
                        with patch.object(ingestion_service, "_update_tasks"):
                            stats = ingestion_service._process_files_with_stats(
                                [file_info],
                                vault_id=vault_id,
                                vault_config=vault_config,
                                dry_run=False,
                                progress_callback=None,
                            )

    assert stats["updated"] == 1
    assert stats["errors"] == 0
    assert stats["new"] == 0


def test_update_document_updates_ingested_at(
    ingestion_service: IngestionService,
) -> None:
    """_update_document sets document.ingested_at to approximately datetime.now(UTC)."""
    expected_time = datetime(2026, 6, 17, 12, 0, 0, tzinfo=UTC)

    with patch("obsidian_rag.services.ingestion.datetime") as mock_dt:
        mock_dt.now.return_value = expected_time
        mock_dt.UTC = UTC

        document = MagicMock()
        file_info = MagicMock()
        file_info.checksum = "new_checksum"
        file_info.modified_at = expected_time
        mock_session = MagicMock()

        with patch.object(ingestion_service, "_delete_existing_chunks"):
            with patch.object(
                ingestion_service,
                "_generate_embedding",
                return_value=[0.1, 0.2, 0.3],
            ):
                ingestion_service._update_document(
                    mock_session,
                    document,
                    file_info,
                    (None, {}, "content"),
                )

    assert document.ingested_at == expected_time


def test_update_document_ingested_at_is_utc(
    ingestion_service: IngestionService,
) -> None:
    """The ingested_at value set by _update_document uses UTC timezone."""
    expected_time = datetime(2026, 6, 17, 12, 0, 0, tzinfo=UTC)

    with patch("obsidian_rag.services.ingestion.datetime") as mock_dt:
        mock_dt.now.return_value = expected_time
        mock_dt.UTC = UTC

        document = MagicMock()
        file_info = MagicMock()
        file_info.checksum = "new_checksum"
        file_info.modified_at = expected_time
        mock_session = MagicMock()

        with patch.object(ingestion_service, "_delete_existing_chunks"):
            with patch.object(
                ingestion_service,
                "_generate_embedding",
                return_value=[0.1, 0.2, 0.3],
            ):
                ingestion_service._update_document(
                    mock_session,
                    document,
                    file_info,
                    (None, {}, "content"),
                )

    assert document.ingested_at.tzinfo is UTC


def test_ingest_single_file_update_path_updates_ingested_at(
    ingestion_service: IngestionService,
    tmp_path: Path,
) -> None:
    """_ingest_single_file updates ingested_at when existing doc has different checksum."""
    vault_config = VaultConfig(
        container_path=str(tmp_path),
        host_path=str(tmp_path),
    )
    file_info = _create_file_info(tmp_path)
    vault_id = uuid.uuid4()

    mock_session = MagicMock()
    mock_session_context = MagicMock()
    mock_session_context.__enter__ = MagicMock(return_value=mock_session)
    mock_session_context.__exit__ = MagicMock(return_value=None)
    ingestion_service.db_manager.get_session.return_value = mock_session_context  # type: ignore[attr-defined]

    existing_doc = MagicMock()
    existing_doc.checksum_md5 = "old_checksum"
    existing_doc.id = uuid.uuid4()
    existing_doc.tags = None

    mock_session.query.return_value.filter_by.return_value.first.return_value = (
        existing_doc
    )

    expected_time = datetime(2026, 6, 17, 12, 0, 0, tzinfo=UTC)

    with patch(
        "obsidian_rag.services.ingestion.parse_frontmatter",
        return_value=(None, {}, "content"),
    ):
        with patch(
            "obsidian_rag.services.ingestion.parse_tasks_from_content",
            return_value=[],
        ):
            with patch(
                "obsidian_rag.services.ingestion.should_chunk_document",
                return_value=False,
            ):
                with patch(
                    "obsidian_rag.services.ingestion.datetime"
                ) as mock_dt:
                    mock_dt.now.return_value = expected_time
                    mock_dt.UTC = UTC

                    ingestion_service._ingest_single_file(
                        file_info,
                        vault_id=vault_id,
                        vault_config=vault_config,
                    )

    assert existing_doc.ingested_at == expected_time


def test_ingest_single_file_integrity_recovery_updates_ingested_at(
    ingestion_service: IngestionService,
    tmp_path: Path,
) -> None:
    """IntegrityError recovery updates ingested_at on the recovered document."""
    vault_config = VaultConfig(
        container_path=str(tmp_path),
        host_path=str(tmp_path),
    )
    file_info = _create_file_info(tmp_path)
    vault_id = uuid.uuid4()

    mock_session = MagicMock()
    mock_session_context = MagicMock()
    mock_session_context.__enter__ = MagicMock(return_value=mock_session)
    mock_session_context.__exit__ = MagicMock(return_value=None)
    ingestion_service.db_manager.get_session.return_value = mock_session_context  # type: ignore[attr-defined]

    existing_doc = MagicMock()
    existing_doc.id = uuid.uuid4()
    existing_doc.tags = None

    mock_session.query.return_value.filter_by.return_value.first.side_effect = [
        None,
        existing_doc,
    ]

    err = _create_integrity_error("uq_document_vault_path")
    mock_session.flush.side_effect = err

    expected_time = datetime(2026, 6, 17, 12, 0, 0, tzinfo=UTC)

    with patch(
        "obsidian_rag.services.ingestion.parse_frontmatter",
        return_value=(None, {}, "content"),
    ):
        with patch(
            "obsidian_rag.services.ingestion.parse_tasks_from_content",
            return_value=[],
        ):
            with patch(
                "obsidian_rag.services.ingestion.should_chunk_document",
                return_value=False,
            ):
                with patch.object(
                    ingestion_service,
                    "_create_document",
                    return_value=(MagicMock(), 0),
                ):
                    with patch(
                        "obsidian_rag.services.ingestion.datetime"
                    ) as mock_dt:
                        mock_dt.now.return_value = expected_time
                        mock_dt.UTC = UTC

                        ingestion_service._ingest_single_file(
                            file_info,
                            vault_id=vault_id,
                            vault_config=vault_config,
                        )

    assert existing_doc.ingested_at == expected_time


def test_ingest_single_file_logs_lookup_diagnostic(
    ingestion_service: IngestionService,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """_ingest_single_file emits a debug log with vault_id and file_path before lookup."""
    vault_config = VaultConfig(
        container_path=str(tmp_path),
        host_path=str(tmp_path),
    )
    file_info = _create_file_info(tmp_path)
    vault_id = uuid.uuid4()

    mock_session = MagicMock()
    mock_session_context = MagicMock()
    mock_session_context.__enter__ = MagicMock(return_value=mock_session)
    mock_session_context.__exit__ = MagicMock(return_value=None)
    ingestion_service.db_manager.get_session.return_value = mock_session_context  # type: ignore[attr-defined]

    mock_session.query.return_value.filter_by.return_value.first.return_value = None

    with caplog.at_level(logging.DEBUG):
        with patch(
            "obsidian_rag.services.ingestion.parse_frontmatter",
            return_value=(None, {}, "content"),
        ):
            with patch(
                "obsidian_rag.services.ingestion.parse_tasks_from_content",
                return_value=[],
            ):
                with patch(
                    "obsidian_rag.services.ingestion.should_chunk_document",
                    return_value=False,
                ):
                    with patch.object(
                        ingestion_service,
                        "_create_document",
                        return_value=(MagicMock(), 0),
                    ):
                        with patch.object(
                            ingestion_service,
                            "_create_chunks_for_new_document",
                            return_value=0,
                        ):
                            ingestion_service._ingest_single_file(
                                file_info,
                                vault_id=vault_id,
                                vault_config=vault_config,
                            )

    assert "Looking up document:" in caplog.text
    assert str(vault_id) in caplog.text
    assert "test.md" in caplog.text


def test_ingest_single_file_diagnostic_log_uses_exact_values(
    ingestion_service: IngestionService,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The lookup diagnostic log uses actual vault_id and relative_path values."""
    vault_config = VaultConfig(
        container_path=str(tmp_path),
        host_path=str(tmp_path),
    )
    file_info = _create_file_info(tmp_path)
    vault_id = uuid.uuid4()

    mock_session = MagicMock()
    mock_session_context = MagicMock()
    mock_session_context.__enter__ = MagicMock(return_value=mock_session)
    mock_session_context.__exit__ = MagicMock(return_value=None)
    ingestion_service.db_manager.get_session.return_value = mock_session_context  # type: ignore[attr-defined]

    mock_session.query.return_value.filter_by.return_value.first.return_value = None

    with caplog.at_level(logging.DEBUG):
        with patch(
            "obsidian_rag.services.ingestion.parse_frontmatter",
            return_value=(None, {}, "content"),
        ):
            with patch(
                "obsidian_rag.services.ingestion.parse_tasks_from_content",
                return_value=[],
            ):
                with patch(
                    "obsidian_rag.services.ingestion.should_chunk_document",
                    return_value=False,
                ):
                    with patch.object(
                        ingestion_service,
                        "_create_document",
                        return_value=(MagicMock(), 0),
                    ):
                        with patch.object(
                            ingestion_service,
                            "_create_chunks_for_new_document",
                            return_value=0,
                        ):
                            ingestion_service._ingest_single_file(
                                file_info,
                                vault_id=vault_id,
                                vault_config=vault_config,
                            )

    # Find the exact log message
    diagnostic_records = [
        record
        for record in caplog.records
        if "Looking up document:" in record.message
    ]
    assert len(diagnostic_records) == 1
    log_message = diagnostic_records[0].message
    assert str(vault_id) in log_message
    assert "test.md" in log_message
    assert "vault_id=" in log_message
    assert "file_path=" in log_message
