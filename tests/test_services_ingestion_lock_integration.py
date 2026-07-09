"""Integration tests for lock acquire/release wiring in ingest_vault."""

import uuid
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
    settings.ingestion.max_chunk_chars = 24000
    settings.ingestion.chunk_overlap_chars = 800
    settings.ingestion.ingest_lock_ttl_seconds = 300
    settings.ingestion.ingest_lock_heartbeat_interval = 50
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


def _create_file_info(path: Path, idx: int = 0) -> FileInfo:
    """Create a FileInfo for testing."""
    return FileInfo(
        path=path / f"note_{idx}.md",
        name=f"note_{idx}.md",
        content=f"content {idx}",
        checksum=f"sum{idx}",
        created_at=MagicMock(),
        modified_at=MagicMock(),
    )


# ─── ingest_vault lock acquire/release tests ───


def test_ingest_vault_acquires_lock_on_start(
    ingestion_service: IngestionService,
    tmp_path: Path,
) -> None:
    """ingest_vault calls acquire_ingest_lock when not dry_run and vault_id exists."""
    vault_config = VaultConfig(container_path=str(tmp_path), host_path=str(tmp_path))
    vault_id = uuid.uuid4()

    mock_acquisition = MagicMock()
    mock_acquisition.vault_id = vault_id

    with patch(
        "obsidian_rag.services.ingestion_lock.acquire_ingest_lock",
        return_value=mock_acquisition,
    ) as mock_acquire:
        with patch.object(
            ingestion_service, "_resolve_vault_config", return_value=vault_config
        ):
            with patch.object(
                ingestion_service, "_get_or_create_vault", return_value=vault_id
            ):
                with patch.object(
                    ingestion_service, "_get_file_info_list", return_value=([], 0)
                ):
                    options = IngestVaultOptions(vault=vault_config)
                    ingestion_service.ingest_vault(tmp_path, options)

    mock_acquire.assert_called_once_with(
        ingestion_service.db_manager,
        vault_id,
        force=options.force,
        ttl_seconds=ingestion_service.settings.ingestion.ingest_lock_ttl_seconds,
    )


def test_ingest_vault_releases_lock_on_normal_completion(
    ingestion_service: IngestionService,
    tmp_path: Path,
) -> None:
    """ingest_vault releases lock with failed=False on happy path."""
    vault_config = VaultConfig(container_path=str(tmp_path), host_path=str(tmp_path))
    vault_id = uuid.uuid4()

    mock_acquisition = MagicMock()

    with patch(
        "obsidian_rag.services.ingestion_lock.acquire_ingest_lock",
        return_value=mock_acquisition,
    ):
        with patch(
            "obsidian_rag.services.ingestion.release_ingest_lock"
        ) as mock_release:
            with patch.object(
                ingestion_service, "_resolve_vault_config", return_value=vault_config
            ):
                with patch.object(
                    ingestion_service, "_get_or_create_vault", return_value=vault_id
                ):
                    with patch.object(
                        ingestion_service, "_get_file_info_list", return_value=([], 0)
                    ):
                        options = IngestVaultOptions(vault=vault_config)
                        ingestion_service.ingest_vault(tmp_path, options)

    mock_release.assert_called_once_with(
        ingestion_service.db_manager,
        vault_id,
        failed=False,
    )


def test_ingest_vault_releases_lock_failed_on_exception(
    ingestion_service: IngestionService,
    tmp_path: Path,
) -> None:
    """ingest_vault releases lock with failed=True when an exception occurs."""
    vault_config = VaultConfig(container_path=str(tmp_path), host_path=str(tmp_path))
    vault_id = uuid.uuid4()

    mock_acquisition = MagicMock()

    with patch(
        "obsidian_rag.services.ingestion_lock.acquire_ingest_lock",
        return_value=mock_acquisition,
    ):
        with patch(
            "obsidian_rag.services.ingestion.release_ingest_lock"
        ) as mock_release:
            with patch.object(
                ingestion_service, "_resolve_vault_config", return_value=vault_config
            ):
                with patch.object(
                    ingestion_service, "_get_or_create_vault", return_value=vault_id
                ):
                    with patch.object(
                        ingestion_service,
                        "_get_file_info_list",
                        side_effect=RuntimeError("boom"),
                    ):
                        options = IngestVaultOptions(vault=vault_config)
                        with pytest.raises(RuntimeError, match="boom"):
                            ingestion_service.ingest_vault(tmp_path, options)

    mock_release.assert_called_once_with(
        ingestion_service.db_manager,
        vault_id,
        failed=True,
    )


def test_ingest_vault_releases_lock_on_total_zero_early_return(
    ingestion_service: IngestionService,
    tmp_path: Path,
) -> None:
    """Empty file list early return still releases the lock."""
    vault_config = VaultConfig(container_path=str(tmp_path), host_path=str(tmp_path))
    vault_id = uuid.uuid4()

    mock_acquisition = MagicMock()

    with patch(
        "obsidian_rag.services.ingestion_lock.acquire_ingest_lock",
        return_value=mock_acquisition,
    ):
        with patch(
            "obsidian_rag.services.ingestion.release_ingest_lock"
        ) as mock_release:
            with patch.object(
                ingestion_service, "_resolve_vault_config", return_value=vault_config
            ):
                with patch.object(
                    ingestion_service, "_get_or_create_vault", return_value=vault_id
                ):
                    with patch.object(
                        ingestion_service, "_get_file_info_list", return_value=([], 0)
                    ):
                        options = IngestVaultOptions(vault=vault_config)
                        result = ingestion_service.ingest_vault(tmp_path, options)

    assert result.total == 0
    mock_release.assert_called_once_with(
        ingestion_service.db_manager,
        vault_id,
        failed=False,
    )


def test_ingest_vault_dry_run_does_not_acquire_lock(
    ingestion_service: IngestionService,
    tmp_path: Path,
) -> None:
    """When dry_run=True, acquire_ingest_lock is never called."""
    vault_config = VaultConfig(container_path=str(tmp_path), host_path=str(tmp_path))
    vault_id = uuid.uuid4()

    with patch(
        "obsidian_rag.services.ingestion_lock.acquire_ingest_lock"
    ) as mock_acquire:
        with patch.object(
            ingestion_service, "_resolve_vault_config", return_value=vault_config
        ):
            with patch.object(
                ingestion_service, "_get_or_create_vault", return_value=vault_id
            ):
                with patch.object(
                    ingestion_service, "_get_file_info_list", return_value=([], 0)
                ):
                    options = IngestVaultOptions(vault=vault_config, dry_run=True)
                    ingestion_service.ingest_vault(tmp_path, options)

    mock_acquire.assert_not_called()


def test_ingest_vault_no_op_skip_returns_synthetic_result(
    ingestion_service: IngestionService,
    tmp_path: Path,
) -> None:
    """When acquire_ingest_lock returns None, a synthetic skip result is returned."""
    vault_config = VaultConfig(container_path=str(tmp_path), host_path=str(tmp_path))
    vault_id = uuid.uuid4()

    with patch(
        "obsidian_rag.services.ingestion_lock.acquire_ingest_lock",
        return_value=None,
    ) as mock_acquire:
        with patch(
            "obsidian_rag.services.ingestion.release_ingest_lock"
        ) as mock_release:
            with patch.object(
                ingestion_service, "_resolve_vault_config", return_value=vault_config
            ):
                with patch.object(
                    ingestion_service, "_get_or_create_vault", return_value=vault_id
                ):
                    options = IngestVaultOptions(vault=vault_config)
                    result = ingestion_service.ingest_vault(tmp_path, options)

    mock_acquire.assert_called_once()
    mock_release.assert_not_called()
    assert result.total == 0
    assert "Skipped" in result.message
    assert result.skipped is True  # NEW (REQ-002)


# ─── heartbeat tests ───


def test_heartbeat_fires_every_n_files(
    ingestion_service: IngestionService,
    tmp_path: Path,
) -> None:
    """heartbeat_ingest_lock is called every heartbeat_interval files."""
    vault_config = VaultConfig(container_path=str(tmp_path), host_path=str(tmp_path))
    vault_id = uuid.uuid4()

    # Create 125 fake file infos
    file_infos = [_create_file_info(tmp_path, i) for i in range(125)]

    with patch(
        "obsidian_rag.services.ingestion_lock.heartbeat_ingest_lock",
        return_value=True,
    ) as mock_heartbeat:
        with patch.object(
            ingestion_service, "_ingest_single_file", return_value=("new", 0, False)
        ):
            lock_lost_flag = [False]
            ingestion_service._process_files_with_stats(
                file_infos,
                vault_id=vault_id,
                vault_config=vault_config,
                dry_run=False,
                progress_callback=None,
                force=False,
                heartbeat_interval=50,
            )

    # idx > 0 and idx % 50 == 0  =>  idx 50, 100  =>  2 calls
    assert mock_heartbeat.call_count == 2
    mock_heartbeat.assert_any_call(ingestion_service.db_manager, vault_id)
    assert lock_lost_flag[0] is False


def test_heartbeat_interval_zero_disables_heartbeat(
    ingestion_service: IngestionService,
    tmp_path: Path,
) -> None:
    """When heartbeat_interval=0, heartbeat_ingest_lock is never called."""
    vault_config = VaultConfig(container_path=str(tmp_path), host_path=str(tmp_path))
    vault_id = uuid.uuid4()

    file_infos = [_create_file_info(tmp_path, i) for i in range(10)]

    with patch(
        "obsidian_rag.services.ingestion_lock.heartbeat_ingest_lock"
    ) as mock_heartbeat:
        with patch.object(
            ingestion_service, "_ingest_single_file", return_value=("new", 0, False)
        ):
            ingestion_service._process_files_with_stats(
                file_infos,
                vault_id=vault_id,
                vault_config=vault_config,
                dry_run=False,
                progress_callback=None,
                force=False,
                heartbeat_interval=0,
            )

    mock_heartbeat.assert_not_called()


def test_heartbeat_skipped_when_vault_id_none(
    ingestion_service: IngestionService,
    tmp_path: Path,
) -> None:
    """heartbeat_ingest_lock is not called when vault_id is None."""
    vault_config = VaultConfig(container_path=str(tmp_path), host_path=str(tmp_path))

    file_infos = [_create_file_info(tmp_path, i) for i in range(10)]

    with patch(
        "obsidian_rag.services.ingestion_lock.heartbeat_ingest_lock"
    ) as mock_heartbeat:
        with patch.object(
            ingestion_service, "_ingest_single_file", return_value=("new", 0, False)
        ):
            ingestion_service._process_files_with_stats(
                file_infos,
                vault_id=None,
                vault_config=vault_config,
                dry_run=False,
                progress_callback=None,
                force=False,
                heartbeat_interval=50,
            )

    mock_heartbeat.assert_not_called()


def test_heartbeat_returns_false_breaks_loop_without_flag(
    ingestion_service: IngestionService,
    tmp_path: Path,
) -> None:
    """When heartbeat returns False and no flag container, loop breaks anyway."""
    vault_config = VaultConfig(container_path=str(tmp_path), host_path=str(tmp_path))
    vault_id = uuid.uuid4()

    file_infos = [_create_file_info(tmp_path, i) for i in range(75)]

    with patch(
        "obsidian_rag.services.ingestion_lock.heartbeat_ingest_lock",
        return_value=False,
    ) as mock_heartbeat:
        with patch.object(
            ingestion_service, "_ingest_single_file", return_value=("new", 0, False)
        ):
            stats = ingestion_service._process_files_with_stats(
                file_infos,
                vault_id=vault_id,
                vault_config=vault_config,
                dry_run=False,
                progress_callback=None,
                force=False,
                heartbeat_interval=50,
            )

    assert mock_heartbeat.call_count == 1
    assert stats["new"] == 51


def test_heartbeat_returns_false_breaks_loop(
    ingestion_service: IngestionService,
    tmp_path: Path,
) -> None:
    """When heartbeat returns False, the loop breaks early."""
    vault_config = VaultConfig(container_path=str(tmp_path), host_path=str(tmp_path))
    vault_id = uuid.uuid4()

    file_infos = [_create_file_info(tmp_path, i) for i in range(125)]

    with patch(
        "obsidian_rag.services.ingestion_lock.heartbeat_ingest_lock",
        return_value=False,
    ) as mock_heartbeat:
        with patch.object(
            ingestion_service, "_ingest_single_file", return_value=("new", 0, False)
        ):
            lock_lost_flag = [False]
            stats = ingestion_service._process_files_with_stats(
                file_infos,
                vault_id=vault_id,
                vault_config=vault_config,
                dry_run=False,
                progress_callback=None,
                force=False,
                heartbeat_interval=50,
                lock_lost_flag=lock_lost_flag,
            )

    # First heartbeat at idx 50 returns False -> break
    # By the time heartbeat fires, idx 50 has already been processed.
    assert mock_heartbeat.call_count == 1
    assert stats["new"] == 51  # processed idx 0..50
    assert lock_lost_flag[0] is True


def test_ingest_vault_passes_heartbeat_interval_to_process_files(
    ingestion_service: IngestionService,
    tmp_path: Path,
) -> None:
    """ingest_vault passes heartbeat_interval from settings to _process_files_with_stats."""
    vault_config = VaultConfig(container_path=str(tmp_path), host_path=str(tmp_path))
    vault_id = uuid.uuid4()

    mock_acquisition = MagicMock()
    file_infos = [_create_file_info(tmp_path, i) for i in range(2)]

    with patch(
        "obsidian_rag.services.ingestion_lock.acquire_ingest_lock",
        return_value=mock_acquisition,
    ):
        with patch("obsidian_rag.services.ingestion.release_ingest_lock"):
            with patch.object(
                ingestion_service, "_resolve_vault_config", return_value=vault_config
            ):
                with patch.object(
                    ingestion_service, "_get_or_create_vault", return_value=vault_id
                ):
                    with patch.object(
                        ingestion_service,
                        "_get_file_info_list",
                        return_value=(file_infos, len(file_infos)),
                    ):
                        with patch.object(
                            ingestion_service,
                            "_validate_files_in_vault",
                            return_value=None,
                        ):
                            with patch.object(
                                ingestion_service,
                                "_ingest_single_file",
                                return_value=("new", 0, False),
                            ):
                                with patch.object(
                                    ingestion_service,
                                    "_run_deletion_phase",
                                    return_value=(0, 0),
                                ):
                                    with patch.object(
                                        ingestion_service,
                                        "_process_files_with_stats",
                                        return_value={
                                            "new": 2,
                                            "updated": 0,
                                            "unchanged": 0,
                                            "errors": 0,
                                            "chunks_created": 0,
                                            "empty_documents": 0,
                                        },
                                    ) as mock_process:
                                        options = IngestVaultOptions(vault=vault_config)
                                        ingestion_service.ingest_vault(
                                            tmp_path, options
                                        )

    _, kwargs = mock_process.call_args
    assert kwargs["heartbeat_interval"] == 50


def test_ingest_vault_no_release_when_lock_lost(
    ingestion_service: IngestionService,
    tmp_path: Path,
) -> None:
    """If heartbeat loses the lock, release_ingest_lock is NOT called."""
    vault_config = VaultConfig(container_path=str(tmp_path), host_path=str(tmp_path))
    vault_id = uuid.uuid4()

    file_infos = [_create_file_info(tmp_path, i) for i in range(75)]
    mock_acquisition = MagicMock()

    with patch(
        "obsidian_rag.services.ingestion_lock.acquire_ingest_lock",
        return_value=mock_acquisition,
    ):
        with patch(
            "obsidian_rag.services.ingestion.release_ingest_lock"
        ) as mock_release:
            with patch(
                "obsidian_rag.services.ingestion_lock.heartbeat_ingest_lock",
                return_value=False,
            ):
                with patch.object(
                    ingestion_service,
                    "_resolve_vault_config",
                    return_value=vault_config,
                ):
                    with patch.object(
                        ingestion_service, "_get_or_create_vault", return_value=vault_id
                    ):
                        with patch.object(
                            ingestion_service,
                            "_get_file_info_list",
                            return_value=(file_infos, len(file_infos)),
                        ):
                            with patch.object(
                                ingestion_service,
                                "_validate_files_in_vault",
                                return_value=None,
                            ):
                                with patch.object(
                                    ingestion_service,
                                    "_ingest_single_file",
                                    return_value=("new", 0, False),
                                ):
                                    options = IngestVaultOptions(vault=vault_config)
                                    result = ingestion_service.ingest_vault(
                                        tmp_path, options
                                    )

    mock_release.assert_not_called()
    assert result.total == 75
    assert result.new < 75  # loop broke early at idx 50


def test_ingest_vault_vault_id_none_does_not_acquire_lock(
    ingestion_service: IngestionService,
    tmp_path: Path,
) -> None:
    """When vault_id is None (dry_run), lock is not acquired."""
    vault_config = VaultConfig(container_path=str(tmp_path), host_path=str(tmp_path))

    with patch(
        "obsidian_rag.services.ingestion_lock.acquire_ingest_lock"
    ) as mock_acquire:
        with patch.object(
            ingestion_service, "_resolve_vault_config", return_value=vault_config
        ):
            with patch.object(
                ingestion_service, "_get_or_create_vault", return_value=None
            ):
                with patch.object(
                    ingestion_service, "_get_file_info_list", return_value=([], 0)
                ):
                    options = IngestVaultOptions(vault=vault_config)
                    ingestion_service.ingest_vault(tmp_path, options)

    mock_acquire.assert_not_called()
