"""Tests for CLI ingest lock error handling."""

from pathlib import Path
from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from obsidian_rag.cli import cli
from obsidian_rag.config import VaultConfig
from obsidian_rag.services.ingestion import IngestionResult
from obsidian_rag.services.ingestion_lock import IngestLockError


def _setup_ingest_mocks(
    mock_scan,
    mock_process,
    mock_db_manager,
    mock_get_provider,
    mock_ingestion_service,
    vault_path_str: str,
) -> tuple[MagicMock, MagicMock]:
    """Set up common mocks for CLI ingest tests."""
    mock_scan.return_value = [Path("test.md")]
    mock_process.return_value = []

    mock_settings = MagicMock()
    mock_settings.database.url = "postgresql+psycopg://localhost/test"
    mock_settings.ingestion.batch_size = 100
    mock_settings.ingestion.progress_interval = 10
    mock_settings.logging.level = "INFO"
    mock_settings.logging.format = "text"

    vault_config = VaultConfig(
        container_path=vault_path_str,
        host_path=vault_path_str,
    )
    mock_settings.get_vault.return_value = vault_config
    mock_settings.get_vault_names.return_value = ["test-vault"]

    mock_embedding_provider = MagicMock()
    mock_get_provider.return_value = mock_embedding_provider

    mock_db_instance = MagicMock()
    mock_db_manager.return_value = mock_db_instance

    mock_service = MagicMock()
    mock_ingestion_service.return_value = mock_service
    return mock_settings, mock_service


@patch("obsidian_rag.cli_commands._scan_vault")
@patch("obsidian_rag.cli_commands.process_files_in_batches")
@patch("obsidian_rag.cli_commands.DatabaseManager")
@patch("obsidian_rag.cli_commands._get_embedding_provider")
@patch("obsidian_rag.cli_commands.IngestionService")
def test_cli_ingest_lock_error_prints_to_stderr_and_returns(
    mock_ingestion_service,
    mock_get_provider,
    mock_db_manager,
    mock_process,
    mock_scan,
):
    """Patch ingest_vault to raise IngestLockError; assert stderr message and exit 0."""
    runner = CliRunner()

    with runner.isolated_filesystem() as fs:
        vault_path = Path(fs) / "vault"
        vault_path.mkdir()
        test_file = vault_path / "test.md"
        test_file.write_text("# Test")
        vault_path_str = str(vault_path)

        mock_settings, mock_service = _setup_ingest_mocks(
            mock_scan,
            mock_process,
            mock_db_manager,
            mock_get_provider,
            mock_ingestion_service,
            vault_path_str,
        )

        lock_msg = "Ingest already in progress for vault 'test-vault' (started at 2026-07-09 10:00:00, PID 1234, force=False). Wait for completion or reclaim the stale lock."
        mock_service.ingest_vault.side_effect = IngestLockError(lock_msg)

        with patch("obsidian_rag.cli.get_settings", return_value=mock_settings):
            result = runner.invoke(
                cli,
                ["ingest", vault_path_str, "--vault", "test-vault"],
            )

    assert result.exit_code == 0
    assert "Error:" in result.stderr
    assert lock_msg in result.stderr
    assert "Traceback" not in result.stdout
    assert "Traceback" not in result.stderr


@patch("obsidian_rag.cli_commands._scan_vault")
@patch("obsidian_rag.cli_commands.process_files_in_batches")
@patch("obsidian_rag.cli_commands.DatabaseManager")
@patch("obsidian_rag.cli_commands._get_embedding_provider")
@patch("obsidian_rag.cli_commands.IngestionService")
def test_cli_ingest_normal_when_lock_acquired(
    mock_ingestion_service,
    mock_get_provider,
    mock_db_manager,
    mock_process,
    mock_scan,
):
    """Patch ingest_vault to return normally; assert no error output."""
    runner = CliRunner()

    with runner.isolated_filesystem() as fs:
        vault_path = Path(fs) / "vault"
        vault_path.mkdir()
        test_file = vault_path / "test.md"
        test_file.write_text("# Test")
        vault_path_str = str(vault_path)

        mock_settings, mock_service = _setup_ingest_mocks(
            mock_scan,
            mock_process,
            mock_db_manager,
            mock_get_provider,
            mock_ingestion_service,
            vault_path_str,
        )

        mock_result = IngestionResult(
            total=1,
            new=1,
            updated=0,
            unchanged=0,
            errors=0,
            deleted=0,
            chunks_created=0,
            empty_documents=0,
            processing_time_seconds=1.0,
            message="Ingest completed",
        )
        mock_service.ingest_vault.return_value = mock_result

        with patch("obsidian_rag.cli.get_settings", return_value=mock_settings):
            result = runner.invoke(
                cli,
                ["ingest", vault_path_str, "--vault", "test-vault"],
            )

    assert result.exit_code == 0
    assert "Error:" not in result.stderr
    assert "Successfully ingested" in result.stdout


@patch("obsidian_rag.cli_commands._scan_vault")
@patch("obsidian_rag.cli_commands.process_files_in_batches")
@patch("obsidian_rag.cli_commands.DatabaseManager")
@patch("obsidian_rag.cli_commands._get_embedding_provider")
@patch("obsidian_rag.cli_commands.IngestionService")
def test_cli_ingest_dry_run_no_lock_error(
    mock_ingestion_service,
    mock_get_provider,
    mock_db_manager,
    mock_process,
    mock_scan,
):
    """Dry-run path does not trigger lock error when ingest_vault returns normally."""
    runner = CliRunner()

    with runner.isolated_filesystem() as fs:
        vault_path = Path(fs) / "vault"
        vault_path.mkdir()
        test_file = vault_path / "test.md"
        test_file.write_text("# Test")
        vault_path_str = str(vault_path)

        mock_settings, mock_service = _setup_ingest_mocks(
            mock_scan,
            mock_process,
            mock_db_manager,
            mock_get_provider,
            mock_ingestion_service,
            vault_path_str,
        )

        mock_result = IngestionResult(
            total=1,
            new=0,
            updated=0,
            unchanged=1,
            errors=0,
            deleted=0,
            chunks_created=0,
            empty_documents=0,
            processing_time_seconds=1.0,
            message="Dry run completed",
        )
        mock_service.ingest_vault.return_value = mock_result

        with patch("obsidian_rag.cli.get_settings", return_value=mock_settings):
            result = runner.invoke(
                cli,
                ["ingest", vault_path_str, "--vault", "test-vault", "--dry-run"],
            )

    assert result.exit_code == 0
    assert "Error:" not in result.stderr
    assert "DRY RUN" in result.stdout
