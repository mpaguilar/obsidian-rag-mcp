"""Tests for CLI vault update and delete commands."""

import uuid
from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from obsidian_rag.cli import cli


class TestVaultUpdate:
    """Tests for vault update command."""

    @patch("obsidian_rag.cli_vault_commands.DatabaseManager")
    @patch("obsidian_rag.cli_vault_commands._lookup_vault_by_name")
    @patch("obsidian_rag.cli_vault_commands._has_vault_changed")
    @patch("obsidian_rag.cli_vault_commands._count_vault_documents")
    def test_vault_update_description(
        self,
        mock_count_docs,
        mock_has_changed,
        mock_lookup,
        mock_db_class,
    ) -> None:
        """Verify vault update with description outputs correctly."""
        runner = CliRunner()

        # Create mock vault
        mock_vault = MagicMock()
        mock_vault.name = "TestVault"
        mock_vault.description = "Updated description"
        mock_vault.container_path = "/data/vault"
        mock_vault.host_path = "/host/vault"
        mock_vault.id = uuid.uuid4()
        mock_lookup.return_value = mock_vault
        mock_has_changed.return_value = True
        mock_count_docs.return_value = 5

        # Setup mock db_manager
        mock_db_manager = MagicMock()
        mock_db_manager.get_session.return_value.__enter__ = MagicMock(
            return_value=MagicMock()
        )
        mock_db_manager.get_session.return_value.__exit__ = MagicMock(return_value=None)
        mock_db_class.return_value = mock_db_manager

        # Mock settings with proper logging
        mock_settings = MagicMock()
        mock_settings.database.url = "postgresql+psycopg://localhost/test"
        mock_settings.database.pool_size = 5
        mock_settings.database.max_overflow = 10
        mock_settings.database.pool_timeout = 30
        mock_settings.database.pool_recycle = 3600
        mock_settings.logging.level = "INFO"
        mock_settings.logging.format = "text"

        with patch("obsidian_rag.cli.get_settings", return_value=mock_settings):
            with patch(
                "obsidian_rag.mcp_server.tools.vaults._check_container_path_update",
                return_value=None,
            ):
                with patch("obsidian_rag.mcp_server.tools.vaults._apply_vault_updates"):
                    with patch(
                        "obsidian_rag.mcp_server.tools.vaults._handle_flush_with_integrity_check",
                        return_value=None,
                    ):
                        result = runner.invoke(
                            cli,
                            [
                                "vault",
                                "update",
                                "--name",
                                "TestVault",
                                "--description",
                                "Updated description",
                            ],
                        )

        assert result.exit_code == 0
        assert "TestVault" in result.output

    @patch("obsidian_rag.cli_vault_commands.DatabaseManager")
    @patch("obsidian_rag.cli_vault_commands._lookup_vault_by_name")
    def test_vault_update_container_path_no_force(
        self,
        mock_lookup,
        mock_db_class,
    ) -> None:
        """Verify error displayed when changing container_path without force."""
        runner = CliRunner()

        # Create mock vault
        mock_vault = MagicMock()
        mock_vault.name = "TestVault"
        mock_vault.description = "Test vault"
        mock_vault.container_path = "/data/old"
        mock_vault.host_path = "/host/vault"
        mock_lookup.return_value = mock_vault

        # Setup mock db_manager
        mock_db_manager = MagicMock()
        mock_db_manager.get_session.return_value.__enter__ = MagicMock(
            return_value=MagicMock()
        )
        mock_db_manager.get_session.return_value.__exit__ = MagicMock(return_value=None)
        mock_db_class.return_value = mock_db_manager

        # Mock settings with proper logging
        mock_settings = MagicMock()
        mock_settings.database.url = "postgresql+psycopg://localhost/test"
        mock_settings.database.pool_size = 5
        mock_settings.database.max_overflow = 10
        mock_settings.database.pool_timeout = 30
        mock_settings.database.pool_recycle = 3600
        mock_settings.logging.level = "INFO"
        mock_settings.logging.format = "text"

        with patch("obsidian_rag.cli.get_settings", return_value=mock_settings):
            with patch(
                "obsidian_rag.cli_vault_commands._has_vault_changed",
                return_value=True,
            ):
                with patch(
                    "obsidian_rag.mcp_server.tools.vaults._check_container_path_update",
                    return_value={"success": False, "error": "force required"},
                ):
                    result = runner.invoke(
                        cli,
                        [
                            "vault",
                            "update",
                            "--name",
                            "TestVault",
                            "--container-path",
                            "/data/new",
                        ],
                    )

        assert result.exit_code == 1
        assert "force" in result.output.lower() or "Error" in result.output

    @patch("obsidian_rag.cli_vault_commands.DatabaseManager")
    @patch("obsidian_rag.cli_vault_commands._lookup_vault_by_name")
    @patch("obsidian_rag.cli_vault_commands._has_vault_changed")
    @patch("obsidian_rag.cli_vault_commands._count_vault_documents")
    def test_vault_update_container_path_with_force(
        self,
        mock_count_docs,
        mock_has_changed,
        mock_lookup,
        mock_db_class,
    ) -> None:
        """Verify warning and successful update when using force flag."""
        runner = CliRunner()

        # Create mock vault
        mock_vault = MagicMock()
        mock_vault.name = "TestVault"
        mock_vault.description = "Test vault"
        mock_vault.container_path = "/data/new"
        mock_vault.host_path = "/host/vault"
        mock_vault.id = uuid.uuid4()
        mock_lookup.return_value = mock_vault
        mock_has_changed.return_value = True
        mock_count_docs.return_value = 0

        # Setup mock db_manager
        mock_db_manager = MagicMock()
        mock_db_manager.get_session.return_value.__enter__ = MagicMock(
            return_value=MagicMock()
        )
        mock_db_manager.get_session.return_value.__exit__ = MagicMock(return_value=None)
        mock_db_class.return_value = mock_db_manager

        # Mock settings with proper logging
        mock_settings = MagicMock()
        mock_settings.database.url = "postgresql+psycopg://localhost/test"
        mock_settings.database.pool_size = 5
        mock_settings.database.max_overflow = 10
        mock_settings.database.pool_timeout = 30
        mock_settings.database.pool_recycle = 3600
        mock_settings.logging.level = "INFO"
        mock_settings.logging.format = "text"

        with patch("obsidian_rag.cli.get_settings", return_value=mock_settings):
            with patch(
                "obsidian_rag.mcp_server.tools.vaults._check_container_path_update",
                return_value=None,
            ):
                with patch("obsidian_rag.mcp_server.tools.vaults._apply_vault_updates"):
                    with patch(
                        "obsidian_rag.mcp_server.tools.vaults._handle_flush_with_integrity_check",
                        return_value=None,
                    ):
                        result = runner.invoke(
                            cli,
                            [
                                "vault",
                                "update",
                                "--name",
                                "TestVault",
                                "--container-path",
                                "/data/new",
                                "--force",
                            ],
                        )

        assert result.exit_code == 0
        assert "TestVault" in result.output


class TestVaultDelete:
    """Tests for vault delete command."""

    @patch("obsidian_rag.cli_vault_commands.DatabaseManager")
    @patch("obsidian_rag.cli_vault_commands._lookup_vault_by_name")
    @patch("obsidian_rag.cli_vault_commands._count_vault_cascade_targets")
    def test_vault_delete_confirmed(
        self,
        mock_count_cascade,
        mock_lookup,
        mock_db_class,
    ) -> None:
        """Verify warning and cascade counts displayed when confirmed."""
        runner = CliRunner()

        # Create mock vault
        mock_vault = MagicMock()
        mock_vault.name = "TestVault"
        mock_vault.id = uuid.uuid4()
        mock_lookup.return_value = mock_vault
        mock_count_cascade.return_value = (10, 25, 50)

        # Setup mock session
        mock_session = MagicMock()

        # Setup mock db_manager
        mock_db_manager = MagicMock()
        mock_db_manager.get_session.return_value.__enter__ = MagicMock(
            return_value=mock_session
        )
        mock_db_manager.get_session.return_value.__exit__ = MagicMock(return_value=None)
        mock_db_class.return_value = mock_db_manager

        # Mock settings with proper logging
        mock_settings = MagicMock()
        mock_settings.database.url = "postgresql+psycopg://localhost/test"
        mock_settings.database.pool_size = 5
        mock_settings.database.max_overflow = 10
        mock_settings.database.pool_timeout = 30
        mock_settings.database.pool_recycle = 3600
        mock_settings.logging.level = "INFO"
        mock_settings.logging.format = "text"

        with patch("obsidian_rag.cli.get_settings", return_value=mock_settings):
            result = runner.invoke(
                cli,
                [
                    "vault",
                    "delete",
                    "--name",
                    "TestVault",
                    "--confirm",
                ],
            )

        assert result.exit_code == 0
        assert "deleted" in result.output.lower()
        assert "10" in result.output

    def test_vault_delete_not_confirmed(self) -> None:
        """Verify error displayed when not confirmed."""
        runner = CliRunner()

        # Mock settings with proper logging
        mock_settings = MagicMock()
        mock_settings.database.url = "postgresql+psycopg://localhost/test"
        mock_settings.database.pool_size = 5
        mock_settings.database.max_overflow = 10
        mock_settings.database.pool_timeout = 30
        mock_settings.database.pool_recycle = 3600
        mock_settings.logging.level = "INFO"
        mock_settings.logging.format = "text"

        with patch("obsidian_rag.cli.get_settings", return_value=mock_settings):
            result = runner.invoke(
                cli,
                [
                    "vault",
                    "delete",
                    "--name",
                    "TestVault",
                ],
            )

        assert result.exit_code == 1
        assert "confirm" in result.output.lower()

    @patch("obsidian_rag.cli_vault_commands.DatabaseManager")
    @patch("obsidian_rag.cli_vault_commands._lookup_vault_by_name")
    def test_vault_delete_vault_not_found(
        self,
        mock_lookup,
        mock_db_class,
    ) -> None:
        """Verify error when deleting non-existent vault (TASK-025)."""
        runner = CliRunner()

        # Vault not found
        mock_lookup.return_value = None

        # Setup mock db_manager
        mock_db_manager = MagicMock()
        mock_db_manager.get_session.return_value.__enter__ = MagicMock(
            return_value=MagicMock()
        )
        mock_db_manager.get_session.return_value.__exit__ = MagicMock(return_value=None)
        mock_db_class.return_value = mock_db_manager

        # Mock settings with proper logging
        mock_settings = MagicMock()
        mock_settings.database.url = "postgresql+psycopg://localhost/test"
        mock_settings.database.pool_size = 5
        mock_settings.database.max_overflow = 10
        mock_settings.database.pool_timeout = 30
        mock_settings.database.pool_recycle = 3600
        mock_settings.logging.level = "INFO"
        mock_settings.logging.format = "text"

        with patch("obsidian_rag.cli.get_settings", return_value=mock_settings):
            result = runner.invoke(
                cli,
                [
                    "vault",
                    "delete",
                    "--name",
                    "NonExistentVault",
                    "--confirm",
                ],
            )

        assert result.exit_code == 1
        assert "not found" in result.output.lower()


class TestVaultUpdateEdgeCases:
    """Tests for vault update command edge cases."""

    @patch("obsidian_rag.cli_vault_commands.DatabaseManager")
    @patch("obsidian_rag.cli_vault_commands._lookup_vault_by_name")
    def test_vault_update_vault_not_found(
        self,
        mock_lookup,
        mock_db_class,
    ) -> None:
        """Verify error when updating non-existent vault (TASK-025)."""
        runner = CliRunner()

        # Vault not found
        mock_lookup.return_value = None

        # Setup mock db_manager
        mock_db_manager = MagicMock()
        mock_db_manager.get_session.return_value.__enter__ = MagicMock(
            return_value=MagicMock()
        )
        mock_db_manager.get_session.return_value.__exit__ = MagicMock(return_value=None)
        mock_db_class.return_value = mock_db_manager

        # Mock settings with proper logging
        mock_settings = MagicMock()
        mock_settings.database.url = "postgresql+psycopg://localhost/test"
        mock_settings.database.pool_size = 5
        mock_settings.database.max_overflow = 10
        mock_settings.database.pool_timeout = 30
        mock_settings.database.pool_recycle = 3600
        mock_settings.logging.level = "INFO"
        mock_settings.logging.format = "text"

        with patch("obsidian_rag.cli.get_settings", return_value=mock_settings):
            result = runner.invoke(
                cli,
                [
                    "vault",
                    "update",
                    "--name",
                    "NonExistentVault",
                    "--description",
                    "New description",
                ],
            )

        assert result.exit_code == 1
        assert "not found" in result.output.lower()

    @patch("obsidian_rag.cli_vault_commands.DatabaseManager")
    @patch("obsidian_rag.cli_vault_commands._lookup_vault_by_name")
    @patch("obsidian_rag.cli_vault_commands._has_vault_changed")
    @patch("obsidian_rag.cli_vault_commands._count_vault_documents")
    def test_vault_update_no_changes(
        self,
        mock_count_docs,
        mock_has_changed,
        mock_lookup,
        mock_db_class,
    ) -> None:
        """Verify message when no changes to apply (TASK-025)."""
        runner = CliRunner()

        # Create mock vault
        mock_vault = MagicMock()
        mock_vault.name = "TestVault"
        mock_vault.description = "Same description"
        mock_vault.container_path = "/data/vault"
        mock_vault.host_path = "/host/vault"
        mock_vault.id = uuid.uuid4()
        mock_lookup.return_value = mock_vault
        mock_has_changed.return_value = False  # No changes
        mock_count_docs.return_value = 5

        # Setup mock db_manager
        mock_db_manager = MagicMock()
        mock_db_manager.get_session.return_value.__enter__ = MagicMock(
            return_value=MagicMock()
        )
        mock_db_manager.get_session.return_value.__exit__ = MagicMock(return_value=None)
        mock_db_class.return_value = mock_db_manager

        # Mock settings with proper logging
        mock_settings = MagicMock()
        mock_settings.database.url = "postgresql+psycopg://localhost/test"
        mock_settings.database.pool_size = 5
        mock_settings.database.max_overflow = 10
        mock_settings.database.pool_timeout = 30
        mock_settings.database.pool_recycle = 3600
        mock_settings.logging.level = "INFO"
        mock_settings.logging.format = "text"

        with patch("obsidian_rag.cli.get_settings", return_value=mock_settings):
            result = runner.invoke(
                cli,
                [
                    "vault",
                    "update",
                    "--name",
                    "TestVault",
                    "--description",
                    "Same description",
                ],
            )

        assert result.exit_code == 0
        assert "No changes" in result.output

    @patch("obsidian_rag.cli_vault_commands.DatabaseManager")
    @patch("obsidian_rag.cli_vault_commands._lookup_vault_by_name")
    @patch("obsidian_rag.cli_vault_commands._has_vault_changed")
    @patch("obsidian_rag.cli_vault_commands._count_vault_documents")
    def test_vault_update_integrity_error(
        self,
        mock_count_docs,
        mock_has_changed,
        mock_lookup,
        mock_db_class,
    ) -> None:
        """Verify error handling for integrity error during update (TASK-025)."""
        runner = CliRunner()

        # Create mock vault
        mock_vault = MagicMock()
        mock_vault.name = "TestVault"
        mock_vault.description = "Test vault"
        mock_vault.container_path = "/data/vault"
        mock_vault.host_path = "/host/vault"
        mock_vault.id = uuid.uuid4()
        mock_lookup.return_value = mock_vault
        mock_has_changed.return_value = True
        mock_count_docs.return_value = 5

        # Setup mock db_manager
        mock_db_manager = MagicMock()
        mock_db_manager.get_session.return_value.__enter__ = MagicMock(
            return_value=MagicMock()
        )
        mock_db_manager.get_session.return_value.__exit__ = MagicMock(return_value=None)
        mock_db_class.return_value = mock_db_manager

        # Mock settings with proper logging
        mock_settings = MagicMock()
        mock_settings.database.url = "postgresql+psycopg://localhost/test"
        mock_settings.database.pool_size = 5
        mock_settings.database.max_overflow = 10
        mock_settings.database.pool_timeout = 30
        mock_settings.database.pool_recycle = 3600
        mock_settings.logging.level = "INFO"
        mock_settings.logging.format = "text"

        with patch("obsidian_rag.cli.get_settings", return_value=mock_settings):
            with patch(
                "obsidian_rag.cli_vault_commands._check_container_path_update",
                return_value=None,
            ):
                with patch(
                    "obsidian_rag.cli_vault_commands._apply_vault_updates",
                ):
                    with patch(
                        "obsidian_rag.cli_vault_commands._handle_flush_with_integrity_check",
                        return_value={"success": False, "error": "Integrity error"},
                    ):
                        result = runner.invoke(
                            cli,
                            [
                                "vault",
                                "update",
                                "--name",
                                "TestVault",
                                "--description",
                                "New description",
                            ],
                        )

        assert result.exit_code == 1
        assert "Integrity error" in result.output or "Error" in result.output
