"""Tests for CLI vault list and get commands."""

import json
import uuid
from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from obsidian_rag.cli import cli


class TestVaultGroup:
    """Tests for the vault Click group."""

    def test_vault_group_exists(self) -> None:
        """Verify that the vault group command exists."""
        runner = CliRunner()
        result = runner.invoke(cli, ["vault", "--help"])

        assert result.exit_code == 0
        assert "Manage Obsidian vaults" in result.output

    def test_vault_list_command_exists(self) -> None:
        """Verify that the vault list subcommand exists."""
        runner = CliRunner()
        result = runner.invoke(cli, ["vault", "list", "--help"])

        assert result.exit_code == 0

    def test_vault_get_command_exists(self) -> None:
        """Verify that the vault get subcommand exists."""
        runner = CliRunner()
        result = runner.invoke(cli, ["vault", "get", "--help"])

        assert result.exit_code == 0


class TestVaultList:
    """Tests for vault list command."""

    @patch("obsidian_rag.cli_vault_commands.DatabaseManager")
    def test_vault_list_table_format(self, mock_db_manager_class) -> None:
        """Test vault list with table output format."""
        runner = CliRunner()

        # Create mock vault data
        vault_id = uuid.uuid4()
        mock_vault = MagicMock()
        mock_vault.id = vault_id
        mock_vault.name = "Test Vault"
        mock_vault.description = "A test vault"
        mock_vault.container_path = "/data/test"
        mock_vault.host_path = "/home/user/test"
        mock_vault.created_at = datetime(2024, 1, 15, 10, 30, 0, tzinfo=UTC)

        # Mock the query result (vault, document_count)
        mock_result = [(mock_vault, 42)]

        # Setup mock session and query chain
        mock_session = MagicMock()
        mock_subquery = MagicMock()
        mock_subquery.c = MagicMock()
        mock_subquery.c.vault_id = "vault_id"
        mock_subquery.c.doc_count = "doc_count"

        mock_session.query.return_value.group_by.return_value.subquery.return_value = (
            mock_subquery
        )

        # Setup the main query chain
        mock_query = MagicMock()
        mock_query.outerjoin.return_value.order_by.return_value = mock_query
        mock_query.count.return_value = 1
        mock_query.offset.return_value.limit.return_value.all.return_value = mock_result

        mock_session.query.return_value = mock_query

        # Setup mock db_manager
        mock_db_manager = MagicMock()
        mock_db_manager.get_session.return_value.__enter__ = MagicMock(
            return_value=mock_session
        )
        mock_db_manager.get_session.return_value.__exit__ = MagicMock(return_value=None)
        mock_db_manager_class.return_value = mock_db_manager

        # Mock settings
        mock_settings = MagicMock()
        mock_settings.database.url = "postgresql+psycopg://localhost/test"
        mock_settings.database.pool_size = 5
        mock_settings.database.max_overflow = 10
        mock_settings.database.pool_timeout = 30
        mock_settings.database.pool_recycle = 3600
        mock_settings.logging.level = "INFO"
        mock_settings.logging.format = "text"

        with patch("obsidian_rag.cli.get_settings", return_value=mock_settings):
            result = runner.invoke(cli, ["vault", "list"])

        assert result.exit_code == 0
        assert "Test Vault" in result.output
        assert "A test vault" in result.output
        assert "/data/test" in result.output
        assert "/home/user/test" in result.output
        assert "42" in result.output  # document count

    @patch("obsidian_rag.cli_vault_commands.DatabaseManager")
    def test_vault_list_json_format(self, mock_db_manager_class) -> None:
        """Test vault list with JSON output format."""
        runner = CliRunner()

        # Create mock vault data
        vault_id = uuid.uuid4()
        mock_vault = MagicMock()
        mock_vault.id = vault_id
        mock_vault.name = "Test Vault"
        mock_vault.description = "A test vault"
        mock_vault.container_path = "/data/test"
        mock_vault.host_path = "/home/user/test"
        mock_vault.created_at = datetime(2024, 1, 15, 10, 30, 0, tzinfo=UTC)

        # Mock the query result (vault, document_count)
        mock_result = [(mock_vault, 42)]

        # Setup mock session and query chain
        mock_session = MagicMock()
        mock_subquery = MagicMock()
        mock_subquery.c = MagicMock()
        mock_subquery.c.vault_id = "vault_id"
        mock_subquery.c.doc_count = "doc_count"

        mock_session.query.return_value.group_by.return_value.subquery.return_value = (
            mock_subquery
        )

        # Setup the main query chain
        mock_query = MagicMock()
        mock_query.outerjoin.return_value.order_by.return_value = mock_query
        mock_query.count.return_value = 1
        mock_query.offset.return_value.limit.return_value.all.return_value = mock_result

        mock_session.query.return_value = mock_query

        # Setup mock db_manager
        mock_db_manager = MagicMock()
        mock_db_manager.get_session.return_value.__enter__ = MagicMock(
            return_value=mock_session
        )
        mock_db_manager.get_session.return_value.__exit__ = MagicMock(return_value=None)
        mock_db_manager_class.return_value = mock_db_manager

        # Mock settings
        mock_settings = MagicMock()
        mock_settings.database.url = "postgresql+psycopg://localhost/test"
        mock_settings.database.pool_size = 5
        mock_settings.database.max_overflow = 10
        mock_settings.database.pool_timeout = 30
        mock_settings.database.pool_recycle = 3600
        mock_settings.logging.level = "INFO"
        mock_settings.logging.format = "text"

        with patch("obsidian_rag.cli.get_settings", return_value=mock_settings):
            result = runner.invoke(cli, ["vault", "list", "--format", "json"])

        assert result.exit_code == 0
        # Parse JSON output
        output_data = json.loads(result.output)
        assert len(output_data) == 1
        assert output_data[0]["name"] == "Test Vault"
        assert output_data[0]["description"] == "A test vault"
        assert output_data[0]["container_path"] == "/data/test"
        assert output_data[0]["host_path"] == "/home/user/test"
        assert output_data[0]["document_count"] == 42

    @patch("obsidian_rag.cli_vault_commands.DatabaseManager")
    def test_vault_list_empty(self, mock_db_manager_class) -> None:
        """Test vault list with no vaults."""
        runner = CliRunner()

        # Setup mock session and query chain
        mock_session = MagicMock()
        mock_subquery = MagicMock()
        mock_subquery.c = MagicMock()
        mock_subquery.c.vault_id = "vault_id"
        mock_subquery.c.doc_count = "doc_count"

        mock_session.query.return_value.group_by.return_value.subquery.return_value = (
            mock_subquery
        )

        # Setup the main query chain
        mock_query = MagicMock()
        mock_query.outerjoin.return_value.order_by.return_value = mock_query
        mock_query.count.return_value = 0
        mock_query.offset.return_value.limit.return_value.all.return_value = []

        mock_session.query.return_value = mock_query

        # Setup mock db_manager
        mock_db_manager = MagicMock()
        mock_db_manager.get_session.return_value.__enter__ = MagicMock(
            return_value=mock_session
        )
        mock_db_manager.get_session.return_value.__exit__ = MagicMock(return_value=None)
        mock_db_manager_class.return_value = mock_db_manager

        # Mock settings
        mock_settings = MagicMock()
        mock_settings.database.url = "postgresql+psycopg://localhost/test"
        mock_settings.database.pool_size = 5
        mock_settings.database.max_overflow = 10
        mock_settings.database.pool_timeout = 30
        mock_settings.database.pool_recycle = 3600
        mock_settings.logging.level = "INFO"
        mock_settings.logging.format = "text"

        with patch("obsidian_rag.cli.get_settings", return_value=mock_settings):
            result = runner.invoke(cli, ["vault", "list"])

        assert result.exit_code == 0
        assert "No vaults found" in result.output

    @patch("obsidian_rag.cli_vault_commands.DatabaseManager")
    def test_vault_list_pagination(self, mock_db_manager_class) -> None:
        """Test vault list with pagination options."""
        runner = CliRunner()

        # Create mock vault data
        mock_vault = MagicMock()
        mock_vault.id = uuid.uuid4()
        mock_vault.name = "Test Vault"
        mock_vault.description = None
        mock_vault.container_path = "/data/test"
        mock_vault.host_path = "/home/user/test"
        mock_vault.created_at = datetime(2024, 1, 15, 10, 30, 0, tzinfo=UTC)

        mock_result = [(mock_vault, 10)]

        # Setup mock session and query chain
        mock_session = MagicMock()
        mock_subquery = MagicMock()
        mock_subquery.c = MagicMock()
        mock_subquery.c.vault_id = "vault_id"
        mock_subquery.c.doc_count = "doc_count"

        mock_session.query.return_value.group_by.return_value.subquery.return_value = (
            mock_subquery
        )

        # Setup the main query chain
        mock_query = MagicMock()
        mock_query.outerjoin.return_value.order_by.return_value = mock_query
        mock_query.count.return_value = 1
        mock_query.offset.return_value.limit.return_value.all.return_value = mock_result

        mock_session.query.return_value = mock_query

        # Setup mock db_manager
        mock_db_manager = MagicMock()
        mock_db_manager.get_session.return_value.__enter__ = MagicMock(
            return_value=mock_session
        )
        mock_db_manager.get_session.return_value.__exit__ = MagicMock(return_value=None)
        mock_db_manager_class.return_value = mock_db_manager

        # Mock settings
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
                cli, ["vault", "list", "--limit", "5", "--offset", "10"]
            )

        assert result.exit_code == 0
        # Verify that offset and limit were called
        mock_query.offset.assert_called_once_with(10)
        mock_query.offset.return_value.limit.assert_called_once_with(5)

    @patch("obsidian_rag.cli_vault_commands.DatabaseManager")
    def test_vault_list_limit_validation(self, mock_db_manager_class) -> None:
        """Test vault list with limit validation (edge cases)."""
        runner = CliRunner()

        # Test limit < 1 (should be clamped to 1)
        mock_vault = MagicMock()
        mock_vault.id = uuid.uuid4()
        mock_vault.name = "Test Vault"
        mock_vault.description = None
        mock_vault.container_path = "/data/test"
        mock_vault.host_path = "/home/user/test"
        mock_vault.created_at = datetime(2024, 1, 15, 10, 30, 0, tzinfo=UTC)

        mock_result = [(mock_vault, 10)]

        # Setup mock session and query chain
        mock_session = MagicMock()
        mock_subquery = MagicMock()
        mock_subquery.c = MagicMock()
        mock_subquery.c.vault_id = "vault_id"
        mock_subquery.c.doc_count = "doc_count"

        mock_session.query.return_value.group_by.return_value.subquery.return_value = (
            mock_subquery
        )

        # Setup the main query chain
        mock_query = MagicMock()
        mock_query.outerjoin.return_value.order_by.return_value = mock_query
        mock_query.count.return_value = 1
        mock_query.offset.return_value.limit.return_value.all.return_value = mock_result

        mock_session.query.return_value = mock_query

        # Setup mock db_manager
        mock_db_manager = MagicMock()
        mock_db_manager.get_session.return_value.__enter__ = MagicMock(
            return_value=mock_session
        )
        mock_db_manager.get_session.return_value.__exit__ = MagicMock(return_value=None)
        mock_db_manager_class.return_value = mock_db_manager

        # Mock settings
        mock_settings = MagicMock()
        mock_settings.database.url = "postgresql+psycopg://localhost/test"
        mock_settings.database.pool_size = 5
        mock_settings.database.max_overflow = 10
        mock_settings.database.pool_timeout = 30
        mock_settings.database.pool_recycle = 3600
        mock_settings.logging.level = "INFO"
        mock_settings.logging.format = "text"

        # Test with limit=0 (should be clamped to 1)
        with patch("obsidian_rag.cli.get_settings", return_value=mock_settings):
            result = runner.invoke(cli, ["vault", "list", "--limit", "0"])

        assert result.exit_code == 0
        # Verify that limit was clamped to 1
        mock_query.offset.return_value.limit.assert_called_with(1)

        # Reset mock for next test
        mock_query.reset_mock()
        mock_query.offset.return_value.limit.return_value.all.return_value = mock_result

        # Test with limit=20000 (should be clamped to 10000)
        with patch("obsidian_rag.cli.get_settings", return_value=mock_settings):
            result = runner.invoke(cli, ["vault", "list", "--limit", "20000"])

        assert result.exit_code == 0
        # Verify that limit was clamped to 10000
        mock_query.offset.return_value.limit.assert_called_with(10000)

    @patch("obsidian_rag.cli_vault_commands.DatabaseManager")
    def test_vault_list_offset_validation(self, mock_db_manager_class) -> None:
        """Test vault list with offset validation (edge cases)."""
        runner = CliRunner()

        mock_vault = MagicMock()
        mock_vault.id = uuid.uuid4()
        mock_vault.name = "Test Vault"
        mock_vault.description = None
        mock_vault.container_path = "/data/test"
        mock_vault.host_path = "/home/user/test"
        mock_vault.created_at = datetime(2024, 1, 15, 10, 30, 0, tzinfo=UTC)

        mock_result = [(mock_vault, 10)]

        # Setup mock session and query chain
        mock_session = MagicMock()
        mock_subquery = MagicMock()
        mock_subquery.c = MagicMock()
        mock_subquery.c.vault_id = "vault_id"
        mock_subquery.c.doc_count = "doc_count"

        mock_session.query.return_value.group_by.return_value.subquery.return_value = (
            mock_subquery
        )

        # Setup the main query chain
        mock_query = MagicMock()
        mock_query.outerjoin.return_value.order_by.return_value = mock_query
        mock_query.count.return_value = 1
        mock_query.offset.return_value.limit.return_value.all.return_value = mock_result

        mock_session.query.return_value = mock_query

        # Setup mock db_manager
        mock_db_manager = MagicMock()
        mock_db_manager.get_session.return_value.__enter__ = MagicMock(
            return_value=mock_session
        )
        mock_db_manager.get_session.return_value.__exit__ = MagicMock(return_value=None)
        mock_db_manager_class.return_value = mock_db_manager

        # Mock settings
        mock_settings = MagicMock()
        mock_settings.database.url = "postgresql+psycopg://localhost/test"
        mock_settings.database.pool_size = 5
        mock_settings.database.max_overflow = 10
        mock_settings.database.pool_timeout = 30
        mock_settings.database.pool_recycle = 3600
        mock_settings.logging.level = "INFO"
        mock_settings.logging.format = "text"

        # Test with offset=-5 (should be clamped to 0)
        with patch("obsidian_rag.cli.get_settings", return_value=mock_settings):
            result = runner.invoke(cli, ["vault", "list", "--offset", "-5"])

        assert result.exit_code == 0
        # Verify that offset was clamped to 0
        mock_query.offset.assert_called_with(0)


class TestVaultGet:
    """Tests for vault get command."""

    @patch("obsidian_rag.cli_vault_commands.DatabaseManager")
    def test_vault_get_by_name(self, mock_db_manager_class) -> None:
        """Test getting a vault by name."""
        runner = CliRunner()

        # Create mock vault data
        vault_id = uuid.uuid4()
        mock_vault = MagicMock()
        mock_vault.id = vault_id
        mock_vault.name = "Test Vault"
        mock_vault.description = "A test vault"
        mock_vault.container_path = "/data/test"
        mock_vault.host_path = "/home/user/test"
        mock_vault.created_at = datetime(2024, 1, 15, 10, 30, 0, tzinfo=UTC)

        # Setup mock session
        mock_session = MagicMock()
        mock_session.query.return_value.filter.return_value.first.return_value = (
            mock_vault
        )
        mock_session.query.return_value.filter.return_value.scalar.return_value = 42

        # Setup mock db_manager
        mock_db_manager = MagicMock()
        mock_db_manager.get_session.return_value.__enter__ = MagicMock(
            return_value=mock_session
        )
        mock_db_manager.get_session.return_value.__exit__ = MagicMock(return_value=None)
        mock_db_manager_class.return_value = mock_db_manager

        # Mock settings
        mock_settings = MagicMock()
        mock_settings.database.url = "postgresql+psycopg://localhost/test"
        mock_settings.database.pool_size = 5
        mock_settings.database.max_overflow = 10
        mock_settings.database.pool_timeout = 30
        mock_settings.database.pool_recycle = 3600
        mock_settings.logging.level = "INFO"
        mock_settings.logging.format = "text"

        with patch("obsidian_rag.cli.get_settings", return_value=mock_settings):
            result = runner.invoke(cli, ["vault", "get", "--name", "Test Vault"])

        assert result.exit_code == 0
        assert "Test Vault" in result.output
        assert "A test vault" in result.output
        assert "/data/test" in result.output
        assert "/home/user/test" in result.output
        assert "42" in result.output  # document count

    @patch("obsidian_rag.cli_vault_commands.DatabaseManager")
    def test_vault_get_by_id(self, mock_db_manager_class) -> None:
        """Test getting a vault by ID."""
        runner = CliRunner()

        # Create mock vault data
        vault_id = uuid.uuid4()
        mock_vault = MagicMock()
        mock_vault.id = vault_id
        mock_vault.name = "Test Vault"
        mock_vault.description = "A test vault"
        mock_vault.container_path = "/data/test"
        mock_vault.host_path = "/home/user/test"
        mock_vault.created_at = datetime(2024, 1, 15, 10, 30, 0, tzinfo=UTC)

        # Setup mock session
        mock_session = MagicMock()
        mock_session.query.return_value.filter.return_value.first.return_value = (
            mock_vault
        )
        mock_session.query.return_value.filter.return_value.scalar.return_value = 42

        # Setup mock db_manager
        mock_db_manager = MagicMock()
        mock_db_manager.get_session.return_value.__enter__ = MagicMock(
            return_value=mock_session
        )
        mock_db_manager.get_session.return_value.__exit__ = MagicMock(return_value=None)
        mock_db_manager_class.return_value = mock_db_manager

        # Mock settings
        mock_settings = MagicMock()
        mock_settings.database.url = "postgresql+psycopg://localhost/test"
        mock_settings.database.pool_size = 5
        mock_settings.database.max_overflow = 10
        mock_settings.database.pool_timeout = 30
        mock_settings.database.pool_recycle = 3600
        mock_settings.logging.level = "INFO"
        mock_settings.logging.format = "text"

        with patch("obsidian_rag.cli.get_settings", return_value=mock_settings):
            result = runner.invoke(cli, ["vault", "get", "--id", str(vault_id)])

        assert result.exit_code == 0
        assert "Test Vault" in result.output
        assert str(vault_id) in result.output

    @patch("obsidian_rag.cli_vault_commands.DatabaseManager")
    def test_vault_get_by_id_not_found(self, mock_db_manager_class) -> None:
        """Test getting a vault by ID when vault doesn't exist."""
        runner = CliRunner()

        vault_id = uuid.uuid4()

        # Setup mock session - vault not found (valid UUID but no vault)
        mock_session = MagicMock()
        mock_session.query.return_value.filter.return_value.first.return_value = None

        # Setup mock db_manager
        mock_db_manager = MagicMock()
        mock_db_manager.get_session.return_value.__enter__ = MagicMock(
            return_value=mock_session
        )
        mock_db_manager.get_session.return_value.__exit__ = MagicMock(return_value=None)
        mock_db_manager_class.return_value = mock_db_manager

        # Mock settings
        mock_settings = MagicMock()
        mock_settings.database.url = "postgresql+psycopg://localhost/test"
        mock_settings.database.pool_size = 5
        mock_settings.database.max_overflow = 10
        mock_settings.database.pool_timeout = 30
        mock_settings.database.pool_recycle = 3600
        mock_settings.logging.level = "INFO"
        mock_settings.logging.format = "text"

        with patch("obsidian_rag.cli.get_settings", return_value=mock_settings):
            result = runner.invoke(cli, ["vault", "get", "--id", str(vault_id)])

        assert result.exit_code == 1
        assert "not found" in result.output.lower()

    @patch("obsidian_rag.cli_vault_commands.DatabaseManager")
    def test_vault_get_no_description(self, mock_db_manager_class) -> None:
        """Test getting a vault without description."""
        runner = CliRunner()

        # Create mock vault data without description
        vault_id = uuid.uuid4()
        mock_vault = MagicMock()
        mock_vault.id = vault_id
        mock_vault.name = "Test Vault"
        mock_vault.description = None  # No description
        mock_vault.container_path = "/data/test"
        mock_vault.host_path = "/home/user/test"
        mock_vault.created_at = datetime(2024, 1, 15, 10, 30, 0, tzinfo=UTC)

        # Setup mock session
        mock_session = MagicMock()
        mock_session.query.return_value.filter.return_value.first.return_value = (
            mock_vault
        )
        mock_session.query.return_value.filter.return_value.scalar.return_value = 42

        # Setup mock db_manager
        mock_db_manager = MagicMock()
        mock_db_manager.get_session.return_value.__enter__ = MagicMock(
            return_value=mock_session
        )
        mock_db_manager.get_session.return_value.__exit__ = MagicMock(return_value=None)
        mock_db_manager_class.return_value = mock_db_manager

        # Mock settings
        mock_settings = MagicMock()
        mock_settings.database.url = "postgresql+psycopg://localhost/test"
        mock_settings.database.pool_size = 5
        mock_settings.database.max_overflow = 10
        mock_settings.database.pool_timeout = 30
        mock_settings.database.pool_recycle = 3600
        mock_settings.logging.level = "INFO"
        mock_settings.logging.format = "text"

        with patch("obsidian_rag.cli.get_settings", return_value=mock_settings):
            result = runner.invoke(cli, ["vault", "get", "--name", "Test Vault"])

        assert result.exit_code == 0
        assert "Test Vault" in result.output
        # Description line should not appear when description is None
        lines = result.output.split("\n")
        description_lines = [l for l in lines if l.startswith("Description:")]
        assert len(description_lines) == 0

    @patch("obsidian_rag.cli_vault_commands.DatabaseManager")
    def test_vault_get_not_found(self, mock_db_manager_class) -> None:
        """Test getting a non-existent vault."""
        runner = CliRunner()

        # Setup mock session - vault not found
        mock_session = MagicMock()
        mock_session.query.return_value.filter.return_value.first.return_value = None

        # Setup mock db_manager
        mock_db_manager = MagicMock()
        mock_db_manager.get_session.return_value.__enter__ = MagicMock(
            return_value=mock_session
        )
        mock_db_manager.get_session.return_value.__exit__ = MagicMock(return_value=None)
        mock_db_manager_class.return_value = mock_db_manager

        # Mock settings
        mock_settings = MagicMock()
        mock_settings.database.url = "postgresql+psycopg://localhost/test"
        mock_settings.database.pool_size = 5
        mock_settings.database.max_overflow = 10
        mock_settings.database.pool_timeout = 30
        mock_settings.database.pool_recycle = 3600
        mock_settings.logging.level = "INFO"
        mock_settings.logging.format = "text"

        with patch("obsidian_rag.cli.get_settings", return_value=mock_settings):
            result = runner.invoke(cli, ["vault", "get", "--name", "NonExistent"])

        assert result.exit_code == 1
        assert "not found" in result.output.lower()

    def test_vault_get_missing_name_and_id(self) -> None:
        """Test vault get without name or id."""
        runner = CliRunner()

        result = runner.invoke(cli, ["vault", "get"])

        assert result.exit_code != 0
        assert "name" in result.output.lower() or "id" in result.output.lower()

    @patch("obsidian_rag.cli_vault_commands.DatabaseManager")
    def test_vault_get_invalid_uuid(self, mock_db_manager_class) -> None:
        """Test getting a vault with invalid UUID."""
        runner = CliRunner()

        # Setup mock db_manager
        mock_db_manager = MagicMock()
        mock_db_manager.get_session.return_value.__enter__ = MagicMock(
            return_value=MagicMock()
        )
        mock_db_manager.get_session.return_value.__exit__ = MagicMock(return_value=None)
        mock_db_manager_class.return_value = mock_db_manager

        # Mock settings
        mock_settings = MagicMock()
        mock_settings.database.url = "postgresql+psycopg://localhost/test"
        mock_settings.database.pool_size = 5
        mock_settings.database.max_overflow = 10
        mock_settings.database.pool_timeout = 30
        mock_settings.database.pool_recycle = 3600
        mock_settings.logging.level = "INFO"
        mock_settings.logging.format = "text"

        with patch("obsidian_rag.cli.get_settings", return_value=mock_settings):
            result = runner.invoke(cli, ["vault", "get", "--id", "not-a-valid-uuid"])

        assert result.exit_code == 1
        assert "invalid" in result.output.lower() or "uuid" in result.output.lower()
