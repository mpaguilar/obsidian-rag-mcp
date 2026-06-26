"""Tests for exact query CLI command helpers."""

from unittest.mock import MagicMock, patch


from obsidian_rag.cli_query_exact import (
    _display_document_list,
    _display_single_document,
    _run_exact_query_command,
)


class TestDisplaySingleDocument:
    """Tests for _display_single_document."""

    @patch("obsidian_rag.cli_query_exact.click.echo")
    def test_display_single_document_with_tags(self, mock_echo):
        """Display document with tags."""
        doc = MagicMock()
        doc.vault_name = "TestVault"
        doc.file_path = "notes/test.md"
        doc.tags = ["tag1", "tag2"]
        doc.content = "Hello world"

        _display_single_document(doc)

        mock_echo.assert_any_call("Vault: TestVault")
        mock_echo.assert_any_call("Path: notes/test.md")
        mock_echo.assert_any_call("Tags: tag1, tag2")
        mock_echo.assert_any_call()
        mock_echo.assert_any_call("Hello world")

    @patch("obsidian_rag.cli_query_exact.click.echo")
    def test_display_single_document_no_tags(self, mock_echo):
        """Display document without tags."""
        doc = MagicMock()
        doc.vault_name = "Vault"
        doc.file_path = "a.md"
        doc.tags = []
        doc.content = "Content"

        _display_single_document(doc)

        mock_echo.assert_any_call("Tags: none")


class TestDisplayDocumentList:
    """Tests for _display_document_list."""

    @patch("obsidian_rag.cli_query_exact.click.echo")
    def test_display_document_list_with_results(self, mock_echo):
        """Display list with documents."""
        doc1 = MagicMock()
        doc1.file_name = "a.md"
        doc1.file_path = "notes/a.md"
        doc1.vault_name = "Vault1"
        doc1.tags = ["tag1"]

        doc2 = MagicMock()
        doc2.file_name = "b.md"
        doc2.file_path = "notes/b.md"
        doc2.vault_name = "Vault2"
        doc2.tags = []

        docs = MagicMock()
        docs.results = [doc1, doc2]
        docs.total_count = 2

        _display_document_list(docs)

        mock_echo.assert_any_call("Found 2 results:\n")
        mock_echo.assert_any_call("File: a.md")
        mock_echo.assert_any_call("File: b.md")

    @patch("obsidian_rag.cli_query_exact.click.echo")
    def test_display_document_list_empty(self, mock_echo):
        """Display empty list."""
        docs = MagicMock()
        docs.results = []
        docs.total_count = 0

        _display_document_list(docs)

        mock_echo.assert_called_once_with("No matching documents found.")


class TestRunExactQueryCommand:
    """Tests for _run_exact_query_command."""

    @patch("obsidian_rag.cli_commands.sys.exit")
    @patch("obsidian_rag.cli_query_exact.click.echo")
    @patch("obsidian_rag.cli_query_exact.get_document_impl")
    @patch("obsidian_rag.cli_query_exact.DatabaseManager")
    def test_run_exact_query_by_document_id(
        self, mock_db_manager, mock_get, mock_echo, mock_exit
    ):
        """Lookup by document_id calls get_document and displays table."""
        mock_settings = MagicMock()
        mock_settings.database.url = "postgresql+psycopg://localhost/test"
        mock_settings.database.pool_size = 5
        mock_settings.database.max_overflow = 10
        mock_settings.database.pool_timeout = 30
        mock_settings.database.pool_recycle = 3600

        ctx = MagicMock()
        ctx.obj = {"settings": mock_settings}

        mock_session = MagicMock()
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)

        mock_db_instance = MagicMock()
        mock_db_instance.get_session.return_value = mock_session
        mock_db_manager.return_value = mock_db_instance

        mock_result = MagicMock()
        mock_result.vault_name = "Vault"
        mock_result.file_path = "a.md"
        mock_result.tags = []
        mock_result.content = "Hello"
        mock_get.return_value = mock_result

        _run_exact_query_command(
            ctx,
            vault=None,
            path=None,
            name=None,
            document_id="doc-123",
            limit=20,
            output_format="table",
        )

        mock_get.assert_called_once_with(
            mock_session,
            vault_name=None,
            file_path=None,
            document_id="doc-123",
        )
        mock_exit.assert_not_called()
        mock_echo.assert_any_call("Vault: Vault")

    @patch("obsidian_rag.cli_commands.sys.exit")
    @patch("obsidian_rag.cli_query_exact.click.echo")
    @patch("obsidian_rag.cli_query_exact.get_document_impl")
    @patch("obsidian_rag.cli_query_exact.DatabaseManager")
    def test_run_exact_query_by_path(
        self, mock_db_manager, mock_get, _mock_echo, mock_exit
    ):
        """Lookup by path calls get_document with vault."""
        mock_settings = MagicMock()
        mock_settings.database.url = "postgresql+psycopg://localhost/test"
        mock_settings.database.pool_size = 5
        mock_settings.database.max_overflow = 10
        mock_settings.database.pool_timeout = 30
        mock_settings.database.pool_recycle = 3600

        ctx = MagicMock()
        ctx.obj = {"settings": mock_settings}

        mock_session = MagicMock()
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)

        mock_db_instance = MagicMock()
        mock_db_instance.get_session.return_value = mock_session
        mock_db_manager.return_value = mock_db_instance

        mock_result = MagicMock()
        mock_result.vault_name = "Vault"
        mock_result.file_path = "notes/b.md"
        mock_result.tags = ["tag1"]
        mock_result.content = "Content"
        mock_get.return_value = mock_result

        _run_exact_query_command(
            ctx,
            vault="Vault",
            path="notes/b.md",
            name=None,
            document_id=None,
            limit=20,
            output_format="table",
        )

        mock_get.assert_called_once_with(
            mock_session,
            vault_name="Vault",
            file_path="notes/b.md",
            document_id=None,
        )
        mock_exit.assert_not_called()

    @patch("obsidian_rag.cli_commands.sys.exit")
    @patch("obsidian_rag.cli_query_exact.click.echo")
    @patch("obsidian_rag.cli_query_exact.list_documents_impl")
    @patch("obsidian_rag.cli_query_exact.DatabaseManager")
    def test_run_exact_query_by_name(
        self, mock_db_manager, mock_list, mock_echo, mock_exit
    ):
        """Lookup by name calls list_documents."""
        mock_settings = MagicMock()
        mock_settings.database.url = "postgresql+psycopg://localhost/test"
        mock_settings.database.pool_size = 5
        mock_settings.database.max_overflow = 10
        mock_settings.database.pool_timeout = 30
        mock_settings.database.pool_recycle = 3600

        ctx = MagicMock()
        ctx.obj = {"settings": mock_settings}

        mock_session = MagicMock()
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)

        mock_db_instance = MagicMock()
        mock_db_instance.get_session.return_value = mock_session
        mock_db_manager.return_value = mock_db_instance

        mock_doc = MagicMock()
        mock_doc.file_name = "c.md"
        mock_doc.file_path = "c.md"
        mock_doc.vault_name = "Vault"
        mock_doc.tags = []

        mock_result = MagicMock()
        mock_result.results = [mock_doc]
        mock_result.total_count = 1
        mock_list.return_value = mock_result

        _run_exact_query_command(
            ctx,
            vault=None,
            path=None,
            name="c.md",
            document_id=None,
            limit=20,
            output_format="table",
        )

        mock_list.assert_called_once_with(
            mock_session,
            file_name="c.md",
            vault_name=None,
            limit=20,
        )
        mock_exit.assert_not_called()
        mock_echo.assert_any_call("Found 1 results:\n")

    @patch("obsidian_rag.cli_commands.sys.exit")
    @patch("obsidian_rag.cli_query_exact.click.echo")
    @patch("obsidian_rag.cli_query_exact.list_documents_impl")
    @patch("obsidian_rag.cli_query_exact.DatabaseManager")
    def test_run_exact_query_by_name_with_vault(
        self, mock_db_manager, mock_list, _mock_echo, _mock_exit
    ):
        """Lookup by name with vault filter."""
        mock_settings = MagicMock()
        mock_settings.database.url = "postgresql+psycopg://localhost/test"
        mock_settings.database.pool_size = 5
        mock_settings.database.max_overflow = 10
        mock_settings.database.pool_timeout = 30
        mock_settings.database.pool_recycle = 3600

        ctx = MagicMock()
        ctx.obj = {"settings": mock_settings}

        mock_session = MagicMock()
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)

        mock_db_instance = MagicMock()
        mock_db_instance.get_session.return_value = mock_session
        mock_db_manager.return_value = mock_db_instance

        mock_result = MagicMock()
        mock_result.results = []
        mock_result.total_count = 0
        mock_list.return_value = mock_result

        _run_exact_query_command(
            ctx,
            vault="MyVault",
            path=None,
            name="d.md",
            document_id=None,
            limit=10,
            output_format="table",
        )

        mock_list.assert_called_once_with(
            mock_session,
            file_name="d.md",
            vault_name="MyVault",
            limit=10,
        )

    @patch("obsidian_rag.cli_commands.sys.exit")
    @patch("obsidian_rag.cli_query_exact.click.echo")
    @patch("obsidian_rag.cli_query_exact.get_document_impl")
    @patch("obsidian_rag.cli_query_exact.DatabaseManager")
    def test_run_exact_query_document_not_found(
        self, mock_db_manager, mock_get, mock_echo, mock_exit
    ):
        """ValueError from get_document prints error and exits."""
        mock_settings = MagicMock()
        mock_settings.database.url = "postgresql+psycopg://localhost/test"
        mock_settings.database.pool_size = 5
        mock_settings.database.max_overflow = 10
        mock_settings.database.pool_timeout = 30
        mock_settings.database.pool_recycle = 3600

        ctx = MagicMock()
        ctx.obj = {"settings": mock_settings}

        mock_session = MagicMock()
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)

        mock_db_instance = MagicMock()
        mock_db_instance.get_session.return_value = mock_session
        mock_db_manager.return_value = mock_db_instance

        mock_get.side_effect = ValueError("Document not found")

        _run_exact_query_command(
            ctx,
            vault=None,
            path=None,
            name=None,
            document_id="missing",
            limit=20,
            output_format="table",
        )

        mock_echo.assert_any_call("Error: Document not found", err=True)
        mock_exit.assert_called_once_with(1)

    @patch("obsidian_rag.cli_commands.sys.exit")
    @patch("obsidian_rag.cli_query_exact.click.echo")
    @patch("obsidian_rag.cli_query_exact.get_document_impl")
    @patch("obsidian_rag.cli_query_exact.DatabaseManager")
    def test_run_exact_query_output_json(
        self, mock_db_manager, mock_get, mock_echo, mock_exit
    ):
        """JSON format outputs model_dump_json."""
        mock_settings = MagicMock()
        mock_settings.database.url = "postgresql+psycopg://localhost/test"
        mock_settings.database.pool_size = 5
        mock_settings.database.max_overflow = 10
        mock_settings.database.pool_timeout = 30
        mock_settings.database.pool_recycle = 3600

        ctx = MagicMock()
        ctx.obj = {"settings": mock_settings}

        mock_session = MagicMock()
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)

        mock_db_instance = MagicMock()
        mock_db_instance.get_session.return_value = mock_session
        mock_db_manager.return_value = mock_db_instance

        mock_result = MagicMock()
        mock_result.model_dump_json.return_value = '{"id": "123"}'
        mock_get.return_value = mock_result

        _run_exact_query_command(
            ctx,
            vault=None,
            path=None,
            name=None,
            document_id="doc-123",
            limit=20,
            output_format="json",
        )

        mock_result.model_dump_json.assert_called_once_with(indent=2)
        mock_echo.assert_called_once_with('{"id": "123"}')
        mock_exit.assert_not_called()

    @patch("obsidian_rag.cli_commands.sys.exit")
    @patch("obsidian_rag.cli_query_exact.click.echo")
    @patch("obsidian_rag.cli_query_exact.list_documents_impl")
    @patch("obsidian_rag.cli_query_exact.DatabaseManager")
    def test_run_exact_query_output_table(
        self, mock_db_manager, mock_list, mock_echo, mock_exit
    ):
        """Table format uses display helper."""
        mock_settings = MagicMock()
        mock_settings.database.url = "postgresql+psycopg://localhost/test"
        mock_settings.database.pool_size = 5
        mock_settings.database.max_overflow = 10
        mock_settings.database.pool_timeout = 30
        mock_settings.database.pool_recycle = 3600

        ctx = MagicMock()
        ctx.obj = {"settings": mock_settings}

        mock_session = MagicMock()
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)

        mock_db_instance = MagicMock()
        mock_db_instance.get_session.return_value = mock_session
        mock_db_manager.return_value = mock_db_instance

        mock_doc = MagicMock()
        mock_doc.file_name = "x.md"
        mock_doc.file_path = "x.md"
        mock_doc.vault_name = "V"
        mock_doc.tags = ["t1"]

        mock_result = MagicMock()
        mock_result.results = [mock_doc]
        mock_result.total_count = 1
        mock_list.return_value = mock_result

        _run_exact_query_command(
            ctx,
            vault=None,
            path=None,
            name="x.md",
            document_id=None,
            limit=20,
            output_format="table",
        )

        mock_echo.assert_any_call("Found 1 results:\n")
        mock_exit.assert_not_called()

    @patch("obsidian_rag.cli_commands.sys.exit")
    @patch("obsidian_rag.cli_query_exact.click.echo")
    @patch("obsidian_rag.cli_query_exact.list_documents_impl")
    @patch("obsidian_rag.cli_query_exact.DatabaseManager")
    def test_run_exact_query_by_name_json(
        self, mock_db_manager, mock_list, mock_echo, mock_exit
    ):
        """JSON format for name lookup uses model_dump_json."""
        mock_settings = MagicMock()
        mock_settings.database.url = "postgresql+psycopg://localhost/test"
        mock_settings.database.pool_size = 5
        mock_settings.database.max_overflow = 10
        mock_settings.database.pool_timeout = 30
        mock_settings.database.pool_recycle = 3600

        ctx = MagicMock()
        ctx.obj = {"settings": mock_settings}

        mock_session = MagicMock()
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)

        mock_db_instance = MagicMock()
        mock_db_instance.get_session.return_value = mock_session
        mock_db_manager.return_value = mock_db_instance

        mock_result = MagicMock()
        mock_result.model_dump_json.return_value = '[{"name": "x.md"}]'
        mock_list.return_value = mock_result

        _run_exact_query_command(
            ctx,
            vault=None,
            path=None,
            name="x.md",
            document_id=None,
            limit=20,
            output_format="json",
        )

        mock_result.model_dump_json.assert_called_once_with(indent=2)
        mock_echo.assert_called_once_with('[{"name": "x.md"}]')
        mock_exit.assert_not_called()

    @patch("obsidian_rag.cli_commands.sys.exit")
    @patch("obsidian_rag.cli_query_exact.click.echo")
    @patch("obsidian_rag.cli_query_exact.list_documents_impl")
    @patch("obsidian_rag.cli_query_exact.DatabaseManager")
    def test_run_exact_query_by_name_error(
        self, mock_db_manager, mock_list, mock_echo, mock_exit
    ):
        """ValueError from list_documents prints error and exits."""
        mock_settings = MagicMock()
        mock_settings.database.url = "postgresql+psycopg://localhost/test"
        mock_settings.database.pool_size = 5
        mock_settings.database.max_overflow = 10
        mock_settings.database.pool_timeout = 30
        mock_settings.database.pool_recycle = 3600

        ctx = MagicMock()
        ctx.obj = {"settings": mock_settings}

        mock_session = MagicMock()
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)

        mock_db_instance = MagicMock()
        mock_db_instance.get_session.return_value = mock_session
        mock_db_manager.return_value = mock_db_instance

        mock_list.side_effect = ValueError("No such vault")

        _run_exact_query_command(
            ctx,
            vault=None,
            path=None,
            name="x.md",
            document_id=None,
            limit=20,
            output_format="table",
        )

        mock_echo.assert_any_call("Error: No such vault", err=True)
        mock_exit.assert_called_once_with(1)

    @patch("obsidian_rag.cli_commands.sys.exit")
    @patch("obsidian_rag.cli_query_exact.click.echo")
    @patch("obsidian_rag.cli_query_exact.get_document_impl")
    @patch("obsidian_rag.cli_query_exact.DatabaseManager")
    def test_run_exact_query_no_lookup_params(
        self, mock_db_manager, mock_get, _mock_echo, mock_exit
    ):
        """When all lookup params are None, nothing happens in session block."""
        mock_settings = MagicMock()
        mock_settings.database.url = "postgresql+psycopg://localhost/test"
        mock_settings.database.pool_size = 5
        mock_settings.database.max_overflow = 10
        mock_settings.database.pool_timeout = 30
        mock_settings.database.pool_recycle = 3600

        ctx = MagicMock()
        ctx.obj = {"settings": mock_settings}

        mock_session = MagicMock()
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)

        mock_db_instance = MagicMock()
        mock_db_instance.get_session.return_value = mock_session
        mock_db_manager.return_value = mock_db_instance

        _run_exact_query_command(
            ctx,
            vault=None,
            path=None,
            name=None,
            document_id=None,
            limit=20,
            output_format="table",
        )

        mock_get.assert_not_called()
        mock_exit.assert_not_called()
