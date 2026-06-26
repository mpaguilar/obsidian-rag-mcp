"""End-to-end integration tests for CLI --exact document retrieval."""

from unittest.mock import Mock, patch

from click.testing import CliRunner

from obsidian_rag.cli import cli


class TestCLIExactPathLookup:
    """Test exact path lookup via CLI."""

    @patch("obsidian_rag.cli_query_exact.get_document_impl")
    @patch("obsidian_rag.cli_query_exact.DatabaseManager")
    def test_cli_exact_path_lookup(self, mock_db_cls, mock_get_doc):
        """query --exact --vault Personal --path notes.md."""
        mock_db = Mock()
        mock_db_cls.return_value = mock_db
        mock_session = Mock()
        mock_db.get_session.return_value.__enter__ = Mock(return_value=mock_session)
        mock_db.get_session.return_value.__exit__ = Mock(return_value=False)

        mock_doc = Mock()
        mock_doc.model_dump_json.return_value = '{"id": "abc"}'
        mock_doc.tags = []
        mock_get_doc.return_value = mock_doc

        runner = CliRunner()
        result = runner.invoke(
            cli, ["query", "--exact", "--vault", "Personal", "--path", "notes.md"]
        )
        assert result.exit_code == 0
        mock_get_doc.assert_called_once()
        call_kwargs = mock_get_doc.call_args.kwargs
        assert call_kwargs["vault_name"] == "Personal"
        assert call_kwargs["file_path"] == "notes.md"

    @patch("obsidian_rag.cli_query_exact.list_documents_impl")
    @patch("obsidian_rag.cli_query_exact.DatabaseManager")
    def test_cli_exact_name_lookup(self, mock_db_cls, mock_list_docs):
        """query --exact --name notes.md."""
        mock_db = Mock()
        mock_db_cls.return_value = mock_db
        mock_session = Mock()
        mock_db.get_session.return_value.__enter__ = Mock(return_value=mock_session)
        mock_db.get_session.return_value.__exit__ = Mock(return_value=False)

        mock_result = Mock()
        mock_result.model_dump_json.return_value = '{"documents": []}'
        mock_result.results = []
        mock_result.total_count = 0
        mock_list_docs.return_value = mock_result

        runner = CliRunner()
        result = runner.invoke(cli, ["query", "--exact", "--name", "notes.md"])
        assert result.exit_code == 0
        mock_list_docs.assert_called_once()
        call_kwargs = mock_list_docs.call_args.kwargs
        assert call_kwargs["file_name"] == "notes.md"

    @patch("obsidian_rag.cli_query_exact.get_document_impl")
    @patch("obsidian_rag.cli_query_exact.DatabaseManager")
    def test_cli_exact_id_lookup(self, mock_db_cls, mock_get_doc):
        """query --exact --id abc-123."""
        mock_db = Mock()
        mock_db_cls.return_value = mock_db
        mock_session = Mock()
        mock_db.get_session.return_value.__enter__ = Mock(return_value=mock_session)
        mock_db.get_session.return_value.__exit__ = Mock(return_value=False)

        mock_doc = Mock()
        mock_doc.model_dump_json.return_value = '{"id": "abc"}'
        mock_doc.tags = []
        mock_get_doc.return_value = mock_doc

        runner = CliRunner()
        result = runner.invoke(cli, ["query", "--exact", "--id", "abc-123"])
        assert result.exit_code == 0
        mock_get_doc.assert_called_once()
        call_kwargs = mock_get_doc.call_args.kwargs
        assert call_kwargs["document_id"] == "abc-123"

    def test_cli_exact_path_without_vault(self):
        """Click error requires --vault."""
        runner = CliRunner()
        result = runner.invoke(cli, ["query", "--exact", "--path", "notes.md"])
        assert result.exit_code != 0
        assert "--vault" in result.output or "--vault" in str(result.exception)

    def test_cli_exact_with_query_text(self):
        """Click error when --exact with query_text."""
        runner = CliRunner()
        result = runner.invoke(cli, ["query", "--exact", "search text"])
        assert result.exit_code != 0
        assert "Cannot use" in result.output or "Cannot use" in str(result.exception)

    def test_cli_exact_no_lookup_params(self):
        """Click error when --exact alone."""
        runner = CliRunner()
        result = runner.invoke(cli, ["query", "--exact"])
        assert result.exit_code != 0
        assert "Must provide" in result.output or "Must provide" in str(
            result.exception
        )

    def test_cli_semantic_search_still_works(self):
        """query search text unchanged."""
        runner = CliRunner()
        result = runner.invoke(cli, ["query", "search text"])
        assert result.exit_code == 0

    @patch("obsidian_rag.cli_query_exact.list_documents_impl")
    @patch("obsidian_rag.cli_query_exact.DatabaseManager")
    def test_cli_exact_output_json(self, mock_db_cls, mock_list_docs):
        """JSON format output."""
        mock_db = Mock()
        mock_db_cls.return_value = mock_db
        mock_session = Mock()
        mock_db.get_session.return_value.__enter__ = Mock(return_value=mock_session)
        mock_db.get_session.return_value.__exit__ = Mock(return_value=False)

        mock_result = Mock()
        mock_result.model_dump_json.return_value = '{"documents": [], "total_count": 0}'
        mock_list_docs.return_value = mock_result

        runner = CliRunner()
        result = runner.invoke(
            cli, ["query", "--exact", "--name", "notes.md", "--format", "json"]
        )
        assert result.exit_code == 0
        assert '"documents"' in result.output

    @patch("obsidian_rag.cli_query_exact.list_documents_impl")
    @patch("obsidian_rag.cli_query_exact.DatabaseManager")
    def test_cli_exact_name_with_vault(self, mock_db_cls, mock_list_docs):
        """query --exact --vault Personal --name notes.md."""
        mock_db = Mock()
        mock_db_cls.return_value = mock_db
        mock_session = Mock()
        mock_db.get_session.return_value.__enter__ = Mock(return_value=mock_session)
        mock_db.get_session.return_value.__exit__ = Mock(return_value=False)

        mock_result = Mock()
        mock_result.model_dump_json.return_value = '{"documents": []}'
        mock_result.results = []
        mock_result.total_count = 0
        mock_list_docs.return_value = mock_result

        runner = CliRunner()
        result = runner.invoke(
            cli, ["query", "--exact", "--vault", "Personal", "--name", "notes.md"]
        )
        assert result.exit_code == 0
        mock_list_docs.assert_called_once()
        call_kwargs = mock_list_docs.call_args.kwargs
        assert call_kwargs["file_name"] == "notes.md"
        assert call_kwargs["vault_name"] == "Personal"

    @patch("obsidian_rag.cli_query_exact.get_document_impl")
    @patch("obsidian_rag.cli_query_exact.DatabaseManager")
    def test_cli_exact_document_not_found(self, mock_db_cls, mock_get_doc):
        """Error message to stderr."""
        mock_db = Mock()
        mock_db_cls.return_value = mock_db
        mock_session = Mock()
        mock_db.get_session.return_value.__enter__ = Mock(return_value=mock_session)
        mock_db.get_session.return_value.__exit__ = Mock(return_value=False)

        mock_get_doc.side_effect = ValueError("Document not found")

        runner = CliRunner()
        result = runner.invoke(
            cli, ["query", "--exact", "--vault", "Personal", "--path", "missing.md"]
        )
        assert result.exit_code == 1
        assert "not found" in result.output or "Error" in result.output

    def test_cli_exact_help_shows_new_options(self):
        """Help text includes --exact, --path, --name, --id."""
        runner = CliRunner()
        result = runner.invoke(cli, ["query", "--help"])
        assert result.exit_code == 0
        assert "--exact" in result.output
        assert "--path" in result.output
        assert "--name" in result.output
        assert "--id" in result.output
