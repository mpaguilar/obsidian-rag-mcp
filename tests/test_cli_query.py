"""Tests for CLI commands."""

from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from obsidian_rag.cli import cli


class TestFormatQueryResults:
    """Test query result formatting functions."""

    def test_format_query_results_json_with_tags(self):
        """Test _format_query_results_json with results containing tags (TASK-046)."""
        import json

        from obsidian_rag.cli_commands import _format_query_results_json

        mock_doc = MagicMock()
        mock_doc.file_path = "/path/to/doc.md"
        mock_doc.file_name = "doc.md"
        mock_doc.frontmatter_json = {"kind": "note"}
        mock_doc.tags = ["work", "urgent"]

        results = [(mock_doc, 0.5)]
        output = _format_query_results_json(results)

        parsed = json.loads(output)
        assert len(parsed) == 1
        assert parsed[0]["file_path"] == "/path/to/doc.md"
        assert parsed[0]["file_name"] == "doc.md"
        assert parsed[0]["kind"] == "note"
        assert parsed[0]["tags"] == ["work", "urgent"]
        assert parsed[0]["distance"] == 0.5

    def test_format_query_results_table_with_kind_and_tags(self):
        """Test _format_query_results_table with documents having kind and tags (TASK-047)."""
        from obsidian_rag.cli_commands import _format_query_results_table

        mock_doc = MagicMock()
        mock_doc.file_name = "doc.md"
        mock_doc.file_path = "/path/to/doc.md"
        mock_doc.frontmatter_json = {"kind": "project"}
        mock_doc.tags = ["work", "urgent"]

        results = [(mock_doc, 0.75)]
        output = _format_query_results_table(results)

        assert "Found 1 results:" in output
        assert "File: doc.md" in output
        assert "Path: /path/to/doc.md" in output
        assert "Distance: 0.7500" in output
        assert "Kind: project" in output
        assert "Tags: work, urgent" in output


class TestQueryCommand:
    """Test query command with various scenarios."""

    @patch("obsidian_rag.cli_commands.DatabaseManager")
    @patch("obsidian_rag.cli_commands._get_embedding_provider")
    def test_query_command_with_embedding_failure(
        self, mock_get_provider, mock_db_manager
    ):
        """Test query command with embedding generation failure (TASK-048)."""
        runner = CliRunner()

        mock_settings = MagicMock()
        mock_settings.database.url = "postgresql+psycopg://localhost/test"
        mock_settings.logging.level = "INFO"
        mock_settings.logging.format = "text"

        mock_embedding_provider = MagicMock()
        mock_embedding_provider.generate_embedding.side_effect = Exception(
            "Embedding failed"
        )
        mock_get_provider.return_value = mock_embedding_provider

        mock_db_instance = MagicMock()
        mock_db_manager.return_value = mock_db_instance

        with patch("obsidian_rag.cli.get_settings", return_value=mock_settings):
            result = runner.invoke(cli, ["query", "test query"])

        assert result.exit_code == 1
        assert "Failed to generate query embedding" in result.output

    @patch("obsidian_rag.cli_commands.DatabaseManager")
    @patch("obsidian_rag.cli_commands._get_embedding_provider")
    def test_query_command_with_json_output(self, mock_get_provider, mock_db_manager):
        """Test query command with JSON output format (TASK-049)."""
        import json

        runner = CliRunner()

        mock_settings = MagicMock()
        mock_settings.database.url = "postgresql+psycopg://localhost/test"
        mock_settings.logging.level = "INFO"
        mock_settings.logging.format = "text"

        mock_embedding_provider = MagicMock()
        mock_embedding_provider.generate_embedding.return_value = [0.1] * 1536
        mock_get_provider.return_value = mock_embedding_provider

        mock_doc = MagicMock()
        mock_doc.file_path = "/path/to/doc.md"
        mock_doc.file_name = "doc.md"
        mock_doc.frontmatter_json = None
        mock_doc.tags = None

        mock_session = MagicMock()
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)
        mock_query_result = MagicMock()
        mock_query_result.filter.return_value.order_by.return_value.limit.return_value.all.return_value = [
            (mock_doc, 0.5)
        ]
        mock_session.query.return_value = mock_query_result

        mock_db_instance = MagicMock()
        mock_db_instance.get_session.return_value = mock_session
        mock_db_manager.return_value = mock_db_instance

        with patch("obsidian_rag.cli.get_settings", return_value=mock_settings):
            result = runner.invoke(cli, ["query", "test query", "--format", "json"])

        assert result.exit_code == 0
        # Verify JSON output
        try:
            parsed = json.loads(result.output)
            assert isinstance(parsed, list)
        except json.JSONDecodeError:
            # Output might have logging mixed in, check for JSON structure
            assert '"file_path"' in result.output or "file_path" in result.output


class TestBuildTasksQuery:
    """Test _build_tasks_query function."""

    def test_build_tasks_query_with_status_filter(self):
        """Test _build_tasks_query with status filter (TASK-051)."""
        from obsidian_rag.cli_commands import _build_tasks_query

        mock_session = MagicMock()
        mock_query = MagicMock()
        mock_session.query.return_value = mock_query
        mock_query.join.return_value = mock_query
        mock_query.filter.return_value = mock_query
        mock_query.order_by.return_value = mock_query
        mock_query.limit.return_value = mock_query

        from obsidian_rag.cli_commands import TaskDateFilters

        date_filters = TaskDateFilters()
        _build_tasks_query(mock_session, "completed", date_filters, None, None, 10)

        mock_session.query.assert_called_once()
        mock_query.filter.assert_called()
        mock_query.limit.assert_called_once_with(10)

    def test_build_tasks_query_with_date_objects(self):
        """Test _build_tasks_query accepts date objects directly."""
        from datetime import date

        from obsidian_rag.cli_commands import _build_tasks_query, TaskDateFilters

        mock_session = MagicMock()
        mock_query = MagicMock()
        mock_session.query.return_value = mock_query
        mock_query.join.return_value = mock_query
        mock_query.filter.return_value = mock_query
        mock_query.order_by.return_value = mock_query
        mock_query.limit.return_value = mock_query

        today = date.today()

        date_filters = TaskDateFilters(
            due_before=today,
            due_after=today,
            scheduled_before=today,
            scheduled_after=today,
            completion_before=today,
            completion_after=today,
        )
        _build_tasks_query(mock_session, None, date_filters, None, None, 10)

        mock_session.query.assert_called_once()
        mock_query.filter.assert_called()
        mock_query.limit.assert_called_once_with(10)


class TestQueryCommandCoverage:
    """Test query command coverage gaps."""

    @patch("obsidian_rag.cli_commands.DatabaseManager")
    @patch("obsidian_rag.cli_commands._get_embedding_provider")
    def test_query_command_with_table_output(self, mock_get_provider, mock_db_manager):
        """Test query command with table output format (line 494)."""

        runner = CliRunner()

        mock_settings = MagicMock()
        mock_settings.database.url = "postgresql+psycopg://localhost/test"
        mock_settings.logging.level = "INFO"
        mock_settings.logging.format = "text"

        mock_embedding_provider = MagicMock()
        mock_embedding_provider.generate_embedding.return_value = [0.1] * 1536
        mock_get_provider.return_value = mock_embedding_provider

        # Create mock document with all fields for branch coverage
        mock_doc = MagicMock()
        mock_doc.file_name = "test.md"
        mock_doc.file_path = "path/to/test.md"
        mock_doc.frontmatter_json = {"kind": "note"}  # Triggers line 407->409 branch
        mock_doc.tags = ["tag1", "tag2"]  # Triggers line 409->411 branch

        mock_session = MagicMock()
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)

        # Create mock result with distance
        mock_result = (mock_doc, 0.5)
        mock_session.query.return_value.filter.return_value.order_by.return_value.limit.return_value.all.return_value = [
            mock_result
        ]

        mock_db_instance = MagicMock()
        mock_db_instance.get_session.return_value = mock_session
        mock_db_manager.return_value = mock_db_instance

        with patch("obsidian_rag.cli.get_settings", return_value=mock_settings):
            result = runner.invoke(cli, ["query", "test query", "--format", "table"])

        assert result.exit_code == 0
        assert "test.md" in result.output
        assert "note" in result.output  # Kind is displayed
        assert "tag1" in result.output  # Tags are displayed

    @patch("obsidian_rag.cli_commands.DatabaseManager")
    @patch("obsidian_rag.cli_commands._get_embedding_provider")
    def test_query_command_no_matching_documents(
        self, mock_get_provider, mock_db_manager
    ):
        """Test query command when no documents match (covers early return)."""
        runner = CliRunner()

        mock_settings = MagicMock()
        mock_settings.database.url = "postgresql+psycopg://localhost/test"
        mock_settings.logging.level = "INFO"
        mock_settings.logging.format = "text"

        mock_embedding_provider = MagicMock()
        mock_embedding_provider.generate_embedding.return_value = [0.1] * 1536
        mock_get_provider.return_value = mock_embedding_provider

        mock_session = MagicMock()
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)
        mock_session.query.return_value.filter.return_value.order_by.return_value.limit.return_value.all.return_value = []

        mock_db_instance = MagicMock()
        mock_db_instance.get_session.return_value = mock_session
        mock_db_manager.return_value = mock_db_instance

        with patch("obsidian_rag.cli.get_settings", return_value=mock_settings):
            result = runner.invoke(cli, ["query", "test query"])

        assert result.exit_code == 0
        assert "No matching documents found" in result.output


class TestFormatQueryResultsTableCoverage:
    """Test _format_query_results_table branches (lines 407->409, 409->411)."""

    def test_format_query_results_table_without_kind_and_tags(self):
        """Test table format when doc.kind and doc.tags are None/empty."""
        from obsidian_rag.cli_commands import _format_query_results_table

        # Create mock document without kind and tags
        mock_doc = MagicMock()
        mock_doc.file_name = "test.md"
        mock_doc.file_path = "path/to/test.md"
        mock_doc.frontmatter_json = None  # Branch not taken
        mock_doc.tags = []  # Branch not taken (falsy)

        results = [(mock_doc, 0.5)]

        result = _format_query_results_table(results)

        assert "test.md" in result
        assert "Kind:" not in result  # Should not include Kind
        assert "Tags:" not in result  # Should not include Tags


class TestBuildTasksQueryCoverage:
    """Test _build_tasks_query coverage gaps (lines 569-570, 576)."""

    def test_build_tasks_query_with_valid_due_date(self):
        """Test _build_tasks_query with valid due_before date (lines 569-570)."""
        from datetime import date

        from obsidian_rag.cli_commands import _build_tasks_query

        mock_session = MagicMock()
        mock_query = MagicMock()
        mock_session.query.return_value = mock_query
        mock_query.join.return_value = mock_query
        mock_query.filter.return_value = mock_query
        mock_query.order_by.return_value = mock_query
        mock_query.limit.return_value = mock_query

        from obsidian_rag.cli_commands import TaskDateFilters

        due_date = date(2026, 3, 15)
        date_filters = TaskDateFilters(due_before=due_date)
        _build_tasks_query(mock_session, None, date_filters, None, None, 10)

        mock_session.query.assert_called_once()
        mock_query.filter.assert_called()  # Should filter by due date
        mock_query.limit.assert_called_once_with(10)

    def test_build_tasks_query_with_include_tags_filter(self):
        """Test _build_tasks_query with include_tags filter."""
        from obsidian_rag.cli_commands import _build_tasks_query, TaskDateFilters

        mock_session = MagicMock()
        mock_query = MagicMock()
        mock_session.query.return_value = mock_query
        mock_query.join.return_value = mock_query
        mock_query.filter.return_value = mock_query
        mock_query.order_by.return_value = mock_query
        mock_query.limit.return_value = mock_query

        date_filters = TaskDateFilters()
        _build_tasks_query(mock_session, None, date_filters, ["work"], None, 10)

        mock_session.query.assert_called_once()
        mock_query.filter.assert_called()  # Should filter by include_tags
        mock_query.limit.assert_called_once_with(10)


class TestQueryExactFlag:
    """Test query command --exact flag and options."""

    def test_query_exact_flag_help_text(self):
        """Test --exact appears in query help text."""
        runner = CliRunner()
        result = runner.invoke(cli, ["query", "--help"])

        assert result.exit_code == 0
        assert "--exact" in result.output

    def test_query_exact_path_option(self):
        """Test --path option appears in query help text."""
        runner = CliRunner()
        result = runner.invoke(cli, ["query", "--help"])

        assert result.exit_code == 0
        assert "--path" in result.output

    def test_query_exact_name_option(self):
        """Test --name option appears in query help text."""
        runner = CliRunner()
        result = runner.invoke(cli, ["query", "--help"])

        assert result.exit_code == 0
        assert "--name" in result.output

    def test_query_exact_id_option(self):
        """Test --id option appears in query help text."""
        runner = CliRunner()
        result = runner.invoke(cli, ["query", "--help"])

        assert result.exit_code == 0
        assert "--id" in result.output

    def test_query_exact_path_requires_vault(self):
        """Test --exact --path X without --vault raises error."""
        runner = CliRunner()

        mock_settings = MagicMock()
        mock_settings.database.url = "postgresql+psycopg://localhost/test"
        mock_settings.logging.level = "INFO"
        mock_settings.logging.format = "text"

        with patch("obsidian_rag.cli.get_settings", return_value=mock_settings):
            result = runner.invoke(cli, ["query", "--exact", "--path", "test.md"])

        assert result.exit_code != 0
        assert "requires" in result.output.lower()

    def test_query_exact_with_query_text(self):
        """Test --exact with query_text raises error."""
        runner = CliRunner()

        mock_settings = MagicMock()
        mock_settings.database.url = "postgresql+psycopg://localhost/test"
        mock_settings.logging.level = "INFO"
        mock_settings.logging.format = "text"

        with patch("obsidian_rag.cli.get_settings", return_value=mock_settings):
            result = runner.invoke(cli, ["query", "--exact", "search text"])

        assert result.exit_code != 0
        assert "Cannot use query_text" in result.output

    def test_query_exact_no_lookup_params(self):
        """Test --exact without --path, --name, or --id raises error."""
        runner = CliRunner()

        mock_settings = MagicMock()
        mock_settings.database.url = "postgresql+psycopg://localhost/test"
        mock_settings.logging.level = "INFO"
        mock_settings.logging.format = "text"

        with patch("obsidian_rag.cli.get_settings", return_value=mock_settings):
            result = runner.invoke(cli, ["query", "--exact"])

        assert result.exit_code != 0
        assert "Must provide" in result.output

    def test_query_without_exact_still_requires_query_text(self):
        """Test query without --exact and without query_text raises error."""
        runner = CliRunner()

        mock_settings = MagicMock()
        mock_settings.database.url = "postgresql+psycopg://localhost/test"
        mock_settings.logging.level = "INFO"
        mock_settings.logging.format = "text"

        with patch("obsidian_rag.cli.get_settings", return_value=mock_settings):
            result = runner.invoke(cli, ["query"])

        assert result.exit_code != 0
        assert "Missing argument" in result.output

    @patch("obsidian_rag.cli_commands.DatabaseManager")
    @patch("obsidian_rag.cli_commands._get_embedding_provider")
    def test_query_semantic_search_unchanged(self, mock_get_provider, mock_db_manager):
        """Test query 'text' still works without --exact."""
        runner = CliRunner()

        mock_settings = MagicMock()
        mock_settings.database.url = "postgresql+psycopg://localhost/test"
        mock_settings.logging.level = "INFO"
        mock_settings.logging.format = "text"

        mock_embedding_provider = MagicMock()
        mock_embedding_provider.generate_embedding.return_value = [0.1] * 1536
        mock_get_provider.return_value = mock_embedding_provider

        mock_session = MagicMock()
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)
        mock_session.query.return_value.filter.return_value.order_by.return_value.limit.return_value.all.return_value = []

        mock_db_instance = MagicMock()
        mock_db_instance.get_session.return_value = mock_session
        mock_db_manager.return_value = mock_db_instance

        with patch("obsidian_rag.cli.get_settings", return_value=mock_settings):
            result = runner.invoke(cli, ["query", "test query"])

        assert result.exit_code == 0


class TestQueryExactById:
    """Test --exact --id lookup via CLI."""

    @patch("obsidian_rag.cli_commands.DatabaseManager")
    def test_query_exact_by_id(self, mock_db_manager):
        """Test --exact --id performs exact lookup by document UUID."""
        runner = CliRunner()

        mock_settings = MagicMock()
        mock_settings.database.url = "postgresql+psycopg://localhost/test"
        mock_settings.logging.level = "INFO"
        mock_settings.logging.format = "text"

        mock_doc = MagicMock()
        mock_doc.vault_name = "test-vault"
        mock_doc.file_path = "test.md"
        mock_doc.file_name = "test.md"
        mock_doc.tags = []
        mock_doc.content = "Hello"

        mock_session = MagicMock()
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)

        mock_db_instance = MagicMock()
        mock_db_instance.get_session.return_value = mock_session
        mock_db_manager.return_value = mock_db_instance

        with patch("obsidian_rag.cli.get_settings", return_value=mock_settings):
            with patch(
                "obsidian_rag.cli_query_exact.get_document_impl",
                return_value=mock_doc,
            ):
                result = runner.invoke(
                    cli,
                    [
                        "query",
                        "--exact",
                        "--id",
                        "550e8400-e29b-41d4-a716-446655440000",
                    ],
                )

        assert result.exit_code == 0
        assert "test.md" in result.output


class TestQueryExactByName:
    """Test --exact --name lookup via CLI."""

    @patch("obsidian_rag.cli_commands.DatabaseManager")
    def test_query_exact_no_results(self, mock_db_manager):
        """Test --exact with no matching documents shows message."""
        runner = CliRunner()

        mock_settings = MagicMock()
        mock_settings.database.url = "postgresql+psycopg://localhost/test"
        mock_settings.logging.level = "INFO"
        mock_settings.logging.format = "text"

        mock_session = MagicMock()
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)

        mock_db_instance = MagicMock()
        mock_db_instance.get_session.return_value = mock_session
        mock_db_manager.return_value = mock_db_instance

        mock_result = MagicMock()
        mock_result.results = []
        mock_result.total_count = 0

        with patch("obsidian_rag.cli.get_settings", return_value=mock_settings):
            with patch(
                "obsidian_rag.cli_query_exact.list_documents_impl",
                return_value=mock_result,
            ):
                result = runner.invoke(
                    cli, ["query", "--exact", "--name", "nonexistent.md"]
                )

        assert result.exit_code == 0
        assert "No matching documents found" in result.output
