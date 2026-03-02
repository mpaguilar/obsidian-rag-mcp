"""Tests for CLI module."""

from datetime import date
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import pytest
from click.testing import CliRunner

from obsidian_rag.cli import (
    cli,
    _setup_logging,
    _get_embedding_provider,
    _update_stats,
    _format_query_results_json,
    _format_query_results_table,
    _format_task_results,
    ProcessingContext,
    _scan_vault,
    _create_progress_callback,
    _report_ingest_results,
    _process_single_file_safe,
    _process_files,
    _create_document,
    _update_document,
    _create_tasks,
    _update_tasks,
    _build_tasks_query,
    _search_documents,
)
from obsidian_rag.parsing.scanner import FileInfo


class TestCli:
    """Test cases for CLI."""

    def test_cli_help(self):
        """Test CLI help command."""
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])

        assert result.exit_code == 0
        assert "Obsidian RAG" in result.output

    def test_cli_version(self):
        """Test CLI version."""
        from obsidian_rag import __version__

        assert __version__ == "0.1.0"

    def test_ingest_help(self):
        """Test ingest command help."""
        runner = CliRunner()
        result = runner.invoke(cli, ["ingest", "--help"])

        assert result.exit_code == 0
        assert "ingest" in result.output.lower()

    def test_query_help(self):
        """Test query command help."""
        runner = CliRunner()
        result = runner.invoke(cli, ["query", "--help"])

        assert result.exit_code == 0
        assert "query" in result.output.lower()

    def test_tasks_help(self):
        """Test tasks command help."""
        runner = CliRunner()
        result = runner.invoke(cli, ["tasks", "--help"])

        assert result.exit_code == 0
        assert "tasks" in result.output.lower()

    def test_ingest_nonexistent_path(self):
        """Test ingest with nonexistent path fails."""
        runner = CliRunner()
        result = runner.invoke(cli, ["ingest", "/nonexistent/path"])

        assert result.exit_code != 0


class TestSetupLogging:
    """Test cases for _setup_logging function."""

    def test_setup_logging_text_format(self):
        """Test setting up logging with text format."""
        # Should not raise
        _setup_logging("INFO", "text")

    def test_setup_logging_json_format(self):
        """Test setting up logging with json format."""
        # Should not raise
        _setup_logging("DEBUG", "json")


class TestUpdateStats:
    """Test cases for _update_stats function."""

    def test_update_new(self):
        """Test updating stats with new result."""
        stats = {"new": 0, "updated": 0, "unchanged": 0, "errors": 0}
        _update_stats(stats, "new")
        assert stats["new"] == 1

    def test_update_updated(self):
        """Test updating stats with updated result."""
        stats = {"new": 0, "updated": 0, "unchanged": 0, "errors": 0}
        _update_stats(stats, "updated")
        assert stats["updated"] == 1

    def test_update_unchanged(self):
        """Test updating stats with unchanged result."""
        stats = {"new": 0, "updated": 0, "unchanged": 0, "errors": 0}
        _update_stats(stats, "unchanged")
        assert stats["unchanged"] == 1

    def test_update_unknown_result(self):
        """Test updating stats with unknown result."""
        stats = {"new": 0, "updated": 0, "unchanged": 0, "errors": 0}
        _update_stats(stats, "unknown")  # Should not crash
        assert stats["new"] == 0


class TestFormatQueryResultsJson:
    """Test cases for _format_query_results_json function."""

    def test_format_empty_results(self):
        """Test formatting empty results."""
        result = _format_query_results_json([])
        assert result == "[]"

    def test_format_results(self):
        """Test formatting results."""

        # Create mock document
        class MockDoc:
            def __init__(self):
                self.file_path = "/path/to/file.md"
                self.file_name = "file.md"
                self.kind = "note"
                self.tags = ["tag1", "tag2"]

        mock_doc = MockDoc()
        results = [(mock_doc, 0.5)]

        result = _format_query_results_json(results)
        assert "file.md" in result
        assert "0.5" in result


class TestFormatQueryResultsTable:
    """Test cases for _format_query_results_table function."""

    def test_format_empty_results(self):
        """Test formatting empty results."""
        result = _format_query_results_table([])
        assert "Found 0 results" in result

    def test_format_results(self):
        """Test formatting results."""

        class MockDoc:
            def __init__(self):
                self.file_path = "/path/to/file.md"
                self.file_name = "file.md"
                self.kind = "note"
                self.tags = ["tag1"]

        mock_doc = MockDoc()
        results = [(mock_doc, 0.5)]

        result = _format_query_results_table(results)
        assert "file.md" in result
        assert "0.5000" in result
        assert "note" in result


class TestFormatTaskResults:
    """Test cases for _format_task_results function."""

    def test_format_empty_results(self):
        """Test formatting empty results."""
        result = _format_task_results([])
        assert "Found 0 tasks" in result

    def test_format_task_with_all_fields(self):
        """Test formatting task with all fields."""

        class MockDoc:
            file_name = "test.md"

        class MockTask:
            def __init__(self):
                self.status = "completed"
                self.description = "Test task"
                self.due = "2024-03-15"
                self.priority = "high"
                self.tags = ["work"]
                self.document = MockDoc()

        result = _format_task_results([MockTask()])
        assert "[x] Test task" in result  # completed status
        assert "test.md" in result
        assert "Due: 2024-03-15" in result
        assert "Priority: high" in result
        assert "work" in result

    def test_format_task_normal_priority_not_shown(self):
        """Test that normal priority is not shown in output."""

        class MockDoc:
            file_name = "test.md"

        class MockTask:
            def __init__(self):
                self.status = "not_completed"
                self.description = "Test task"
                self.due = None
                self.priority = "normal"
                self.tags = None
                self.document = MockDoc()

        result = _format_task_results([MockTask()])
        assert "[ ] Test task" in result
        assert "Priority" not in result


class TestProcessingContext:
    """Test cases for ProcessingContext class."""

    def test_context_creation(self):
        """Test creating ProcessingContext."""
        mock_db = Mock()
        ctx = ProcessingContext(
            db_manager=mock_db,
            embedding_provider="mock_provider",
            dry_run=True,
            verbose=False,
            stats={"new": 0},
        )

        assert ctx.db_manager == mock_db
        assert ctx.embedding_provider == "mock_provider"
        assert ctx.dry_run is True
        assert ctx.verbose is False
        assert ctx.stats == {"new": 0}


class TestScanVault:
    """Test cases for _scan_vault function."""

    def test_scan_vault_success(self, tmp_path):
        """Test scanning vault successfully."""
        # Create a markdown file
        (tmp_path / "test.md").write_text("content")

        result = _scan_vault(tmp_path)
        assert len(result) == 1

    @patch("obsidian_rag.cli.scan_markdown_files")
    @patch("obsidian_rag.cli.click.echo")
    @patch("obsidian_rag.cli.sys.exit")
    def test_scan_vault_failure(self, mock_exit, mock_echo, mock_scan):
        """Test scanning vault with error."""
        mock_scan.side_effect = Exception("Scan failed")

        _scan_vault(Path("/fake/path"))

        mock_exit.assert_called_once_with(1)


class TestCreateProgressCallback:
    """Test cases for _create_progress_callback function."""

    def test_callback_verbose(self):
        """Test callback when verbose is True."""
        callback = _create_progress_callback(verbose=True)

        with patch("obsidian_rag.cli.click.echo") as mock_echo:
            callback(10, 20, 8, 2)
            mock_echo.assert_called_once()
            assert "10/20" in mock_echo.call_args[0][0]

    def test_callback_not_verbose(self):
        """Test callback when verbose is False."""
        callback = _create_progress_callback(verbose=False)

        with patch("obsidian_rag.cli.click.echo") as mock_echo:
            callback(10, 20, 8, 2)
            mock_echo.assert_not_called()


class TestReportIngestResults:
    """Test cases for _report_ingest_results function."""

    def test_report_success(self):
        """Test reporting successful ingestion."""
        stats = {"new": 5, "updated": 3, "unchanged": 2, "errors": 0}

        with patch("obsidian_rag.cli.click.echo") as mock_echo:
            _report_ingest_results(10, stats, 5.5)

            assert mock_echo.call_count == 2
            output = " ".join(str(call) for call in mock_echo.call_args_list)
            assert "5 new" in output
            assert "3 updated" in output
            assert "2 unchanged" in output

    def test_report_with_errors(self):
        """Test reporting with errors."""
        stats = {"new": 5, "updated": 3, "unchanged": 2, "errors": 1}

        with patch("obsidian_rag.cli.click.echo") as mock_echo:
            _report_ingest_results(10, stats, 5.5)

            assert mock_echo.call_count == 3
            output = " ".join(str(call) for call in mock_echo.call_args_list)
            assert "Errors: 1" in output


class TestProcessSingleFileSafe:
    """Test cases for _process_single_file_safe function."""

    def test_process_success(self):
        """Test successful file processing."""
        file_info = Mock(spec=FileInfo)
        file_info.path = Path("/test.md")
        file_info.name = "test.md"

        mock_db = Mock()
        mock_provider = Mock()
        stats = {"new": 0, "updated": 0, "unchanged": 0, "errors": 0}

        ctx = ProcessingContext(mock_db, mock_provider, False, False, stats)

        with patch("obsidian_rag.cli._process_single_file", return_value="new"):
            _process_single_file_safe(file_info, ctx)

        assert stats["new"] == 1

    def test_process_error(self):
        """Test file processing with error."""
        file_info = Mock(spec=FileInfo)
        file_info.path = Path("/test.md")
        file_info.name = "test.md"

        mock_db = Mock()
        mock_provider = Mock()
        stats = {"new": 0, "updated": 0, "unchanged": 0, "errors": 0}

        ctx = ProcessingContext(mock_db, mock_provider, False, False, stats)

        with patch(
            "obsidian_rag.cli._process_single_file", side_effect=Exception("Failed")
        ):
            _process_single_file_safe(file_info, ctx)

        assert stats["errors"] == 1

    def test_process_error_verbose(self):
        """Test file processing with error in verbose mode."""
        file_info = Mock(spec=FileInfo)
        file_info.path = Path("/test.md")
        file_info.name = "test.md"

        mock_db = Mock()
        mock_provider = Mock()
        stats = {"new": 0, "updated": 0, "unchanged": 0, "errors": 0}

        ctx = ProcessingContext(mock_db, mock_provider, False, True, stats)

        with patch(
            "obsidian_rag.cli._process_single_file", side_effect=Exception("Failed")
        ):
            with patch("obsidian_rag.cli.click.echo") as mock_echo:
                _process_single_file_safe(file_info, ctx)

                mock_echo.assert_called_once()


class TestProcessFiles:
    """Test cases for _process_files function."""

    def test_process_all_files(self):
        """Test processing all files."""
        file_info1 = Mock(spec=FileInfo)
        file_info2 = Mock(spec=FileInfo)

        mock_db = Mock()
        mock_provider = Mock()

        with patch("obsidian_rag.cli._process_single_file_safe") as mock_process:
            stats = _process_files(
                [file_info1, file_info2], mock_db, mock_provider, False, False
            )

            assert mock_process.call_count == 2
            assert isinstance(stats, dict)


class TestCreateDocument:
    """Test cases for _create_document function."""

    def test_create_with_embedding(self):
        """Test creating document with successful embedding."""
        file_info = Mock(spec=FileInfo)
        file_info.path = Path("/test.md")
        file_info.name = "test.md"
        file_info.checksum = "abc123"
        file_info.created_at = Mock()
        file_info.modified_at = Mock()

        parsed_data = ("note", ["tag"], {"key": "value"}, "content")
        mock_provider = Mock()
        mock_provider.generate_embedding.return_value = [0.1, 0.2, 0.3]

        doc = _create_document(file_info, parsed_data, mock_provider)

        assert doc.file_name == "test.md"
        assert doc.content == "content"
        assert doc.content_vector == [0.1, 0.2, 0.3]
        mock_provider.generate_embedding.assert_called_once_with("content")

    def test_create_embedding_failure(self):
        """Test creating document when embedding generation fails."""
        file_info = Mock(spec=FileInfo)
        file_info.path = Path("/test.md")
        file_info.name = "test.md"
        file_info.checksum = "abc123"
        file_info.created_at = Mock()
        file_info.modified_at = Mock()

        parsed_data = ("note", ["tag"], {"key": "value"}, "content")
        mock_provider = Mock()
        mock_provider.generate_embedding.side_effect = Exception("Failed")

        doc = _create_document(file_info, parsed_data, mock_provider)

        assert doc.content_vector is None


class TestUpdateDocument:
    """Test cases for _update_document function."""

    def test_update_document(self):
        """Test updating document fields."""
        doc = Mock()
        file_info = Mock(spec=FileInfo)
        file_info.checksum = "new_checksum"
        file_info.modified_at = Mock()

        parsed_data = ("note", ["tag"], {"key": "value"}, "new content")

        _update_document(doc, file_info, parsed_data)

        assert doc.content == "new content"
        assert doc.checksum_md5 == "new_checksum"


class TestCreateTasks:
    """Test cases for _create_tasks function."""

    def test_create_tasks(self):
        """Test creating tasks from parsed data."""
        mock_session = Mock()
        mock_doc = Mock()
        mock_doc.id = 1

        # Create a mock parsed task
        mock_parsed_task = Mock()
        mock_parsed_task.raw_text = "- [ ] Task"
        mock_parsed_task.status = "not_completed"
        mock_parsed_task.description = "Task"
        mock_parsed_task.tags = ["work"]
        mock_parsed_task.repeat = None
        mock_parsed_task.scheduled = None
        mock_parsed_task.due = None
        mock_parsed_task.completion = None
        mock_parsed_task.priority = "normal"
        mock_parsed_task.custom_metadata = {}

        parsed_tasks = [(10, mock_parsed_task)]

        _create_tasks(mock_session, mock_doc, parsed_tasks)

        mock_session.add.assert_called_once()


class TestUpdateTasks:
    """Test cases for _update_tasks function."""

    def test_update_tasks(self):
        """Test updating tasks (delete old, create new)."""
        mock_session = Mock()
        mock_doc = Mock()
        mock_doc.id = 1

        mock_query = Mock()
        mock_session.query.return_value.filter_by.return_value = mock_query

        with patch("obsidian_rag.cli._create_tasks") as mock_create:
            _update_tasks(mock_session, mock_doc, [])

            mock_query.delete.assert_called_once()
            mock_create.assert_called_once()


class TestBuildTasksQuery:
    """Test cases for _build_tasks_query function."""

    def test_build_basic_query(self):
        """Test building basic query without filters."""
        mock_session = Mock()
        mock_query = Mock()
        mock_ordered = Mock()
        mock_limited = Mock()
        mock_query.order_by.return_value = mock_ordered
        mock_ordered.limit.return_value = mock_limited
        mock_session.query.return_value.join.return_value = mock_query

        result = _build_tasks_query(mock_session, None, None, None, 10)

        assert result == mock_limited
        mock_query.order_by.assert_called_once()
        mock_ordered.limit.assert_called_once_with(10)

    def test_build_query_with_status(self):
        """Test building query with status filter."""
        mock_session = Mock()
        mock_query = Mock()
        mock_session.query.return_value.join.return_value = mock_query

        _build_tasks_query(mock_session, "completed", None, None, 10)

        mock_query.filter.assert_called_once()

    def test_build_query_with_due_before(self):
        """Test building query with due_before filter."""
        mock_session = Mock()
        mock_query = Mock()
        mock_session.query.return_value.join.return_value = mock_query

        _build_tasks_query(mock_session, None, "2024-12-31", None, 10)

        mock_query.filter.assert_called_once()

    @patch("obsidian_rag.cli.click.echo")
    @patch("obsidian_rag.cli.sys.exit")
    def test_build_query_with_invalid_date(self, mock_exit, mock_echo):
        """Test building query with invalid date format."""
        mock_session = Mock()
        mock_query = Mock()
        mock_session.query.return_value.join.return_value = mock_query

        _build_tasks_query(mock_session, None, "invalid-date", None, 10)

        mock_exit.assert_called_once_with(1)

    def test_build_query_with_tag(self):
        """Test building query with tag filter."""
        mock_session = Mock()
        mock_query = Mock()
        mock_session.query.return_value.join.return_value = mock_query

        _build_tasks_query(mock_session, None, None, "work", 10)

        mock_query.filter.assert_called_once()


class TestSearchDocuments:
    """Test cases for _search_documents function."""

    def test_search_documents(self):
        """Test searching documents."""
        mock_session = Mock()
        mock_query = Mock()
        mock_query.filter.return_value.order_by.return_value.limit.return_value.all.return_value = []
        mock_session.query.return_value = mock_query

        results = _search_documents(mock_session, [0.1, 0.2], 5)

        assert results == []
        mock_session.query.assert_called_once()


class TestGetEmbeddingProvider:
    """Test cases for _get_embedding_provider function."""

    def test_get_provider_with_config(self):
        """Test getting provider with valid config."""
        mock_settings = Mock()
        mock_config = Mock()
        mock_config.provider = "openai"
        mock_config.api_key = "test-key"
        mock_config.model = "text-embedding-3-small"
        mock_config.base_url = None
        mock_settings.get_endpoint_config.return_value = mock_config

        with patch(
            "obsidian_rag.cli.ProviderFactory.create_embedding_provider"
        ) as mock_create:
            _get_embedding_provider(mock_settings)
            mock_create.assert_called_once_with(
                "openai",
                api_key="test-key",
                model="text-embedding-3-small",
                base_url=None,
            )

    def test_get_provider_no_config(self):
        """Test getting provider without config."""
        mock_settings = Mock()
        mock_settings.get_endpoint_config.return_value = None

        with patch(
            "obsidian_rag.cli.ProviderFactory.create_embedding_provider"
        ) as mock_create:
            with patch("obsidian_rag.cli.log.warning"):
                _get_embedding_provider(mock_settings)
                mock_create.assert_called_once_with("openai")


class TestProcessSingleFile:
    """Test cases for _process_single_file function."""

    def test_process_new_document(self):
        """Test processing a new document."""
        from obsidian_rag.cli import _process_single_file

        mock_db = Mock()
        mock_session = Mock()
        mock_session.__enter__ = Mock(return_value=mock_session)
        mock_session.__exit__ = Mock(return_value=None)
        mock_db.get_session.return_value = mock_session

        file_info = Mock()
        file_info.path = Path("/test.md")
        file_info.checksum = "abc123"
        file_info.content = "# Test\n\nContent"

        mock_provider = Mock()
        mock_provider.generate_embedding.return_value = [0.1, 0.2]

        # No existing document
        mock_session.query.return_value.filter_by.return_value.first.return_value = None

        result = _process_single_file(mock_db, file_info, mock_provider, dry_run=False)

        assert result == "new"
        mock_session.add.assert_called_once()

    def test_process_unchanged_document(self):
        """Test processing an unchanged document."""
        from obsidian_rag.cli import _process_single_file

        mock_db = Mock()
        mock_session = Mock()
        mock_session.__enter__ = Mock(return_value=mock_session)
        mock_session.__exit__ = Mock(return_value=None)
        mock_db.get_session.return_value = mock_session

        file_info = Mock()
        file_info.path = Path("/test.md")
        file_info.checksum = "abc123"
        file_info.content = "# Test\n\nContent"

        mock_provider = Mock()

        # Existing document with same checksum
        existing_doc = Mock()
        existing_doc.checksum_md5 = "abc123"
        mock_session.query.return_value.filter_by.return_value.first.return_value = (
            existing_doc
        )

        result = _process_single_file(mock_db, file_info, mock_provider, dry_run=False)

        assert result == "unchanged"

    def test_process_updated_document(self):
        """Test processing an updated document."""
        from obsidian_rag.cli import _process_single_file

        mock_db = Mock()
        mock_session = Mock()
        mock_session.__enter__ = Mock(return_value=mock_session)
        mock_session.__exit__ = Mock(return_value=None)
        mock_db.get_session.return_value = mock_session

        file_info = Mock()
        file_info.path = Path("/test.md")
        file_info.checksum = "new_checksum"
        file_info.content = "# Updated\n\nContent"
        file_info.modified_at = Mock()

        mock_provider = Mock()

        # Existing document with different checksum
        existing_doc = Mock()
        existing_doc.checksum_md5 = "old_checksum"
        existing_doc.id = 1
        mock_session.query.return_value.filter_by.return_value.first.return_value = (
            existing_doc
        )

        # Mock task deletion
        mock_task_query = Mock()
        mock_session.query.return_value.filter_by.return_value = mock_task_query

        result = _process_single_file(mock_db, file_info, mock_provider, dry_run=False)

        assert result == "updated"

    def test_process_dry_run(self):
        """Test processing in dry run mode."""
        from obsidian_rag.cli import _process_single_file

        mock_db = Mock()
        file_info = Mock()
        file_info.content = "# Test\n\nContent"

        mock_provider = Mock()

        result = _process_single_file(mock_db, file_info, mock_provider, dry_run=True)

        assert result == "new"
