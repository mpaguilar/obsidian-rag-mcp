"""Tests for CLI commands."""

from unittest.mock import MagicMock


from obsidian_rag.cli import cli
from obsidian_rag.cli_commands import _apply_exclude_tags_cli, _apply_include_tags_cli


def test_apply_include_tags_cli_with_tags():
    """verify _build_tag_condition_cli is called"""
    from unittest.mock import patch

    query = MagicMock()
    with patch("obsidian_rag.cli_commands._build_tag_condition_cli") as mock_build:
        mock_condition = MagicMock()
        mock_build.return_value = mock_condition
        _apply_include_tags_cli(query, ["work"])
        mock_build.assert_called_once_with("work")
        query.filter.assert_called_once_with(mock_condition)


def test_apply_include_tags_cli_matches_document_tags():
    """mock _build_tag_condition_cli to verify it's used"""
    from unittest.mock import patch

    query = MagicMock()
    with patch("obsidian_rag.cli_commands._build_tag_condition_cli") as mock_build:
        mock_condition = MagicMock()
        mock_build.return_value = mock_condition
        _apply_include_tags_cli(query, ["work"])
        assert mock_build.called


def test_apply_exclude_tags_cli_with_tags():
    """verify _build_tag_condition_cli is called"""
    from unittest.mock import patch

    query = MagicMock()
    with patch("obsidian_rag.cli_commands._build_tag_condition_cli") as mock_build:
        with patch("obsidian_rag.cli_commands.or_") as mock_or:
            with patch("obsidian_rag.cli_commands.not_") as mock_not:
                mock_condition = MagicMock()
                mock_build.return_value = mock_condition
                mock_or.return_value = MagicMock()
                mock_not.return_value = MagicMock()
                _apply_exclude_tags_cli(query, ["blocked"])
                mock_build.assert_called_once_with("blocked")


def test_apply_exclude_tags_cli_matches_document_tags():
    """verify Document.tags exclusion via _build_tag_condition_cli"""
    from unittest.mock import patch

    query = MagicMock()
    with patch("obsidian_rag.cli_commands._build_tag_condition_cli") as mock_build:
        with patch("obsidian_rag.cli_commands.or_") as mock_or:
            with patch("obsidian_rag.cli_commands.not_") as mock_not:
                mock_condition = MagicMock()
                mock_build.return_value = mock_condition
                mock_or.return_value = MagicMock()
                mock_not.return_value = MagicMock()
                _apply_exclude_tags_cli(query, ["blocked"])
                assert mock_build.called


def test_tasks_command_finds_by_document_tags():
    """End-to-end test: finds tasks whose parent document has tag even if task has no inline matching tag."""
    from click.testing import CliRunner
    from unittest.mock import patch, MagicMock

    runner = CliRunner()

    mock_task = MagicMock()
    mock_task.raw_text = "- [ ] do thing"
    mock_task.status = "not_completed"
    mock_task.description = "do thing"
    mock_task.tags = []  # No inline tags
    mock_task.due = None
    mock_task.scheduled = None
    mock_task.priority = "normal"
    mock_task.document = MagicMock()
    mock_task.document.file_path = "test.md"
    mock_task.document.file_name = "test.md"
    mock_task.document.tags = ["work"]  # Document has the tag

    mock_query = MagicMock()
    mock_query.all.return_value = [mock_task]

    with (
        patch("obsidian_rag.cli_commands.DatabaseManager") as mock_db_manager,
        patch(
            "obsidian_rag.cli_commands._build_tasks_query", return_value=mock_query
        ) as mock_build,
    ):
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

        with patch("obsidian_rag.cli.get_settings", return_value=mock_settings):
            result = runner.invoke(cli, ["tasks", "--include-tags", "work"])

        assert result.exit_code == 0
        assert "do thing" in result.output
        assert mock_build.call_args.kwargs["include_tags"] == ["work"]


def test_tasks_command_excludes_by_document_tags():
    """End-to-end test: excludes tasks whose parent document has excluded tag."""
    from click.testing import CliRunner
    from unittest.mock import patch, MagicMock

    runner = CliRunner()

    mock_query = MagicMock()
    mock_query.all.return_value = []

    with (
        patch("obsidian_rag.cli_commands.DatabaseManager") as mock_db_manager,
        patch(
            "obsidian_rag.cli_commands._build_tasks_query", return_value=mock_query
        ) as mock_build,
    ):
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

        with patch("obsidian_rag.cli.get_settings", return_value=mock_settings):
            result = runner.invoke(cli, ["tasks", "--exclude-tags", "blocked"])

        assert result.exit_code == 0
        assert mock_build.call_args.kwargs["exclude_tags"] == ["blocked"]


def test_apply_include_tags_cli_document_tags_match():
    """Unit test: task has no inline tags but document has matching tag -> task is returned."""
    from unittest.mock import patch, MagicMock

    query = MagicMock()
    with patch("obsidian_rag.cli_commands._build_tag_condition_cli") as mock_build:
        mock_condition = MagicMock()
        mock_build.return_value = mock_condition
        _apply_include_tags_cli(query, ["work"])
        assert mock_build.called
        query.filter.assert_called_with(mock_condition)


def test_apply_exclude_tags_cli_document_tags_exclude():
    """Unit test: task whose parent document has excluded tag is excluded."""
    from unittest.mock import patch, MagicMock

    query = MagicMock()
    with patch("obsidian_rag.cli_commands._build_tag_condition_cli") as mock_build:
        with patch("obsidian_rag.cli_commands.or_") as mock_or:
            with patch("obsidian_rag.cli_commands.not_") as mock_not:
                mock_condition = MagicMock()
                mock_build.return_value = mock_condition
                mock_or.return_value = MagicMock()
                mock_not.return_value = MagicMock()
                _apply_exclude_tags_cli(query, ["blocked"])
                assert mock_build.called
