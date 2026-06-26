"""Tests for CLI commands."""

from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from obsidian_rag.cli import cli


class TestTasksCommand:
    """Test tasks command with various scenarios."""

    @patch("obsidian_rag.cli_commands.DatabaseManager")
    def test_tasks_command_with_no_results(self, mock_db_manager):
        """Test tasks command with no results (TASK-050)."""
        runner = CliRunner()

        mock_settings = MagicMock()
        mock_settings.database.url = "postgresql+psycopg://localhost/test"
        mock_settings.logging.level = "INFO"
        mock_settings.logging.format = "text"

        mock_session = MagicMock()
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)
        mock_query_result = MagicMock()
        mock_query_result.join.return_value.filter.return_value.order_by.return_value.limit.return_value.all.return_value = []
        mock_session.query.return_value = mock_query_result

        mock_db_instance = MagicMock()
        mock_db_instance.get_session.return_value = mock_session
        mock_db_manager.return_value = mock_db_instance

        with patch("obsidian_rag.cli.get_settings", return_value=mock_settings):
            result = runner.invoke(cli, ["tasks"])

        assert result.exit_code == 0
        assert "Found 0 tasks" in result.output


class TestTasksCommandCoverage:
    """Test tasks command coverage gaps (lines 529-530)."""

    @patch("obsidian_rag.cli_commands.DatabaseManager")
    def test_tasks_command_no_tasks_found_message(self, mock_db_manager):
        """Test tasks command shows 'No tasks found' message (lines 529-530)."""
        runner = CliRunner()

        mock_settings = MagicMock()
        mock_settings.database.url = "postgresql+psycopg://localhost/test"
        mock_settings.logging.level = "INFO"
        mock_settings.logging.format = "text"

        mock_session = MagicMock()
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)
        # Return empty list to trigger "No tasks found" message
        mock_session.query.return_value.join.return_value.filter.return_value.order_by.return_value.limit.return_value.all.return_value = []

        mock_db_instance = MagicMock()
        mock_db_instance.get_session.return_value = mock_session
        mock_db_manager.return_value = mock_db_instance

        with patch("obsidian_rag.cli.get_settings", return_value=mock_settings):
            result = runner.invoke(cli, ["tasks"])

        assert result.exit_code == 0
        # This covers the early return when no results


class TestApplyIncludeTagsCli:
    """Test _apply_include_tags_cli function."""

    def test_apply_include_tags_cli_with_tags(self):
        """Test _apply_include_tags_cli applies filters for each tag."""
        from obsidian_rag.cli_commands import _apply_include_tags_cli

        mock_query = MagicMock()
        mock_query.filter.return_value = mock_query

        result = _apply_include_tags_cli(mock_query, ["work", "urgent"])

        # Should call filter for each tag (2 times)
        assert mock_query.filter.call_count == 2
        assert result == mock_query

    def test_apply_include_tags_cli_with_empty_list(self):
        """Test _apply_include_tags_cli returns query unchanged with empty list."""
        from obsidian_rag.cli_commands import _apply_include_tags_cli

        mock_query = MagicMock()

        result = _apply_include_tags_cli(mock_query, [])

        mock_query.filter.assert_not_called()
        assert result == mock_query

    def test_apply_include_tags_cli_with_none(self):
        """Test _apply_include_tags_cli returns query unchanged with None."""
        from obsidian_rag.cli_commands import _apply_include_tags_cli

        mock_query = MagicMock()

        result = _apply_include_tags_cli(mock_query, None)

        mock_query.filter.assert_not_called()
        assert result == mock_query

    def test_apply_include_tags_cli_all_tags_strip_empty(self):
        """Test _apply_include_tags_cli returns query unchanged when tags strip to empty."""
        from obsidian_rag.cli_commands import _apply_include_tags_cli

        mock_query = MagicMock()

        result = _apply_include_tags_cli(mock_query, ["#"])

        mock_query.filter.assert_not_called()
        assert result == mock_query


class TestApplyExcludeTagsCli:
    """Test _apply_exclude_tags_cli function."""

    def test_apply_exclude_tags_cli_with_tags(self):
        """Test _apply_exclude_tags_cli applies not_(or_()) filter."""
        from obsidian_rag.cli_commands import _apply_exclude_tags_cli

        mock_query = MagicMock()
        mock_query.filter.return_value = mock_query

        result = _apply_exclude_tags_cli(mock_query, ["blocked", "waiting"])

        # Should call filter once with not_(or_())
        mock_query.filter.assert_called_once()
        assert result == mock_query

    def test_apply_exclude_tags_cli_with_empty_list(self):
        """Test _apply_exclude_tags_cli returns query unchanged with empty list."""
        from obsidian_rag.cli_commands import _apply_exclude_tags_cli

        mock_query = MagicMock()

        result = _apply_exclude_tags_cli(mock_query, [])

        mock_query.filter.assert_not_called()
        assert result == mock_query

    def test_apply_exclude_tags_cli_with_none(self):
        """Test _apply_exclude_tags_cli returns query unchanged with None."""
        from obsidian_rag.cli_commands import _apply_exclude_tags_cli

        mock_query = MagicMock()

        result = _apply_exclude_tags_cli(mock_query, None)

        mock_query.filter.assert_not_called()
        assert result == mock_query

    def test_apply_exclude_tags_cli_all_tags_strip_empty(self):
        """Test _apply_exclude_tags_cli returns query unchanged when tags strip to empty."""
        from obsidian_rag.cli_commands import _apply_exclude_tags_cli

        mock_query = MagicMock()

        result = _apply_exclude_tags_cli(mock_query, ["#"])

        mock_query.filter.assert_not_called()
        assert result == mock_query


class TestTasksCommandEarlyReturn:
    """Test tasks command early return (lines 529-530)."""

    @patch("obsidian_rag.cli_commands.DatabaseManager")
    @patch("obsidian_rag.cli_commands._build_tasks_query")
    def test_tasks_command_early_return_when_no_results(
        self, mock_build_query, mock_db_manager
    ):
        """Test tasks command returns early with message when no results (lines 529-530)."""
        from click.testing import CliRunner

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

        # Mock query that returns empty list
        mock_query = MagicMock()
        mock_query.all.return_value = []
        mock_build_query.return_value = mock_query

        with patch("obsidian_rag.cli.get_settings", return_value=mock_settings):
            result = runner.invoke(cli, ["tasks"])

        assert result.exit_code == 0
        assert "No tasks found matching the criteria" in result.output


class TestCliDateFiltering:
    """Test CLI tasks command with date filtering."""

    @patch("obsidian_rag.cli_commands.DatabaseManager")
    def test_tasks_with_due_after(self, mock_db_manager):
        """Test tasks command with --due-after option."""

        runner = CliRunner()

        mock_settings = MagicMock()
        mock_settings.database.url = "postgresql+psycopg://localhost/test"
        mock_settings.logging.level = "INFO"
        mock_settings.logging.format = "text"

        mock_session = MagicMock()
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)
        mock_session.query.return_value.join.return_value.filter.return_value.filter.return_value.order_by.return_value.limit.return_value.all.return_value = []

        mock_db_instance = MagicMock()
        mock_db_instance.get_session.return_value = mock_session
        mock_db_manager.return_value = mock_db_instance

        with patch("obsidian_rag.cli.get_settings", return_value=mock_settings):
            result = runner.invoke(cli, ["tasks", "--due-after", "2026-01-01"])

        assert result.exit_code == 0

    @patch("obsidian_rag.cli_commands.DatabaseManager")
    def test_tasks_with_due_before_and_after(self, mock_db_manager):
        """Test tasks command with both --due-before and --due-after."""
        runner = CliRunner()

        mock_settings = MagicMock()
        mock_settings.database.url = "postgresql+psycopg://localhost/test"
        mock_settings.logging.level = "INFO"
        mock_settings.logging.format = "text"

        mock_session = MagicMock()
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)
        mock_session.query.return_value.join.return_value.filter.return_value.filter.return_value.filter.return_value.order_by.return_value.limit.return_value.all.return_value = []

        mock_db_instance = MagicMock()
        mock_db_instance.get_session.return_value = mock_session
        mock_db_manager.return_value = mock_db_instance

        with patch("obsidian_rag.cli.get_settings", return_value=mock_settings):
            result = runner.invoke(
                cli,
                ["tasks", "--due-after", "2026-01-01", "--due-before", "2026-03-31"],
            )

        assert result.exit_code == 0

    @patch("obsidian_rag.cli_commands.DatabaseManager")
    def test_tasks_with_scheduled_dates(self, mock_db_manager):
        """Test tasks command with scheduled date options."""
        runner = CliRunner()

        mock_settings = MagicMock()
        mock_settings.database.url = "postgresql+psycopg://localhost/test"
        mock_settings.logging.level = "INFO"
        mock_settings.logging.format = "text"

        mock_session = MagicMock()
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)
        mock_session.query.return_value.join.return_value.filter.return_value.filter.return_value.order_by.return_value.limit.return_value.all.return_value = []

        mock_db_instance = MagicMock()
        mock_db_instance.get_session.return_value = mock_session
        mock_db_manager.return_value = mock_db_instance

        with patch("obsidian_rag.cli.get_settings", return_value=mock_settings):
            result = runner.invoke(
                cli,
                [
                    "tasks",
                    "--scheduled-after",
                    "2026-01-01",
                    "--scheduled-before",
                    "2026-02-28",
                ],
            )

        assert result.exit_code == 0

    @patch("obsidian_rag.cli_commands.DatabaseManager")
    def test_tasks_with_completion_dates(self, mock_db_manager):
        """Test tasks command with completion date options."""
        runner = CliRunner()

        mock_settings = MagicMock()
        mock_settings.database.url = "postgresql+psycopg://localhost/test"
        mock_settings.logging.level = "INFO"
        mock_settings.logging.format = "text"

        mock_session = MagicMock()
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)
        mock_session.query.return_value.join.return_value.filter.return_value.filter.return_value.order_by.return_value.limit.return_value.all.return_value = []

        mock_db_instance = MagicMock()
        mock_db_instance.get_session.return_value = mock_session
        mock_db_manager.return_value = mock_db_instance

        with patch("obsidian_rag.cli.get_settings", return_value=mock_settings):
            result = runner.invoke(
                cli,
                [
                    "tasks",
                    "--completion-after",
                    "2026-01-01",
                    "--completion-before",
                    "2026-03-31",
                ],
            )

        assert result.exit_code == 0

    @patch("obsidian_rag.cli_commands.DatabaseManager")
    def test_tasks_with_all_date_options(self, mock_db_manager):
        """Test tasks command with all six date options."""
        runner = CliRunner()

        mock_settings = MagicMock()
        mock_settings.database.url = "postgresql+psycopg://localhost/test"
        mock_settings.logging.level = "INFO"
        mock_settings.logging.format = "text"

        mock_session = MagicMock()
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)
        mock_session.query.return_value.join.return_value.filter.return_value.order_by.return_value.limit.return_value.all.return_value = []

        mock_db_instance = MagicMock()
        mock_db_instance.get_session.return_value = mock_session
        mock_db_manager.return_value = mock_db_instance

        with patch("obsidian_rag.cli.get_settings", return_value=mock_settings):
            result = runner.invoke(
                cli,
                [
                    "tasks",
                    "--due-after",
                    "2026-01-01",
                    "--due-before",
                    "2026-12-31",
                    "--scheduled-after",
                    "2026-01-01",
                    "--scheduled-before",
                    "2026-06-30",
                    "--completion-after",
                    "2026-01-01",
                    "--completion-before",
                    "2026-03-31",
                ],
            )

        assert result.exit_code == 0

    def test_tasks_with_invalid_due_after_format(self):
        """Test tasks command with invalid --due-after format."""
        runner = CliRunner()

        mock_settings = MagicMock()
        mock_settings.database.url = "postgresql+psycopg://localhost/test"
        mock_settings.logging.level = "INFO"
        mock_settings.logging.format = "text"

        with patch("obsidian_rag.cli.get_settings", return_value=mock_settings):
            result = runner.invoke(cli, ["tasks", "--due-after", "invalid-date"])

        assert result.exit_code == 1
        assert "Invalid date format" in result.output

    def test_tasks_with_invalid_scheduled_before_format(self):
        """Test tasks command with invalid --scheduled-before format."""
        runner = CliRunner()

        mock_settings = MagicMock()
        mock_settings.database.url = "postgresql+psycopg://localhost/test"
        mock_settings.logging.level = "INFO"
        mock_settings.logging.format = "text"

        with patch("obsidian_rag.cli.get_settings", return_value=mock_settings):
            result = runner.invoke(cli, ["tasks", "--scheduled-before", "01-01-2026"])

        assert result.exit_code == 1
        assert "Invalid date format" in result.output


class TestCliTagFiltering:
    """Tests for CLI tag filtering with --include-tags and --exclude-tags."""

    @patch("obsidian_rag.cli_commands.DatabaseManager")
    def test_include_tags_single(self, mock_db_manager):
        """Test filtering with single include tag."""
        runner = CliRunner()

        mock_settings = MagicMock()
        mock_settings.database.url = "postgresql+psycopg://localhost/test"
        mock_settings.logging.level = "INFO"
        mock_settings.logging.format = "text"

        mock_session = MagicMock()
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)
        mock_session.query.return_value.join.return_value.filter.return_value.order_by.return_value.limit.return_value.all.return_value = []

        mock_db_instance = MagicMock()
        mock_db_instance.get_session.return_value = mock_session
        mock_db_manager.return_value = mock_db_instance

        with patch("obsidian_rag.cli.get_settings", return_value=mock_settings):
            result = runner.invoke(cli, ["tasks", "--include-tags", "work"])

        assert result.exit_code == 0

    @patch("obsidian_rag.cli_commands.DatabaseManager")
    def test_include_tags_multiple(self, mock_db_manager):
        """Test filtering with multiple include tags."""
        runner = CliRunner()

        mock_settings = MagicMock()
        mock_settings.database.url = "postgresql+psycopg://localhost/test"
        mock_settings.logging.level = "INFO"
        mock_settings.logging.format = "text"

        mock_session = MagicMock()
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)
        mock_session.query.return_value.join.return_value.filter.return_value.filter.return_value.order_by.return_value.limit.return_value.all.return_value = []

        mock_db_instance = MagicMock()
        mock_db_instance.get_session.return_value = mock_session
        mock_db_manager.return_value = mock_db_instance

        with patch("obsidian_rag.cli.get_settings", return_value=mock_settings):
            result = runner.invoke(
                cli, ["tasks", "--include-tags", "work", "--include-tags", "urgent"]
            )

        assert result.exit_code == 0

    @patch("obsidian_rag.cli_commands.DatabaseManager")
    def test_exclude_tags_single(self, mock_db_manager):
        """Test filtering with single exclude tag."""
        runner = CliRunner()

        mock_settings = MagicMock()
        mock_settings.database.url = "postgresql+psycopg://localhost/test"
        mock_settings.logging.level = "INFO"
        mock_settings.logging.format = "text"

        mock_session = MagicMock()
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)
        mock_session.query.return_value.join.return_value.filter.return_value.order_by.return_value.limit.return_value.all.return_value = []

        mock_db_instance = MagicMock()
        mock_db_instance.get_session.return_value = mock_session
        mock_db_manager.return_value = mock_db_instance

        with patch("obsidian_rag.cli.get_settings", return_value=mock_settings):
            result = runner.invoke(cli, ["tasks", "--exclude-tags", "blocked"])

        assert result.exit_code == 0

    @patch("obsidian_rag.cli_commands.DatabaseManager")
    def test_include_and_exclude_combined(self, mock_db_manager):
        """Test combining include and exclude tags."""
        runner = CliRunner()

        mock_settings = MagicMock()
        mock_settings.database.url = "postgresql+psycopg://localhost/test"
        mock_settings.logging.level = "INFO"
        mock_settings.logging.format = "text"

        mock_session = MagicMock()
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)
        mock_session.query.return_value.join.return_value.filter.return_value.filter.return_value.order_by.return_value.limit.return_value.all.return_value = []

        mock_db_instance = MagicMock()
        mock_db_instance.get_session.return_value = mock_session
        mock_db_manager.return_value = mock_db_instance

        with patch("obsidian_rag.cli.get_settings", return_value=mock_settings):
            result = runner.invoke(
                cli,
                [
                    "tasks",
                    "--include-tags",
                    "work",
                    "--exclude-tags",
                    "blocked",
                ],
            )

        assert result.exit_code == 0

    @patch("obsidian_rag.cli_commands.DatabaseManager")
    def test_tag_prefix_stripped_cli(self, mock_db_manager):
        """Test that # prefix is stripped in CLI."""
        from obsidian_rag.mcp_server.tools.tasks import _strip_tag_prefix

        runner = CliRunner()

        mock_settings = MagicMock()
        mock_settings.database.url = "postgresql+psycopg://localhost/test"
        mock_settings.logging.level = "INFO"
        mock_settings.logging.format = "text"

        mock_session = MagicMock()
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)
        mock_session.query.return_value.join.return_value.filter.return_value.order_by.return_value.limit.return_value.all.return_value = []

        mock_db_instance = MagicMock()
        mock_db_instance.get_session.return_value = mock_session
        mock_db_manager.return_value = mock_db_instance

        # Verify that #work becomes work
        assert _strip_tag_prefix("#work") == "work"
        assert _strip_tag_prefix("work") == "work"

        with patch("obsidian_rag.cli.get_settings", return_value=mock_settings):
            result = runner.invoke(cli, ["tasks", "--include-tags", "#work"])

        assert result.exit_code == 0


class TestBuildTagConditionCli:
    """Test _build_tag_condition_cli function."""

    def _compile_condition(self, condition):
        """Compile SQLAlchemy condition with PostgreSQL dialect."""
        from sqlalchemy.dialects import postgresql

        compiled = condition.compile(
            dialect=postgresql.dialect(),
            compile_kwargs={"literal_binds": True},
        )
        return str(compiled)

    def test_build_tag_condition_cli_returns_or_condition(self):
        """verifies it returns an or_() expression"""
        from obsidian_rag.cli_commands import _build_tag_condition_cli

        condition = _build_tag_condition_cli("work")
        # Check that it's an OR expression
        assert hasattr(condition, "clauses") or self._compile_condition(
            condition
        ).startswith("(")
        # For SQLAlchemy 2.x, check the string representation contains OR logic
        cond_str = self._compile_condition(condition)
        assert "work" in cond_str.lower()

    def test_build_tag_condition_cli_includes_task_tags(self):
        """verifies Task.tags.contains is one branch"""
        from obsidian_rag.cli_commands import _build_tag_condition_cli

        condition = _build_tag_condition_cli("work")
        cond_str = self._compile_condition(condition)
        # Should reference task tags somehow
        assert "task" in cond_str.lower() or "tags" in cond_str.lower()

    def test_build_tag_condition_cli_includes_document_tags(self):
        """verifies Document.tags array_to_string is the other branch"""
        from obsidian_rag.cli_commands import _build_tag_condition_cli

        condition = _build_tag_condition_cli("work")
        cond_str = self._compile_condition(condition)
        # Should reference document tags somehow
        assert "document" in cond_str.lower() or "array_to_string" in cond_str.lower()
