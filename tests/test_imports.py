"""Tests for module imports and exports."""


class TestNewModuleImports:
    """Test that all new modules can be imported."""

    def test_cli_dates_import(self):
        """Test cli_dates module imports."""
        from obsidian_rag.cli_dates import parse_cli_date

        assert callable(parse_cli_date)

    def test_tasks_params_import(self):
        """Test tasks_params module imports."""
        from obsidian_rag.mcp_server.tools.tasks_params import (
            GetTasksFilterParams,
        )

        assert GetTasksFilterParams is not None

    def test_tasks_dates_import(self):
        """Test tasks_dates module imports."""
        from obsidian_rag.mcp_server.tools.tasks_dates import parse_iso_date

        assert callable(parse_iso_date)

    def test_get_tasks_import(self):
        """Test get_tasks function imports."""
        from obsidian_rag.mcp_server.tools.tasks import get_tasks

        assert callable(get_tasks)

    def test_get_tasks_handler_import(self):
        """Test _get_tasks_handler imports."""
        from obsidian_rag.mcp_server.handlers import _get_tasks_handler

        assert callable(_get_tasks_handler)

    def test_get_tasks_tool_import(self):
        """Test get_tasks_tool imports."""
        from obsidian_rag.mcp_server.tool_definitions import get_tasks_tool

        assert callable(get_tasks_tool)
