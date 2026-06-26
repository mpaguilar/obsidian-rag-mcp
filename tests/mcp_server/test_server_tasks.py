"""Unit tests for MCP server module."""

from unittest.mock import MagicMock, patch

from click.testing import CliRunner


runner = CliRunner()

runner = CliRunner()


class TestGetTasksServerWrapper:
    """Tests for get_tasks server wrapper function."""

    @patch("obsidian_rag.mcp_server.server._get_registry")
    @patch("obsidian_rag.mcp_server.handlers._get_tasks_handler")
    def test_server_wrapper_passes_tag_filters(self, mock_handler, mock_registry):
        """Test that server wrapper passes tag_filters to handler."""
        mock_registry_instance = MagicMock()
        mock_registry.return_value = mock_registry_instance

        mock_handler.return_value = {
            "results": [],
            "total_count": 0,
            "has_more": False,
            "next_offset": None,
        }

        from obsidian_rag.mcp_server.server import get_tasks
        from obsidian_rag.mcp_server.handlers import GetTasksToolInput, TagFilterStrings

        tag_filters = TagFilterStrings(
            include_tags=["work", "urgent"],
            exclude_tags=["blocked"],
            match_mode="all",
        )

        params = GetTasksToolInput(
            status=["not_completed"],
            tag_filters=tag_filters,
        )
        get_tasks(params=params)

        mock_handler.assert_called_once()
        call_args = mock_handler.call_args
        request = call_args.kwargs["request"]
        assert request.tag_filters is not None
        assert request.tag_filters.include_tags == ["work", "urgent"]
        assert request.tag_filters.exclude_tags == ["blocked"]
        assert request.tag_filters.match_mode == "all"

    @patch("obsidian_rag.mcp_server.server._get_registry")
    @patch("obsidian_rag.mcp_server.handlers._get_tasks_handler")
    def test_server_wrapper_default_tag_filters(self, mock_handler, mock_registry):
        """Test that server wrapper creates default tag_filters when not provided."""
        mock_registry_instance = MagicMock()
        mock_registry.return_value = mock_registry_instance

        mock_handler.return_value = {
            "results": [],
            "total_count": 0,
            "has_more": False,
            "next_offset": None,
        }

        from obsidian_rag.mcp_server.server import get_tasks
        from obsidian_rag.mcp_server.handlers import GetTasksToolInput

        params = GetTasksToolInput(status=["not_completed"])
        get_tasks(params=params)

        mock_handler.assert_called_once()
        call_args = mock_handler.call_args
        request = call_args.kwargs["request"]
        assert request.tag_filters is not None
        assert request.tag_filters.include_tags is None
        assert request.tag_filters.exclude_tags is None
        assert request.tag_filters.match_mode == "all"


class TestGetTasksJsonString:
    """Tests for get_tasks server wrapper with JSON string input."""

    @patch("obsidian_rag.mcp_server.server._get_registry")
    @patch("obsidian_rag.mcp_server.handlers._get_tasks_handler")
    def test_get_tasks_accepts_json_string(self, mock_handler, mock_registry):
        """Test that get_tasks accepts JSON string params."""
        mock_registry_instance = MagicMock()
        mock_registry.return_value = mock_registry_instance

        mock_handler.return_value = {
            "results": [],
            "total_count": 0,
            "has_more": False,
            "next_offset": None,
        }

        from obsidian_rag.mcp_server.server import get_tasks

        # Pass JSON string instead of dataclass
        json_params = '{"status": ["not_completed"], "limit": 10}'
        result = get_tasks(params=json_params)

        assert result == {
            "results": [],
            "total_count": 0,
            "has_more": False,
            "next_offset": None,
        }
        mock_handler.assert_called_once()
        call_args = mock_handler.call_args
        request = call_args.kwargs["request"]
        assert request.status == ["not_completed"]
        assert request.limit == 10

    @patch("obsidian_rag.mcp_server.server._get_registry")
    @patch("obsidian_rag.mcp_server.handlers._get_tasks_handler")
    def test_get_tasks_accepts_dict(self, mock_handler, mock_registry):
        """Test that get_tasks accepts dict params."""
        mock_registry_instance = MagicMock()
        mock_registry.return_value = mock_registry_instance

        mock_handler.return_value = {
            "results": [],
            "total_count": 0,
            "has_more": False,
            "next_offset": None,
        }

        from obsidian_rag.mcp_server.server import get_tasks

        # Pass dict instead of dataclass
        dict_params = {"status": ["completed"], "offset": 5}
        result = get_tasks(params=dict_params)

        assert result == {
            "results": [],
            "total_count": 0,
            "has_more": False,
            "next_offset": None,
        }
        mock_handler.assert_called_once()
        call_args = mock_handler.call_args
        request = call_args.kwargs["request"]
        assert request.status == ["completed"]
        assert request.offset == 5

    @patch("obsidian_rag.mcp_server.server._get_registry")
    @patch("obsidian_rag.mcp_server.handlers._get_tasks_handler")
    def test_get_tasks_accepts_none(self, mock_handler, mock_registry):
        """Test that get_tasks accepts None params and returns all tasks."""
        mock_registry_instance = MagicMock()
        mock_registry.return_value = mock_registry_instance

        mock_handler.return_value = {
            "results": [],
            "total_count": 0,
            "has_more": False,
            "next_offset": None,
        }

        from obsidian_rag.mcp_server.server import get_tasks

        # Pass None
        result = get_tasks(params=None)

        assert result == {
            "results": [],
            "total_count": 0,
            "has_more": False,
            "next_offset": None,
        }
        mock_handler.assert_called_once()
        call_args = mock_handler.call_args
        request = call_args.kwargs["request"]
        # None should create default GetTasksToolInput with default values
        assert request.status is None
        assert request.limit == 20  # Default value

    @patch("obsidian_rag.mcp_server.server._get_registry")
    @patch("obsidian_rag.mcp_server.handlers._get_tasks_handler")
    def test_get_tasks_json_with_nested_filters(self, mock_handler, mock_registry):
        """Test that get_tasks accepts JSON with nested tag_filters.

        Note: When calling directly (not through FastMCP), nested dicts remain
        as dicts rather than being converted to dataclasses. The full Pydantic
        validation with nested conversion only happens when FastMCP invokes
        the tool with AnnotatedGetTasksInput.
        """
        mock_registry_instance = MagicMock()
        mock_registry.return_value = mock_registry_instance

        mock_handler.return_value = {
            "results": [],
            "total_count": 0,
            "has_more": False,
            "next_offset": None,
        }

        from obsidian_rag.mcp_server.server import get_tasks

        # Pass JSON string with nested tag_filters
        json_params = (
            '{"status": ["not_completed"], '
            '"tag_filters": {"include_tags": ["work", "urgent"], "match_mode": "all"}}'
        )
        result = get_tasks(params=json_params)

        assert result == {
            "results": [],
            "total_count": 0,
            "has_more": False,
            "next_offset": None,
        }
        mock_handler.assert_called_once()
        call_args = mock_handler.call_args
        request = call_args.kwargs["request"]
        assert request.status == ["not_completed"]
        # When called directly, nested objects remain as dicts
        assert request.tag_filters == {
            "include_tags": ["work", "urgent"],
            "match_mode": "all",
        }

    @patch("obsidian_rag.mcp_server.server._get_registry")
    @patch("obsidian_rag.mcp_server.handlers._get_tasks_handler")
    def test_get_tasks_with_empty_string(self, mock_handler, mock_registry):
        """Test that get_tasks treats empty string as no params."""
        mock_registry_instance = MagicMock()
        mock_registry.return_value = mock_registry_instance

        mock_handler.return_value = {
            "results": [],
            "total_count": 0,
            "has_more": False,
            "next_offset": None,
        }

        from obsidian_rag.mcp_server.server import get_tasks

        # Pass empty string - should be treated as no params
        result = get_tasks(params="")

        assert result == {
            "results": [],
            "total_count": 0,
            "has_more": False,
            "next_offset": None,
        }
        mock_handler.assert_called_once()
        call_args = mock_handler.call_args
        request = call_args.kwargs["request"]
        # Empty string should create default GetTasksToolInput
        assert request.status is None
        assert request.limit == 20  # Default value

    @patch("obsidian_rag.mcp_server.server._get_registry")
    @patch("obsidian_rag.mcp_server.handlers._get_tasks_handler")
    def test_get_tasks_with_dataclass(self, mock_handler, mock_registry):
        """Test that get_tasks accepts GetTasksToolInput dataclass directly."""
        mock_registry_instance = MagicMock()
        mock_registry.return_value = mock_registry_instance

        mock_handler.return_value = {
            "results": [],
            "total_count": 0,
            "has_more": False,
            "next_offset": None,
        }

        from obsidian_rag.mcp_server.server import get_tasks
        from obsidian_rag.mcp_server.handlers import GetTasksToolInput

        # Pass GetTasksToolInput dataclass directly
        params = GetTasksToolInput(
            status=["completed"],
            limit=15,
            offset=10,
        )
        result = get_tasks(params=params)

        assert result == {
            "results": [],
            "total_count": 0,
            "has_more": False,
            "next_offset": None,
        }
        mock_handler.assert_called_once()
        call_args = mock_handler.call_args
        request = call_args.kwargs["request"]
        assert request.status == ["completed"]
        assert request.limit == 15
        assert request.offset == 10
