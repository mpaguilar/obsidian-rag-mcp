"""Unit tests for MCP server module."""

import asyncio
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


class TestIngestForceParameter:
    """Tests for force parameter in ingest tool."""

    def test_generate_request_id_force_produces_different_id(self):
        """Test force=True vs force=False produces different request IDs."""
        from obsidian_rag.mcp_server.ingest_helpers import _generate_request_id

        id_no_force = _generate_request_id(
            "test-vault", None, no_delete=False, force=False
        )
        id_force = _generate_request_id("test-vault", None, no_delete=False, force=True)

        assert id_no_force != id_force

    def test_generate_request_id_force_same_params_same_id(self):
        """Test identical params including force produce same ID."""
        from obsidian_rag.mcp_server.ingest_helpers import _generate_request_id

        id1 = _generate_request_id("test-vault", "/path", no_delete=True, force=True)
        id2 = _generate_request_id("test-vault", "/path", no_delete=True, force=True)

        assert id1 == id2

    def test_check_and_handle_duplicate_accepts_force(self):
        """Test _check_and_handle_duplicate accepts force parameter."""
        from obsidian_rag.mcp_server.ingest_helpers import _check_and_handle_duplicate
        from obsidian_rag.mcp_server.server import (
            _clear_ingest_tracker,
            _get_ingest_tracker,
        )

        _clear_ingest_tracker()
        tracker = _get_ingest_tracker()
        result = _check_and_handle_duplicate(
            tracker,
            "test-id",
            "test-vault",
            None,
            no_delete=False,
            force=True,
        )
        assert result is None
        _clear_ingest_tracker()

    @patch("obsidian_rag.mcp_server.server._get_registry")
    @patch("obsidian_rag.mcp_server.server._ingest_handler")
    def test_ingest_tool_accepts_force_parameter(self, mock_handler, mock_get_registry):
        """Test ingest() accepts force parameter and passes to IngestHandlerParams."""
        from obsidian_rag.mcp_server.handlers import IngestHandlerParams
        from obsidian_rag.mcp_server.server import (
            _clear_ingest_tracker,
            ingest,
        )

        _clear_ingest_tracker()

        mock_registry = MagicMock()
        mock_registry.settings = MagicMock()
        mock_registry.db_manager = MagicMock()
        mock_registry.embedding_provider = None
        mock_get_registry.return_value = mock_registry

        mock_handler.return_value = {"total": 0}

        ingest("test-vault", force=True)

        # Verify _ingest_handler was called
        mock_handler.assert_called_once()
        call_params = mock_handler.call_args.args[0]
        assert isinstance(call_params, IngestHandlerParams)
        assert call_params.force is True

        _clear_ingest_tracker()

    @patch("obsidian_rag.mcp_server.server._get_registry")
    @patch("obsidian_rag.mcp_server.server._ingest_handler")
    def test_ingest_tool_force_generates_different_request_id(
        self, mock_handler, mock_get_registry
    ):
        """Test ingest with force=True vs False generates different request IDs."""
        import hashlib
        import json

        from obsidian_rag.mcp_server.server import (
            _clear_ingest_tracker,
            ingest,
        )

        _clear_ingest_tracker()

        mock_registry = MagicMock()
        mock_registry.settings = MagicMock()
        mock_registry.db_manager = MagicMock()
        mock_registry.embedding_provider = None
        mock_get_registry.return_value = mock_registry

        mock_handler.return_value = {"total": 0}

        generated_ids = []

        def capture_id(*args, **kwargs):
            params = {
                "vault_name": kwargs.get("vault_name") or args[0],
                "path": kwargs.get("path") if "path" in kwargs else args[1],
                "no_delete": kwargs.get("no_delete"),
                "force": kwargs.get("force"),
            }
            params_json = json.dumps(params, sort_keys=True, separators=(",", ":"))
            request_id = hashlib.md5(params_json.encode()).hexdigest()
            generated_ids.append(request_id)
            return request_id

        with patch(
            "obsidian_rag.mcp_server.server._generate_request_id",
            side_effect=capture_id,
        ):
            ingest("test-vault", force=False)
            ingest("test-vault", force=True)

        assert len(generated_ids) == 2
        assert generated_ids[0] != generated_ids[1]

        _clear_ingest_tracker()

    def test_ingest_force_not_cached_for_non_force(self):
        """Test force=True request does not return non-force cached result."""
        from obsidian_rag.mcp_server.ingest_helpers import _generate_request_id
        from obsidian_rag.mcp_server.ingest_tracker import IngestRequestTracker

        # First call without force
        id_no_force = _generate_request_id(
            "test-vault", None, no_delete=False, force=False
        )
        tracker = IngestRequestTracker()
        asyncio.run(tracker.complete_request(id_no_force, {"cached": True}))

        # Second call with force should have different ID
        id_force = _generate_request_id("test-vault", None, no_delete=False, force=True)
        assert id_no_force != id_force

        # Verify tracker doesn't return cached result for force request
        should_process, cached = asyncio.run(tracker.start_request(id_force, {}))
        assert should_process is True
        assert cached is None

    def test_ingest_handler_params_force_field(self):
        """Test IngestHandlerParams with force=True."""
        from obsidian_rag.mcp_server.handlers import IngestHandlerParams

        params = IngestHandlerParams(
            settings=MagicMock(),
            db_manager=MagicMock(),
            embedding_provider=None,
            vault_name="test",
            path_override=None,
            no_delete=False,
            force=True,
        )
        assert params.force is True


class TestIngestNoDeleteNone:
    """Tests for TASK-003: no_delete parameter accepts None (unspecified)."""

    @patch("obsidian_rag.mcp_server.server._get_registry")
    @patch("obsidian_rag.mcp_server.server._ingest_handler")
    def test_ingest_no_delete_none_propagates_to_handler_params(
        self, mock_handler: MagicMock, mock_get_registry: MagicMock
    ):
        """Test no_delete=None is passed through to IngestHandlerParams."""
        from obsidian_rag.mcp_server.server import (
            _clear_ingest_tracker,
            ingest,
        )

        _clear_ingest_tracker()

        mock_registry = MagicMock()
        mock_get_registry.return_value = mock_registry
        mock_handler.return_value = {"total": 0}

        result = ingest("vault", "/path", no_delete=None)

        call_params = mock_handler.call_args.args[0]
        assert call_params.no_delete is None
        assert result == {"total": 0}

        _clear_ingest_tracker()

    @patch("obsidian_rag.mcp_server.server._check_and_handle_duplicate")
    @patch("obsidian_rag.mcp_server.server._get_registry")
    @patch("obsidian_rag.mcp_server.server._ingest_handler")
    def test_ingest_no_delete_none_propagates_to_request_id(
        self,
        mock_handler: MagicMock,
        mock_get_registry: MagicMock,
        mock_dup_check: MagicMock,
    ):
        """Test no_delete=None flows to _generate_request_id."""
        from obsidian_rag.mcp_server.server import (
            _clear_ingest_tracker,
            ingest,
        )

        _clear_ingest_tracker()

        mock_dup_check.return_value = None
        mock_registry = MagicMock()
        mock_get_registry.return_value = mock_registry
        mock_handler.return_value = {"total": 0}

        with patch("obsidian_rag.mcp_server.server._generate_request_id") as mock_gen:
            mock_gen.return_value = "test-id"
            ingest("vault", "/path")
            mock_gen.assert_called_once_with(
                "vault", "/path", no_delete=None, force=False
            )

        _clear_ingest_tracker()

    @patch("obsidian_rag.mcp_server.server._check_and_handle_duplicate")
    @patch("obsidian_rag.mcp_server.server._get_registry")
    @patch("obsidian_rag.mcp_server.server._ingest_handler")
    def test_ingest_no_delete_none_propagates_to_duplicate_check(
        self,
        mock_handler: MagicMock,
        mock_get_registry: MagicMock,
        mock_dup_check: MagicMock,
    ):
        """Test no_delete=None flows to _check_and_handle_duplicate."""
        from obsidian_rag.mcp_server.server import (
            _clear_ingest_tracker,
            ingest,
        )

        _clear_ingest_tracker()

        mock_dup_check.return_value = None
        mock_registry = MagicMock()
        mock_get_registry.return_value = mock_registry
        mock_handler.return_value = {"total": 0}

        ingest("vault", "/path")

        mock_dup_check.assert_called_once()
        call_kwargs = mock_dup_check.call_args.kwargs
        assert call_kwargs["no_delete"] is None

        _clear_ingest_tracker()

    @patch("obsidian_rag.mcp_server.server._get_registry")
    @patch("obsidian_rag.mcp_server.server._ingest_handler")
    def test_ingest_no_delete_not_passed_defaults_to_none(
        self, mock_handler: MagicMock, mock_get_registry: MagicMock
    ):
        """Test omitting no_delete defaults to None."""
        from obsidian_rag.mcp_server.server import (
            _clear_ingest_tracker,
            ingest,
        )

        _clear_ingest_tracker()

        mock_registry = MagicMock()
        mock_get_registry.return_value = mock_registry
        mock_handler.return_value = {"total": 0}

        ingest("vault", "/path")

        call_params = mock_handler.call_args.args[0]
        assert call_params.no_delete is None

        _clear_ingest_tracker()

    @patch("obsidian_rag.mcp_server.server._get_registry")
    @patch("obsidian_rag.mcp_server.server._ingest_handler")
    def test_ingest_no_delete_false_still_works(
        self, mock_handler: MagicMock, mock_get_registry: MagicMock
    ):
        """Test explicit False is still accepted."""
        from obsidian_rag.mcp_server.server import (
            _clear_ingest_tracker,
            ingest,
        )

        _clear_ingest_tracker()

        mock_registry = MagicMock()
        mock_get_registry.return_value = mock_registry
        mock_handler.return_value = {"total": 0}

        ingest("vault", "/path", no_delete=False)

        call_params = mock_handler.call_args.args[0]
        assert call_params.no_delete is False

        _clear_ingest_tracker()

    @patch("obsidian_rag.mcp_server.server._get_registry")
    @patch("obsidian_rag.mcp_server.server._ingest_handler")
    def test_ingest_no_delete_true_still_works(
        self, mock_handler: MagicMock, mock_get_registry: MagicMock
    ):
        """Test explicit True is still accepted."""
        from obsidian_rag.mcp_server.server import (
            _clear_ingest_tracker,
            ingest,
        )

        _clear_ingest_tracker()

        mock_registry = MagicMock()
        mock_get_registry.return_value = mock_registry
        mock_handler.return_value = {"total": 0}

        ingest("vault", "/path", no_delete=True)

        call_params = mock_handler.call_args.args[0]
        assert call_params.no_delete is True

        _clear_ingest_tracker()
