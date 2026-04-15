"""Integration tests for JSON string validation bugfix.

These tests simulate real MCP client behavior to verify that
JSON-encoded strings are properly handled.
"""

import json
from unittest.mock import MagicMock, patch

import pytest


class TestMCPClientJsonStringIntegration:
    """Integration tests simulating MCP client with JSON string parameters."""

    @pytest.fixture
    def mock_registry(self):
        """Create a mock registry for testing."""
        from obsidian_rag.mcp_server.tool_definitions import MCPToolRegistry

        registry = MCPToolRegistry(
            db_manager=MagicMock(),
            embedding_provider=MagicMock(),
            settings=MagicMock(),
        )
        return registry

    @pytest.fixture
    def setup_registry(self, mock_registry):
        """Setup and teardown for registry tests."""
        from obsidian_rag.mcp_server import tool_definitions as tool_definitions_module

        original_registry = tool_definitions_module._tool_registry
        tool_definitions_module._tool_registry = mock_registry

        yield mock_registry

        tool_definitions_module._tool_registry = original_registry

    def test_get_documents_by_tag_simulates_librechat_client(self, setup_registry):
        """Test simulating LibreChat client that sends JSON string."""
        from obsidian_rag.mcp_server.server import get_documents_by_tag

        with patch(
            "obsidian_rag.mcp_server.server._get_documents_by_tag_handler"
        ) as mock_handler:
            mock_handler.return_value = {
                "results": [],
                "total_count": 0,
                "has_more": False,
                "next_offset": None,
            }

            # Simulate LibreChat double-encoding
            client_filters = json.dumps(
                {"include_tags": ["business"], "exclude_tags": [], "match_mode": "any"}
            )

            result = get_documents_by_tag(filters=client_filters)

            assert result["total_count"] == 0
            mock_handler.assert_called_once()

    def test_query_documents_simulates_librechat_client(self, setup_registry):
        """Test simulating LibreChat client for query_documents."""
        from obsidian_rag.mcp_server.server import query_documents

        mock_registry = setup_registry
        mock_registry.embedding_provider = MagicMock()
        mock_registry.embedding_provider.generate_embedding.return_value = [0.1] * 1536

        with patch("obsidian_rag.mcp_server.server.query_documents_tool") as mock_tool:
            mock_tool.return_value = {"results": []}

            # Simulate LibreChat double-encoding
            client_filters = json.dumps({"include_tags": ["work"], "match_mode": "all"})

            result = query_documents(
                query="test query",
                filters=client_filters,
            )

            assert result == {"results": []}

    def test_get_tasks_simulates_librechat_client(self, setup_registry):
        """Test simulating LibreChat client for get_tasks."""
        from obsidian_rag.mcp_server.server import get_tasks

        with patch(
            "obsidian_rag.mcp_server.handlers._get_tasks_handler"
        ) as mock_handler:
            mock_handler.return_value = {"results": []}

            # Simulate LibreChat double-encoding with nested objects
            client_params = json.dumps(
                {
                    "status": ["not_completed"],
                    "tag_filters": {"include_tags": ["work"], "match_mode": "all"},
                    "date_filters": {"due_after": "2026-01-01", "match_mode": "any"},
                }
            )

            result = get_tasks(params=client_params)

            assert result == {"results": []}


class TestFullFilterSerialization:
    """Tests for full filter object serialization round-trip."""

    def test_query_filter_params_round_trip(self):
        """Test that QueryFilterParams serializes and deserializes correctly."""
        import json
        from obsidian_rag.mcp_server.handlers import (
            AnnotatedQueryFilter,
            QueryFilterParams,
        )
        from pydantic import TypeAdapter

        # Create a full QueryFilterParams
        original = QueryFilterParams(
            include_properties=[
                {"path": "kind", "operator": "equals", "value": "note"}
            ],
            exclude_properties=None,
            include_tags=["work", "urgent"],
            exclude_tags=["blocked"],
            match_mode="any",
        )

        # Serialize to JSON
        json_str = json.dumps(
            {
                "include_properties": original.include_properties,
                "exclude_properties": original.exclude_properties,
                "include_tags": original.include_tags,
                "exclude_tags": original.exclude_tags,
                "match_mode": original.match_mode,
            }
        )

        # Deserialize through AnnotatedQueryFilter
        adapter = TypeAdapter(AnnotatedQueryFilter)
        result = adapter.validate_python(json_str)

        assert isinstance(result, QueryFilterParams)
        assert result.include_tags == ["work", "urgent"]
        assert result.exclude_tags == ["blocked"]
        assert result.match_mode == "any"
        assert result.include_properties == [
            {"path": "kind", "operator": "equals", "value": "note"}
        ]

    def test_get_tasks_tool_input_round_trip(self):
        """Test that GetTasksToolInput serializes and deserializes correctly."""
        import json
        from obsidian_rag.mcp_server.handlers import (
            AnnotatedGetTasksInput,
            GetTasksToolInput,
            TagFilterStrings,
            TaskDateFilterStrings,
        )
        from pydantic import TypeAdapter

        # Create a full GetTasksToolInput
        original = GetTasksToolInput(
            status=["not_completed", "in_progress"],
            tag_filters=TagFilterStrings(
                include_tags=["work"], exclude_tags=["blocked"], match_mode="all"
            ),
            date_filters=TaskDateFilterStrings(
                due_after="2026-01-01", due_before="2026-12-31", match_mode="any"
            ),
            tags=None,
            priority=["high"],
            limit=50,
            offset=10,
        )

        # Serialize to JSON
        json_str = json.dumps(
            {
                "status": original.status,
                "tag_filters": {
                    "include_tags": original.tag_filters.include_tags,
                    "exclude_tags": original.tag_filters.exclude_tags,
                    "match_mode": original.tag_filters.match_mode,
                },
                "date_filters": {
                    "due_after": original.date_filters.due_after,
                    "due_before": original.date_filters.due_before,
                    "match_mode": original.date_filters.match_mode,
                },
                "tags": original.tags,
                "priority": original.priority,
                "limit": original.limit,
                "offset": original.offset,
            }
        )

        # Deserialize through AnnotatedGetTasksInput
        adapter = TypeAdapter(AnnotatedGetTasksInput)
        result = adapter.validate_python(json_str)

        assert isinstance(result, GetTasksToolInput)
        assert result.status == ["not_completed", "in_progress"]
        assert result.tag_filters.include_tags == ["work"]
        assert result.date_filters.due_after == "2026-01-01"
        assert result.limit == 50
        assert result.offset == 10


class TestBackwardCompatibility:
    """Tests to ensure existing clients continue to work."""

    @pytest.fixture
    def mock_registry(self):
        """Create a mock registry for testing."""
        from obsidian_rag.mcp_server.tool_definitions import MCPToolRegistry

        registry = MCPToolRegistry(
            db_manager=MagicMock(),
            embedding_provider=MagicMock(),
            settings=MagicMock(),
        )
        return registry

    @pytest.fixture
    def setup_registry(self, mock_registry):
        """Setup and teardown for registry tests."""
        from obsidian_rag.mcp_server import tool_definitions as tool_definitions_module

        original_registry = tool_definitions_module._tool_registry
        tool_definitions_module._tool_registry = mock_registry

        yield mock_registry

        tool_definitions_module._tool_registry = original_registry

    def test_get_documents_by_tag_with_none_still_works(self, setup_registry):
        """Test that passing None still works."""
        from obsidian_rag.mcp_server.server import get_documents_by_tag

        with patch(
            "obsidian_rag.mcp_server.server._get_documents_by_tag_handler"
        ) as mock_handler:
            mock_handler.return_value = {"results": []}

            result = get_documents_by_tag(filters=None)

            assert result == {"results": []}

    def test_get_documents_by_tag_with_dataclass_still_works(self, setup_registry):
        """Test that passing dataclass instance still works."""
        from obsidian_rag.mcp_server.server import get_documents_by_tag
        from obsidian_rag.mcp_server.handlers import QueryFilterParams

        with patch(
            "obsidian_rag.mcp_server.server._get_documents_by_tag_handler"
        ) as mock_handler:
            mock_handler.return_value = {"results": []}

            filters = QueryFilterParams(include_tags=["work"])
            result = get_documents_by_tag(filters=filters)

            assert result == {"results": []}

    def test_query_documents_with_dataclass_still_works(self, setup_registry):
        """Test that passing dataclass to query_documents still works."""
        from obsidian_rag.mcp_server.server import query_documents
        from obsidian_rag.mcp_server.handlers import QueryFilterParams

        mock_registry = setup_registry
        mock_registry.embedding_provider = MagicMock()
        mock_registry.embedding_provider.generate_embedding.return_value = [0.1] * 1536

        with patch("obsidian_rag.mcp_server.server.query_documents_tool") as mock_tool:
            mock_tool.return_value = {"results": []}

            filters = QueryFilterParams(include_tags=["work"])
            result = query_documents(query="test", filters=filters)

            assert result == {"results": []}

    def test_get_tasks_with_dataclass_still_works(self, setup_registry):
        """Test that passing dataclass to get_tasks still works."""
        from obsidian_rag.mcp_server.server import get_tasks
        from obsidian_rag.mcp_server.handlers import GetTasksToolInput

        with patch(
            "obsidian_rag.mcp_server.handlers._get_tasks_handler"
        ) as mock_handler:
            mock_handler.return_value = {"results": []}

            params = GetTasksToolInput(status=["not_completed"])
            result = get_tasks(params=params)

            assert result == {"results": []}

    def test_get_tasks_with_none_still_works(self, setup_registry):
        """Test that passing None to get_tasks still works (creates defaults)."""
        from obsidian_rag.mcp_server.server import get_tasks
        from obsidian_rag.mcp_server.handlers import GetTasksToolInput

        with patch(
            "obsidian_rag.mcp_server.handlers._get_tasks_handler"
        ) as mock_handler:
            mock_handler.return_value = {"results": []}

            # This should create default GetTasksToolInput
            # Note: get_tasks doesn't accept None, but tests the dataclass path
            params = GetTasksToolInput()  # All defaults
            result = get_tasks(params=params)

            assert result == {"results": []}


class TestErrorHandling:
    """Tests for error handling with JSON string inputs."""

    def test_malformed_json_raises_validation_error(self):
        """Test that malformed JSON raises ValidationError."""
        from obsidian_rag.mcp_server.handlers import AnnotatedQueryFilter
        from pydantic import TypeAdapter, ValidationError

        adapter = TypeAdapter(AnnotatedQueryFilter)
        malformed = '{"include_tags": ["unclosed bracket"'

        with pytest.raises(ValidationError):
            adapter.validate_python(malformed)

    def test_valid_json_invalid_schema_raises_validation_error(self):
        """Test that valid JSON with invalid schema raises ValidationError."""
        from obsidian_rag.mcp_server.handlers import AnnotatedQueryFilter
        from pydantic import TypeAdapter, ValidationError

        adapter = TypeAdapter(AnnotatedQueryFilter)
        # match_mode must be "all" or "any"
        invalid_value = '{"match_mode": "invalid_value"}'

        with pytest.raises(ValidationError):
            adapter.validate_python(invalid_value)

    def test_wrong_type_in_nested_object_raises_error(self):
        """Test that wrong type in nested object raises error."""
        from obsidian_rag.mcp_server.handlers import AnnotatedGetTasksInput
        from pydantic import TypeAdapter, ValidationError

        adapter = TypeAdapter(AnnotatedGetTasksInput)
        # tag_filters.match_mode must be string "all" or "any"
        invalid_nested = '{"tag_filters": {"match_mode": 123}}'

        with pytest.raises(ValidationError):
            adapter.validate_python(invalid_nested)

    def test_empty_json_object_is_valid_for_get_tasks(self):
        """Test that empty JSON object {} creates default GetTasksToolInput."""
        from obsidian_rag.mcp_server.handlers import (
            AnnotatedGetTasksInput,
            GetTasksToolInput,
        )
        from pydantic import TypeAdapter

        adapter = TypeAdapter(AnnotatedGetTasksInput)

        result = adapter.validate_python("{}")

        assert isinstance(result, GetTasksToolInput)
        assert result.status is None
        assert result.limit == 20
        assert result.offset == 0
