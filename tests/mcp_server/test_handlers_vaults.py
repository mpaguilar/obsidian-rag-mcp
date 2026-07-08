"""Tests for vault handler functions and validation utilities."""

import json
from unittest.mock import MagicMock, patch

import pytest
from pydantic import TypeAdapter, ValidationError

from obsidian_rag.mcp_server.handlers import (
    AnnotatedQueryFilter,
    QueryFilterParams,
    _delete_vault_handler,
    _get_all_tags_handler,
    _get_vault_handler,
    _update_vault_handler,
    parse_json_str,
)
from obsidian_rag.mcp_server.tools.vaults_params import VaultUpdateParams


class TestParseJsonStr:
    """Tests for parse_json_str helper function."""

    def test_parse_valid_json_string(self):
        """Test parsing a valid JSON string."""
        json_str = '{"include_tags": ["work"], "match_mode": "any"}'
        result = parse_json_str(json_str)

        assert result == {"include_tags": ["work"], "match_mode": "any"}

    def test_parse_invalid_json_string_raises_error(self):
        """Test that invalid JSON raises ValidationError."""
        invalid_json = "{not valid json}"

        with pytest.raises(json.JSONDecodeError):
            parse_json_str(invalid_json)

    def test_parse_empty_string_returns_none(self):
        """Test that empty string returns None."""
        result = parse_json_str("")
        assert result is None

    def test_parse_whitespace_string_returns_none(self):
        """Test that whitespace-only string returns None."""
        result = parse_json_str("   ")
        assert result is None

    def test_pass_through_dict_unchanged(self):
        """Test that dict is passed through unchanged."""
        input_dict = {"include_tags": ["work"], "match_mode": "any"}
        result = parse_json_str(input_dict)

        assert result is input_dict

    def test_pass_through_none_unchanged(self):
        """Test that None is passed through unchanged."""
        result = parse_json_str(None)
        assert result is None


class TestAnnotatedQueryFilter:
    """Tests for AnnotatedQueryFilter type with BeforeValidator."""

    def test_accepts_dict_input(self):
        """Test that AnnotatedQueryFilter accepts dict input."""
        adapter = TypeAdapter(AnnotatedQueryFilter)
        input_data = {"include_tags": ["work"], "match_mode": "any"}

        result = adapter.validate_python(input_data)

        assert isinstance(result, QueryFilterParams)
        assert result.include_tags == ["work"]
        assert result.match_mode == "any"

    def test_accepts_json_string_input(self):
        """Test that AnnotatedQueryFilter accepts JSON string input."""
        adapter = TypeAdapter(AnnotatedQueryFilter)
        json_str = '{"include_tags": ["work"], "match_mode": "any"}'

        result = adapter.validate_python(json_str)

        assert isinstance(result, QueryFilterParams)
        assert result.include_tags == ["work"]
        assert result.match_mode == "any"

    def test_accepts_none_input(self):
        """Test that AnnotatedQueryFilter accepts None input."""
        adapter = TypeAdapter(AnnotatedQueryFilter)

        result = adapter.validate_python(None)

        assert result is None

    def test_accepts_empty_string_input(self):
        """Test that AnnotatedQueryFilter treats empty string as None."""
        adapter = TypeAdapter(AnnotatedQueryFilter)

        result = adapter.validate_python("")

        assert result is None

    def test_rejects_invalid_json_string(self):
        """Test that AnnotatedQueryFilter rejects invalid JSON."""
        adapter = TypeAdapter(AnnotatedQueryFilter)
        invalid_json = "{not valid}"

        with pytest.raises(ValidationError):
            adapter.validate_python(invalid_json)


class TestParseJsonStrComplex:
    """Additional tests for parse_json_str with complex inputs."""

    def test_parse_nested_object(self):
        """Test parsing JSON string with nested objects."""
        json_str = '{"outer": {"inner": [1, 2, 3]}, "simple": "value"}'
        result = parse_json_str(json_str)

        assert result == {"outer": {"inner": [1, 2, 3]}, "simple": "value"}

    def test_parse_array(self):
        """Test parsing JSON string with array at root."""
        json_str = '["item1", "item2", "item3"]'
        result = parse_json_str(json_str)

        assert result == ["item1", "item2", "item3"]

    def test_parse_boolean(self):
        """Test parsing JSON string with boolean value."""
        json_str = "true"
        result = parse_json_str(json_str)

        assert result is True

    def test_parse_number(self):
        """Test parsing JSON string with number value."""
        json_str = "42"
        result = parse_json_str(json_str)

        assert result == 42

    def test_parse_string(self):
        """Test parsing JSON string with string value."""
        json_str = '"just a string"'
        result = parse_json_str(json_str)

        assert result == "just a string"

    def test_pass_through_list_unchanged(self):
        """Test that list is passed through unchanged."""
        input_list = [1, 2, 3]
        result = parse_json_str(input_list)

        assert result is input_list

    def test_pass_through_int_unchanged(self):
        """Test that int is passed through unchanged."""
        result = parse_json_str(42)
        assert result == 42

    def test_pass_through_bool_unchanged(self):
        """Test that bool is passed through unchanged."""
        result = parse_json_str(True)
        assert result is True


class TestAnnotatedQueryFilterEdgeCases:
    """Edge case tests for AnnotatedQueryFilter."""

    def test_rejects_partial_json(self):
        """Test that partial JSON is rejected."""
        adapter = TypeAdapter(AnnotatedQueryFilter)
        partial_json = '{"include_tags": ["work"]'  # Missing closing brace

        with pytest.raises(ValidationError):
            adapter.validate_python(partial_json)

    def test_rejects_trailing_comma(self):
        """Test that JSON with trailing comma is rejected."""
        adapter = TypeAdapter(AnnotatedQueryFilter)
        json_with_trailing = '{"include_tags": ["work"],}'

        with pytest.raises(ValidationError):
            adapter.validate_python(json_with_trailing)

    def test_rejects_single_quotes(self):
        """Test that JSON with single quotes is rejected."""
        adapter = TypeAdapter(AnnotatedQueryFilter)
        single_quotes = "{'include_tags': ['work']}"

        with pytest.raises(ValidationError):
            adapter.validate_python(single_quotes)

    def test_accepts_unicode_in_json(self):
        """Test that JSON with unicode characters is accepted."""
        adapter = TypeAdapter(AnnotatedQueryFilter)
        json_with_unicode = '{"include_tags": ["café", "naïve"]}'

        result = adapter.validate_python(json_with_unicode)

        assert isinstance(result, QueryFilterParams)
        assert result.include_tags == ["café", "naïve"]

    def test_accepts_whitespace_around_json(self):
        """Test that JSON with surrounding whitespace is accepted."""
        adapter = TypeAdapter(AnnotatedQueryFilter)
        json_with_whitespace = '  \n  {"include_tags": ["work"]}  \n  '

        result = adapter.validate_python(json_with_whitespace)

        assert isinstance(result, QueryFilterParams)
        assert result.include_tags == ["work"]


class TestValidationErrorMessages:
    """Tests for validation error message quality."""

    def test_invalid_json_error_message(self):
        """Test that invalid JSON provides helpful error message."""
        adapter = TypeAdapter(AnnotatedQueryFilter)
        invalid_json = "{not valid json}"

        with pytest.raises(ValidationError) as exc_info:
            adapter.validate_python(invalid_json)

        error_str = str(exc_info.value)
        # Should indicate JSON parsing failed
        assert "JSON" in error_str or "json" in error_str.lower()

    def test_validation_error_includes_location(self):
        """Test that validation error includes field location."""
        adapter = TypeAdapter(AnnotatedQueryFilter)
        # match_mode has wrong type
        json_wrong_type = '{"match_mode": 123}'

        with pytest.raises(ValidationError) as exc_info:
            adapter.validate_python(json_wrong_type)

        error_str = str(exc_info.value)
        # Should mention match_mode field
        assert "match_mode" in error_str


class TestJsonStringSpecialCharacters:
    """Tests for JSON strings with special characters."""

    def test_escapes_quotes(self):
        """Test JSON with escaped quotes."""
        json_str = '{"description": "He said \\"hello\\""}'
        result = parse_json_str(json_str)

        assert result == {"description": 'He said "hello"'}

    def test_escapes_backslash(self):
        """Test JSON with escaped backslash."""
        json_str = '{"path": "C:\\\\Users\\\\test"}'
        result = parse_json_str(json_str)

        assert result == {"path": "C:\\Users\\test"}

    def test_escapes_newline(self):
        """Test JSON with escaped newline."""
        json_str = '{"text": "line1\\nline2"}'
        result = parse_json_str(json_str)

        assert result == {"text": "line1\nline2"}

    def test_null_value(self):
        """Test JSON with explicit null value."""
        json_str = '{"field": null}'
        result = parse_json_str(json_str)

        assert result == {"field": None}


class TestGetVaultHandler:
    """Tests for _get_vault_handler function."""

    @patch("obsidian_rag.mcp_server.handlers.get_vault")
    def test_get_vault_handler_by_name(self, mock_get_vault):
        """Handler returns model_dump when getting vault by name."""
        mock_db_manager = MagicMock()
        mock_session = MagicMock()
        mock_db_manager.get_session.return_value.__enter__.return_value = mock_session
        mock_db_manager.get_session.return_value.__exit__.return_value = False

        mock_vault_response = MagicMock()
        mock_vault_response.model_dump.return_value = {
            "id": "vault-123",
            "name": "TestVault",
            "document_count": 5,
        }
        mock_get_vault.return_value = mock_vault_response

        result = _get_vault_handler(
            db_manager=mock_db_manager,
            vault_name="TestVault",
        )

        mock_get_vault.assert_called_once_with(
            session=mock_session,
            vault_name="TestVault",
            vault_id=None,
        )
        assert result == {"id": "vault-123", "name": "TestVault", "document_count": 5}

    @patch("obsidian_rag.mcp_server.handlers.get_vault")
    def test_get_vault_handler_by_id(self, mock_get_vault):
        """Handler returns model_dump when getting vault by ID."""
        mock_db_manager = MagicMock()
        mock_session = MagicMock()
        mock_db_manager.get_session.return_value.__enter__.return_value = mock_session
        mock_db_manager.get_session.return_value.__exit__.return_value = False

        mock_vault_response = MagicMock()
        mock_vault_response.model_dump.return_value = {
            "id": "vault-456",
            "name": "AnotherVault",
            "document_count": 10,
        }
        mock_get_vault.return_value = mock_vault_response

        result = _get_vault_handler(
            db_manager=mock_db_manager,
            vault_id="vault-456",
        )

        mock_get_vault.assert_called_once_with(
            session=mock_session,
            vault_name=None,
            vault_id="vault-456",
        )
        assert result == {
            "id": "vault-456",
            "name": "AnotherVault",
            "document_count": 10,
        }

    @patch("obsidian_rag.mcp_server.handlers.get_vault")
    def test_get_vault_handler_not_found(self, mock_get_vault):
        """Handler catches ValueError and returns error dict."""
        mock_db_manager = MagicMock()
        mock_session = MagicMock()
        mock_db_manager.get_session.return_value.__enter__.return_value = mock_session
        mock_db_manager.get_session.return_value.__exit__.return_value = False

        mock_get_vault.side_effect = ValueError("Vault 'NonExistent' not found")

        result = _get_vault_handler(
            db_manager=mock_db_manager,
            vault_name="NonExistent",
        )

        assert result == {"success": False, "error": "Vault 'NonExistent' not found"}


class TestUpdateVaultHandler:
    """Tests for _update_vault_handler function."""

    @patch("obsidian_rag.mcp_server.handlers.update_vault")
    def test_update_vault_handler_success(self, mock_update_vault):
        """Handler returns model_dump on successful update."""
        mock_db_manager = MagicMock()
        mock_session = MagicMock()
        mock_db_manager.get_session.return_value.__enter__.return_value = mock_session
        mock_db_manager.get_session.return_value.__exit__.return_value = False

        mock_vault_response = MagicMock()
        mock_vault_response.model_dump.return_value = {
            "id": "vault-123",
            "name": "TestVault",
            "description": "Updated description",
        }
        mock_update_vault.return_value = mock_vault_response

        params = VaultUpdateParams(
            vault_name="TestVault",
            description="Updated description",
        )

        result = _update_vault_handler(
            db_manager=mock_db_manager,
            params=params,
        )

        mock_update_vault.assert_called_once_with(
            session=mock_session,
            params=params,
        )
        assert result == {
            "id": "vault-123",
            "name": "TestVault",
            "description": "Updated description",
        }

    @patch("obsidian_rag.mcp_server.handlers.update_vault")
    def test_update_vault_handler_force_required(self, mock_update_vault):
        """Handler returns error dict when container_path change requires force."""
        mock_db_manager = MagicMock()
        mock_session = MagicMock()
        mock_db_manager.get_session.return_value.__enter__.return_value = mock_session
        mock_db_manager.get_session.return_value.__exit__.return_value = False

        error_dict = {
            "success": False,
            "error": "Changing container_path will delete all documents, tasks, and chunks for this vault and require re-ingestion. Set force=True to confirm.",
        }
        mock_update_vault.return_value = error_dict

        params = VaultUpdateParams(
            vault_name="TestVault",
            container_path="/new/path",
            force=False,
        )

        result = _update_vault_handler(
            db_manager=mock_db_manager,
            params=params,
        )

        assert result == error_dict

    @patch("obsidian_rag.mcp_server.handlers.update_vault")
    def test_update_vault_handler_not_found(self, mock_update_vault):
        """Handler catches ValueError and returns error dict."""
        mock_db_manager = MagicMock()
        mock_session = MagicMock()
        mock_db_manager.get_session.return_value.__enter__.return_value = mock_session
        mock_db_manager.get_session.return_value.__exit__.return_value = False

        mock_update_vault.side_effect = ValueError("Vault 'NonExistent' not found")

        params = VaultUpdateParams(vault_name="NonExistent")

        result = _update_vault_handler(
            db_manager=mock_db_manager,
            params=params,
        )

        assert result == {"success": False, "error": "Vault 'NonExistent' not found"}


class TestDeleteVaultHandler:
    """Tests for _delete_vault_handler function."""

    @patch("obsidian_rag.mcp_server.handlers.delete_vault")
    def test_delete_vault_handler_success(self, mock_delete_vault):
        """Handler returns success dict when deletion confirmed."""
        mock_db_manager = MagicMock()
        mock_session = MagicMock()
        mock_db_manager.get_session.return_value.__enter__.return_value = mock_session
        mock_db_manager.get_session.return_value.__exit__.return_value = False

        success_dict = {
            "success": True,
            "name": "TestVault",
            "id": "vault-123",
            "documents_deleted": 5,
            "tasks_deleted": 10,
            "chunks_deleted": 20,
            "warning": "Vault config entry still exists.",
        }
        mock_delete_vault.return_value = success_dict

        result = _delete_vault_handler(
            db_manager=mock_db_manager,
            vault_name="TestVault",
            confirm=True,
        )

        mock_delete_vault.assert_called_once_with(
            session=mock_session,
            vault_name="TestVault",
            confirm=True,
        )
        assert result == success_dict

    @patch("obsidian_rag.mcp_server.handlers.delete_vault")
    def test_delete_vault_handler_not_confirmed(self, mock_delete_vault):
        """Handler returns error dict when confirm is False."""
        mock_db_manager = MagicMock()
        mock_session = MagicMock()
        mock_db_manager.get_session.return_value.__enter__.return_value = mock_session
        mock_db_manager.get_session.return_value.__exit__.return_value = False

        error_dict = {
            "success": False,
            "error": "confirm=True is required to delete a vault. This action is irreversible and will cascade-delete all associated documents, tasks, and chunks.",
        }
        mock_delete_vault.return_value = error_dict

        result = _delete_vault_handler(
            db_manager=mock_db_manager,
            vault_name="TestVault",
            confirm=False,
        )

        assert result == error_dict

    @patch("obsidian_rag.mcp_server.handlers.delete_vault")
    def test_delete_vault_handler_not_found(self, mock_delete_vault):
        """Handler catches ValueError and returns error dict."""
        mock_db_manager = MagicMock()
        mock_session = MagicMock()
        mock_db_manager.get_session.return_value.__enter__.return_value = mock_session
        mock_db_manager.get_session.return_value.__exit__.return_value = False

        mock_delete_vault.side_effect = ValueError("Vault 'NonExistent' not found")

        result = _delete_vault_handler(
            db_manager=mock_db_manager,
            vault_name="NonExistent",
            confirm=True,
        )

        assert result == {"success": False, "error": "Vault 'NonExistent' not found"}


class TestGetAllTagsHandler:
    """Tests for _get_all_tags_handler function."""

    @patch("obsidian_rag.mcp_server.handlers.get_all_tags_tool")
    def test_get_all_tags_handler_passes_vault_name(self, mock_tool):
        """Handler passes vault_name to get_all_tags_tool."""
        mock_db_manager = MagicMock()
        mock_session = MagicMock()
        mock_db_manager.get_session.return_value.__enter__.return_value = mock_session
        mock_db_manager.get_session.return_value.__exit__.return_value = False

        mock_result = MagicMock()
        mock_result.model_dump.return_value = {
            "tags": ["work"],
            "total_count": 1,
            "has_more": False,
        }
        mock_tool.return_value = mock_result

        result = _get_all_tags_handler(
            mock_db_manager,
            "work*",
            20,
            0,
            vault_name="Personal",
        )

        assert result == {"tags": ["work"], "total_count": 1, "has_more": False}
        mock_tool.assert_called_once_with(
            session=mock_session,
            pattern="work*",
            limit=20,
            offset=0,
            vault_name="Personal",
        )

    @patch("obsidian_rag.mcp_server.handlers.get_all_tags_tool")
    def test_get_all_tags_handler_catches_value_error(self, mock_tool):
        """Handler catches ValueError and returns error dict."""
        mock_db_manager = MagicMock()
        mock_session = MagicMock()
        mock_db_manager.get_session.return_value.__enter__.return_value = mock_session
        mock_db_manager.get_session.return_value.__exit__.return_value = False

        mock_tool.side_effect = ValueError("Vault 'Missing' not found")

        result = _get_all_tags_handler(
            mock_db_manager,
            None,
            20,
            0,
            vault_name="Missing",
        )

        assert result == {"success": False, "error": "Vault 'Missing' not found"}

    @patch("obsidian_rag.mcp_server.handlers.get_all_tags_tool")
    def test_get_all_tags_handler_default_vault_name_none(self, mock_tool):
        """Handler defaults vault_name to None when not provided."""
        mock_db_manager = MagicMock()
        mock_session = MagicMock()
        mock_db_manager.get_session.return_value.__enter__.return_value = mock_session
        mock_db_manager.get_session.return_value.__exit__.return_value = False

        mock_result = MagicMock()
        mock_result.model_dump.return_value = {
            "tags": [],
            "total_count": 0,
            "has_more": False,
        }
        mock_tool.return_value = mock_result

        result = _get_all_tags_handler(mock_db_manager, None, 20, 0)

        assert result == {"tags": [], "total_count": 0, "has_more": False}
        mock_tool.assert_called_once_with(
            session=mock_session,
            pattern=None,
            limit=20,
            offset=0,
            vault_name=None,
        )
