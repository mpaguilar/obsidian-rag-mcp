"""Tests for vault_tools module."""

from unittest.mock import MagicMock, patch


class TestGetVaultToolWrapper:
    """Tests for get_vault tool wrapper."""

    def test_get_vault_tool_wrapper_delegates_through_registry(self):
        """Test get_vault wrapper delegates to get_vault_tool via registry."""
        from obsidian_rag.mcp_server.vault_tools import get_vault

        mock_registry = MagicMock()
        mock_db_manager = MagicMock()
        mock_registry.db_manager = mock_db_manager

        mock_result = {
            "id": "test-uuid",
            "name": "TestVault",
            "description": "Test vault",
            "document_count": 5,
        }

        with patch(
            "obsidian_rag.mcp_server.vault_tools._get_registry"
        ) as mock_get_registry:
            mock_get_registry.return_value = mock_registry

            with patch(
                "obsidian_rag.mcp_server.vault_tools.get_vault_tool"
            ) as mock_tool:
                mock_tool.return_value = mock_result

                result = get_vault(vault_name="TestVault")

        assert result == mock_result
        mock_tool.assert_called_once_with(
            mock_db_manager,
            vault_name="TestVault",
            vault_id=None,
        )

    def test_get_vault_with_vault_id_parameter(self):
        """Test get_vault wrapper with vault_id parameter."""
        from obsidian_rag.mcp_server.vault_tools import get_vault

        mock_registry = MagicMock()
        mock_db_manager = MagicMock()
        mock_registry.db_manager = mock_db_manager

        mock_result = {
            "id": "test-uuid",
            "name": "TestVault",
            "description": "Test vault",
        }

        with patch(
            "obsidian_rag.mcp_server.vault_tools._get_registry"
        ) as mock_get_registry:
            mock_get_registry.return_value = mock_registry

            with patch(
                "obsidian_rag.mcp_server.vault_tools.get_vault_tool"
            ) as mock_tool:
                mock_tool.return_value = mock_result

                result = get_vault(vault_id="test-uuid-123")

        assert result == mock_result
        mock_tool.assert_called_once_with(
            mock_db_manager,
            vault_name=None,
            vault_id="test-uuid-123",
        )

    def test_get_vault_prefers_name_over_vault_id(self):
        """Test get_vault prefers name when both are provided."""
        from obsidian_rag.mcp_server.vault_tools import get_vault

        mock_registry = MagicMock()
        mock_db_manager = MagicMock()
        mock_registry.db_manager = mock_db_manager

        mock_result = {"id": "test-uuid", "name": "TestVault"}

        with patch(
            "obsidian_rag.mcp_server.vault_tools._get_registry"
        ) as mock_get_registry:
            mock_get_registry.return_value = mock_registry

            with patch(
                "obsidian_rag.mcp_server.vault_tools.get_vault_tool"
            ) as mock_tool:
                mock_tool.return_value = mock_result

                result = get_vault(vault_name="TestVault", vault_id="test-uuid-123")

        assert result == mock_result
        # Should pass both parameters to the tool
        mock_tool.assert_called_once_with(
            mock_db_manager,
            vault_name="TestVault",
            vault_id="test-uuid-123",
        )


class TestUpdateVaultToolWrapper:
    """Tests for update_vault tool wrapper."""

    def test_update_vault_tool_wrapper_delegates_through_registry(self):
        """Test update_vault wrapper delegates to update_vault_tool via registry."""
        from obsidian_rag.mcp_server.vault_tools import update_vault

        mock_registry = MagicMock()
        mock_db_manager = MagicMock()
        mock_registry.db_manager = mock_db_manager

        mock_result = {
            "id": "test-uuid",
            "name": "TestVault",
            "description": "Updated description",
        }

        with patch(
            "obsidian_rag.mcp_server.vault_tools._get_registry"
        ) as mock_get_registry:
            mock_get_registry.return_value = mock_registry

            with patch(
                "obsidian_rag.mcp_server.vault_tools.update_vault_tool"
            ) as mock_tool:
                mock_tool.return_value = mock_result

                result = update_vault(
                    vault_name="TestVault",
                    description="Updated description",
                )

        assert result == mock_result
        mock_tool.assert_called_once()
        call_args = mock_tool.call_args
        assert call_args[0][0] == mock_db_manager
        params = call_args[0][1]
        assert params.vault_name == "TestVault"
        assert params.description == "Updated description"

    def test_update_vault_with_all_parameters(self):
        """Test update_vault wrapper with all optional parameters."""
        from obsidian_rag.mcp_server.vault_tools import update_vault

        mock_registry = MagicMock()
        mock_db_manager = MagicMock()
        mock_registry.db_manager = mock_db_manager

        mock_result = {"success": True}

        with patch(
            "obsidian_rag.mcp_server.vault_tools._get_registry"
        ) as mock_get_registry:
            mock_get_registry.return_value = mock_registry

            with patch(
                "obsidian_rag.mcp_server.vault_tools.update_vault_tool"
            ) as mock_tool:
                mock_tool.return_value = mock_result

                result = update_vault(
                    vault_name="TestVault",
                    description="New desc",
                    host_path="/new/host/path",
                    container_path="/new/container/path",
                    force=True,
                )

        assert result == mock_result
        call_args = mock_tool.call_args
        params = call_args[0][1]
        assert params.vault_name == "TestVault"
        assert params.description == "New desc"
        assert params.host_path == "/new/host/path"
        assert params.container_path == "/new/container/path"
        assert params.force is True

    def test_update_vault_with_force_false(self):
        """Test update_vault wrapper with force=False (default)."""
        from obsidian_rag.mcp_server.vault_tools import update_vault

        mock_registry = MagicMock()
        mock_db_manager = MagicMock()
        mock_registry.db_manager = mock_db_manager

        mock_result = {"success": True}

        with patch(
            "obsidian_rag.mcp_server.vault_tools._get_registry"
        ) as mock_get_registry:
            mock_get_registry.return_value = mock_registry

            with patch(
                "obsidian_rag.mcp_server.vault_tools.update_vault_tool"
            ) as mock_tool:
                mock_tool.return_value = mock_result

                result = update_vault(vault_name="TestVault")

        assert result == mock_result
        call_args = mock_tool.call_args
        params = call_args[0][1]
        assert params.force is False


class TestDeleteVaultToolWrapper:
    """Tests for delete_vault tool wrapper."""

    def test_delete_vault_tool_wrapper_delegates_through_registry(self):
        """Test delete_vault wrapper delegates to delete_vault_tool via registry."""
        from obsidian_rag.mcp_server.vault_tools import delete_vault

        mock_registry = MagicMock()
        mock_db_manager = MagicMock()
        mock_registry.db_manager = mock_db_manager

        mock_result = {
            "success": True,
            "name": "TestVault",
            "documents_deleted": 5,
            "tasks_deleted": 10,
            "chunks_deleted": 20,
        }

        with patch(
            "obsidian_rag.mcp_server.vault_tools._get_registry"
        ) as mock_get_registry:
            mock_get_registry.return_value = mock_registry

            with patch(
                "obsidian_rag.mcp_server.vault_tools.delete_vault_tool"
            ) as mock_tool:
                mock_tool.return_value = mock_result

                result = delete_vault(vault_name="TestVault", confirm=True)

        assert result == mock_result
        mock_tool.assert_called_once_with(
            mock_db_manager,
            vault_name="TestVault",
            confirm=True,
        )

    def test_delete_vault_with_confirm_false(self):
        """Test delete_vault wrapper with confirm=False."""
        from obsidian_rag.mcp_server.vault_tools import delete_vault

        mock_registry = MagicMock()
        mock_db_manager = MagicMock()
        mock_registry.db_manager = mock_db_manager

        mock_result = {
            "success": False,
            "error": "confirm=True is required",
        }

        with patch(
            "obsidian_rag.mcp_server.vault_tools._get_registry"
        ) as mock_get_registry:
            mock_get_registry.return_value = mock_registry

            with patch(
                "obsidian_rag.mcp_server.vault_tools.delete_vault_tool"
            ) as mock_tool:
                mock_tool.return_value = mock_result

                result = delete_vault(vault_name="TestVault", confirm=False)

        assert result == mock_result
        mock_tool.assert_called_once_with(
            mock_db_manager,
            vault_name="TestVault",
            confirm=False,
        )


class TestVaultToolsUseCorrectParamNames:
    """Tests to verify vault tools use correct parameter names without built-in shadowing."""

    def test_get_vault_uses_vault_id_not_id(self):
        """Test get_vault uses vault_id parameter name, not id (which shadows built-in)."""
        import inspect

        from obsidian_rag.mcp_server.vault_tools import get_vault

        sig = inspect.signature(get_vault)
        param_names = list(sig.parameters.keys())

        # Should use 'vault_id', not 'id'
        assert "vault_id" in param_names
        assert "id" not in param_names

    def test_update_vault_uses_vault_name_param(self):
        """Test update_vault uses 'vault_name' parameter."""
        import inspect

        from obsidian_rag.mcp_server.vault_tools import update_vault

        sig = inspect.signature(update_vault)
        param_names = list(sig.parameters.keys())

        # Should use 'vault_name', not 'name'
        assert "vault_name" in param_names
        # Should not have 'id' parameter
        assert "id" not in param_names

    def test_delete_vault_uses_vault_name_param(self):
        """Test delete_vault uses 'vault_name' parameter."""
        import inspect

        from obsidian_rag.mcp_server.vault_tools import delete_vault

        sig = inspect.signature(delete_vault)
        param_names = list(sig.parameters.keys())

        # Should use 'vault_name', not 'name'
        assert "vault_name" in param_names
        # Should not have 'id' parameter
        assert "id" not in param_names

    def test_all_vault_tools_avoid_built_in_shadowing(self):
        """Test all vault tool parameter names avoid shadowing Python built-ins."""
        import builtins
        import inspect

        from obsidian_rag.mcp_server.vault_tools import (
            delete_vault,
            get_vault,
            update_vault,
        )

        built_in_names = set(dir(builtins))
        tools = [get_vault, update_vault, delete_vault]

        for tool in tools:
            sig = inspect.signature(tool)
            for param_name in sig.parameters:
                # 'vault_name' is specifically allowed even if it might be in builtins
                # because it's not a commonly shadowed dangerous built-in
                if param_name != "vault_name":
                    assert param_name not in built_in_names, (
                        f"Parameter '{param_name}' in {tool.__name__} shadows a built-in"
                    )
