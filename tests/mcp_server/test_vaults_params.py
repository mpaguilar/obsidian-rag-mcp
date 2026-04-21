"""Tests for vault parameter dataclasses."""

from obsidian_rag.mcp_server.tools.vaults_params import VaultUpdateParams


def test_vault_update_params_defaults():
    """Verify default values for VaultUpdateParams."""
    params = VaultUpdateParams(name="TestVault")

    assert params.name == "TestVault"
    assert params.description is None
    assert params.host_path is None
    assert params.container_path is None
    assert params.force is False


def test_vault_update_params_all_fields():
    """Verify can set all fields in VaultUpdateParams."""
    params = VaultUpdateParams(
        name="TestVault",
        description="A test vault",
        host_path="/data/test",
        container_path="/host/test",
        force=True,
    )

    assert params.name == "TestVault"
    assert params.description == "A test vault"
    assert params.host_path == "/data/test"
    assert params.container_path == "/host/test"
    assert params.force is True


def test_vault_update_params_name_required():
    """Verify name is required for VaultUpdateParams."""
    try:
        params = VaultUpdateParams()  # type: ignore[call-arg]
        assert False, "Should have raised TypeError"
    except TypeError:
        pass  # Expected


def test_vault_update_params_optional_fields():
    """Verify description, host_path, container_path, force are optional."""
    # Test with only description
    params1 = VaultUpdateParams(name="Vault1", description="Desc")
    assert params1.name == "Vault1"
    assert params1.description == "Desc"
    assert params1.host_path is None
    assert params1.container_path is None
    assert params1.force is False

    # Test with only host_path
    params2 = VaultUpdateParams(name="Vault2", host_path="/path")
    assert params2.name == "Vault2"
    assert params2.host_path == "/path"

    # Test with only container_path
    params3 = VaultUpdateParams(name="Vault3", container_path="/container")
    assert params3.name == "Vault3"
    assert params3.container_path == "/container"

    # Test with only force
    params4 = VaultUpdateParams(name="Vault4", force=True)
    assert params4.name == "Vault4"
    assert params4.force is True
