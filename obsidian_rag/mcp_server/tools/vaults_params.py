"""Parameter dataclasses for vault tools."""

from dataclasses import dataclass


@dataclass
class VaultUpdateParams:
    """Parameters for updating a vault.

    Attributes:
        vault_name: Name of the vault to update (required, lookup key).
        description: Optional new description.
        host_path: Optional new host path.
        container_path: Optional new container path (requires force).
        force: Must be True when changing container_path.
    """

    vault_name: str
    description: str | None = None
    host_path: str | None = None
    container_path: str | None = None
    force: bool = False
