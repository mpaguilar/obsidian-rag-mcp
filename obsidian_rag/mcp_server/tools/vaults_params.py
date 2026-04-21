"""Parameter dataclasses for vault tools."""

from dataclasses import dataclass


@dataclass
class VaultUpdateParams:
    """Parameters for update_vault tool.

    Attributes:
        name: Vault name for lookup (not updatable).
        description: New description (optional).
        host_path: New host path (optional).
        container_path: New container path (optional, requires force).
        force: Required when changing container_path.
    """

    name: str
    description: str | None = None
    host_path: str | None = None
    container_path: str | None = None
    force: bool = False
