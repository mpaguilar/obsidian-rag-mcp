"""Tests for config coverage gaps.

This module tests uncovered branches in config.py:
- DatabasePoolConfig validators (lines 402-403)
- MCPConfig validators (lines 489, 497, 505)
- Settings model validator for default vault (lines 803-810)
"""

from unittest.mock import patch

import pytest


class TestDatabasePoolConfigValidators:
    """Test DatabasePoolConfig field validators for edge cases."""

    @patch("obsidian_rag.config._get_config_file_path")
    def test_validate_pool_size_zero(self, mock_get_config):
        """Test pool_size validator with zero value (lines 402-403)."""
        from obsidian_rag.config import DatabaseConfig

        mock_get_config.return_value = None

        # Zero value should raise ValueError
        with pytest.raises(
            ValueError, match="Pool configuration value must be positive"
        ):
            DatabaseConfig(pool_size=0)

    @patch("obsidian_rag.config._get_config_file_path")
    def test_validate_pool_size_negative(self, mock_get_config):
        """Test pool_size validator with negative value (lines 402-403)."""
        from obsidian_rag.config import DatabaseConfig

        mock_get_config.return_value = None

        # Negative value should raise ValueError
        with pytest.raises(
            ValueError, match="Pool configuration value must be positive"
        ):
            DatabaseConfig(pool_size=-5)

    @patch("obsidian_rag.config._get_config_file_path")
    def test_validate_max_overflow_zero(self, mock_get_config):
        """Test max_overflow validator with zero value (lines 402-403)."""
        from obsidian_rag.config import DatabaseConfig

        mock_get_config.return_value = None

        # Zero value should raise ValueError
        with pytest.raises(
            ValueError, match="Pool configuration value must be positive"
        ):
            DatabaseConfig(max_overflow=0)

    @patch("obsidian_rag.config._get_config_file_path")
    def test_validate_pool_timeout_zero(self, mock_get_config):
        """Test pool_timeout validator with zero value (lines 402-403)."""
        from obsidian_rag.config import DatabaseConfig

        mock_get_config.return_value = None

        # Zero value should raise ValueError
        with pytest.raises(
            ValueError, match="Pool configuration value must be positive"
        ):
            DatabaseConfig(pool_timeout=0)

    @patch("obsidian_rag.config._get_config_file_path")
    def test_validate_pool_recycle_zero(self, mock_get_config):
        """Test pool_recycle validator with zero value (lines 402-403)."""
        from obsidian_rag.config import DatabaseConfig

        mock_get_config.return_value = None

        # Zero value should raise ValueError
        with pytest.raises(
            ValueError, match="Pool configuration value must be positive"
        ):
            DatabaseConfig(pool_recycle=0)


class TestMCPConfigValidators:
    """Test MCPConfig field validators for edge cases."""

    @patch("obsidian_rag.config._get_config_file_path")
    def test_validate_max_concurrent_sessions_negative(self, mock_get_config):
        """Test max_concurrent_sessions validator with negative value (line 489)."""
        from obsidian_rag.config import MCPConfig

        mock_get_config.return_value = None

        # Negative value should return default (100)
        config = MCPConfig(max_concurrent_sessions=-5)
        assert config.max_concurrent_sessions == 100

    @patch("obsidian_rag.config._get_config_file_path")
    def test_validate_max_concurrent_sessions_zero(self, mock_get_config):
        """Test max_concurrent_sessions validator with zero (line 489)."""
        from obsidian_rag.config import MCPConfig

        mock_get_config.return_value = None

        # Zero should return default (100)
        config = MCPConfig(max_concurrent_sessions=0)
        assert config.max_concurrent_sessions == 100

    @patch("obsidian_rag.config._get_config_file_path")
    def test_validate_session_timeout_negative(self, mock_get_config):
        """Test session_timeout_seconds validator with negative value (line 497)."""
        from obsidian_rag.config import MCPConfig

        mock_get_config.return_value = None

        # Negative value should return default (300)
        config = MCPConfig(session_timeout_seconds=-10)
        assert config.session_timeout_seconds == 300

    @patch("obsidian_rag.config._get_config_file_path")
    def test_validate_session_timeout_zero(self, mock_get_config):
        """Test session_timeout_seconds validator with zero (line 497)."""
        from obsidian_rag.config import MCPConfig

        mock_get_config.return_value = None

        # Zero should return default (300)
        config = MCPConfig(session_timeout_seconds=0)
        assert config.session_timeout_seconds == 300

    @patch("obsidian_rag.config._get_config_file_path")
    def test_validate_rate_limit_zero(self, mock_get_config):
        """Test rate_limit_per_second validator with zero (line 505)."""
        from obsidian_rag.config import MCPConfig

        mock_get_config.return_value = None

        # Zero should return default (10.0)
        config = MCPConfig(rate_limit_per_second=0.0)
        assert config.rate_limit_per_second == 10.0

    @patch("obsidian_rag.config._get_config_file_path")
    def test_validate_rate_limit_negative(self, mock_get_config):
        """Test rate_limit_per_second validator with negative value (line 505)."""
        from obsidian_rag.config import MCPConfig

        mock_get_config.return_value = None

        # Negative value should return default (10.0)
        config = MCPConfig(rate_limit_per_second=-5.0)
        assert config.rate_limit_per_second == 10.0


class TestSettingsDefaultVault:
    """Test Settings default vault creation (lines 803-810)."""

    @patch("obsidian_rag.config._get_config_file_path")
    def test_settings_creates_default_vault_when_empty(self, mock_get_config):
        """Test that Settings creates default vault when none configured (lines 803-810)."""
        from obsidian_rag.config import Settings, VaultConfig

        mock_get_config.return_value = None

        # Create settings with empty vaults dict
        settings = Settings(vaults={})

        # Should have created default vault
        assert "Obsidian Vault" in settings.vaults
        assert settings.vaults["Obsidian Vault"].container_path == "/data"
        assert settings.vaults["Obsidian Vault"].host_path == "/data"
        assert settings.vaults["Obsidian Vault"].description == "Default vault"

    @patch("obsidian_rag.config._get_config_file_path")
    def test_settings_uses_existing_vaults(self, mock_get_config):
        """Test that Settings preserves existing vaults."""
        from obsidian_rag.config import Settings, VaultConfig

        mock_get_config.return_value = None

        # Create settings with existing vault
        settings = Settings(
            vaults={"CustomVault": VaultConfig(container_path="/custom")}
        )

        # Should preserve the existing vault
        assert "CustomVault" in settings.vaults
        assert settings.vaults["CustomVault"].container_path == "/custom"

    @patch("obsidian_rag.config._get_config_file_path")
    def test_validate_max_concurrent_sessions_positive(self, mock_get_config):
        """Test max_concurrent_sessions validator with positive value (line 489)."""
        from obsidian_rag.config import MCPConfig

        mock_get_config.return_value = None

        # Positive value should be returned as-is (line 489)
        config = MCPConfig(max_concurrent_sessions=50)
        assert config.max_concurrent_sessions == 50

    @patch("obsidian_rag.config._get_config_file_path")
    def test_validate_session_timeout_positive(self, mock_get_config):
        """Test session_timeout_seconds validator with positive value (line 497)."""
        from obsidian_rag.config import MCPConfig

        mock_get_config.return_value = None

        # Positive value should be returned as-is (line 497)
        config = MCPConfig(session_timeout_seconds=600)
        assert config.session_timeout_seconds == 600

    @patch("obsidian_rag.config._get_config_file_path")
    def test_validate_rate_limit_positive(self, mock_get_config):
        """Test rate_limit_per_second validator with positive value (line 505)."""
        from obsidian_rag.config import MCPConfig

        mock_get_config.return_value = None

        # Positive value should be returned as-is (line 505)
        config = MCPConfig(rate_limit_per_second=20.0)
        assert config.rate_limit_per_second == 20.0
