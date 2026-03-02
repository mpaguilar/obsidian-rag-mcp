"""Tests for config module."""

import os
from pathlib import Path
from unittest.mock import mock_open, patch

import pytest

from obsidian_rag.config import (
    Settings,
    _get_config_file_path,
    _interpolate_env_vars,
    _load_yaml_config,
    _merge_configs,
    get_settings,
)


class TestGetConfigFilePath:
    """Test cases for _get_config_file_path function."""

    def test_finds_cwd_config_first(self, tmp_path, monkeypatch):
        """Test that CWD config is found before XDG config."""
        # Change to temp directory
        monkeypatch.chdir(tmp_path)

        # Create config in CWD
        config_file = tmp_path / ".obsidian-rag.yaml"
        config_file.write_text("test: value")

        result = _get_config_file_path()
        assert result == config_file

    def test_finds_xdg_config(self, tmp_path, monkeypatch):
        """Test finding config in XDG directory."""
        # Ensure no CWD config
        monkeypatch.chdir(tmp_path)

        # Set XDG_CONFIG_HOME
        xdg_config = tmp_path / "config"
        xdg_config.mkdir()
        obsidian_config = xdg_config / "obsidian-rag"
        obsidian_config.mkdir()
        config_file = obsidian_config / "config.yaml"
        config_file.write_text("test: value")

        monkeypatch.setenv("XDG_CONFIG_HOME", str(xdg_config))

        result = _get_config_file_path()
        assert result == config_file

    def test_returns_none_when_no_config(self, tmp_path, monkeypatch):
        """Test returning None when no config exists."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)

        result = _get_config_file_path()
        assert result is None


class TestInterpolateEnvVars:
    """Test cases for _interpolate_env_vars function."""

    def test_interpolate_simple_var(self, monkeypatch):
        """Test interpolating ${VAR} syntax."""
        monkeypatch.setenv("TEST_VAR", "test_value")
        result = _interpolate_env_vars("${TEST_VAR}")
        assert result == "test_value"

    def test_interpolate_with_default(self, monkeypatch):
        """Test interpolating ${VAR:-default} syntax."""
        monkeypatch.delenv("MISSING_VAR", raising=False)
        result = _interpolate_env_vars("${MISSING_VAR:-default_value}")
        assert result == "default_value"

    def test_interpolate_uses_env_over_default(self, monkeypatch):
        """Test that env var is used when available."""
        monkeypatch.setenv("EXISTING_VAR", "from_env")
        result = _interpolate_env_vars("${EXISTING_VAR:-default}")
        assert result == "from_env"

    def test_interpolate_in_dict(self, monkeypatch):
        """Test interpolating in dictionary values."""
        monkeypatch.setenv("API_KEY", "secret123")
        result = _interpolate_env_vars({"key": "${API_KEY}", "other": "static"})
        assert result == {"key": "secret123", "other": "static"}

    def test_interpolate_in_list(self, monkeypatch):
        """Test interpolating in list values."""
        monkeypatch.setenv("ITEM", "dynamic")
        result = _interpolate_env_vars(["${ITEM}", "static"])
        assert result == ["dynamic", "static"]

    def test_no_interpolation_for_non_string(self):
        """Test that non-string values are returned as-is."""
        result = _interpolate_env_vars(123)
        assert result == 123

    def test_preserves_original_when_var_not_found(self, monkeypatch):
        """Test preserving original pattern when env var not found."""
        monkeypatch.delenv("UNKNOWN_VAR", raising=False)
        result = _interpolate_env_vars("prefix-${UNKNOWN_VAR}-suffix")
        assert result == "prefix-${UNKNOWN_VAR}-suffix"


class TestMergeConfigs:
    """Test cases for _merge_configs function."""

    def test_merge_simple_dicts(self):
        """Test merging simple dictionaries."""
        base = {"a": 1, "b": 2}
        override = {"b": 3, "c": 4}
        result = _merge_configs(base, override)
        assert result == {"a": 1, "b": 3, "c": 4}

    def test_merge_nested_dicts(self):
        """Test merging nested dictionaries."""
        base = {"section": {"key1": "value1", "key2": "value2"}}
        override = {"section": {"key2": "overridden"}}
        result = _merge_configs(base, override)
        assert result == {"section": {"key1": "value1", "key2": "overridden"}}

    def test_merge_with_empty_override(self):
        """Test merging with empty override."""
        base = {"a": 1}
        result = _merge_configs(base, {})
        assert result == {"a": 1}


class TestLoadYamlConfig:
    """Test cases for _load_yaml_config function."""

    def test_loads_valid_yaml(self, tmp_path, monkeypatch):
        """Test loading valid YAML config."""
        monkeypatch.chdir(tmp_path)
        config_file = tmp_path / ".obsidian-rag.yaml"
        config_file.write_text("database:\n  url: postgresql://localhost/db")

        result = _load_yaml_config()
        assert result == {"database": {"url": "postgresql://localhost/db"}}

    def test_returns_empty_dict_when_no_file(self, tmp_path, monkeypatch):
        """Test returning empty dict when no config file exists."""
        monkeypatch.chdir(tmp_path)
        result = _load_yaml_config()
        assert result == {}

    def test_handles_empty_yaml_file(self, tmp_path, monkeypatch):
        """Test handling empty YAML file."""
        monkeypatch.chdir(tmp_path)
        config_file = tmp_path / ".obsidian-rag.yaml"
        config_file.write_text("")

        result = _load_yaml_config()
        assert result == {}

    def test_interpolates_env_vars(self, tmp_path, monkeypatch):
        """Test environment variable interpolation in YAML."""
        monkeypatch.setenv("DB_PASSWORD", "secret123")
        monkeypatch.chdir(tmp_path)
        config_file = tmp_path / ".obsidian-rag.yaml"
        config_file.write_text("database:\n  password: ${DB_PASSWORD}")

        result = _load_yaml_config()
        assert result == {"database": {"password": "secret123"}}


class TestSettings:
    """Test cases for Settings class."""

    def test_default_settings(self):
        """Test default settings values."""
        settings = Settings()

        assert settings.database.url == "postgresql://localhost/obsidian_rag"
        assert settings.ingestion.batch_size == 100
        assert settings.ingestion.max_file_size_mb == 10
        assert settings.logging.level == "INFO"

    def test_override_with_kwargs(self):
        """Test overriding settings with kwargs."""
        settings = Settings(database={"url": "postgresql://custom/db"})

        assert settings.database.url == "postgresql://custom/db"

    def test_get_endpoint_config_existing(self):
        """Test getting existing endpoint configuration."""
        settings = Settings()
        config = settings.get_endpoint_config("embedding")

        assert config is not None
        assert config.provider == "openai"

    def test_get_endpoint_config_missing(self):
        """Test getting non-existent endpoint configuration."""
        settings = Settings(endpoints={})
        config = settings.get_endpoint_config("nonexistent")

        assert config is None


class TestGetSettings:
    """Test cases for get_settings function."""

    def test_returns_settings_instance(self):
        """Test that get_settings returns a Settings instance."""
        settings = get_settings()
        assert isinstance(settings, Settings)

    def test_applies_verbose_logging(self):
        """Test that verbose flag sets DEBUG logging level."""
        settings = get_settings(verbose=True)
        assert settings.logging.level == "DEBUG"


class TestSettingsVerboseWithExistingLogging:
    """Test verbose flag with existing logging config."""

    def test_verbose_overrides_existing_logging(self, tmp_path, monkeypatch):
        """Test that verbose flag overrides existing logging config."""
        monkeypatch.chdir(tmp_path)
        config_file = tmp_path / ".obsidian-rag.yaml"
        config_file.write_text("logging:\n  level: WARNING")

        settings = get_settings(verbose=True)
        assert settings.logging.level == "DEBUG"


class TestLoadYamlConfigErrors:
    """Test YAML config loading error handling."""

    def test_handles_yaml_error(self, tmp_path, monkeypatch):
        """Test handling invalid YAML file."""
        from obsidian_rag.config import _load_yaml_config

        monkeypatch.chdir(tmp_path)
        config_file = tmp_path / ".obsidian-rag.yaml"
        config_file.write_text("invalid: yaml: [content")

        result = _load_yaml_config()
        assert result == {}


class TestSettingsLoggingLevelHandling:
    """Test Settings logging level initialization."""

    def test_settings_init_with_logging_dict(self):
        """Test Settings initialization with logging as dict."""
        settings = Settings(logging={"level": "ERROR", "format": "json"})
        assert settings.logging.level == "ERROR"
        assert settings.logging.format == "json"


class TestReplaceEnvVar:
    """Test cases for _replace_env_var function."""

    def test_replace_env_var_simple(self, monkeypatch):
        """Test replacing simple env var."""
        from obsidian_rag.config import _replace_env_var
        from re import Match, compile

        monkeypatch.setenv("TEST_VAR", "test_value")
        pattern = compile(r"\$\{([^}]+)\}")
        match = pattern.match("${TEST_VAR}")
        assert match is not None
        result = _replace_env_var(match)
        assert result == "test_value"

    def test_replace_env_var_with_default(self, monkeypatch):
        """Test replacing env var with default."""
        from obsidian_rag.config import _replace_env_var
        from re import Match, compile

        monkeypatch.delenv("MISSING_VAR", raising=False)
        pattern = compile(r"\$\{([^}]+)\}")
        match = pattern.match("${MISSING_VAR:-default}")
        assert match is not None
        result = _replace_env_var(match)
        assert result == "default"
