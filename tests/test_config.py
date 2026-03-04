"""Tests for config module."""

import os
from pathlib import Path
from unittest.mock import mock_open, patch

import pytest

from obsidian_rag.config import (
    DatabaseConfig,
    Settings,
    _get_config_file_path,
    _interpolate_env_vars,
    _load_yaml_config,
    _deep_merge,
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

    @patch("pathlib.Path.exists")
    def test_returns_none_when_no_config(self, mock_exists, tmp_path, monkeypatch):
        """Test returning None when no config exists."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
        mock_exists.return_value = False

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


class TestDeepMerge:
    """Test cases for _deep_merge function."""

    def test_merge_simple_dicts(self):
        """Test merging simple dictionaries."""
        base = {"a": 1, "b": 2}
        override = {"b": 3, "c": 4}
        result = _deep_merge(base, override)
        assert result == {"a": 1, "b": 3, "c": 4}

    def test_merge_nested_dicts(self):
        """Test merging nested dictionaries."""
        base = {"section": {"key1": "value1", "key2": "value2"}}
        override = {"section": {"key2": "overridden"}}
        result = _deep_merge(base, override)
        assert result == {"section": {"key1": "value1", "key2": "overridden"}}

    def test_merge_with_empty_override(self):
        """Test merging with empty override."""
        base = {"a": 1}
        result = _deep_merge(base, {})
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

    @patch("pathlib.Path.exists")
    def test_returns_empty_dict_when_no_file(self, mock_exists, tmp_path, monkeypatch):
        """Test returning empty dict when no config file exists."""
        monkeypatch.chdir(tmp_path)
        mock_exists.return_value = False
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

    @patch("obsidian_rag.config._get_config_file_path")
    def test_default_settings(self, mock_get_config, monkeypatch):
        """Test default settings values with no config file."""
        mock_get_config.return_value = None
        monkeypatch.delenv("OBSIDIAN_RAG_DATABASE_URL", raising=False)
        settings = Settings()

        assert settings.database.url == "postgresql+psycopg://localhost/obsidian_rag"
        assert settings.ingestion.batch_size == 100
        assert settings.ingestion.max_file_size_mb == 10
        assert settings.logging.level == "INFO"

    @patch("obsidian_rag.config._get_config_file_path")
    def test_override_with_kwargs(self, mock_get_config):
        """Test overriding settings with kwargs."""
        mock_get_config.return_value = None
        settings = Settings(database={"url": "postgresql://custom/db"})

        assert settings.database.url == "postgresql://custom/db"

    @patch("obsidian_rag.config._get_config_file_path")
    def test_get_endpoint_config_existing(self, mock_get_config):
        """Test getting existing endpoint configuration."""
        mock_get_config.return_value = None
        settings = Settings()
        config = settings.get_endpoint_config("embedding")

        assert config is not None
        assert config.provider == "openai"

    @patch("obsidian_rag.config._get_config_file_path")
    def test_get_endpoint_config_missing(self, mock_get_config):
        """Test getting non-existent endpoint configuration."""
        mock_get_config.return_value = None
        settings = Settings(endpoints={})
        config = settings.get_endpoint_config("nonexistent")

        assert config is None


class TestSettingsEnvVarOverride:
    """Test that environment variables override config files."""

    @patch("obsidian_rag.config._get_config_file_path")
    @patch(
        "builtins.open", mock_open(read_data="database:\n  url: postgresql://config/db")
    )
    def test_env_var_overrides_config_file(self, mock_get_config, monkeypatch):
        """Test that OBSIDIAN_RAG_DATABASE_URL overrides config file value."""
        from pathlib import Path

        mock_get_config.return_value = Path("/fake/config.yaml")
        monkeypatch.setenv("OBSIDIAN_RAG_DATABASE_URL", "postgresql://env/db")

        settings = Settings()
        assert settings.database.url == "postgresql://env/db"


class TestGetSettings:
    """Test cases for get_settings function."""

    @patch("obsidian_rag.config._get_config_file_path")
    def test_returns_settings_instance(self, mock_get_config):
        """Test that get_settings returns a Settings instance."""
        mock_get_config.return_value = None
        settings = get_settings()
        assert isinstance(settings, Settings)

    @patch("obsidian_rag.config._get_config_file_path")
    def test_applies_verbose_logging(self, mock_get_config):
        """Test that verbose flag sets DEBUG logging level."""
        mock_get_config.return_value = None
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

    @patch("obsidian_rag.config._get_config_file_path")
    def test_settings_init_with_logging_dict(self, mock_get_config):
        """Test Settings initialization with logging as dict."""
        mock_get_config.return_value = None
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


class TestDatabaseConfigVectorDimension:
    """Test cases for DatabaseConfig vector_dimension field."""

    def test_default_vector_dimension(self):
        """Test that vector_dimension defaults to 1536."""
        config = DatabaseConfig()
        assert config.vector_dimension == 1536

    def test_custom_vector_dimension(self):
        """Test that vector_dimension can be set to custom value."""
        config = DatabaseConfig(vector_dimension=768)
        assert config.vector_dimension == 768

    def test_validation_returns_default_for_zero(self):
        """Test that validation returns default (1536) for zero value."""
        config = DatabaseConfig(vector_dimension=0)
        assert config.vector_dimension == 1536

    def test_validation_returns_default_for_negative(self):
        """Test that validation returns default (1536) for negative value."""
        config = DatabaseConfig(vector_dimension=-100)
        assert config.vector_dimension == 1536

    def test_validation_accepts_positive_value(self):
        """Test that validation accepts positive values."""
        config = DatabaseConfig(vector_dimension=1024)
        assert config.vector_dimension == 1024


class TestIngestionConfig:
    """Test cases for IngestionConfig."""

    def test_default_batch_size(self):
        """Test that batch_size defaults to 100."""
        from obsidian_rag.config import IngestionConfig

        config = IngestionConfig()
        assert config.batch_size == 100

    def test_batch_size_validation_returns_default_for_zero(self):
        """Test that batch_size validation returns default for zero."""
        from obsidian_rag.config import IngestionConfig

        config = IngestionConfig(batch_size=0)
        assert config.batch_size == 100

    def test_batch_size_validation_returns_default_for_negative(self):
        """Test that batch_size validation returns default for negative."""
        from obsidian_rag.config import IngestionConfig

        config = IngestionConfig(batch_size=-5)
        assert config.batch_size == 100

    def test_custom_batch_size(self):
        """Test that batch_size can be set to custom value."""
        from obsidian_rag.config import IngestionConfig

        config = IngestionConfig(batch_size=50)
        assert config.batch_size == 50


class TestDatabaseConfigYamlOverride:
    """Test YAML config override for vector_dimension."""

    def test_yaml_config_overrides_vector_dimension(self, tmp_path, monkeypatch):
        """Test that vector_dimension can be set via YAML config file."""
        monkeypatch.chdir(tmp_path)
        config_file = tmp_path / ".obsidian-rag.yaml"
        # Use 1536 to match the default embedding model (text-embedding-3-small)
        config_file.write_text("database:\n  vector_dimension: 1536")

        settings = Settings()
        assert settings.database.vector_dimension == 1536

    def test_yaml_config_invalid_uses_validation(self, tmp_path, monkeypatch):
        """Test that invalid YAML config value triggers validation."""
        monkeypatch.chdir(tmp_path)
        config_file = tmp_path / ".obsidian-rag.yaml"
        config_file.write_text("database:\n  vector_dimension: -1")

        settings = Settings()
        assert settings.database.vector_dimension == 1536


class TestVectorDimensionMaxValidation:
    """Test cases for vector_dimension maximum validation (2000 limit)."""

    def test_vector_dimension_2000_is_accepted(self):
        """Test that vector_dimension = 2000 is accepted (at the limit)."""
        config = DatabaseConfig(vector_dimension=2000)
        assert config.vector_dimension == 2000

    def test_vector_dimension_2001_raises_error(self):
        """Test that vector_dimension > 2000 raises validation error."""
        import pytest

        with pytest.raises(ValueError) as exc_info:
            DatabaseConfig(vector_dimension=2001)

        error_msg = str(exc_info.value)
        assert "vector_dimension must be <= 2000" in error_msg
        assert "pgvector" in error_msg.lower()

    def test_vector_dimension_4096_raises_error(self):
        """Test that vector_dimension = 4096 raises validation error."""
        import pytest

        with pytest.raises(ValueError) as exc_info:
            DatabaseConfig(vector_dimension=4096)

        error_msg = str(exc_info.value)
        assert "vector_dimension must be <= 2000" in error_msg

    def test_error_message_includes_compatible_models(self):
        """Test that error message lists compatible models."""
        import pytest

        with pytest.raises(ValueError) as exc_info:
            DatabaseConfig(vector_dimension=3000)

        error_msg = str(exc_info.value)
        assert "text-embedding-3-small" in error_msg
        assert "all-MiniLM-L6-v2" in error_msg
        assert "all-mpnet-base-v2" in error_msg


class TestEmbeddingDimensionCrossValidation:
    """Test cases for cross-validation between provider and vector_dimension."""

    @patch("obsidian_rag.config._get_config_file_path")
    def test_matching_dimensions_succeeds(self, mock_get_config):
        """Test that matching provider dimension and vector_dimension succeeds."""
        mock_get_config.return_value = None

        settings = Settings(
            endpoints={
                "embedding": {
                    "provider": "openai",
                    "model": "text-embedding-3-small",
                }
            },
            database={"vector_dimension": 1536},
        )

        assert settings.database.vector_dimension == 1536

    @patch("obsidian_rag.config._get_config_file_path")
    def test_mismatched_dimensions_raises_error(self, mock_get_config):
        """Test that mismatched dimensions raise validation error."""
        import pytest

        mock_get_config.return_value = None

        with pytest.raises(ValueError) as exc_info:
            Settings(
                endpoints={
                    "embedding": {
                        "provider": "openai",
                        "model": "text-embedding-3-small",
                    }
                },
                database={"vector_dimension": 768},
            )

        error_msg = str(exc_info.value)
        assert "dimension mismatch" in error_msg.lower()
        assert "1536" in error_msg
        assert "768" in error_msg

    @patch("obsidian_rag.config._get_config_file_path")
    def test_openai_ada002_dimension_validation(self, mock_get_config):
        """Test validation for OpenAI ada-002 model (1536 dims)."""
        mock_get_config.return_value = None

        settings = Settings(
            endpoints={
                "embedding": {
                    "provider": "openai",
                    "model": "text-embedding-ada-002",
                }
            },
            database={"vector_dimension": 1536},
        )

        assert settings.database.vector_dimension == 1536

    @patch("obsidian_rag.config._get_config_file_path")
    def test_huggingface_minilm_dimension_validation(self, mock_get_config):
        """Test validation for HuggingFace MiniLM model (384 dims)."""
        mock_get_config.return_value = None

        settings = Settings(
            endpoints={
                "embedding": {
                    "provider": "huggingface",
                    "model": "all-MiniLM-L6-v2",
                }
            },
            database={"vector_dimension": 384},
        )

        assert settings.database.vector_dimension == 384

    @patch("obsidian_rag.config._get_config_file_path")
    def test_high_dimension_model_rejected(self, mock_get_config):
        """Test that high-dimension models (>2000) are rejected."""
        import pytest

        mock_get_config.return_value = None

        with pytest.raises(ValueError) as exc_info:
            Settings(
                endpoints={
                    "embedding": {
                        "provider": "openrouter",
                        "model": "qwen/qwen3-embedding-8b",
                    }
                },
                database={"vector_dimension": 4096},
            )

        error_msg = str(exc_info.value)
        assert "4096" in error_msg
        assert "exceeds" in error_msg.lower() or "2000" in error_msg

    @patch("obsidian_rag.config._get_config_file_path")
    def test_unknown_provider_skips_validation(self, mock_get_config):
        """Test that unknown providers skip dimension validation."""
        mock_get_config.return_value = None

        settings = Settings(
            endpoints={
                "embedding": {
                    "provider": "unknown_provider",
                    "model": "some-model",
                }
            },
            database={"vector_dimension": 1536},
        )

        assert settings.database.vector_dimension == 1536

    @patch("obsidian_rag.config._get_config_file_path")
    def test_unknown_model_skips_validation(self, mock_get_config):
        """Test that unknown models skip dimension validation."""
        mock_get_config.return_value = None

        settings = Settings(
            endpoints={
                "embedding": {
                    "provider": "openai",
                    "model": "unknown-model-v1",
                }
            },
            database={"vector_dimension": 768},
        )

        assert settings.database.vector_dimension == 768

    @patch("obsidian_rag.config._get_config_file_path")
    def test_no_embedding_config_skips_validation(self, mock_get_config):
        """Test that missing embedding config skips validation."""
        mock_get_config.return_value = None

        settings = Settings(
            endpoints={},
            database={"vector_dimension": 1536},
        )

        assert settings.database.vector_dimension == 1536

    @patch("obsidian_rag.config._get_config_file_path")
    def test_empty_model_skips_validation(self, mock_get_config):
        """Test that empty model name skips validation."""
        mock_get_config.return_value = None

        settings = Settings(
            endpoints={
                "embedding": {
                    "provider": "openai",
                    "model": "",
                }
            },
            database={"vector_dimension": 768},
        )

        assert settings.database.vector_dimension == 768


class TestYamlConfigSettingsSource:
    """Test cases for YamlConfigSettingsSource."""

    @patch("obsidian_rag.config._get_config_file_path")
    def test_get_field_value_returns_none_tuple(self, mock_get_config):
        """Test that get_field_value returns (None, '', False)."""
        from obsidian_rag.config import YamlConfigSettingsSource, Settings

        mock_get_config.return_value = None
        settings_cls = Settings
        source = YamlConfigSettingsSource(settings_cls)

        # This method is required by Pydantic but we don't use it
        result = source.get_field_value(None, "test_field")
        assert result == (None, "", False)


class TestSettingsWithInitKwargs:
    """Test Settings initialization with various kwargs."""

    @patch("obsidian_rag.config._get_config_file_path")
    def test_settings_init_logs_debug_message(self, mock_get_config, caplog):
        """Test that Settings.__init__ logs debug message."""
        from obsidian_rag.config import log

        mock_get_config.return_value = None

        with caplog.at_level("DEBUG", logger="obsidian_rag.config"):
            Settings()

        assert "Initializing application settings" in caplog.text

    @patch("obsidian_rag.config._get_config_file_path")
    def test_get_settings_logs_debug_message(self, mock_get_config, caplog):
        """Test that get_settings logs debug message."""
        mock_get_config.return_value = None

        with caplog.at_level("DEBUG", logger="obsidian_rag.config"):
            get_settings()

        assert "Creating settings instance" in caplog.text

    @patch("obsidian_rag.config._get_config_file_path")
    def test_get_endpoint_config_logs_debug_message(self, mock_get_config, caplog):
        """Test that get_endpoint_config logs debug message."""
        mock_get_config.return_value = None

        settings = Settings()

        with caplog.at_level("DEBUG", logger="obsidian_rag.config"):
            settings.get_endpoint_config("embedding")

        assert "Getting endpoint config for: embedding" in caplog.text


class TestMCPConfig:
    """Test cases for MCPConfig."""

    def test_mcp_config_port_validation_raises_for_zero(self):
        """Test that port validation raises for port 0."""
        from obsidian_rag.config import MCPConfig
        import pytest

        with pytest.raises(ValueError, match="Port must be between 1 and 65535"):
            MCPConfig(port=0)

    def test_mcp_config_port_validation_raises_for_negative(self):
        """Test that port validation raises for negative port."""
        from obsidian_rag.config import MCPConfig
        import pytest

        with pytest.raises(ValueError, match="Port must be between 1 and 65535"):
            MCPConfig(port=-1)

    def test_mcp_config_port_validation_raises_for_too_large(self):
        """Test that port validation raises for port > 65535."""
        from obsidian_rag.config import MCPConfig
        import pytest

        with pytest.raises(ValueError, match="Port must be between 1 and 65535"):
            MCPConfig(port=70000)


class TestValidateEmbeddingDimensionCompatibilityEdgeCases:
    """Test edge cases for dimension validation."""

    @patch("obsidian_rag.config._get_config_file_path")
    def test_validate_dimension_match_method(self, mock_get_config):
        """Test the _validate_dimension_match method directly."""
        from obsidian_rag.config import Settings

        mock_get_config.return_value = None

        settings = Settings()
        # This should not raise
        settings._validate_dimension_match(1536, 1536, "test-model")

    @patch("obsidian_rag.config._get_config_file_path")
    def test_validate_dimension_match_raises_on_mismatch(self, mock_get_config):
        """Test that _validate_dimension_match raises on mismatch."""
        import pytest

        from obsidian_rag.config import Settings

        mock_get_config.return_value = None

        settings = Settings()
        with pytest.raises(ValueError) as exc_info:
            settings._validate_dimension_match(1536, 768, "test-model")

        assert "dimension mismatch" in str(exc_info.value).lower()

    @patch("obsidian_rag.config._get_config_file_path")
    def test_openrouter_model_dimensions(self, mock_get_config):
        """Test that openrouter model dimensions are looked up correctly."""
        import pytest

        mock_get_config.return_value = None

        # This should fail because qwen/qwen3-embedding-8b has 4096 dims which exceeds 2000 limit
        with pytest.raises(ValueError) as exc_info:
            Settings(
                endpoints={
                    "embedding": {
                        "provider": "openrouter",
                        "model": "qwen/qwen3-embedding-8b",
                    }
                },
                database={"vector_dimension": 768},  # Wrong dimension
            )

        error_msg = str(exc_info.value)
        assert "4096" in error_msg
        assert "2000" in error_msg

    @patch("obsidian_rag.config._get_config_file_path")
    def test_get_expected_dimension_no_embedding_config(self, mock_get_config):
        """Test _get_expected_dimension returns None when no embedding config."""
        from obsidian_rag.config import Settings

        mock_get_config.return_value = None

        # Create settings with endpoints that doesn't include "embedding"
        settings = Settings()
        # Manually remove the embedding config to test the None case
        settings.endpoints = {}
        result = settings._get_expected_dimension()

        assert result is None
