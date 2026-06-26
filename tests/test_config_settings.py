"""Tests for advanced Settings validation, MCP config, vaults, and endpoint parsing."""

import os
from unittest.mock import patch

import pytest

from obsidian_rag.config import (
    Settings,
    get_settings,
)


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


class TestVaultConfigModelValidator:
    """Test VaultConfig model validator for host_path default."""

    def test_vault_config_default_host_path(self):
        """Test host_path defaults to container_path when not provided (line 440)."""
        from obsidian_rag.config import VaultConfig

        config = VaultConfig(container_path="/data")
        assert config.host_path == "/data"

    def test_vault_config_explicit_host_path(self):
        """Test host_path is used when explicitly provided."""
        from obsidian_rag.config import VaultConfig

        config = VaultConfig(container_path="/data", host_path="/host/data")
        assert config.host_path == "/host/data"


class TestMCPConfigValidators:
    """Test MCPConfig field validators."""

    def test_mcp_config_max_concurrent_default(self):
        """Test max_concurrent_sessions returns default when v < 1 (lines 487-489)."""
        from obsidian_rag.config import MCPConfig

        config = MCPConfig(max_concurrent_sessions=0)
        assert config.max_concurrent_sessions == 100

        config = MCPConfig(max_concurrent_sessions=-5)
        assert config.max_concurrent_sessions == 100

    def test_mcp_config_session_timeout_default(self):
        """Test session_timeout_seconds returns default when v < 1 (lines 495-497)."""
        from obsidian_rag.config import MCPConfig

        config = MCPConfig(session_timeout_seconds=0)
        assert config.session_timeout_seconds == 300

        config = MCPConfig(session_timeout_seconds=-10)
        assert config.session_timeout_seconds == 300

    def test_mcp_config_rate_limit_default(self):
        """Test rate_limit_per_second returns default when v <= 0 (lines 503-505)."""
        from obsidian_rag.config import MCPConfig

        config = MCPConfig(rate_limit_per_second=0)
        assert config.rate_limit_per_second == 10.0

        config = MCPConfig(rate_limit_per_second=-5.0)
        assert config.rate_limit_per_second == 10.0

    def test_mcp_config_host_strips_quotes(self):
        """Test host validator strips surrounding quotes (lines 603-621)."""
        from obsidian_rag.config import MCPConfig

        # Test double quotes
        config = MCPConfig(host='"0.0.0.0"')
        assert config.host == "0.0.0.0"

        # Test single quotes
        config = MCPConfig(host="'127.0.0.1'")
        assert config.host == "127.0.0.1"

        # Test whitespace and quotes
        config = MCPConfig(host='  "localhost"  ')
        assert config.host == "localhost"

    def test_mcp_config_host_empty_raises(self):
        """Test host validator raises for empty string after stripping (lines 616-617)."""
        from obsidian_rag.config import MCPConfig

        with pytest.raises(ValueError) as exc_info:
            MCPConfig(host='""')

        assert "Host cannot be empty" in str(exc_info.value)

    def test_mcp_config_host_empty_whitespace_raises(self):
        """Test host validator raises for whitespace-only string (lines 616-617)."""
        from obsidian_rag.config import MCPConfig

        with pytest.raises(ValueError) as exc_info:
            MCPConfig(host="   ")

        assert "Host cannot be empty" in str(exc_info.value)


class TestSettingsVaultValidation:
    """Test Settings vault validation."""

    @patch("obsidian_rag.config._get_config_file_path")
    def test_settings_creates_default_vault(self, mock_get_config):
        """Test default vault created when no vaults configured (lines 803-810)."""
        from obsidian_rag.config import Settings

        mock_get_config.return_value = None

        settings = Settings(vaults={})
        assert "Obsidian Vault" in settings.vaults
        assert settings.vaults["Obsidian Vault"].container_path == "/data"

    @patch("obsidian_rag.config._get_config_file_path")
    def test_settings_vault_name_pattern_invalid(self, mock_get_config):
        """Test vault name with invalid pattern raises error (lines 772-777)."""
        from obsidian_rag.config import Settings, VaultConfig

        mock_get_config.return_value = None

        with pytest.raises(ValueError) as exc_info:
            Settings(vaults={"Invalid.Name": VaultConfig(container_path="/data")})

        assert "Invalid vault name" in str(exc_info.value)

    @patch("obsidian_rag.config._get_config_file_path")
    def test_settings_vault_name_too_long(self, mock_get_config):
        """Test vault name exceeding max length raises error (lines 780-783)."""
        from obsidian_rag.config import Settings, VaultConfig

        mock_get_config.return_value = None

        long_name = "A" * 101  # Exceeds 100 char limit

        with pytest.raises(ValueError) as exc_info:
            Settings(vaults={long_name: VaultConfig(container_path="/data")})

        assert "exceeds" in str(exc_info.value).lower()

    @patch("obsidian_rag.config._get_config_file_path")
    def test_settings_max_vaults_exceeded(self, mock_get_config):
        """Test exceeding max vaults raises error (lines 814-815)."""
        from obsidian_rag.config import Settings, VaultConfig

        mock_get_config.return_value = None

        # Create 101 vaults (max is 100)
        vaults = {
            f"Vault{i}": VaultConfig(container_path=f"/data{i}") for i in range(101)
        }

        with pytest.raises(ValueError) as exc_info:
            Settings(vaults=vaults)

        assert "Maximum" in str(exc_info.value)


class TestSettingsVaultMethods:
    """Test Settings vault helper methods."""

    @patch("obsidian_rag.config._get_config_file_path")
    def test_get_vault_returns_config(self, mock_get_config):
        """Test get_vault returns vault config (line 833-835)."""
        from obsidian_rag.config import Settings, VaultConfig

        mock_get_config.return_value = None

        settings = Settings(vaults={"TestVault": VaultConfig(container_path="/data")})

        vault = settings.get_vault("TestVault")
        assert vault is not None
        assert vault.container_path == "/data"

    @patch("obsidian_rag.config._get_config_file_path")
    def test_get_vault_returns_none_for_missing(self, mock_get_config):
        """Test get_vault returns None for non-existent vault."""
        from obsidian_rag.config import Settings

        mock_get_config.return_value = None

        settings = Settings()  # Creates default vault
        vault = settings.get_vault("NonExistent")
        assert vault is None

    @patch("obsidian_rag.config._get_config_file_path")
    def test_get_vault_names(self, mock_get_config):
        """Test get_vault_names returns list of names (line 844)."""
        from obsidian_rag.config import Settings, VaultConfig

        mock_get_config.return_value = None

        settings = Settings(
            vaults={
                "Vault1": VaultConfig(container_path="/data1"),
                "Vault2": VaultConfig(container_path="/data2"),
            }
        )

        names = settings.get_vault_names()
        # Only explicitly configured vaults should appear
        assert sorted(names) == ["Vault1", "Vault2"]


class TestSettingsEndpointEnvVarParsing:
    """Tests for endpoint environment variable parsing."""

    def setup_method(self):
        """Clear all endpoint-related env vars before each test."""
        # Clear any existing endpoint env vars
        for key in list(os.environ.keys()):
            if key.startswith("OBSIDIAN_RAG_ENDPOINTS_"):
                del os.environ[key]

    @patch("obsidian_rag.config._get_config_file_path")
    def test_endpoint_env_var_parsing_basic(self, mock_get_config):
        """Test that endpoint env vars are parsed into endpoints dict."""
        mock_get_config.return_value = None

        # Set env vars
        os.environ["OBSIDIAN_RAG_ENDPOINTS_EMBEDDING_PROVIDER"] = "openrouter"
        os.environ["OBSIDIAN_RAG_ENDPOINTS_EMBEDDING_MODEL"] = (
            "openai/text-embedding-3-small"
        )
        os.environ["OBSIDIAN_RAG_ENDPOINTS_EMBEDDING_API_KEY"] = "test-api-key"
        os.environ["OBSIDIAN_RAG_ENDPOINTS_EMBEDDING_BASE_URL"] = (
            "https://openrouter.ai/api/v1"
        )

        try:
            settings = Settings()

            assert settings.endpoints["embedding"].provider == "openrouter"
            assert (
                settings.endpoints["embedding"].model == "openai/text-embedding-3-small"
            )
            assert settings.endpoints["embedding"].api_key == "test-api-key"
            assert (
                settings.endpoints["embedding"].base_url
                == "https://openrouter.ai/api/v1"
            )
        finally:
            # Clean up
            del os.environ["OBSIDIAN_RAG_ENDPOINTS_EMBEDDING_PROVIDER"]
            del os.environ["OBSIDIAN_RAG_ENDPOINTS_EMBEDDING_MODEL"]
            del os.environ["OBSIDIAN_RAG_ENDPOINTS_EMBEDDING_API_KEY"]
            del os.environ["OBSIDIAN_RAG_ENDPOINTS_EMBEDDING_BASE_URL"]

    @patch("obsidian_rag.config._get_config_file_path")
    def test_endpoint_env_var_type_conversion(self, mock_get_config):
        """Test that endpoint env vars are converted to proper types."""
        mock_get_config.return_value = None

        # Set env vars with types
        os.environ["OBSIDIAN_RAG_ENDPOINTS_EMBEDDING_TEMPERATURE"] = "0.5"
        os.environ["OBSIDIAN_RAG_ENDPOINTS_EMBEDDING_MAX_TOKENS"] = "1000"
        os.environ["OBSIDIAN_RAG_ENDPOINTS_EMBEDDING_ENABLED"] = "true"
        os.environ["OBSIDIAN_RAG_ENDPOINTS_EMBEDDING_EMPTY"] = ""

        try:
            settings = Settings()

            assert settings.endpoints["embedding"].temperature == 0.5
            assert isinstance(settings.endpoints["embedding"].temperature, float)
            assert settings.endpoints["embedding"].max_tokens == 1000
            assert isinstance(settings.endpoints["embedding"].max_tokens, int)
        finally:
            # Clean up
            del os.environ["OBSIDIAN_RAG_ENDPOINTS_EMBEDDING_TEMPERATURE"]
            del os.environ["OBSIDIAN_RAG_ENDPOINTS_EMBEDDING_MAX_TOKENS"]
            del os.environ["OBSIDIAN_RAG_ENDPOINTS_EMBEDDING_ENABLED"]
            del os.environ["OBSIDIAN_RAG_ENDPOINTS_EMBEDDING_EMPTY"]

    @patch("obsidian_rag.config._get_config_file_path")
    def test_endpoint_env_var_explicit_overrides_env(self, mock_get_config):
        """Test that explicit kwargs override env vars."""
        mock_get_config.return_value = None

        os.environ["OBSIDIAN_RAG_ENDPOINTS_EMBEDDING_PROVIDER"] = "openrouter"
        os.environ["OBSIDIAN_RAG_ENDPOINTS_EMBEDDING_MODEL"] = "wrong-model"

        try:
            settings = Settings(
                endpoints={
                    "embedding": {
                        "provider": "openai",
                        "model": "text-embedding-3-small",
                    }
                }
            )

            # Explicit values should override env vars
            assert settings.endpoints["embedding"].provider == "openai"
            assert settings.endpoints["embedding"].model == "text-embedding-3-small"
        finally:
            del os.environ["OBSIDIAN_RAG_ENDPOINTS_EMBEDDING_PROVIDER"]
            del os.environ["OBSIDIAN_RAG_ENDPOINTS_EMBEDDING_MODEL"]

    def test_convert_endpoint_value_numeric(self):
        """Test _convert_endpoint_value for numeric fields."""
        from obsidian_rag.config import Settings

        # Temperature should be float
        result = Settings._convert_endpoint_value("temperature", "0.75")
        assert result == 0.75
        assert isinstance(result, float)

        # Max tokens should be int
        result = Settings._convert_endpoint_value("max_tokens", "2048")
        assert result == 2048
        assert isinstance(result, int)

    def test_convert_endpoint_value_boolean(self):
        """Test _convert_endpoint_value for boolean fields."""
        from obsidian_rag.config import Settings

        result = Settings._convert_endpoint_value("enabled", "true")
        assert result is True

        result = Settings._convert_endpoint_value("enabled", "TRUE")
        assert result is True

        result = Settings._convert_endpoint_value("enabled", "false")
        assert result is False

        result = Settings._convert_endpoint_value("enabled", "FALSE")
        assert result is False

    def test_convert_endpoint_value_empty(self):
        """Test _convert_endpoint_value for empty string."""
        from obsidian_rag.config import Settings

        result = Settings._convert_endpoint_value("api_key", "")
        assert result is None

    def test_convert_endpoint_value_string(self):
        """Test _convert_endpoint_value for regular strings."""
        from obsidian_rag.config import Settings

        result = Settings._convert_endpoint_value("model", "gpt-4")
        assert result == "gpt-4"

    def test_parse_env_var_key_valid(self):
        """Test _parse_env_var_key with valid key."""
        from obsidian_rag.config import Settings

        result = Settings._parse_env_var_key(
            "OBSIDIAN_RAG_ENDPOINTS_EMBEDDING_PROVIDER"
        )
        assert result == ("embedding", "provider")

        result = Settings._parse_env_var_key("OBSIDIAN_RAG_ENDPOINTS_CHAT_MODEL")
        assert result == ("chat", "model")

    def test_parse_env_var_key_invalid(self):
        """Test _parse_env_var_key with invalid keys."""
        from obsidian_rag.config import Settings

        # Wrong prefix
        result = Settings._parse_env_var_key("SOME_OTHER_VAR")
        assert result is None

        # No underscore after endpoint name
        result = Settings._parse_env_var_key("OBSIDIAN_RAG_ENDPOINTS_EMBEDDING")
        assert result is None

    def test_try_parse_numeric_temperature(self):
        """Test _try_parse_numeric for temperature."""
        from obsidian_rag.config import Settings

        result = Settings._try_parse_numeric("temperature", "0.5")
        assert result == 0.5
        assert isinstance(result, float)

        # Invalid float returns original
        result = Settings._try_parse_numeric("temperature", "not-a-number")
        assert result == "not-a-number"

    def test_try_parse_numeric_max_tokens(self):
        """Test _try_parse_numeric for max_tokens."""
        from obsidian_rag.config import Settings

        result = Settings._try_parse_numeric("max_tokens", "1000")
        assert result == 1000
        assert isinstance(result, int)

        # Invalid int returns original
        result = Settings._try_parse_numeric("max_tokens", "not-an-int")
        assert result == "not-an-int"

    def test_try_parse_numeric_other_fields(self):
        """Test _try_parse_numeric for non-numeric fields."""
        from obsidian_rag.config import Settings

        # Other fields should return original value
        result = Settings._try_parse_numeric("model", "gpt-4")
        assert result == "gpt-4"

    def test_merge_endpoints_into_data_no_endpoints(self):
        """Test _merge_endpoints_into_data with no endpoints."""
        from obsidian_rag.config import Settings

        data = {"database": {"url": "test"}}
        result = Settings._merge_endpoints_into_data({}, data)
        assert result == data

    def test_merge_endpoints_into_data_with_endpoints(self):
        """Test _merge_endpoints_into_data with endpoints."""
        from obsidian_rag.config import Settings

        data = {"database": {"url": "test"}}
        endpoints = {"embedding": {"provider": "openai"}}
        result = Settings._merge_endpoints_into_data(endpoints, data)

        assert result["endpoints"]["embedding"]["provider"] == "openai"
        assert result["database"]["url"] == "test"
