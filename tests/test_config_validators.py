"""Tests for config_validators.py.

This module tests the standalone validation helpers in config_validators.py
to ensure 100% branch coverage of all defensive code paths.
"""

from obsidian_rag.config_validators import (
    apply_endpoint_merge,
    merge_endpoints_into_data,
    parse_env_var_key,
    try_parse_numeric,
)


def test_try_parse_numeric_temperature_invalid_float():
    """Test try_parse_numeric with invalid float for temperature.

    Covers the ValueError branch in config_validators.py:160.
    """
    result = try_parse_numeric("temperature", "not-a-number")
    assert result == "not-a-number"


def test_try_parse_numeric_max_tokens_invalid_int():
    """Test try_parse_numeric with invalid int for max_tokens.

    Covers the ValueError branch in config_validators.py:165.
    """
    result = try_parse_numeric("max_tokens", "not-an-int")
    assert result == "not-an-int"


def test_parse_env_var_key_no_underscore_after_prefix():
    """Test parse_env_var_key with no underscore after endpoint name.

    Covers the len(parts) != expected_parts_count branch in config_validators.py:213.
    """
    result = parse_env_var_key("OBSIDIAN_RAG_ENDPOINTS_EMBEDDING")
    assert result is None


def test_apply_endpoint_merge_non_dict_existing():
    """Test apply_endpoint_merge when existing value is not a dict.

    Covers the isinstance(existing, dict) defensive branch in config_validators.py:257.
    When existing is not a dict, the endpoint config should not be merged.
    """
    existing_endpoints = {"embedding": "not_a_dict"}
    apply_endpoint_merge(existing_endpoints, "embedding", {"provider": "openai"})
    assert existing_endpoints["embedding"] == "not_a_dict"


def test_merge_endpoints_into_data_non_dict_existing_endpoints():
    """Test merge_endpoints_into_data when existing_endpoints is not a dict.

    Covers the isinstance(existing_endpoints, dict) defensive branch in config_validators.py:282.
    When existing endpoints is not a dict, it should be replaced entirely.
    """
    data = {"endpoints": "not_a_dict"}
    endpoints = {"embedding": {"provider": "openai"}}
    result = merge_endpoints_into_data(endpoints, data)
    assert result["endpoints"] == {"embedding": {"provider": "openai"}}
