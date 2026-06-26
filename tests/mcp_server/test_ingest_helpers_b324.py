from unittest.mock import patch


@patch("obsidian_rag.mcp_server.ingest_helpers.hashlib.md5")
def test_md5_called_with_usedforsecurity_false(mock_md5: object) -> None:
    """B324: hashlib.md5 must be called with usedforsecurity=False."""
    from obsidian_rag.mcp_server.ingest_helpers import _generate_request_id

    mock_md5.return_value.hexdigest.return_value = "abc123"
    _generate_request_id(vault_name="v", path=None, no_delete=None, force=False)
    _, kwargs = mock_md5.call_args
    assert kwargs.get("usedforsecurity") is False


def test_request_id_still_deterministic() -> None:
    """Same params produce the same request id after the B324 fix."""
    from obsidian_rag.mcp_server.ingest_helpers import _generate_request_id

    id1 = _generate_request_id(vault_name="v", path=None, no_delete=None, force=False)
    id2 = _generate_request_id(vault_name="v", path=None, no_delete=None, force=False)
    assert id1 == id2
