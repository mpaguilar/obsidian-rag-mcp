"""Tests for properties display in CLI exact document query."""

from unittest.mock import MagicMock, patch

from obsidian_rag.cli_query_exact import (
    _display_document_list,
    _display_single_document,
    _execute_get_document_lookup,
    _execute_list_documents_lookup,
)


def test_display_single_document_shows_properties():
    """Display document with properties."""
    doc = MagicMock()
    doc.vault_name = "TestVault"
    doc.file_path = "notes/test.md"
    doc.tags = []
    doc.content = "Hello world"
    doc.properties = {"status": "active", "priority": "high"}

    with patch("obsidian_rag.cli_query_exact.click.echo") as mock_echo:
        _display_single_document(doc)

    mock_echo.assert_any_call("Vault: TestVault")
    mock_echo.assert_any_call("Path: notes/test.md")
    mock_echo.assert_any_call("Tags: none")
    mock_echo.assert_any_call("Properties:")
    mock_echo.assert_any_call("  status: active")
    mock_echo.assert_any_call("  priority: high")


def test_display_single_document_no_properties():
    """Display document without properties."""
    doc = MagicMock()
    doc.vault_name = "Vault"
    doc.file_path = "a.md"
    doc.tags = []
    doc.content = "Content"
    doc.properties = None

    with patch("obsidian_rag.cli_query_exact.click.echo") as mock_echo:
        _display_single_document(doc)

    mock_echo.assert_any_call("Vault: Vault")
    mock_echo.assert_any_call("Path: a.md")
    mock_echo.assert_any_call("Tags: none")
    # Properties section should not appear
    properties_calls = [
        call for call in mock_echo.call_args_list if "Properties:" in str(call)
    ]
    assert not properties_calls


def test_display_document_list_shows_properties():
    """Display list with documents that have properties."""
    doc = MagicMock()
    doc.file_name = "a.md"
    doc.file_path = "notes/a.md"
    doc.vault_name = "Vault1"
    doc.tags = []
    doc.properties = {"category": "notes"}

    docs = MagicMock()
    docs.results = [doc]
    docs.total_count = 1

    with patch("obsidian_rag.cli_query_exact.click.echo") as mock_echo:
        _display_document_list(docs)

    mock_echo.assert_any_call("Found 1 results:\n")
    mock_echo.assert_any_call("File: a.md")
    mock_echo.assert_any_call("Path: notes/a.md")
    mock_echo.assert_any_call("Vault: Vault1")
    mock_echo.assert_any_call("Properties:")
    mock_echo.assert_any_call("  category: notes")


def test_display_document_list_no_properties():
    """Display list with documents without properties."""
    doc = MagicMock()
    doc.file_name = "b.md"
    doc.file_path = "notes/b.md"
    doc.vault_name = "Vault2"
    doc.tags = []
    doc.properties = None

    docs = MagicMock()
    docs.results = [doc]
    docs.total_count = 1

    with patch("obsidian_rag.cli_query_exact.click.echo") as mock_echo:
        _display_document_list(docs)

    mock_echo.assert_any_call("File: b.md")
    mock_echo.assert_any_call("Path: notes/b.md")
    mock_echo.assert_any_call("Vault: Vault2")
    # Properties section should not appear
    properties_calls = [
        call for call in mock_echo.call_args_list if "Properties:" in str(call)
    ]
    assert not properties_calls


def test_json_output_includes_properties():
    """JSON output includes properties from DocumentResponse."""
    mock_doc = MagicMock()
    mock_doc.model_dump_json.return_value = '{"properties": {"status": "active"}}'

    mock_session = MagicMock()

    with (
        patch(
            "obsidian_rag.cli_query_exact.get_document_impl",
            return_value=mock_doc,
        ) as mock_get,
        patch("obsidian_rag.cli_query_exact.click.echo") as mock_echo,
    ):
        _execute_get_document_lookup(
            mock_session,
            vault="TestVault",
            path="notes/test.md",
            document_id=None,
            output_format="json",
        )

    mock_get.assert_called_once()
    mock_echo.assert_called_once_with('{"properties": {"status": "active"}}')


def test_json_output_list_includes_properties():
    """JSON list output includes properties from DocumentListResponse."""
    mock_result = MagicMock()
    mock_result.model_dump_json.return_value = (
        '{"results": [{"properties": {"key": "val"}}]}'
    )

    mock_session = MagicMock()

    with (
        patch(
            "obsidian_rag.cli_query_exact.list_documents_impl",
            return_value=mock_result,
        ) as mock_list,
        patch("obsidian_rag.cli_query_exact.click.echo") as mock_echo,
    ):
        _execute_list_documents_lookup(
            mock_session,
            vault="TestVault",
            name="test.md",
            limit=10,
            output_format="json",
        )

    mock_list.assert_called_once()
    mock_echo.assert_called_once_with('{"results": [{"properties": {"key": "val"}}]}')
