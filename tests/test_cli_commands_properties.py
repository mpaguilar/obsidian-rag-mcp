"""Tests for _format_query_results_json and _format_query_results_table properties display."""

import json
from unittest.mock import MagicMock

from obsidian_rag.cli_commands import (
    _format_query_results_json,
    _format_query_results_table,
)


def test_format_query_results_json_includes_properties():
    """JSON output includes properties from frontmatter_json excluding tags."""
    mock_doc = MagicMock()
    mock_doc.file_path = "/path/to/doc.md"
    mock_doc.file_name = "doc.md"
    mock_doc.frontmatter_json = {"kind": "note", "author": "alice", "tags": ["work"]}
    mock_doc.tags = ["work"]

    results = [(mock_doc, 0.5)]
    output = _format_query_results_json(results)

    parsed = json.loads(output)
    assert len(parsed) == 1
    assert parsed[0]["properties"] == {"kind": "note", "author": "alice"}


def test_format_query_results_json_properties_none_no_frontmatter():
    """JSON output has properties=None when frontmatter_json is None."""
    mock_doc = MagicMock()
    mock_doc.file_path = "/path/to/doc.md"
    mock_doc.file_name = "doc.md"
    mock_doc.frontmatter_json = None
    mock_doc.tags = []

    results = [(mock_doc, 0.5)]
    output = _format_query_results_json(results)

    parsed = json.loads(output)
    assert parsed[0]["properties"] is None


def test_format_query_results_json_properties_excludes_tags():
    """JSON properties exclude the 'tags' key from frontmatter_json."""
    mock_doc = MagicMock()
    mock_doc.file_path = "/path/to/doc.md"
    mock_doc.file_name = "doc.md"
    mock_doc.frontmatter_json = {"tags": ["work", "urgent"], "project": "alpha"}
    mock_doc.tags = ["work", "urgent"]

    results = [(mock_doc, 0.5)]
    output = _format_query_results_json(results)

    parsed = json.loads(output)
    assert "tags" not in parsed[0]["properties"]
    assert parsed[0]["properties"] == {"project": "alpha"}


def test_format_query_results_table_shows_properties():
    """Table output shows properties after tags."""
    mock_doc = MagicMock()
    mock_doc.file_name = "doc.md"
    mock_doc.file_path = "/path/to/doc.md"
    mock_doc.frontmatter_json = {"kind": "note", "author": "alice", "tags": ["work"]}
    mock_doc.tags = ["work"]

    results = [(mock_doc, 0.5)]
    output = _format_query_results_table(results)

    assert "File: doc.md" in output
    assert "Kind: note" in output
    assert "Tags: work" in output
    assert "author: alice" in output


def test_format_query_results_table_no_properties_no_frontmatter():
    """Table output omits properties section when frontmatter_json is None."""
    mock_doc = MagicMock()
    mock_doc.file_name = "doc.md"
    mock_doc.file_path = "/path/to/doc.md"
    mock_doc.frontmatter_json = None
    mock_doc.tags = []

    results = [(mock_doc, 0.5)]
    output = _format_query_results_table(results)

    assert "File: doc.md" in output
    assert "Kind:" not in output
    assert "Tags:" not in output


def test_format_query_results_table_properties_empty_dict():
    """Table output omits properties when frontmatter_json only contains tags."""
    mock_doc = MagicMock()
    mock_doc.file_name = "doc.md"
    mock_doc.file_path = "/path/to/doc.md"
    mock_doc.frontmatter_json = {"tags": ["work"]}
    mock_doc.tags = ["work"]

    results = [(mock_doc, 0.5)]
    output = _format_query_results_table(results)

    assert "File: doc.md" in output
    assert "Tags: work" in output
    assert "  tags:" not in output
    assert "author:" not in output
