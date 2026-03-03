"""Tests for FrontMatter parsing module."""

import pytest

from obsidian_rag.parsing.frontmatter import (
    extract_frontmatter,
    normalize_tags,
    parse_frontmatter,
)


class TestExtractFrontmatter:
    """Test cases for extract_frontmatter function."""

    def test_extracts_valid_frontmatter(self):
        """Test extracting valid YAML frontmatter."""
        content = """---
title: My Document
kind: note
tags: [test, markdown]
---

This is the content.
"""
        frontmatter, remaining = extract_frontmatter(content)

        assert frontmatter == {
            "title": "My Document",
            "kind": "note",
            "tags": ["test", "markdown"],
        }
        assert remaining == "This is the content.\n"

    def test_no_frontmatter_returns_empty_dict(self):
        """Test content without frontmatter returns empty dict."""
        content = "Just some content without frontmatter."
        frontmatter, remaining = extract_frontmatter(content)

        assert frontmatter == {}
        assert remaining == content

    def test_empty_frontmatter(self):
        """Test handling of empty frontmatter block."""
        content = "---\n---\n\nContent here."
        frontmatter, remaining = extract_frontmatter(content)

        assert frontmatter == {}
        # Empty frontmatter doesn't match our pattern, so content is unchanged
        assert remaining == "---\n---\n\nContent here."

    def test_corrupted_frontmatter_logs_warning(self, caplog):
        """Test that corrupted frontmatter is handled gracefully."""
        content = """---
invalid: yaml: [
---

Content.
"""
        with caplog.at_level("WARNING"):
            frontmatter, remaining = extract_frontmatter(content)

        assert frontmatter == {}
        assert "Failed to parse FrontMatter" in caplog.text
        assert remaining == "Content.\n"


class TestNormalizeTags:
    """Test cases for normalize_tags function."""

    def test_normalize_string_tags(self):
        """Test normalizing tags from string."""
        result = normalize_tags("tag1, tag2, tag3")
        assert result == ["tag1", "tag2", "tag3"]

    def test_normalize_list_tags(self):
        """Test normalizing tags from list."""
        result = normalize_tags(["tag1", "tag2", "tag3"])
        assert result == ["tag1", "tag2", "tag3"]

    def test_normalize_deduplicates_tags(self):
        """Test that duplicate tags are removed."""
        result = normalize_tags(["tag1", "tag2", "tag1", "tag3"])
        assert result == ["tag1", "tag2", "tag3"]

    def test_normalize_empty_tags(self):
        """Test normalizing empty/None tags."""
        assert normalize_tags(None) is None
        assert normalize_tags([]) is None

    def test_normalize_preserves_order(self):
        """Test that tag order is preserved during deduplication."""
        result = normalize_tags(["c", "a", "b", "a", "c"])
        assert result == ["c", "a", "b"]


class TestParseFrontmatter:
    """Test cases for parse_frontmatter function."""

    def test_parses_all_fields(self):
        """Test parsing complete frontmatter."""
        content = """---
kind: article
tags: [python, testing]
author: John Doe
---

Body content.
"""
        kind, tags, metadata, remaining = parse_frontmatter(content)

        assert kind == "article"
        assert tags == ["python", "testing"]
        assert metadata == {"author": "John Doe"}
        assert remaining == "Body content.\n"

    def test_parses_kind_only(self):
        """Test parsing frontmatter with only kind."""
        content = """---
kind: note
---

Content.
"""
        kind, tags, metadata, remaining = parse_frontmatter(content)

        assert kind == "note"
        assert tags is None
        assert metadata == {}

    def test_no_frontmatter(self):
        """Test parsing content without frontmatter."""
        content = "Just content."
        kind, tags, metadata, remaining = parse_frontmatter(content)

        assert kind is None
        assert tags is None
        assert metadata == {}
        assert remaining == content

    def test_tags_as_string(self):
        """Test parsing tags provided as string."""
        content = """---
tags: "python, testing"
---

Content.
"""
        kind, tags, metadata, remaining = parse_frontmatter(content)

        assert tags == ["python", "testing"]

    def test_frontmatter_returns_none(self, caplog):
        """Test handling when yaml.safe_load returns None."""
        from obsidian_rag.parsing.frontmatter import extract_frontmatter

        # YAML with just whitespace/comments returns None
        content = """---
# Just a comment
---

Content.
"""
        with caplog.at_level("DEBUG"):
            frontmatter, remaining = extract_frontmatter(content)

        assert frontmatter == {}
        assert "Successfully extracted FrontMatter" in caplog.text

    def test_frontmatter_not_dictionary(self, caplog):
        """Test handling when frontmatter is not a dictionary."""
        from obsidian_rag.parsing.frontmatter import extract_frontmatter

        # YAML that parses to a list, not a dict
        content = """---
- item1
- item2
---

Content.
"""
        with caplog.at_level("WARNING"):
            frontmatter, remaining = extract_frontmatter(content)

        assert frontmatter == {}
        assert "FrontMatter is not a dictionary" in caplog.text

    def test_normalize_tags_non_string_non_list(self):
        """Test normalizing tags that are neither string nor list."""
        from obsidian_rag.parsing.frontmatter import normalize_tags

        # Pass an integer as tags (edge case)
        result = normalize_tags(123)
        assert result == ["123"]

    def test_normalize_tags_boolean(self):
        """Test normalizing boolean tags (edge case)."""
        from obsidian_rag.parsing.frontmatter import normalize_tags

        # Boolean values should be converted to strings
        result = normalize_tags(True)
        assert result == ["True"]

    def test_normalize_tags_integer_in_list(self):
        """Test normalizing list with integer values."""
        from obsidian_rag.parsing.frontmatter import normalize_tags

        result = normalize_tags([1, 2, 3])
        assert result == ["1", "2", "3"]
