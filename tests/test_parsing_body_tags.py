"""Tests for body tag extraction module."""

from obsidian_rag.parsing.body_tags import _strip_code_blocks, extract_body_tags


class TestStripCodeBlocks:
    """Test cases for _strip_code_blocks function."""

    def test_strip_code_blocks_no_code_blocks(self) -> None:
        """Content with no code blocks is unchanged."""
        content = "This is regular content with #tag."
        result = _strip_code_blocks(content)
        assert result == content

    def test_strip_code_blocks_single_fenced(self) -> None:
        """Single fenced block is removed."""
        content = "before\n```\ncode\n```\nafter"
        result = _strip_code_blocks(content)
        assert result == "before\n\nafter"

    def test_strip_code_blocks_multiple_fenced(self) -> None:
        """Multiple fenced blocks are removed."""
        content = "a\n```\ncode1\n```\nb\n```\ncode2\n```\nc"
        result = _strip_code_blocks(content)
        assert result == "a\n\nb\n\nc"

    def test_strip_code_blocks_inline_code(self) -> None:
        """Inline code is removed."""
        content = "text `code #tag` more"
        result = _strip_code_blocks(content)
        assert result == "text  more"

    def test_strip_code_blocks_preserves_other_content(self) -> None:
        """Content outside code blocks is preserved."""
        content = "keep this #tag\n```\ncode\n```\nkeep this too"
        result = _strip_code_blocks(content)
        assert "keep this #tag" in result
        assert "keep this too" in result
        assert "code" not in result

    def test_strip_code_blocks_empty_code_block(self) -> None:
        """Empty fenced block is removed."""
        content = "before\n```\n```\nafter"
        result = _strip_code_blocks(content)
        assert result == "before\n\nafter"

    def test_strip_code_blocks_language_identifier(self) -> None:
        """Fenced block with language identifier is removed."""
        content = "before\n```python\n#tag\n```\nafter"
        result = _strip_code_blocks(content)
        assert result == "before\n\nafter"

    def test_strip_code_blocks_unclosed_fenced(self) -> None:
        """Unclosed fenced block strips to end of content."""
        content = "before\n```\ncode #tag"
        result = _strip_code_blocks(content)
        assert result == "before\n"

    def test_strip_code_blocks_empty_content(self) -> None:
        """Empty string returns empty string."""
        assert _strip_code_blocks("") == ""

    def test_strip_code_blocks_nested_content(self) -> None:
        """Content before and after blocks is preserved."""
        content = "start #keep\n```\ncode #drop\n```\nend #keep"
        result = _strip_code_blocks(content)
        assert "start #keep" in result
        assert "end #keep" in result
        assert "drop" not in result


class TestExtractBodyTags:
    """Test cases for extract_body_tags function."""

    def test_extract_body_tags_no_tags(self) -> None:
        """Content with no tags returns None."""
        result = extract_body_tags("Just plain text.")
        assert result is None

    def test_extract_body_tags_none_content(self) -> None:
        """None content returns None."""
        assert extract_body_tags(None) is None

    def test_extract_body_tags_single_tag(self) -> None:
        """Single #tag returns [tag]."""
        result = extract_body_tags("This has a #tag.")
        assert result == ["tag"]

    def test_extract_body_tags_multiple_tags(self) -> None:
        """Multiple different tags are returned."""
        result = extract_body_tags("#first and #second here.")
        assert result == ["first", "second"]

    def test_extract_body_tags_hierarchical_tag(self) -> None:
        """Hierarchical tag personal/expenses is extracted."""
        result = extract_body_tags("#personal/expenses")
        assert result == ["personal/expenses"]

    def test_extract_body_tags_deduplication(self) -> None:
        """Same tag appearing twice becomes a single entry."""
        result = extract_body_tags("#tag #tag")
        assert result == ["tag"]

    def test_extract_body_tags_case_insensitive_dedup(self) -> None:
        """#Work and #work are deduplicated case-insensitively."""
        result = extract_body_tags("#Work #work")
        assert result == ["work"]

    def test_extract_body_tags_empty_content(self) -> None:
        """Empty string returns None."""
        assert extract_body_tags("") is None

    def test_extract_body_tags_task_line(self) -> None:
        """Tag inside a task line is extracted as a body tag."""
        result = extract_body_tags("- [ ] do something #important")
        assert result == ["important"]

    def test_extract_body_tags_excludes_heading(self) -> None:
        """# Heading is not extracted as a tag."""
        result = extract_body_tags("# Heading\n")
        assert result is None

    def test_extract_body_tags_excludes_h2_heading(self) -> None:
        """## Heading is not extracted as a tag."""
        result = extract_body_tags("## Heading\n")
        assert result is None

    def test_extract_body_tags_excludes_all_numeric(self) -> None:
        """#1984 is not extracted as a tag."""
        result = extract_body_tags("Read #1984 today.")
        assert "1984" not in (result or [])

    def test_extract_body_tags_includes_mixed_alphanumeric(self) -> None:
        """#y1984 is extracted as a tag."""
        result = extract_body_tags("#y1984")
        assert result == ["y1984"]

    def test_extract_body_tags_excludes_tags_in_code_block(self) -> None:
        """Tags inside fenced code blocks are not extracted."""
        result = extract_body_tags("```\n#code-tag\n```\n#real-tag")
        assert result == ["real-tag"]

    def test_extract_body_tags_excludes_tags_in_inline_code(self) -> None:
        """Tags inside inline code are not extracted."""
        result = extract_body_tags("Use `#inline` and #real-tag")
        assert result == ["real-tag"]

    def test_extract_body_tags_includes_tags_in_blockquote(self) -> None:
        """Tags in blockquotes are extracted."""
        result = extract_body_tags("> #blockquote-tag")
        assert result == ["blockquote-tag"]

    def test_extract_body_tags_heading_and_tag_same_line(self) -> None:
        """Heading with tag on same line extracts only the tag."""
        result = extract_body_tags("# Heading with #tag")
        assert result == ["tag"]

    def test_extract_body_tags_tag_at_line_start(self) -> None:
        """Tag at start of line without space after is extracted."""
        result = extract_body_tags("#personal/expenses at line start")
        assert result == ["personal/expenses"]

    def test_extract_body_tags_no_frontmatter_content(self) -> None:
        """Works on pure body content."""
        result = extract_body_tags("Some body text with #tag.")
        assert result == ["tag"]

    def test_extract_body_tags_dot_tags(self) -> None:
        """Version-style tag with dot is extracted."""
        result = extract_body_tags("#v1.0/release")
        assert result == ["v1.0/release"]
