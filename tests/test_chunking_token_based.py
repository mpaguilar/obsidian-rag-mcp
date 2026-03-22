"""Tests for token-based chunking."""

from unittest.mock import Mock, patch

import pytest

from obsidian_rag.chunking import (
    Chunk,
    ChunkType,
    TokenChunk,
    chunk_document,
    should_chunk_document,
    split_into_token_chunks,
)
from obsidian_rag.tokenizer import TokenizerConfig


class TestTokenChunk:
    """Test cases for TokenChunk dataclass."""

    def test_token_chunk_creation(self):
        """Test TokenChunk creation with all fields."""
        chunk = TokenChunk(
            text="Test content",
            start_char=0,
            end_char=12,
            index=0,
            token_count=512,
            chunk_type=ChunkType.CONTENT,
        )
        assert chunk.text == "Test content"
        assert chunk.start_char == 0
        assert chunk.end_char == 12
        assert chunk.index == 0
        assert chunk.token_count == 512
        assert chunk.chunk_type == ChunkType.CONTENT

    def test_token_chunk_defaults(self):
        """Test TokenChunk with default values."""
        chunk = TokenChunk(
            text="Test",
            start_char=0,
            end_char=4,
            index=0,
        )
        assert chunk.token_count is None
        assert chunk.chunk_type is None


class TestChunkType:
    """Test cases for ChunkType enum."""

    def test_chunk_type_values(self):
        """Test ChunkType enum values."""
        assert ChunkType.CONTENT.value == "content"
        assert ChunkType.TASK.value == "task"


class TestSplitIntoTokenChunks:
    """Test cases for split_into_token_chunks function."""

    @patch("obsidian_rag.tokenizer.get_tokenizer")
    @patch("obsidian_rag.tokenizer.count_tokens")
    def test_single_chunk_short_content(self, mock_count, mock_get_tokenizer):
        """Test that short content returns single chunk."""
        mock_tokenizer = Mock()
        mock_get_tokenizer.return_value = mock_tokenizer
        mock_count.return_value = 100  # Less than 512

        config = TokenizerConfig(chunk_size=512, chunk_overlap=50)
        result = split_into_token_chunks("Short content", config)

        assert len(result) == 1
        assert result[0].text == "Short content"
        assert result[0].token_count == 100

    @patch("obsidian_rag.tokenizer.get_tokenizer")
    @patch("obsidian_rag.tokenizer.count_tokens")
    def test_multiple_chunks_long_content(self, mock_count, mock_get_tokenizer):
        """Test that long content is split into multiple chunks."""
        mock_tokenizer = Mock()
        mock_get_tokenizer.return_value = mock_tokenizer

        # Simulate content that needs splitting
        def side_effect(text, tokenizer):
            return len(text) // 4  # ~4 chars per token

        mock_count.side_effect = side_effect

        # Create content of ~2000 tokens (8000 chars)
        content = "word " * 1600  # ~8000 chars

        config = TokenizerConfig(chunk_size=512, chunk_overlap=50)
        result = split_into_token_chunks(content, config)

        # Should create multiple chunks
        assert len(result) > 1

        # Each chunk should be approximately 512 tokens
        for chunk in result:
            assert chunk.token_count is not None
            assert chunk.token_count <= 512 + 50  # Allow for overlap

    @patch("obsidian_rag.tokenizer.get_tokenizer")
    def test_empty_content(self, mock_get_tokenizer):
        """Test that empty content returns empty list."""
        mock_get_tokenizer.return_value = Mock()

        config = TokenizerConfig()
        result = split_into_token_chunks("", config)

        assert result == []

    @patch("obsidian_rag.tokenizer.get_tokenizer")
    @patch("obsidian_rag.tokenizer.count_tokens")
    def test_chunk_type_task_detection(self, mock_count, mock_get_tokenizer):
        """Test that task lines are detected as task chunks."""
        mock_tokenizer = Mock()
        mock_get_tokenizer.return_value = mock_tokenizer
        mock_count.return_value = 50

        content = "- [ ] This is a task line\nSome other content"
        config = TokenizerConfig()

        result = split_into_token_chunks(content, config)

        # First chunk should be task type
        assert result[0].chunk_type == ChunkType.TASK

    @patch("obsidian_rag.tokenizer.get_tokenizer")
    @patch("obsidian_rag.tokenizer.count_tokens")
    def test_chunk_overlap(self, mock_count, mock_get_tokenizer):
        """Test that chunks have proper overlap."""
        mock_tokenizer = Mock()
        mock_get_tokenizer.return_value = mock_tokenizer
        # Return realistic token counts: ~4 chars per token
        mock_count.side_effect = lambda text, _: max(1, len(text) // 4)

        content = "word " * 2000  # ~10000 chars = ~2500 tokens
        config = TokenizerConfig(chunk_size=512, chunk_overlap=50)

        result = split_into_token_chunks(content, config)

        # Check that consecutive chunks overlap
        if len(result) > 1:
            for i in range(len(result) - 1):
                # End of current chunk should overlap with start of next
                # Allow for edge case where they might be equal
                assert result[i].end_char >= result[i + 1].start_char


class TestShouldChunkDocument:
    """Test cases for should_chunk_document function."""

    @patch("obsidian_rag.tokenizer.get_tokenizer")
    @patch("obsidian_rag.tokenizer.count_tokens")
    def test_should_chunk_when_over_limit(self, mock_count, mock_get_tokenizer):
        """Test that document over token limit should be chunked."""
        mock_get_tokenizer.return_value = Mock()
        mock_count.return_value = 600  # Over 512 limit

        result = should_chunk_document("Long content", 512, "/cache")

        assert result is True

    @patch("obsidian_rag.tokenizer.get_tokenizer")
    @patch("obsidian_rag.tokenizer.count_tokens")
    def test_should_not_chunk_when_under_limit(self, mock_count, mock_get_tokenizer):
        """Test that document under token limit should not be chunked."""
        mock_get_tokenizer.return_value = Mock()
        mock_count.return_value = 400  # Under 512 limit

        result = should_chunk_document("Short content", 512, "/cache")

        assert result is False


class TestChunkDocument:
    """Test cases for chunk_document function."""

    @patch("obsidian_rag.chunking.split_into_token_chunks")
    @patch("obsidian_rag.chunking.should_chunk_document")
    def test_chunk_document_creates_chunks(self, mock_should, mock_split):
        """Test that chunk_document creates chunks for large documents."""
        mock_should.return_value = True
        mock_split.return_value = [
            TokenChunk(
                text="Chunk 1",
                start_char=0,
                end_char=7,
                index=0,
                token_count=512,
                chunk_type=ChunkType.CONTENT,
            ),
        ]

        result = chunk_document("Long content", "doc-uuid", 512, 50)

        assert len(result) == 1
        assert result[0]["chunk_text"] == "Chunk 1"
        assert result[0]["token_count"] == 512
        assert result[0]["chunk_type"] == "content"
        assert result[0]["document_id"] == "doc-uuid"

    @patch("obsidian_rag.chunking.should_chunk_document")
    def test_chunk_document_single_chunk(self, mock_should):
        """Test that small documents return single chunk."""
        mock_should.return_value = False

        result = chunk_document("Short", "doc-uuid", 512, 50)

        assert len(result) == 1
        assert result[0]["chunk_text"] == "Short"
        assert result[0]["chunk_index"] == 0
