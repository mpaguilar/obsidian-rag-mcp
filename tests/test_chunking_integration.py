"""Integration tests for token-based chunking."""

from unittest.mock import Mock, patch

from obsidian_rag.chunking import chunk_document, split_into_token_chunks
from obsidian_rag.tokenizer import TokenizerConfig


class TestTokenChunkingIntegration:
    """Integration tests for token-based chunking."""

    @patch("obsidian_rag.tokenizer.get_tokenizer")
    @patch("obsidian_rag.tokenizer.count_tokens")
    def test_chunk_real_document(
        self,
        mock_count_tokens,
        mock_get_tokenizer,
        tmp_path,
    ):
        """Test chunking a real document with ~2000 tokens."""
        _msg = "test_chunk_real_document starting"
        print(_msg)

        # Setup mock tokenizer
        mock_tokenizer = Mock()
        mock_get_tokenizer.return_value = mock_tokenizer

        # Simulate realistic token counting (~4 chars per token)
        def count_tokens_side_effect(text, tokenizer):
            return len(text) // 4

        mock_count_tokens.side_effect = count_tokens_side_effect

        # Create a test document with ~2000 tokens worth of content
        # (~8000 characters, ~4 chars per token)
        content = (
            """
# Test Document

This is a paragraph with some content. It should be long enough to test chunking.
"""
            + "word " * 1600
        )  # ~8000 chars, ~2000 tokens

        config = TokenizerConfig(chunk_size=512, chunk_overlap=50)

        chunks = split_into_token_chunks(content, config)

        # Should create multiple chunks
        assert len(chunks) > 1

        # Each chunk should have token count
        for chunk in chunks:
            assert chunk.token_count is not None
            assert chunk.token_count <= 512 + 50  # Allow for overlap

        _msg = f"test_chunk_real_document returning: {len(chunks)} chunks created"
        print(_msg)

    @patch("obsidian_rag.tokenizer.get_tokenizer")
    @patch("obsidian_rag.tokenizer.count_tokens")
    def test_chunk_with_tasks(self, mock_count_tokens, mock_get_tokenizer):
        """Test that task lines are detected as task chunks."""
        _msg = "test_chunk_with_tasks starting"
        print(_msg)

        # Setup mock tokenizer
        mock_tokenizer = Mock()
        mock_get_tokenizer.return_value = mock_tokenizer

        # Simulate realistic token counting
        def count_tokens_side_effect(text, tokenizer):
            return len(text) // 4

        mock_count_tokens.side_effect = count_tokens_side_effect

        content = (
            """
- [ ] This is a task
- [x] This is a completed task

Regular paragraph content here.
"""
            + "word " * 1000
        )  # Add enough content to ensure chunking

        config = TokenizerConfig(chunk_size=512)

        chunks = split_into_token_chunks(content, config)

        # First chunk should contain tasks
        assert any(c.chunk_type.value == "task" for c in chunks if c.chunk_type)

        _msg = "test_chunk_with_tasks returning: task chunks detected"
        print(_msg)

    @patch("obsidian_rag.tokenizer.get_tokenizer")
    @patch("obsidian_rag.tokenizer.count_tokens")
    def test_chunk_document_for_database(
        self,
        mock_count_tokens,
        mock_get_tokenizer,
    ):
        """Test chunk_document function returns proper database-ready dicts."""
        _msg = "test_chunk_document_for_database starting"
        print(_msg)

        # Setup mock tokenizer
        mock_tokenizer = Mock()
        mock_get_tokenizer.return_value = mock_tokenizer

        # Simulate realistic token counting
        def count_tokens_side_effect(text, tokenizer):
            return len(text) // 4

        mock_count_tokens.side_effect = count_tokens_side_effect

        content = "word " * 2000  # ~8000 chars, ~2000 tokens

        chunks = chunk_document(
            content,
            "doc-uuid-123",
            chunk_size=512,
            chunk_overlap=50,
            model_name="gpt2",
        )

        assert len(chunks) > 0

        for chunk in chunks:
            assert "document_id" in chunk
            assert "chunk_index" in chunk
            assert "chunk_text" in chunk
            assert "token_count" in chunk
            assert "chunk_type" in chunk
            assert chunk["document_id"] == "doc-uuid-123"

        _msg = f"test_chunk_document_for_database returning: {len(chunks)} chunks"
        print(_msg)
