"""Tests for document chunking algorithm.

This module tests the chunking functionality including:
- Token estimation
- Split point finding
- Chunk creation with various content sizes
- Overlap handling
- Edge cases

"""

import pytest

from obsidian_rag.chunking import (
    Chunk,
    _calculate_next_start,
    _create_chunks_from_content,
    _create_single_chunk,
    _estimate_tokens,
    _find_split_point,
    _normalize_chunking_params,
    should_chunk_document,
    split_into_chunks,
)


def test_estimate_tokens_basic():
    """Test token estimation for basic text."""
    text = "a" * 400  # 400 chars = ~100 tokens
    result = _estimate_tokens(text)
    assert result == 100


def test_estimate_tokens_empty():
    """Test token estimation for empty text."""
    result = _estimate_tokens("")
    assert result == 0


def test_estimate_tokens_long():
    """Test token estimation for long text."""
    text = "word " * 1000  # ~5000 chars
    result = _estimate_tokens(text)
    assert result == 1250  # 5000 // 4


def test_find_split_point_paragraph_boundary():
    """Test finding split at paragraph boundary."""
    text = "Paragraph 1.\n\nParagraph 2.\n\nParagraph 3."
    target = len("Paragraph 1.\n\nParagraph 2.") + 5
    result = _find_split_point(text, target, len(text))
    # Result includes the paragraph delimiter (\n\n), so it's 28 not 26
    assert result == len("Paragraph 1.\n\nParagraph 2.\n\n")


def test_find_split_point_sentence_boundary():
    """Test finding split at sentence boundary."""
    text = "First sentence. Second sentence. Third sentence."
    target = len("First sentence. Second se") + 5
    result = _find_split_point(text, target, len(text))
    # Should find the period after "First sentence."
    assert result == len("First sentence. ")


def test_find_split_point_newline_boundary():
    """Test finding split at newline boundary."""
    text = "Line 1\nLine 2\nLine 3"
    target = len("Line 1\nLine 2") + 2
    result = _find_split_point(text, target, len(text))
    assert result == len("Line 1\nLine 2\n")


def test_find_split_point_fallback():
    """Test fallback to target when no boundary found."""
    text = "wordwordwordwordword"
    target = 10
    result = _find_split_point(text, target, len(text))
    assert result == target


def test_find_split_point_respects_max():
    """Test that split point respects max_pos limit."""
    text = "a" * 100
    target = 80
    max_pos = 50
    result = _find_split_point(text, target, max_pos)
    assert result <= max_pos


def test_split_into_chunks_empty_text():
    """Test chunking empty text returns empty list."""
    result = split_into_chunks("", 24000, 800)
    assert result == []


def test_split_into_chunks_whitespace_only():
    """Test chunking whitespace-only text returns single chunk."""
    result = split_into_chunks("   \n\t  ", 24000, 800)
    # Whitespace-only content is still valid content, returns single chunk
    assert len(result) == 1
    assert result[0].text == "   \n\t  "


def test_split_into_chunks_small_document():
    """Test small document returns single chunk."""
    text = "Small document content."
    result = split_into_chunks(text, 24000, 800)
    assert len(result) == 1
    assert result[0].text == text
    assert result[0].index == 0
    assert result[0].start_char == 0
    assert result[0].end_char == len(text)


def test_split_into_chunks_exact_size():
    """Test document exactly at chunk size returns single chunk."""
    text = "a" * 24000
    result = split_into_chunks(text, 24000, 800)
    assert len(result) == 1
    assert result[0].text == text


def test_split_into_chunks_large_document():
    """Test large document splits into multiple chunks."""
    # Create text larger than max_chunk_chars
    paragraph = "Word " * 100 + ".\n\n"  # ~500 chars per paragraph
    text = paragraph * 100  # ~50K chars, should split into 3 chunks

    result = split_into_chunks(text, 24000, 800)

    assert len(result) >= 2
    assert all(len(chunk.text) <= 24000 for chunk in result)

    # Check indices are sequential
    for i, chunk in enumerate(result):
        assert chunk.index == i


def test_split_into_chunks_overlap_content():
    """Test that overlap contains text from previous chunk."""
    chunk1_text = "A" * 20000 + "\n\n" + "B" * 5000
    text = chunk1_text + "\n\n" + "C" * 10000

    result = split_into_chunks(text, 24000, 800)

    if len(result) > 1:
        # Second chunk should contain some text from first chunk's end
        chunk2 = result[1]
        assert "B" * 100 in chunk2.text  # Overlap contains end of chunk 1


def test_split_into_chunks_no_infinite_loop():
    """Test that chunking always makes progress."""
    text = "word " * 50000  # Very long text without paragraph breaks

    result = split_into_chunks(text, 24000, 800)

    assert len(result) > 1
    # Each chunk should be different
    for i in range(1, len(result)):
        assert result[i].start_char > result[i - 1].start_char


def test_split_into_chunks_preserves_content():
    """Test that all content is preserved across chunks (except overlap)."""
    text = "Paragraph one.\n\nParagraph two.\n\nParagraph three."
    result = split_into_chunks(text, 50, 10)  # Small chunks for testing

    # Reconstruct text from chunks (accounting for overlap)
    reconstructed = result[0].text
    for i in range(1, len(result)):
        chunk = result[i]
        prev_chunk = result[i - 1]
        overlap_len = prev_chunk.end_char - chunk.start_char
        reconstructed += chunk.text[overlap_len:]

    assert reconstructed == text


def test_split_into_chunks_chunk_attributes():
    """Test that all chunk attributes are correctly set."""
    text = "A" * 10000 + "\n\n" + "B" * 10000 + "\n\n" + "C" * 10000
    result = split_into_chunks(text, 24000, 800)

    for chunk in result:
        assert isinstance(chunk, Chunk)
        assert isinstance(chunk.text, str)
        assert isinstance(chunk.start_char, int)
        assert isinstance(chunk.end_char, int)
        assert isinstance(chunk.index, int)
        assert chunk.start_char >= 0
        assert chunk.end_char <= len(text)
        assert chunk.start_char < chunk.end_char


def test_should_chunk_document_true():
    """Test should_chunk_document returns True for large content."""
    text = "a" * 25000
    result = should_chunk_document(text, 24000)
    assert result is True


def test_should_chunk_document_false():
    """Test should_chunk_document returns False for small content."""
    text = "a" * 1000
    result = should_chunk_document(text, 24000)
    assert result is False


def test_should_chunk_document_exact_size():
    """Test should_chunk_document at exact size boundary."""
    text = "a" * 24000
    result = should_chunk_document(text, 24000)
    assert result is False  # Equal size doesn't need chunking


def test_should_chunk_document_empty():
    """Test should_chunk_document with empty content."""
    result = should_chunk_document("", 24000)
    assert result is False


def test_should_chunk_document_invalid_max():
    """Test should_chunk_document with invalid max_chunk_chars."""
    text = "a" * 1000
    # Should handle invalid input gracefully
    result = should_chunk_document(text, "invalid")  # type: ignore[arg-type]
    assert result is False  # Uses default 24000


def test_split_into_chunks_invalid_params_type():
    """Test that invalid parameter types are handled gracefully."""
    text = "Some content"
    # Should not raise, uses defaults
    result = split_into_chunks(text, "invalid", "invalid")  # type: ignore[arg-type]
    assert len(result) == 1


def test_split_into_chunks_zero_max_chars():
    """Test that zero max_chunk_chars uses default."""
    text = "Some content"
    result = split_into_chunks(text, 0, 800)
    assert len(result) == 1


def test_split_into_chunks_negative_overlap():
    """Test that negative overlap uses default."""
    text = "Some content"
    result = split_into_chunks(text, 24000, -100)
    assert len(result) == 1


def test_split_into_chunks_large_overlap():
    """Test chunking with large overlap."""
    text = "A" * 10000 + "\n\n" + "B" * 10000
    # Overlap larger than half the chunk size
    result = split_into_chunks(text, 12000, 8000)

    assert len(result) >= 2
    # Verify overlap is applied
    if len(result) > 1:
        overlap = result[0].end_char - result[1].start_char
        assert overlap > 0


def test_chunk_dataclass_equality():
    """Test Chunk dataclass equality."""
    chunk1 = Chunk(text="hello", start_char=0, end_char=5, index=0)
    chunk2 = Chunk(text="hello", start_char=0, end_char=5, index=0)
    chunk3 = Chunk(text="world", start_char=0, end_char=5, index=0)

    assert chunk1 == chunk2
    assert chunk1 != chunk3


def test_chunk_dataclass_repr():
    """Test Chunk dataclass representation."""
    chunk = Chunk(text="hello", start_char=0, end_char=5, index=0)
    repr_str = repr(chunk)
    assert "Chunk" in repr_str
    assert "hello" in repr_str


def test_split_into_chunks_progress_protection():
    """Test that chunking makes progress even with problematic content.

    This test triggers the edge case where _find_split_point returns
    a position at or before start_pos, requiring the progress protection
    logic at line 202.
    """
    # Create content where split point finding might not make progress
    # Very long string without any delimiters
    text = "a" * 50000  # 50K chars of continuous 'a'

    result = split_into_chunks(text, 24000, 800)

    # Should create multiple chunks
    assert len(result) > 1
    # Each chunk should make progress
    for i in range(1, len(result)):
        assert result[i].start_char > result[i - 1].start_char


def test_split_into_chunks_overlap_protection():
    """Test overlap reduction when overlap would prevent progress.

    This test triggers the edge case at line 224 where next_start
    would be <= start_pos, requiring overlap reduction.
    """
    # Create content with very small chunk size and large overlap
    text = "Word " * 1000  # ~5K chars

    # Small max_chars (1000) with large overlap (900)
    # This forces the overlap protection logic
    result = split_into_chunks(text, 1000, 900)

    # Should create multiple chunks
    assert len(result) > 1
    # Verify all chunks have valid positions
    for chunk in result:
        assert chunk.start_char >= 0
        assert chunk.end_char <= len(text)
        assert chunk.start_char < chunk.end_char


def test_normalize_chunking_params_valid():
    """Test parameter normalization with valid values."""
    max_chars, overlap = _normalize_chunking_params(1000, 100)
    assert max_chars == 1000
    assert overlap == 100


def test_normalize_chunking_params_invalid_max():
    """Test parameter normalization with invalid max_chunk_chars."""
    from obsidian_rag.chunking import DEFAULT_MAX_CHUNK_CHARS

    max_chars, overlap = _normalize_chunking_params(0, 100)
    assert max_chars == DEFAULT_MAX_CHUNK_CHARS


def test_normalize_chunking_params_invalid_overlap():
    """Test parameter normalization with invalid overlap."""
    from obsidian_rag.chunking import DEFAULT_CHUNK_OVERLAP_CHARS

    max_chars, overlap = _normalize_chunking_params(1000, -1)
    assert overlap == DEFAULT_CHUNK_OVERLAP_CHARS


def test_normalize_chunking_params_type_error():
    """Test parameter normalization with type errors."""
    from obsidian_rag.chunking import (
        DEFAULT_CHUNK_OVERLAP_CHARS,
        DEFAULT_MAX_CHUNK_CHARS,
    )

    max_chars, overlap = _normalize_chunking_params("invalid", "invalid")  # type: ignore[arg-type]
    assert max_chars == DEFAULT_MAX_CHUNK_CHARS
    assert overlap == DEFAULT_CHUNK_OVERLAP_CHARS


def test_create_single_chunk():
    """Test creating a single chunk."""
    content = "Test content for single chunk"
    result = _create_single_chunk(content)

    assert len(result) == 1
    assert result[0].text == content
    assert result[0].start_char == 0
    assert result[0].end_char == len(content)
    assert result[0].index == 0


def test_calculate_next_start_normal():
    """Test next start calculation with normal overlap."""
    result = _calculate_next_start(1000, 100, 0)
    assert result == 900  # 1000 - 100


def test_calculate_next_start_no_progress():
    """Test next start when overlap would prevent progress."""
    # When next_start (1000 - 900 = 100) <= current_start (100)
    # Should return actual_end (1000) instead
    result = _calculate_next_start(1000, 900, 100)
    assert result == 1000


def test_create_chunks_from_content():
    """Test creating multiple chunks from content."""
    content = "Word " * 500  # ~2500 chars
    result = _create_chunks_from_content(content, 1000, 100)

    assert len(result) > 1
    # Verify chunk attributes
    for i, chunk in enumerate(result):
        assert chunk.index == i
        assert chunk.text == content[chunk.start_char : chunk.end_char]


def test_create_chunks_from_content_progress_protection():
    """Test progress protection in chunk creation.

    This triggers the edge case where actual_end <= start_pos
    and we need to force progress.
    """
    # Create content that might trigger the progress protection
    content = "a" * 100
    # Very small max_chars to force many chunks
    result = _create_chunks_from_content(content, 50, 0)

    assert len(result) >= 2
    # Verify we made progress
    for i in range(len(result) - 1):
        assert result[i + 1].start_char > result[i].start_char


def test_create_chunks_from_content_loop_exit():
    """Test that the while loop properly exits when start_pos >= content_len.

    This covers line 210->237 (the while loop exit branch).
    """
    # Create content that will be fully processed in one iteration
    content = "a" * 50  # Small content that fits in one chunk

    result = _create_chunks_from_content(content, 100, 10)

    # Should create exactly one chunk
    assert len(result) == 1
    assert result[0].start_char == 0
    assert result[0].end_char == 50
    assert result[0].text == content


def test_create_chunks_from_content_actual_end_fallback():
    """Test the actual_end fallback when _find_split_point returns <= start_pos.

    This covers line 219 where actual_end is forced to make progress.
    To trigger this, we need _find_split_point to return a value <= start_pos,
    which can happen when the search window has no delimiters.
    """
    # Create a scenario where _find_split_point might not find a good split
    # Use very small max_chars with content that has no delimiters in the search window
    content = "a" * 2000  # Long content without delimiters

    # With max_chars=1000 and overlap=900, the second chunk's next_start would be
    # 1000 - 900 = 100, but we need to trigger the actual_end <= start_pos case
    # This happens when _find_split_point returns a position at or before start_pos

    # Use a small chunk size to force the edge case
    result = _create_chunks_from_content(content, 500, 0)

    # Should create multiple chunks without getting stuck
    assert len(result) >= 3
    # Verify all chunks make progress
    for i in range(len(result) - 1):
        assert result[i + 1].start_char > result[i].start_char


def test_create_chunks_from_content_exact_boundary():
    """Test chunking when content length exactly matches chunk boundaries."""
    # Content that exactly fills 2 chunks with no overlap
    content = "a" * 1000  # Exactly 1000 chars

    result = _create_chunks_from_content(content, 500, 0)

    # Should create exactly 2 chunks
    assert len(result) == 2
    assert result[0].text == "a" * 500
    assert result[1].text == "a" * 500
    assert result[1].start_char == 500


def test_create_chunks_from_content_single_char():
    """Test chunking single character content."""
    content = "x"

    result = _create_chunks_from_content(content, 1000, 100)

    assert len(result) == 1
    assert result[0].text == "x"
    assert result[0].start_char == 0
    assert result[0].end_char == 1


def test_create_chunks_from_content_with_mocked_split_point():
    """Test chunking when _find_split_point returns invalid position.

    This directly tests line 219 where actual_end <= start_pos.
    We mock _find_split_point to return an invalid position.
    """
    from unittest.mock import patch

    content = "a" * 100

    # Mock _find_split_point to return a position <= start_pos
    # This should trigger the fallback at line 219
    with patch("obsidian_rag.chunking._find_split_point", return_value=0) as mock_find:
        result = _create_chunks_from_content(content, 50, 0)

        # Verify the mock was called
        assert mock_find.called
        # Should still create chunks (using the fallback)
        assert len(result) >= 2
        # Verify progress was made
        for i in range(len(result) - 1):
            assert result[i + 1].start_char > result[i].start_char


def test_create_chunks_from_content_loop_exit_branch():
    """Test the while loop exit branch (line 210->237).

    This test ensures the loop exit condition is covered.
    """
    # Empty content should immediately exit the loop
    result = _create_chunks_from_content("", 100, 10)
    assert result == []

    # Single chunk content should exit after one iteration
    content = "short"
    result = _create_chunks_from_content(content, 100, 10)
    assert len(result) == 1
    assert result[0].text == content
