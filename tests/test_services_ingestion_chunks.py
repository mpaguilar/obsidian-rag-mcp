"""Tests for ingestion_chunks module.

Tests chunking operations extracted from ingestion.py to comply with
the 1000 line limit.
"""

import uuid
from unittest.mock import MagicMock, patch

import pytest

from obsidian_rag.services.ingestion_chunks import (
    BATCH_SIZE,
    MAX_RETRIES,
    _create_document_chunk,
    _generate_embedding_with_retry,
    _process_chunk_batch,
    create_chunks_with_embeddings,
)


class TestGenerateEmbeddingWithRetry:
    """Tests for _generate_embedding_with_retry function."""

    def test_no_embedding_provider_returns_none(self):
        """Test that None is returned when no embedding provider."""
        result = _generate_embedding_with_retry("test text", None)
        assert result is None

    def test_successful_embedding_first_attempt(self):
        """Test successful embedding on first attempt."""
        mock_provider = MagicMock()
        mock_provider.generate_embedding.return_value = [0.1, 0.2, 0.3]

        result = _generate_embedding_with_retry("test text", mock_provider)

        assert result == [0.1, 0.2, 0.3]
        mock_provider.generate_embedding.assert_called_once_with("test text")

    def test_successful_embedding_after_retry(self):
        """Test successful embedding after retry on failure."""
        mock_provider = MagicMock()
        mock_provider.generate_embedding.side_effect = [
            RuntimeError("Network error"),
            [0.1, 0.2, 0.3],
        ]

        result = _generate_embedding_with_retry("test text", mock_provider)

        assert result == [0.1, 0.2, 0.3]
        assert mock_provider.generate_embedding.call_count == 2

    def test_max_retries_exceeded_returns_none(self):
        """Test that None is returned after max retries exceeded."""
        mock_provider = MagicMock()
        mock_provider.generate_embedding.side_effect = RuntimeError("Persistent error")

        result = _generate_embedding_with_retry(
            "test text", mock_provider, max_retries=3
        )

        assert result is None
        assert mock_provider.generate_embedding.call_count == 3

    def test_oserror_triggers_retry(self):
        """Test that OSError triggers retry logic."""
        mock_provider = MagicMock()
        mock_provider.generate_embedding.side_effect = [
            OSError("Disk error"),
            [0.1, 0.2, 0.3],
        ]

        result = _generate_embedding_with_retry("test text", mock_provider)

        assert result == [0.1, 0.2, 0.3]
        assert mock_provider.generate_embedding.call_count == 2

    def test_value_error_triggers_retry(self):
        """Test that ValueError triggers retry logic."""
        mock_provider = MagicMock()
        mock_provider.generate_embedding.side_effect = [
            ValueError("Invalid input"),
            [0.1, 0.2, 0.3],
        ]

        result = _generate_embedding_with_retry("test text", mock_provider)

        assert result == [0.1, 0.2, 0.3]
        assert mock_provider.generate_embedding.call_count == 2

    def test_unexpected_exception_not_caught(self):
        """Test that unexpected exceptions are not caught."""
        mock_provider = MagicMock()
        mock_provider.generate_embedding.side_effect = TypeError("Unexpected")

        with pytest.raises(TypeError):
            _generate_embedding_with_retry("test text", mock_provider)


class TestCreateDocumentChunk:
    """Tests for _create_document_chunk function."""

    def test_create_chunk_with_all_fields(self):
        """Test creating a chunk with all fields populated."""
        mock_session = MagicMock()
        document_id = uuid.uuid4()
        chunk_data = {
            "chunk_index": 0,
            "chunk_text": "Test chunk content",
            "start_char": 0,
            "end_char": 100,
            "token_count": 50,
            "chunk_type": "content",
        }
        embedding = [0.1, 0.2, 0.3]

        result = _create_document_chunk(
            mock_session, document_id, chunk_data, embedding
        )

        assert result.document_id == document_id
        assert result.chunk_index == 0
        assert result.chunk_text == "Test chunk content"
        assert result.chunk_vector == embedding
        assert result.start_char == 0
        assert result.end_char == 100
        assert result.token_count == 50
        assert result.chunk_type == "content"
        mock_session.add.assert_called_once_with(result)

    def test_create_chunk_without_optional_fields(self):
        """Test creating a chunk without optional fields."""
        mock_session = MagicMock()
        document_id = uuid.uuid4()
        chunk_data = {
            "chunk_index": 1,
            "chunk_text": "Another chunk",
            "start_char": 100,
            "end_char": 200,
        }
        embedding = [0.4, 0.5, 0.6]

        result = _create_document_chunk(
            mock_session, document_id, chunk_data, embedding
        )

        assert result.document_id == document_id
        assert result.chunk_index == 1
        assert result.chunk_text == "Another chunk"
        assert result.token_count is None
        assert result.chunk_type is None


class TestProcessChunkBatch:
    """Tests for _process_chunk_batch function."""

    @patch("obsidian_rag.services.ingestion_chunks._generate_embedding_with_retry")
    @patch("obsidian_rag.services.ingestion_chunks._create_document_chunk")
    def test_process_batch_all_success(
        self, mock_create_chunk, mock_generate_embedding
    ):
        """Test processing a batch where all chunks succeed."""
        mock_session = MagicMock()
        document_id = uuid.uuid4()
        batch = [
            {"chunk_text": "Chunk 1"},
            {"chunk_text": "Chunk 2"},
        ]
        mock_provider = MagicMock()

        # Mock embedding generation to return valid embeddings
        mock_generate_embedding.side_effect = [
            [0.1, 0.2],
            [0.3, 0.4],
        ]

        result = _process_chunk_batch(mock_session, document_id, batch, mock_provider)

        assert result == 2
        assert mock_generate_embedding.call_count == 2
        assert mock_create_chunk.call_count == 2

    @patch("obsidian_rag.services.ingestion_chunks._generate_embedding_with_retry")
    @patch("obsidian_rag.services.ingestion_chunks._create_document_chunk")
    def test_process_batch_with_failures(
        self, mock_create_chunk, mock_generate_embedding
    ):
        """Test processing a batch with some embedding failures."""
        mock_session = MagicMock()
        document_id = uuid.uuid4()
        batch = [
            {"chunk_text": "Chunk 1"},
            {"chunk_text": "Chunk 2"},
        ]
        mock_provider = MagicMock()

        # First succeeds, second fails
        mock_generate_embedding.side_effect = [
            [0.1, 0.2],  # Success
            None,  # Failure
        ]

        result = _process_chunk_batch(mock_session, document_id, batch, mock_provider)

        assert result == 1  # Only one chunk created
        assert mock_create_chunk.call_count == 1

    def test_process_batch_no_provider(self):
        """Test processing a batch without embedding provider."""
        mock_session = MagicMock()
        document_id = uuid.uuid4()
        batch = [
            {"chunk_text": "Chunk 1"},
        ]

        result = _process_chunk_batch(mock_session, document_id, batch, None)

        assert result == 0  # No chunks created without provider

    def test_process_batch_empty(self):
        """Test processing an empty batch."""
        mock_session = MagicMock()
        document_id = uuid.uuid4()
        mock_provider = MagicMock()

        result = _process_chunk_batch(mock_session, document_id, [], mock_provider)

        assert result == 0
        mock_provider.generate_embedding.assert_not_called()


class TestCreateChunksWithEmbeddings:
    """Tests for create_chunks_with_embeddings function."""

    @patch("obsidian_rag.services.ingestion_chunks.chunk_document")
    @patch("obsidian_rag.services.ingestion_chunks._process_chunk_batch")
    def test_create_chunks_success(self, mock_process_batch, mock_chunk_document):
        """Test successful chunk creation."""
        mock_session = MagicMock()
        document_id = uuid.uuid4()
        content = "This is test content for chunking."
        mock_provider = MagicMock()

        # Mock chunk_document to return chunks
        mock_chunks = [
            {"chunk_text": "Chunk 1", "chunk_index": 0},
            {"chunk_text": "Chunk 2", "chunk_index": 1},
        ]
        mock_chunk_document.return_value = mock_chunks
        mock_process_batch.return_value = 2

        result = create_chunks_with_embeddings(
            mock_session,
            document_id,
            content,
            mock_provider,
            chunk_size=512,
            chunk_overlap=50,
            model_name="test-model",
        )

        assert result == 2
        mock_chunk_document.assert_called_once_with(
            content, str(document_id), 512, 50, "test-model"
        )
        mock_process_batch.assert_called_once()

    @patch("obsidian_rag.services.ingestion_chunks.chunk_document")
    def test_empty_content_returns_zero(self, mock_chunk_document):
        """Test that empty content returns 0 chunks."""
        mock_session = MagicMock()
        document_id = uuid.uuid4()
        mock_provider = MagicMock()

        result = create_chunks_with_embeddings(
            mock_session,
            document_id,
            "   ",  # Whitespace only
            mock_provider,
            chunk_size=512,
            chunk_overlap=50,
            model_name="test-model",
        )

        assert result == 0
        mock_chunk_document.assert_not_called()

    @patch("obsidian_rag.services.ingestion_chunks.chunk_document")
    def test_no_chunks_returned(self, mock_chunk_document):
        """Test when chunk_document returns empty list."""
        mock_session = MagicMock()
        document_id = uuid.uuid4()
        content = "Some content"
        mock_provider = MagicMock()

        mock_chunk_document.return_value = []

        result = create_chunks_with_embeddings(
            mock_session,
            document_id,
            content,
            mock_provider,
            chunk_size=512,
            chunk_overlap=50,
            model_name="test-model",
        )

        assert result == 0

    @patch("obsidian_rag.services.ingestion_chunks.chunk_document")
    @patch("obsidian_rag.services.ingestion_chunks._process_chunk_batch")
    def test_batch_processing_multiple_batches(
        self, mock_process_batch, mock_chunk_document
    ):
        """Test processing multiple batches when chunks exceed BATCH_SIZE."""
        mock_session = MagicMock()
        document_id = uuid.uuid4()
        content = "Test content"
        mock_provider = MagicMock()

        # Create more chunks than BATCH_SIZE to trigger multiple batches
        mock_chunks = [
            {"chunk_text": f"Chunk {i}", "chunk_index": i} for i in range(25)
        ]
        mock_chunk_document.return_value = mock_chunks
        mock_process_batch.return_value = 10  # Each batch creates 10 chunks

        result = create_chunks_with_embeddings(
            mock_session,
            document_id,
            content,
            mock_provider,
            chunk_size=512,
            chunk_overlap=50,
            model_name="test-model",
        )

        # Should process 3 batches: 10 + 10 + 5 chunks
        assert mock_process_batch.call_count == 3
        assert result == 30  # 3 batches * 10 chunks each

    @patch("obsidian_rag.services.ingestion_chunks.chunk_document")
    @patch("obsidian_rag.services.ingestion_chunks._process_chunk_batch")
    def test_batch_processing_exact_batch_size(
        self, mock_process_batch, mock_chunk_document
    ):
        """Test processing when chunks exactly match BATCH_SIZE."""
        mock_session = MagicMock()
        document_id = uuid.uuid4()
        content = "Test content"
        mock_provider = MagicMock()

        # Create exactly BATCH_SIZE chunks
        mock_chunks = [{"chunk_text": f"Chunk {i}"} for i in range(BATCH_SIZE)]
        mock_chunk_document.return_value = mock_chunks
        mock_process_batch.return_value = BATCH_SIZE

        result = create_chunks_with_embeddings(
            mock_session,
            document_id,
            content,
            mock_provider,
            chunk_size=512,
            chunk_overlap=50,
            model_name="test-model",
        )

        assert mock_process_batch.call_count == 1
        assert result == BATCH_SIZE


class TestConstants:
    """Tests for module constants."""

    def test_batch_size_value(self):
        """Test that BATCH_SIZE is set correctly."""
        assert BATCH_SIZE == 10

    def test_max_retries_value(self):
        """Test that MAX_RETRIES is set correctly."""
        assert MAX_RETRIES == 3
