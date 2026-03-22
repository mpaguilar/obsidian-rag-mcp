"""Tests for IngestionResult chunk statistics fields."""

from obsidian_rag.services.ingestion import IngestionResult


class TestIngestionResultChunkFields:
    """Test cases for IngestionResult chunk statistics fields."""

    def test_ingestion_result_has_chunk_fields(self):
        """IngestionResult should have chunk statistics fields."""
        result = IngestionResult(
            total=10,
            new=5,
            updated=3,
            unchanged=2,
            errors=0,
            deleted=0,
            chunks_created=3,
            empty_documents=0,
            processing_time_seconds=1.5,
            message="Test message",
            total_chunks=15,
            avg_chunk_tokens=250,
            task_chunk_count=5,
            content_chunk_count=10,
        )

        assert result.total_chunks == 15
        assert result.avg_chunk_tokens == 250
        assert result.task_chunk_count == 5
        assert result.content_chunk_count == 10

    def test_ingestion_result_chunk_fields_defaults(self):
        """Chunk fields should have default values of 0."""
        result = IngestionResult(
            total=10,
            new=5,
            updated=3,
            unchanged=2,
            errors=0,
            deleted=0,
            chunks_created=3,
            empty_documents=0,
            processing_time_seconds=1.5,
            message="Test message",
        )

        # Should have default values
        assert result.total_chunks == 0
        assert result.avg_chunk_tokens == 0
        assert result.task_chunk_count == 0
        assert result.content_chunk_count == 0

    def test_ingestion_result_to_dict_includes_chunk_fields(self):
        """to_dict() should include chunk statistics fields."""
        result = IngestionResult(
            total=10,
            new=5,
            updated=3,
            unchanged=2,
            errors=0,
            deleted=0,
            chunks_created=3,
            empty_documents=0,
            processing_time_seconds=1.5,
            message="Test message",
            total_chunks=15,
            avg_chunk_tokens=250,
            task_chunk_count=5,
            content_chunk_count=10,
        )

        result_dict = result.to_dict()

        assert result_dict["total_chunks"] == 15
        assert result_dict["avg_chunk_tokens"] == 250
        assert result_dict["task_chunk_count"] == 5
        assert result_dict["content_chunk_count"] == 10

    def test_ingestion_result_to_dict_chunk_field_defaults(self):
        """to_dict() should include default chunk field values."""
        result = IngestionResult(
            total=10,
            new=5,
            updated=3,
            unchanged=2,
            errors=0,
            deleted=0,
            chunks_created=3,
            empty_documents=0,
            processing_time_seconds=1.5,
            message="Test message",
        )

        result_dict = result.to_dict()

        # Should have default values in dict
        assert result_dict["total_chunks"] == 0
        assert result_dict["avg_chunk_tokens"] == 0
        assert result_dict["task_chunk_count"] == 0
        assert result_dict["content_chunk_count"] == 0

    def test_ingestion_result_all_fields_in_dict(self):
        """All IngestionResult fields should be present in to_dict()."""
        result = IngestionResult(
            total=10,
            new=5,
            updated=3,
            unchanged=2,
            errors=0,
            deleted=0,
            chunks_created=3,
            empty_documents=0,
            processing_time_seconds=1.5,
            message="Test message",
            total_chunks=15,
            avg_chunk_tokens=250,
            task_chunk_count=5,
            content_chunk_count=10,
        )

        result_dict = result.to_dict()

        # Verify all expected keys are present
        expected_keys = {
            "total",
            "new",
            "updated",
            "unchanged",
            "errors",
            "deleted",
            "chunks_created",
            "empty_documents",
            "total_chunks",
            "avg_chunk_tokens",
            "task_chunk_count",
            "content_chunk_count",
            "processing_time_seconds",
            "message",
        }

        assert set(result_dict.keys()) == expected_keys
