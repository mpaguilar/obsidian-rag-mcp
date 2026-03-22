"""Tests for DocumentChunk model with token-based chunking fields."""

from datetime import UTC, datetime

from obsidian_rag.database.models import (
    Document,
    DocumentChunk,
    Vault,
)


class TestDocumentChunkTokenFields:
    """Test cases for DocumentChunk token_count and chunk_type fields."""

    def test_chunk_has_token_count_field(self, db_session):
        """Test that DocumentChunk has token_count field."""
        vault = Vault(name="Test Vault", container_path="/test", host_path="/test")
        db_session.add(vault)
        db_session.flush()

        doc = Document(
            vault_id=vault.id,
            file_path="test.md",
            file_name="test.md",
            content="Test content",
            checksum_md5="abc123",
            created_at_fs=datetime.now(UTC),
            modified_at_fs=datetime.now(UTC),
        )
        db_session.add(doc)
        db_session.flush()

        chunk = DocumentChunk(
            document_id=doc.id,
            chunk_index=0,
            chunk_text="Test chunk",
            chunk_vector=[0.1] * 1536,
            start_char=0,
            end_char=10,
            token_count=512,
            chunk_type="content",
        )
        db_session.add(chunk)
        db_session.commit()

        expected_token_count = 512
        assert chunk.token_count == expected_token_count
        assert chunk.chunk_type == "content"

    def test_chunk_type_task_value(self, db_session):
        """Test that chunk_type accepts 'task' value."""
        vault = Vault(name="Test Vault 2", container_path="/test2", host_path="/test2")
        db_session.add(vault)
        db_session.flush()

        doc = Document(
            vault_id=vault.id,
            file_path="test2.md",
            file_name="test2.md",
            content="Test content",
            checksum_md5="def456",
            created_at_fs=datetime.now(UTC),
            modified_at_fs=datetime.now(UTC),
        )
        db_session.add(doc)
        db_session.flush()

        # Test 'task' type
        task_chunk = DocumentChunk(
            document_id=doc.id,
            chunk_index=0,
            chunk_text="Task chunk",
            chunk_vector=[0.2] * 1536,
            start_char=0,
            end_char=10,
            token_count=128,
            chunk_type="task",
        )
        db_session.add(task_chunk)
        db_session.commit()

        assert task_chunk.chunk_type == "task"

    def test_token_count_nullable(self, db_session):
        """Test that token_count can be null for backward compatibility."""
        vault = Vault(name="Test Vault 3", container_path="/test3", host_path="/test3")
        db_session.add(vault)
        db_session.flush()

        doc = Document(
            vault_id=vault.id,
            file_path="test3.md",
            file_name="test3.md",
            content="Test content",
            checksum_md5="ghi789",
            created_at_fs=datetime.now(UTC),
            modified_at_fs=datetime.now(UTC),
        )
        db_session.add(doc)
        db_session.flush()

        chunk = DocumentChunk(
            document_id=doc.id,
            chunk_index=0,
            chunk_text="Test chunk",
            chunk_vector=[0.3] * 1536,
            start_char=0,
            end_char=10,
            token_count=None,
            chunk_type=None,
        )
        db_session.add(chunk)
        db_session.commit()

        assert chunk.token_count is None
        assert chunk.chunk_type is None


class TestDocumentChunkHNSWIndex:
    """Test cases for DocumentChunk HNSW index parameters."""

    def test_hnsw_index_parameters_updated(self):
        """Test that HNSW index uses updated parameters (M=32, ef_construction=128)."""
        # Access the __table_args__ from the DocumentChunk class
        table_args = DocumentChunk.__table_args__

        # Find the Index entry
        hnsw_index = None
        for arg in table_args:
            arg_name = getattr(arg, "name", None)
            if arg_name and "hnsw" in arg_name:
                hnsw_index = arg
                break

        assert hnsw_index is not None, "HNSW index not found in __table_args__"

        # Check the postgresql_with parameters
        index_kwargs = getattr(hnsw_index, "kwargs", {})
        assert index_kwargs.get("postgresql_with") == {
            "M": 32,
            "ef_construction": 128,
        }, "HNSW index parameters not updated correctly"

    def test_hnsw_index_uses_cosine_ops(self):
        """Test that HNSW index uses vector_cosine_ops."""
        table_args = DocumentChunk.__table_args__

        hnsw_index = None
        for arg in table_args:
            arg_name = getattr(arg, "name", None)
            if arg_name and "hnsw" in arg_name:
                hnsw_index = arg
                break

        assert hnsw_index is not None
        index_kwargs = getattr(hnsw_index, "kwargs", {})
        assert index_kwargs.get("postgresql_ops") == {
            "chunk_vector": "vector_cosine_ops",
        }
