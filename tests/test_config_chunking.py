"""Tests for chunking configuration."""

from obsidian_rag.config import ChunkingConfig, Settings


class TestChunkingConfig:
    """Test cases for ChunkingConfig."""

    def test_default_chunking_config(self):
        """Test default chunking configuration values."""
        config = ChunkingConfig()
        assert config.chunk_size == 512
        assert config.chunk_overlap == 50
        assert config.tokenizer_cache_dir == "~/.cache/obsidian-rag/tokenizers"
        assert config.tokenizer_model == "gpt2"
        assert config.flashrank_enabled is True
        assert config.flashrank_model == "ms-marco-MiniLM-L-12-v2"
        assert config.flashrank_top_k == 10

    def test_custom_chunking_config(self):
        """Test custom chunking configuration."""
        config = ChunkingConfig(
            chunk_size=256,
            chunk_overlap=25,
            tokenizer_cache_dir="/custom/cache",
            tokenizer_model="cl100k_base",
            flashrank_enabled=False,
            flashrank_model="rank-T5-flan",
            flashrank_top_k=5,
        )
        assert config.chunk_size == 256
        assert config.chunk_overlap == 25
        assert config.tokenizer_cache_dir == "/custom/cache"
        assert config.tokenizer_model == "cl100k_base"
        assert config.flashrank_enabled is False
        assert config.flashrank_model == "rank-T5-flan"
        assert config.flashrank_top_k == 5

    def test_chunk_size_validation(self):
        """Test chunk size validation."""
        # Too small
        config = ChunkingConfig(chunk_size=32)
        assert config.chunk_size == 64  # Should clamp to minimum

        # Too large
        config = ChunkingConfig(chunk_size=3000)
        assert config.chunk_size == 2048  # Should clamp to maximum

    def test_chunk_overlap_validation(self):
        """Test chunk overlap validation."""
        # Overlap >= chunk_size
        config = ChunkingConfig(chunk_size=512, chunk_overlap=600)
        assert config.chunk_overlap == 511  # Should be less than chunk_size


class TestSettingsChunking:
    """Test cases for Settings with chunking config."""

    def test_settings_has_chunking_config(self):
        """Test that Settings includes chunking configuration."""
        settings = Settings()
        assert hasattr(settings, "chunking")
        assert settings.chunking.chunk_size == 512

    def test_settings_chunking_from_dict(self):
        """Test loading chunking config from dict."""
        settings = Settings(
            chunking={
                "chunk_size": 256,
                "flashrank_enabled": False,
            }
        )
        assert settings.chunking.chunk_size == 256
        assert settings.chunking.flashrank_enabled is False
