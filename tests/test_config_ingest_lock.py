"""Tests for IngestionConfig ingest lock fields."""

from obsidian_rag.config import IngestionConfig


def test_ingest_lock_heartbeat_interval_default():
    """IngestionConfig() has field == 50."""
    config = IngestionConfig()
    assert config.ingest_lock_heartbeat_interval == 50


def test_ingest_lock_heartbeat_interval_invalid_returns_default():
    """IngestionConfig(ingest_lock_heartbeat_interval=0) -> 50; same for negative."""
    config_zero = IngestionConfig(ingest_lock_heartbeat_interval=0)
    assert config_zero.ingest_lock_heartbeat_interval == 50

    config_negative = IngestionConfig(ingest_lock_heartbeat_interval=-1)
    assert config_negative.ingest_lock_heartbeat_interval == 50


def test_ingest_lock_heartbeat_interval_valid():
    """IngestionConfig(ingest_lock_heartbeat_interval=10) -> 10."""
    config = IngestionConfig(ingest_lock_heartbeat_interval=10)
    assert config.ingest_lock_heartbeat_interval == 10


def test_ingest_lock_ttl_seconds_default():
    """default 300."""
    config = IngestionConfig()
    assert config.ingest_lock_ttl_seconds == 300


def test_ingest_lock_ttl_seconds_below_minimum_returns_default():
    """ingest_lock_ttl_seconds=59 -> 300; =0 -> 300."""
    config_59 = IngestionConfig(ingest_lock_ttl_seconds=59)
    assert config_59.ingest_lock_ttl_seconds == 300

    config_0 = IngestionConfig(ingest_lock_ttl_seconds=0)
    assert config_0.ingest_lock_ttl_seconds == 300


def test_ingest_lock_ttl_seconds_at_minimum():
    """=60 -> 60 (boundary inclusive)."""
    config = IngestionConfig(ingest_lock_ttl_seconds=60)
    assert config.ingest_lock_ttl_seconds == 60


def test_ingest_lock_ttl_seconds_large():
    """=3600 -> 3600."""
    config = IngestionConfig(ingest_lock_ttl_seconds=3600)
    assert config.ingest_lock_ttl_seconds == 3600
