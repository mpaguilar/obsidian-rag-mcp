from obsidian_rag.services.ingestion_models import IngestionResult


def _make_result(**overrides: object) -> IngestionResult:
    """Build an IngestionResult with required fields pre-filled and overrides applied."""
    defaults: dict[str, object] = {
        "total": 0,
        "new": 0,
        "updated": 0,
        "unchanged": 0,
        "errors": 0,
        "deleted": 0,
        "chunks_created": 0,
        "empty_documents": 0,
        "processing_time_seconds": 0.0,
        "message": "",
    }
    defaults.update(overrides)
    return IngestionResult(**defaults)  # type: ignore[arg-type]


def test_ingestion_result_skipped_defaults_to_false() -> None:
    """A default-constructed IngestionResult has skipped=False (REQ-002 zero state)."""
    result = _make_result()
    assert result.skipped is False


def test_ingestion_result_skipped_true_round_trips_through_to_dict() -> None:
    """to_dict() propagates skipped=True (REQ-002 round-trip)."""
    result = _make_result(skipped=True)
    assert result.to_dict()["skipped"] is True


def test_ingestion_result_to_dict_includes_skipped_key_default_false() -> None:
    """to_dict() always includes the skipped key, defaulting to False (REQ-002)."""
    result = _make_result()
    assert "skipped" in result.to_dict()
    assert result.to_dict()["skipped"] is False


def test_ingestion_result_to_dict_preserves_all_existing_keys() -> None:
    """Regression: all pre-existing keys still present with correct values."""
    result = IngestionResult(
        total=1,
        new=2,
        updated=3,
        unchanged=4,
        errors=5,
        deleted=6,
        chunks_created=7,
        empty_documents=8,
        total_chunks=9,
        avg_chunk_tokens=10,
        task_chunk_count=11,
        content_chunk_count=12,
        processing_time_seconds=1.5,
        message="msg",
    )
    d = result.to_dict()
    assert d["total"] == 1
    assert d["new"] == 2
    assert d["updated"] == 3
    assert d["unchanged"] == 4
    assert d["errors"] == 5
    assert d["deleted"] == 6
    assert d["chunks_created"] == 7
    assert d["empty_documents"] == 8
    assert d["total_chunks"] == 9
    assert d["avg_chunk_tokens"] == 10
    assert d["task_chunk_count"] == 11
    assert d["content_chunk_count"] == 12
    assert d["processing_time_seconds"] == 1.5
    assert d["message"] == "msg"
    assert "skipped" in d


def test_ingestion_result_existing_call_sites_unchanged() -> None:
    """Constructing with only originally-required positional args succeeds and skipped is False."""
    result = IngestionResult(
        total=0,
        new=0,
        updated=0,
        unchanged=0,
        errors=0,
        deleted=0,
        chunks_created=0,
        empty_documents=0,
        processing_time_seconds=0.0,
        message="",
    )
    assert result.skipped is False
