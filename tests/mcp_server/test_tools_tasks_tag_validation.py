"""Tests for tag filter validation."""

import pytest

from obsidian_rag.mcp_server.tools.tasks import _validate_tag_filters


def test_no_tags_passes_validation():
    """Test that None tags pass validation."""
    _validate_tag_filters(None, None)  # Should not raise
    _validate_tag_filters(["work"], None)  # Should not raise
    _validate_tag_filters(None, ["blocked"])  # Should not raise


def test_empty_tags_passes_validation():
    """Test that empty lists pass validation."""
    _validate_tag_filters([], [])  # Should not raise
    _validate_tag_filters(["work"], [])  # Should not raise
    _validate_tag_filters([], ["blocked"])  # Should not raise


def test_no_conflicts_passes_validation():
    """Test that non-overlapping tags pass validation."""
    _validate_tag_filters(["work"], ["blocked"])  # Should not raise
    _validate_tag_filters(
        ["work", "urgent"], ["blocked", "waiting"]
    )  # Should not raise


def test_conflicting_tags_raises_error():
    """Test that conflicting tags raise ValueError."""
    with pytest.raises(ValueError, match="Conflicting tags found"):
        _validate_tag_filters(["work"], ["work"])


def test_case_insensitive_conflict_detection():
    """Test that conflicts are detected case-insensitively."""
    with pytest.raises(ValueError, match="Conflicting tags found"):
        _validate_tag_filters(["Work"], ["work"])

    with pytest.raises(ValueError, match="Conflicting tags found"):
        _validate_tag_filters(["WORK", "URGENT"], ["work", "blocked"])


def test_multiple_conflicts_in_error_message():
    """Test that all conflicts are listed in error message."""
    with pytest.raises(ValueError) as exc_info:
        _validate_tag_filters(
            ["work", "urgent", "personal"], ["work", "urgent", "blocked"]
        )

    error_msg = str(exc_info.value)
    assert "work" in error_msg
    assert "urgent" in error_msg
    assert "personal" not in error_msg  # Not a conflict
    assert "blocked" not in error_msg  # Not a conflict


def test_partial_overlap_raises_error():
    """Test that partial overlap still raises error."""
    with pytest.raises(ValueError, match="Conflicting tags found"):
        _validate_tag_filters(["work", "urgent"], ["work", "blocked"])
