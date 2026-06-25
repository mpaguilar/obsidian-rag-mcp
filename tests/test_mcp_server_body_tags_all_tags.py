"""Tests verifying get_all_tags returns inline body tags from Document.tags."""

from unittest.mock import MagicMock

from obsidian_rag.mcp_server.tools.documents import get_all_tags


def _create_mock_tag_rows(tags: list[str | None]) -> list[MagicMock]:
    """Create mock row objects with .tag attribute."""
    rows = []
    for tag in tags:
        row = MagicMock()
        row.tag = tag
        rows.append(row)
    return rows


def _configure_mock_for_tags(db_session: MagicMock, tags: list[str | None]) -> None:
    """Configure mock session to return specific tags from query chain."""
    rows = _create_mock_tag_rows(tags)
    query_mock = db_session.query.return_value
    query_mock.filter.return_value = query_mock
    query_mock.order_by.return_value = query_mock
    query_mock.all.return_value = rows


def test_get_all_tags_includes_body_inline_tags(db_session: MagicMock) -> None:
    """Verify get_all_tags returns body-only inline tags present in Document.tags.

    Body tags are tags found in the markdown body (e.g., #meeting, #project)
    that get stored in Document.tags alongside frontmatter tags. The
    get_all_tags function queries Document.tags via UNNEST and should
    return all tags including body-only ones.
    """
    expected_count = 2
    _configure_mock_for_tags(db_session, ["meeting", "project"])

    result = get_all_tags(db_session, pattern=None, limit=20, offset=0)

    assert result.total_count == expected_count
    assert "meeting" in result.tags
    assert "project" in result.tags


def test_get_all_tags_includes_mixed_frontmatter_body_tags(
    db_session: MagicMock,
) -> None:
    """Verify get_all_tags returns both frontmatter and body tags.

    When Document.tags contains a mix of frontmatter-derived tags and
    body inline tags, get_all_tags should return all of them.
    """
    expected_count = 4
    _configure_mock_for_tags(
        db_session, ["frontmatter-tag", "meeting", "project", "work"]
    )

    result = get_all_tags(db_session, pattern=None, limit=20, offset=0)

    assert result.total_count == expected_count
    assert "frontmatter-tag" in result.tags
    assert "meeting" in result.tags
    assert "project" in result.tags
    assert "work" in result.tags


def test_get_all_tags_pattern_filter_works_with_body_tags(
    db_session: MagicMock,
) -> None:
    """Verify pattern filtering works correctly on body tags.

    When a glob pattern is applied, body tags should be filtered
    just like frontmatter tags.
    """
    expected_count = 2
    _configure_mock_for_tags(db_session, ["project-alpha", "project-beta"])

    result = get_all_tags(db_session, pattern="project*", limit=20, offset=0)

    assert result.total_count == expected_count
    assert "project-alpha" in result.tags
    assert "project-beta" in result.tags


def test_get_all_tags_pagination_with_body_tags(db_session: MagicMock) -> None:
    """Verify pagination works correctly when body tags are present.

    Tests the has_more branch when there are more body tags than the
    requested limit.
    """
    total_tags = 4
    page_size = 2
    next_offset_value = 2
    _configure_mock_for_tags(
        db_session, ["meeting", "project-alpha", "project-beta", "retro"]
    )

    result = get_all_tags(db_session, pattern=None, limit=page_size, offset=0)

    assert result.total_count == total_tags
    assert len(result.tags) == page_size
    assert result.has_more is True
    assert result.next_offset == next_offset_value


def test_get_all_tags_filters_none_tags(db_session: MagicMock) -> None:
    """Verify None tags from the query are filtered out.

    The _extract_tags_postgresql function filters out rows where
    tag is None. This test verifies that behavior with body tags.
    """
    expected_count = 2
    _configure_mock_for_tags(
        db_session, ["meeting", None, "project"]
    )

    result = get_all_tags(db_session, pattern=None, limit=20, offset=0)

    assert result.total_count == expected_count
    assert None not in result.tags
    assert "meeting" in result.tags
    assert "project" in result.tags
