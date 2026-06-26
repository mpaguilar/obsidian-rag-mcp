"""Tests for subquery SQL structure in tag extraction."""

from sqlalchemy import func, select as sa_select
from sqlalchemy.dialects import postgresql

from obsidian_rag.database.models import Document


def test_subquery_sql_without_pattern():
    """Verify subquery generates correct SQL without pattern filter."""
    tag_subq = (
        sa_select(
            func.unnest(Document.tags).label("tag"),
        )
        .select_from(Document)
        .subquery("tag_subq")
    )

    tags_query = sa_select(
        func.distinct(tag_subq.c.tag).label("tag"),
    ).where(tag_subq.c.tag.isnot(None))

    compiled = tags_query.compile(
        dialect=postgresql.dialect(),
        compile_kwargs={"render_postcompile": True},
    )
    sql_str = str(compiled)

    # Verify subquery structure
    assert "unnest" in sql_str.lower()
    assert "tag_subq" in sql_str.lower()
    assert "documents" in sql_str.lower()
    # The subquery should wrap unnest in a SELECT with AS tag
    assert "AS tag" in sql_str or "as tag" in sql_str.lower()


def test_subquery_sql_with_pattern():
    """Verify pattern filter works through subquery."""
    tag_subq = (
        sa_select(
            func.unnest(Document.tags).label("tag"),
        )
        .select_from(Document)
        .subquery("tag_subq")
    )

    tags_query = (
        sa_select(
            func.distinct(tag_subq.c.tag).label("tag"),
        )
        .where(tag_subq.c.tag.isnot(None))
        .where(
            func.lower(tag_subq.c.tag).ilike(func.lower("work%")),
        )
    )

    compiled = tags_query.compile(
        dialect=postgresql.dialect(),
        compile_kwargs={"render_postcompile": True},
    )
    sql_str = str(compiled)

    assert "unnest" in sql_str.lower()
    assert "tag_subq" in sql_str.lower()
    assert "ILIKE" in sql_str or "ilike" in sql_str.lower()


def test_subquery_column_accessible():
    """Verify that tag_subq.c.tag is a valid column reference."""
    tag_subq = (
        sa_select(
            func.unnest(Document.tags).label("tag"),
        )
        .select_from(Document)
        .subquery("tag_subq")
    )

    # tag_subq.c.tag should be a real column object, not None
    assert tag_subq.c.tag is not None


def test_subquery_not_table_valued():
    """Verify generated SQL does not contain buggy table_valued artifacts."""
    tag_subq = (
        sa_select(
            func.unnest(Document.tags).label("tag"),
        )
        .select_from(Document)
        .subquery("tag_subq")
    )

    tags_query = sa_select(
        func.distinct(tag_subq.c.tag).label("tag"),
    ).where(tag_subq.c.tag.isnot(None))

    compiled = tags_query.compile(
        dialect=postgresql.dialect(),
        compile_kwargs={"render_postcompile": True},
    )
    sql_str = str(compiled)

    # Should NOT contain the buggy table_valued pattern
    assert "table_valued" not in sql_str.lower()
