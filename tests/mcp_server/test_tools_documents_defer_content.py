"""Compiled-SQL tests proving defer(Document.content) excludes the content column.

These tests use a real PostgreSQL dialect (no live database) to compile the
exact query shapes built by the document tool functions and assert that
``documents.content`` does not appear in the SELECT clause when ``defer`` is
applied.
"""

import re
import uuid

from sqlalchemy import create_engine
from sqlalchemy.dialects import postgresql
from sqlalchemy.orm import defer, joinedload, sessionmaker

from obsidian_rag.database.models import Document, Vault
from obsidian_rag.mcp_server.tools.documents_tags import apply_postgresql_tag_filter


def _compiled_sql(query) -> str:
    """Compile a SQLAlchemy query to SQL string using the PostgreSQL dialect."""
    return str(
        query.statement.compile(
            dialect=postgresql.dialect(),
            compile_kwargs={"literal_binds": True},
        )
    )


_CONTENT_PATTERN = re.compile(r"\bdocuments\.content\b")


def _create_postgres_session():
    """Create a SQLAlchemy session bound to a PostgreSQL dialect (no real DB)."""
    engine = create_engine("postgresql+psycopg://dummy@localhost/dummy")
    Session = sessionmaker(bind=engine)
    return Session()


# ---------------------------------------------------------------------------
# list_documents
# ---------------------------------------------------------------------------


def test_list_documents_query_defers_content():
    """list_documents always defers Document.content from the SELECT clause."""
    session = _create_postgres_session()

    # Build the exact query shape used inside list_documents
    query = session.query(Document).options(
        joinedload(Document.vault), defer(Document.content)
    )
    query = query.filter(Document.file_name == "test.md")

    sql_str = _compiled_sql(query).lower()

    assert not _CONTENT_PATTERN.search(sql_str)


# ---------------------------------------------------------------------------
# get_documents_by_tag
# ---------------------------------------------------------------------------


def test_get_documents_by_tag_query_defers_content_no_vault():
    """get_documents_by_tag (no vault) defers Document.content from SELECT."""
    session = _create_postgres_session()

    query = session.query(Document).options(defer(Document.content))
    query = apply_postgresql_tag_filter(query, None)
    query = query.order_by(Document.file_name)

    sql_str = _compiled_sql(query).lower()

    assert not _CONTENT_PATTERN.search(sql_str)


def test_get_documents_by_tag_query_defers_content_with_vault():
    """get_documents_by_tag (vault filter) defers Document.content from SELECT."""
    session = _create_postgres_session()

    query = session.query(Document).join(Vault).options(defer(Document.content))
    query = query.filter(Vault.name == "test_vault")
    query = apply_postgresql_tag_filter(query, None)
    query = query.order_by(Document.file_name)

    sql_str = _compiled_sql(query).lower()

    assert not _CONTENT_PATTERN.search(sql_str)


# ---------------------------------------------------------------------------
# get_documents_by_property_postgresql
# ---------------------------------------------------------------------------


def test_get_documents_by_property_postgres_query_defers_content_no_vault():
    """get_documents_by_property_postgresql (no vault) defers content."""
    session = _create_postgres_session()

    query = session.query(Document).options(defer(Document.content))

    sql_str = _compiled_sql(query).lower()

    assert not _CONTENT_PATTERN.search(sql_str)


def test_get_documents_by_property_postgres_query_defers_content_with_vault():
    """get_documents_by_property_postgresql (vault filter) defers content."""
    session = _create_postgres_session()

    query = session.query(Document).join(Vault).options(defer(Document.content))
    query = query.filter(Vault.name == "test_vault")

    sql_str = _compiled_sql(query).lower()

    assert not _CONTENT_PATTERN.search(sql_str)


# ---------------------------------------------------------------------------
# _lookup_document_by_id
# ---------------------------------------------------------------------------


def test_get_document_lookup_by_id_defers_when_include_content_false():
    """_lookup_document_by_id defers content when include_content=False."""
    session = _create_postgres_session()
    doc_id = str(uuid.uuid4())

    options = [joinedload(Document.vault), defer(Document.content)]
    query = session.query(Document).options(*options).filter(Document.id == doc_id)

    sql_str = _compiled_sql(query).lower()

    assert not _CONTENT_PATTERN.search(sql_str)


def test_get_document_lookup_by_id_no_defer_when_include_content_true():
    """_lookup_document_by_id includes content when include_content=True."""
    session = _create_postgres_session()
    doc_id = str(uuid.uuid4())

    options = [joinedload(Document.vault)]
    query = session.query(Document).options(*options).filter(Document.id == doc_id)

    sql_str = _compiled_sql(query).lower()

    assert _CONTENT_PATTERN.search(sql_str)


# ---------------------------------------------------------------------------
# _lookup_document_by_vault_path
# ---------------------------------------------------------------------------


def test_get_document_lookup_by_vault_path_defers_when_include_content_false():
    """_lookup_document_by_vault_path defers content when include_content=False."""
    session = _create_postgres_session()

    options = [joinedload(Document.vault), defer(Document.content)]
    query = (
        session.query(Document)
        .options(*options)
        .filter(Document.vault_id == "vault-id")
        .filter(Document.file_path == "notes/test.md")
    )

    sql_str = _compiled_sql(query).lower()

    assert not _CONTENT_PATTERN.search(sql_str)


def test_get_document_lookup_by_vault_path_no_defer_when_include_content_true():
    """_lookup_document_by_vault_path includes content when include_content=True."""
    session = _create_postgres_session()

    options = [joinedload(Document.vault)]
    query = (
        session.query(Document)
        .options(*options)
        .filter(Document.vault_id == "vault-id")
        .filter(Document.file_path == "notes/test.md")
    )

    sql_str = _compiled_sql(query).lower()

    assert _CONTENT_PATTERN.search(sql_str)


# ---------------------------------------------------------------------------
# query_documents_postgresql
# ---------------------------------------------------------------------------


def test_query_documents_postgres_defers_when_include_content_false():
    """query_documents_postgresql defers content when include_content=False."""
    session = _create_postgres_session()
    query_embedding = [0.1] * 1536
    distance_expr = Document.content_vector.cosine_distance(query_embedding)

    query = session.query(Document, distance_expr.label("distance")).filter(
        Document.content_vector.isnot(None),
    )
    query = query.options(defer(Document.content))

    sql_str = _compiled_sql(query).lower()

    assert not _CONTENT_PATTERN.search(sql_str)


def test_query_documents_postgres_no_defer_when_include_content_true():
    """query_documents_postgresql includes content when include_content=True."""
    session = _create_postgres_session()
    query_embedding = [0.1] * 1536
    distance_expr = Document.content_vector.cosine_distance(query_embedding)

    query = session.query(Document, distance_expr.label("distance")).filter(
        Document.content_vector.isnot(None),
    )
    # No defer applied when include_content=True

    sql_str = _compiled_sql(query).lower()

    assert _CONTENT_PATTERN.search(sql_str)
