"""End-to-end regression tests for garbage tags from triple-backtick prose mentions.

These tests prove that a document whose body contains triple-backtick prose
mentions produces NO garbage tags in ``get_all_tags`` / ``get_documents_by_tag``
after ingestion. They must FAIL on the current buggy code and PASS after the fix.

The real ``extract_body_tags`` and ``_merge_tags`` implementations are used.
Database interactions are mocked per CONVENTIONS.md.
"""

import uuid
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from obsidian_rag.config import VaultConfig
from obsidian_rag.database.models import Document, Task
from obsidian_rag.mcp_server.models import TagFilter
from obsidian_rag.mcp_server.tools.documents import get_all_tags, get_documents_by_tag
from obsidian_rag.parsing.body_tags import extract_body_tags
from obsidian_rag.services.ingestion import IngestionService
from obsidian_rag.services.tag_merging import _merge_tags


_AFFECTED_DOC_BODY = (
    "## Body Tag Extraction\n\n"
    "`_strip_code_blocks()` removes fenced code blocks (``` ... ```) "
    "and inline code (`#tag`).\n\n"
    "All-numeric `#1984` is NOT a tag, `#y1984` IS. "
    'Inline tag "#personal/expenses" is valid.\n'
)

_GARBAGE_TAGS = {
    "1984get_documents_by_tagget_all_tags_merge_tags",
    "tagparse_frontmatter",
    "tag_strip_code_blocks",
    "obsidian_rag/services/ingestion.pyparse_frontmatter",
    "tag",
    "strip_hash",
}


def _make_settings() -> MagicMock:
    """Build mock settings for an IngestionService.

    Returns:
        A MagicMock configured with ingestion and chunking attributes.

    """
    settings = MagicMock()
    settings.ingestion.batch_size = 100
    settings.ingestion.progress_interval = 10
    settings.chunking.chunk_size = 512
    settings.chunking.chunk_overlap = 50
    settings.chunking.model_name = "gpt2"
    settings.chunking.cache_dir = None
    return settings


@pytest.fixture
def ingestion_service() -> IngestionService:
    """Create an IngestionService with mocked dependencies.

    Returns:
        An IngestionService wired to mock database and embedding providers.

    """
    provider = MagicMock()
    provider.generate_embedding.return_value = [0.1, 0.2, 0.3]
    return IngestionService(
        db_manager=MagicMock(),
        embedding_provider=provider,
        settings=_make_settings(),
    )


def _make_session(service: IngestionService) -> MagicMock:
    """Attach a mock session to the service's db_manager and return it.

    Args:
        service: The IngestionService whose db_manager will be wired.

    Returns:
        The mock session that ``get_session()`` will yield.

    """
    session = MagicMock()
    context = MagicMock()
    context.__enter__ = MagicMock(return_value=session)
    context.__exit__ = MagicMock(return_value=None)
    service.db_manager.get_session.return_value = context  # type: ignore[attr-defined]
    return session


def _make_file_info(tmp_path: Path, content: str) -> MagicMock:
    """Build a mock FileInfo object.

    Args:
        tmp_path: Temporary directory used as the vault root.
        content: Raw file content (frontmatter + body).

    Returns:
        A MagicMock simulating a scanned markdown file.

    """
    file_info = MagicMock()
    file_info.path = tmp_path / "note.md"
    file_info.name = "note.md"
    file_info.content = content
    file_info.checksum = "new-checksum"
    file_info.created_at = datetime.now(timezone.utc)
    file_info.modified_at = datetime.now(timezone.utc)
    return file_info


def _added_documents(session: MagicMock) -> list[Document]:
    """Return Document instances passed to session.add.

    Args:
        session: The mock session to inspect.

    Returns:
        List of Document objects added during ingestion.

    """
    return [
        call.args[0]
        for call in session.add.call_args_list
        if isinstance(call.args[0], Document)
    ]


def _added_tasks(session: MagicMock) -> list[Task]:
    """Return Task instances passed to session.add.

    Args:
        session: The mock session to inspect.

    Returns:
        List of Task objects added during ingestion.

    """
    return [
        call.args[0]
        for call in session.add.call_args_list
        if isinstance(call.args[0], Task)
    ]


def _run_new_ingest(
    service: IngestionService,
    tmp_path: Path,
    *,
    frontmatter_tags: list[str] | None,
    body: str,
) -> MagicMock:
    """Run _ingest_single_file for a NEW document and return the session.

    Args:
        service: The IngestionService to exercise.
        tmp_path: Temporary directory used as the vault root.
        frontmatter_tags: Tags returned by the mocked parse_frontmatter.
        body: Body content returned by the mocked parse_frontmatter.

    Returns:
        The mock session so callers can inspect added Document/Task objects.

    """
    session = _make_session(service)
    session.query.return_value.filter_by.return_value.first.return_value = None

    vault_config = VaultConfig(
        container_path=str(tmp_path),
        host_path=str(tmp_path),
    )
    file_info = _make_file_info(tmp_path, body)

    with patch("obsidian_rag.services.ingestion.parse_frontmatter") as mock_fm:
        mock_fm.return_value = (frontmatter_tags, {}, body)
        with patch(
            "obsidian_rag.services.ingestion.parse_tasks_from_content",
        ) as mock_pt:
            mock_pt.return_value = []
            with patch(
                "obsidian_rag.services.ingestion.should_chunk_document",
            ) as mock_sc:
                mock_sc.return_value = False
                service._ingest_single_file(
                    file_info,
                    vault_id=uuid.uuid4(),
                    vault_config=vault_config,
                )

    return session


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
    query_mock.options.return_value = query_mock
    query_mock.filter.return_value = query_mock
    query_mock.order_by.return_value = query_mock
    query_mock.all.return_value = rows


def test_extract_body_tags_on_affected_doc_no_garbage() -> None:
    """Direct call on _AFFECTED_DOC_BODY: forbidden tags are absent, valid tag present.

    This test calls ``extract_body_tags`` directly with the prose document that
    mentions triple backticks, inline code, and function names. It asserts that
    none of the known garbage tags appear in the result and that the legitimate
    tag ``personal/expenses`` is correctly extracted.
    """
    result = extract_body_tags(_AFFECTED_DOC_BODY)

    assert result is not None
    result_set = set(result)

    intersection = result_set & _GARBAGE_TAGS
    assert intersection == set(), f"Garbage tags found: {intersection}"

    assert "personal/expenses" in result_set


def test_ingest_then_get_all_tags_no_garbage(
    ingestion_service: IngestionService,
    tmp_path: Path,
) -> None:
    """Ingest affected doc, then get_all_tags must contain personal/expenses and no garbage.

    This test runs the ingestion pipeline with the affected document body,
    extracts the stored Document.tags, and then queries ``get_all_tags`` with
    a mock session pre-loaded with those exact tags. It asserts that the
    returned tag list contains ``personal/expenses`` and none of the garbage
    strings.
    """
    session = _run_new_ingest(
        ingestion_service,
        tmp_path,
        frontmatter_tags=None,
        body=_AFFECTED_DOC_BODY,
    )

    docs = _added_documents(session)
    assert docs
    doc = docs[0]
    doc.id = uuid.uuid4()
    stored_tags = doc.tags

    assert stored_tags is not None
    assert "personal/expenses" in stored_tags

    # Ensure garbage tags were not stored
    stored_set = set(stored_tags)
    intersection = stored_set & _GARBAGE_TAGS
    assert intersection == set(), f"Garbage tags stored: {intersection}"

    # Now query via get_all_tags using the stored tags
    mock_session = MagicMock()
    mock_session.bind.dialect.name = "postgresql"
    _configure_mock_for_tags(mock_session, stored_tags)

    result = get_all_tags(mock_session, pattern=None, limit=20, offset=0)

    returned_set = set(result.tags)
    assert "personal/expenses" in returned_set
    garbage_in_result = returned_set & _GARBAGE_TAGS
    assert garbage_in_result == set(), (
        f"Garbage tags in get_all_tags: {garbage_in_result}"
    )


def test_ingest_then_get_documents_by_tag_no_garbage_match(
    ingestion_service: IngestionService,
    tmp_path: Path,
) -> None:
    """get_documents_by_tag with a garbage include_tag must return ZERO matches.

    This test ingests the affected document, then calls ``get_documents_by_tag``
    with ``include_tags=["tag"]`` (a known garbage string produced by the bug).
    Because the garbage tag must NOT be indexed, the query should return zero
    matching documents. The test currently fails because the buggy code does
    store ``"tag"`` in Document.tags.
    """
    session = _run_new_ingest(
        ingestion_service,
        tmp_path,
        frontmatter_tags=None,
        body=_AFFECTED_DOC_BODY,
    )

    docs = _added_documents(session)
    assert docs
    doc = docs[0]
    doc.id = uuid.uuid4()
    stored_tags = doc.tags

    assert stored_tags is not None
    stored_set = set(stored_tags)
    assert "tag" not in stored_set, "Garbage tag 'tag' was stored in Document.tags"

    # get_documents_by_tag with the garbage tag should return 0 results
    mock_session = MagicMock()
    mock_session.bind.dialect.name = "postgresql"
    mock_query = mock_session.query.return_value
    mock_query.options.return_value = mock_query
    mock_query.filter.return_value = mock_query
    mock_query.order_by.return_value = mock_query
    mock_query.count.return_value = 0
    mock_query.offset.return_value.limit.return_value.all.return_value = []

    result = get_documents_by_tag(
        mock_session,
        TagFilter(include_tags=["tag"]),
        limit=20,
        offset=0,
    )

    assert result.total_count == 0
    assert len(result.results) == 0


def test_task_tags_inherit_no_garbage(
    ingestion_service: IngestionService,
    tmp_path: Path,
) -> None:
    """_merge_tags(document_tags, task_tags) produces no garbage in merged task tags.

    This test mirrors the ``_create_tasks`` flow: document-level tags are
    ``extract_body_tags(_AFFECTED_DOC_BODY)``, task tags are a legitimate
    task-level tag. The merged result must contain the legitimate tags and
    none of the garbage strings.
    """
    document_tags = extract_body_tags(_AFFECTED_DOC_BODY)
    task_tags = ["urgent"]

    assert document_tags is not None

    merged = _merge_tags(document_tags, task_tags)

    assert merged is not None
    merged_set = set(merged)

    assert "personal/expenses" in merged_set
    assert "urgent" in merged_set

    garbage_in_merged = merged_set & _GARBAGE_TAGS
    assert garbage_in_merged == set(), (
        f"Garbage tags in merged task tags: {garbage_in_merged}"
    )
