"""Integration tests for body tag extraction through the full pipeline.

These tests verify the end-to-end flow:
body content -> extract_body_tags -> merge with frontmatter -> stored in
Document.tags -> queryable via get_documents_by_tag and get_all_tags.

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
from obsidian_rag.mcp_server.tools.documents import get_documents_by_tag
from obsidian_rag.services.ingestion import IngestionService


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


def _make_task(tag: str | None = None) -> MagicMock:
    """Build a mock parsed task with the given inline tags.

    Args:
        tag: A single inline tag for the task, or None for no task tags.

    Returns:
        A MagicMock simulating a parsed task.

    """
    task = MagicMock()
    task.raw_text = "- [ ] do thing"
    task.status = "not_completed"
    task.description = "do thing"
    task.tags = [tag] if tag else None
    task.repeat = None
    task.scheduled = None
    task.due = None
    task.completion = None
    task.priority = "normal"
    task.custom_metadata = {}
    return task


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
    task: MagicMock | None = None,
) -> MagicMock:
    """Run _ingest_single_file for a NEW document and return the session.

    Args:
        service: The IngestionService to exercise.
        tmp_path: Temporary directory used as the vault root.
        frontmatter_tags: Tags returned by the mocked parse_frontmatter.
        body: Body content returned by the mocked parse_frontmatter.
        task: Optional parsed task to return from parse_tasks_from_content.

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
    parsed_tasks = [(1, task)] if task is not None else []

    with patch("obsidian_rag.services.ingestion.parse_frontmatter") as mock_fm:
        mock_fm.return_value = (frontmatter_tags, {}, body)
        with patch(
            "obsidian_rag.services.ingestion.parse_tasks_from_content",
        ) as mock_pt:
            mock_pt.return_value = parsed_tasks
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


def _mock_session_for_tag_query(documents: list[Document]) -> MagicMock:
    """Create a mock session that returns the given documents from tag queries.

    Args:
        documents: List of Document objects to return from the query.

    Returns:
        A mock SQLAlchemy session configured for get_documents_by_tag.

    """
    mock_session = MagicMock()
    mock_query = MagicMock()
    mock_query.filter.return_value = mock_query
    mock_query.order_by.return_value = mock_query
    mock_query.count.return_value = len(documents)
    mock_query.offset.return_value.limit.return_value.all.return_value = documents
    mock_session.query.return_value = mock_query
    return mock_session


def test_document_with_body_tags_found_by_get_documents_by_tag(
    ingestion_service: IngestionService,
    tmp_path: Path,
) -> None:
    """Body tag #meeting is stored in Document.tags and found by get_documents_by_tag."""
    session = _run_new_ingest(
        ingestion_service,
        tmp_path,
        frontmatter_tags=None,
        body="Discussed project roadmap in the #meeting today",
    )

    docs = _added_documents(session)
    assert docs
    doc = docs[0]
    doc.id = uuid.uuid4()
    assert doc.tags is not None
    assert "meeting" in doc.tags

    mock_session = _mock_session_for_tag_query([doc])
    result = get_documents_by_tag(
        mock_session,
        TagFilter(include_tags=["meeting"]),
        limit=20,
        offset=0,
    )

    assert result.total_count == 1
    assert len(result.results) == 1
    assert "meeting" in result.results[0].tags


def test_document_with_no_frontmatter_body_tags_stored(
    ingestion_service: IngestionService,
    tmp_path: Path,
) -> None:
    """Document with only inline body tags stores those tags in Document.tags."""
    session = _run_new_ingest(
        ingestion_service,
        tmp_path,
        frontmatter_tags=None,
        body="Body with #tag1 and #tag2 only",
    )

    docs = _added_documents(session)
    assert docs
    assert docs[0].tags == ["tag1", "tag2"]


def test_document_with_both_frontmatter_and_body_tags_deduped(
    ingestion_service: IngestionService,
    tmp_path: Path,
) -> None:
    """Frontmatter tags and body tags are merged and deduplicated in Document.tags."""
    session = _run_new_ingest(
        ingestion_service,
        tmp_path,
        frontmatter_tags=["work"],
        body="Body text with #work and #project tags",
    )

    docs = _added_documents(session)
    assert docs
    tags = docs[0].tags
    assert tags == ["work", "project"]
    assert tags.count("work") == 1


def test_heading_not_extracted_as_tag(
    ingestion_service: IngestionService,
    tmp_path: Path,
) -> None:
    """A markdown heading is not treated as a body tag in Document.tags."""
    session = _run_new_ingest(
        ingestion_service,
        tmp_path,
        frontmatter_tags=None,
        body="# Heading\n\nSome text under the heading",
    )

    docs = _added_documents(session)
    assert docs
    assert docs[0].tags is None


def test_code_block_tags_not_extracted(
    ingestion_service: IngestionService,
    tmp_path: Path,
) -> None:
    """Tags inside fenced code blocks are not extracted into Document.tags."""
    session = _run_new_ingest(
        ingestion_service,
        tmp_path,
        frontmatter_tags=None,
        body="```\n#code-tag\n```\n\nAlso discussed #real-tag",
    )

    docs = _added_documents(session)
    assert docs
    tags = docs[0].tags
    assert tags == ["real-tag"]
    assert "code-tag" not in tags


def test_inline_code_tags_not_extracted(
    ingestion_service: IngestionService,
    tmp_path: Path,
) -> None:
    """Tags inside inline code are not extracted into Document.tags."""
    session = _run_new_ingest(
        ingestion_service,
        tmp_path,
        frontmatter_tags=None,
        body="Use `#inline-tag` for styling, but #real-tag is better",
    )

    docs = _added_documents(session)
    assert docs
    tags = docs[0].tags
    assert tags == ["real-tag"]
    assert "inline-tag" not in tags


def test_all_numeric_not_extracted(
    ingestion_service: IngestionService,
    tmp_path: Path,
) -> None:
    """An all-numeric token like #1984 is not treated as a tag in Document.tags."""
    session = _run_new_ingest(
        ingestion_service,
        tmp_path,
        frontmatter_tags=None,
        body="Book #1984 is dystopian, but #scifi is fun",
    )

    docs = _added_documents(session)
    assert docs
    tags = docs[0].tags
    assert tags == ["scifi"]
    assert "1984" not in tags


def test_task_tags_inherit_body_tags(
    ingestion_service: IngestionService,
    tmp_path: Path,
) -> None:
    """Tasks inherit document-level body tags via _merge_tags in _create_tasks."""
    session = _run_new_ingest(
        ingestion_service,
        tmp_path,
        frontmatter_tags=None,
        body="Project tasks #team\n- [ ] review code",
        task=_make_task("urgent"),
    )

    tasks = _added_tasks(session)
    assert tasks
    assert tasks[0].tags == ["team", "urgent"]


def test_task_line_body_tag_stored_in_document(
    ingestion_service: IngestionService,
    tmp_path: Path,
) -> None:
    """Tag inside a task line is stored in Document.tags."""
    session = _run_new_ingest(
        ingestion_service,
        tmp_path,
        frontmatter_tags=None,
        body="- [ ] do something #important",
    )

    docs = _added_documents(session)
    assert docs
    assert docs[0].tags == ["important"]


def test_duplicate_body_tags_deduped_in_document(
    ingestion_service: IngestionService,
    tmp_path: Path,
) -> None:
    """Duplicate body tags are deduplicated in Document.tags."""
    session = _run_new_ingest(
        ingestion_service,
        tmp_path,
        frontmatter_tags=None,
        body="Discussed #tag and #tag again",
    )

    docs = _added_documents(session)
    assert docs
    assert docs[0].tags == ["tag"]


def test_empty_body_no_tags_extracted(
    ingestion_service: IngestionService,
    tmp_path: Path,
) -> None:
    """Empty body content yields None tags in Document.tags."""
    session = _run_new_ingest(
        ingestion_service,
        tmp_path,
        frontmatter_tags=None,
        body="",
    )

    docs = _added_documents(session)
    assert docs
    assert docs[0].tags is None


def test_update_existing_document_merges_body_tags(
    ingestion_service: IngestionService,
    tmp_path: Path,
) -> None:
    """Update path also merges body tags into the existing document."""
    session = _make_session(ingestion_service)

    existing_doc = MagicMock()
    existing_doc.id = uuid.uuid4()
    existing_doc.checksum_md5 = "old-checksum"
    existing_doc.tags = ["old"]
    session.query.return_value.filter_by.return_value.first.return_value = existing_doc

    vault_config = VaultConfig(
        container_path=str(tmp_path),
        host_path=str(tmp_path),
    )
    body = "Updated content with #inline tag"
    file_info = _make_file_info(tmp_path, body)

    with patch("obsidian_rag.services.ingestion.parse_frontmatter") as mock_fm:
        mock_fm.return_value = (["updated"], {}, body)
        with patch(
            "obsidian_rag.services.ingestion.parse_tasks_from_content",
        ) as mock_pt:
            mock_pt.return_value = [(1, _make_task("inline"))]
            with patch(
                "obsidian_rag.services.ingestion.should_chunk_document",
            ) as mock_sc:
                mock_sc.return_value = False
                result, _chunks, _is_empty = ingestion_service._ingest_single_file(
                    file_info,
                    vault_id=uuid.uuid4(),
                    vault_config=vault_config,
                )

                assert result == "updated"

    assert existing_doc.tags == ["updated", "inline"]
    tasks = _added_tasks(session)
    assert tasks
    assert tasks[0].tags == ["updated", "inline"]


def test_dry_run_extracts_body_tags_without_database_write(
    ingestion_service: IngestionService,
    tmp_path: Path,
) -> None:
    """Dry run extracts body tags but does not persist a Document."""
    session = _make_session(ingestion_service)
    session.query.return_value.filter_by.return_value.first.return_value = None

    vault_config = VaultConfig(
        container_path=str(tmp_path),
        host_path=str(tmp_path),
    )
    body = "Discussed #meeting today"
    file_info = _make_file_info(tmp_path, body)

    with patch("obsidian_rag.services.ingestion.parse_frontmatter") as mock_fm:
        mock_fm.return_value = (None, {}, body)
        with patch(
            "obsidian_rag.services.ingestion.parse_tasks_from_content",
        ) as mock_pt:
            mock_pt.return_value = []
            with patch(
                "obsidian_rag.services.ingestion.should_chunk_document",
            ) as mock_sc:
                mock_sc.return_value = False
                result, chunks, is_empty = ingestion_service._ingest_single_file(
                    file_info,
                    vault_id=uuid.uuid4(),
                    vault_config=vault_config,
                    dry_run=True,
                )

                assert result == "new"
                assert chunks == 0
                assert is_empty is False

    docs = _added_documents(session)
    assert len(docs) == 0


def test_unchanged_checksum_skips_update(
    ingestion_service: IngestionService,
    tmp_path: Path,
) -> None:
    """Existing document with matching checksum is skipped without update."""
    session = _make_session(ingestion_service)

    existing_doc = MagicMock()
    existing_doc.id = uuid.uuid4()
    existing_doc.checksum_md5 = "same-checksum"
    existing_doc.tags = ["old"]
    session.query.return_value.filter_by.return_value.first.return_value = existing_doc

    vault_config = VaultConfig(
        container_path=str(tmp_path),
        host_path=str(tmp_path),
    )
    body = "Content with #new-tag"
    file_info = _make_file_info(tmp_path, body)
    file_info.checksum = "same-checksum"

    with patch("obsidian_rag.services.ingestion.parse_frontmatter") as mock_fm:
        mock_fm.return_value = (None, {}, body)
        with patch(
            "obsidian_rag.services.ingestion.parse_tasks_from_content",
        ) as mock_pt:
            mock_pt.return_value = []
            with patch(
                "obsidian_rag.services.ingestion.should_chunk_document",
            ) as mock_sc:
                mock_sc.return_value = False
                result, _chunks, _is_empty = ingestion_service._ingest_single_file(
                    file_info,
                    vault_id=uuid.uuid4(),
                    vault_config=vault_config,
                )

                assert result == "unchanged"

    assert existing_doc.tags == ["old"]
