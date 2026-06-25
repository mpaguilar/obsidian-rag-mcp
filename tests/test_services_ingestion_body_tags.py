"""Tests for body tag extraction and merging in _ingest_single_file.

These tests verify that ``_ingest_single_file`` extracts inline body tags from
the content returned by ``parse_frontmatter`` (i.e. the body with frontmatter
removed) and merges them with frontmatter tags before the merged tag list flows
into ``Document.tags`` (and onwards into task tags via ``_create_tasks``).

The real ``extract_body_tags`` and ``_merge_tags`` implementations are used
(matching the existing pattern in ``tests/test_services_ingestion.py`` which
exercises the real ``_merge_tags``). Only ``parse_frontmatter``,
``parse_tasks_from_content`` and ``should_chunk_document`` are patched so the
frontmatter tags and body content can be controlled independently.
"""

import uuid
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from obsidian_rag.config import VaultConfig
from obsidian_rag.database.models import Document, Task
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


def _make_task(tag: str | None) -> MagicMock:
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
    return [call.args[0] for call in session.add.call_args_list if isinstance(call.args[0], Document)]


def _added_tasks(session: MagicMock) -> list[Task]:
    """Return Task instances passed to session.add.

    Args:
        session: The mock session to inspect.

    Returns:
        List of Task objects added during ingestion.

    """
    return [call.args[0] for call in session.add.call_args_list if isinstance(call.args[0], Task)]


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
        with patch("obsidian_rag.services.ingestion.parse_tasks_from_content") as mock_pt:
            mock_pt.return_value = parsed_tasks
            with patch("obsidian_rag.services.ingestion.should_chunk_document") as mock_sc:
                mock_sc.return_value = False
                service._ingest_single_file(
                    file_info,
                    vault_id=uuid.uuid4(),
                    vault_config=vault_config,
                )

    return session


def test_ingest_single_file_merges_body_tags(
    ingestion_service: IngestionService,
    tmp_path: Path,
) -> None:
    """Frontmatter tags plus body inline tags are merged into Document.tags."""
    session = _run_new_ingest(
        ingestion_service,
        tmp_path,
        frontmatter_tags=["work"],
        body="Some intro text #meeting and #project here",
    )

    docs = _added_documents(session)
    assert docs
    assert docs[0].tags == ["work", "meeting", "project"]


def test_ingest_single_file_body_tags_only(
    ingestion_service: IngestionService,
    tmp_path: Path,
) -> None:
    """A document with no frontmatter but body tags gets those tags."""
    session = _run_new_ingest(
        ingestion_service,
        tmp_path,
        frontmatter_tags=None,
        body="Body with #tag1 and #tag2 only",
    )

    docs = _added_documents(session)
    assert docs
    assert docs[0].tags == ["tag1", "tag2"]


def test_ingest_single_file_no_body_tags(
    ingestion_service: IngestionService,
    tmp_path: Path,
) -> None:
    """Frontmatter tags with no body tags preserve unchanged behavior."""
    session = _run_new_ingest(
        ingestion_service,
        tmp_path,
        frontmatter_tags=["work"],
        body="Just plain text without any inline tags here",
    )

    docs = _added_documents(session)
    assert docs
    assert docs[0].tags == ["work"]


def test_ingest_single_file_body_tags_excluded_heading(
    ingestion_service: IngestionService,
    tmp_path: Path,
) -> None:
    """A markdown heading is not treated as a body tag."""
    session = _run_new_ingest(
        ingestion_service,
        tmp_path,
        frontmatter_tags=["work"],
        body="# Heading\n\nSome text under the heading",
    )

    docs = _added_documents(session)
    assert docs
    tags = docs[0].tags
    assert tags == ["work"]
    assert "heading" not in tags
    assert "Heading" not in tags


def test_ingest_single_file_body_tags_excluded_code_block(
    ingestion_service: IngestionService,
    tmp_path: Path,
) -> None:
    """Tags inside code blocks are not extracted into Document.tags."""
    session = _run_new_ingest(
        ingestion_service,
        tmp_path,
        frontmatter_tags=["work"],
        body="```\n#code-tag\n```\n\n#real-tag",
    )

    docs = _added_documents(session)
    assert docs
    tags = docs[0].tags
    assert tags == ["work", "real-tag"]
    assert "code-tag" not in tags
    assert "real-tag" in tags


def test_ingest_single_file_body_tags_case_insensitive_dedup(
    ingestion_service: IngestionService,
    tmp_path: Path,
) -> None:
    """Frontmatter 'Work' and body '#work' collapse to a single 'work' tag."""
    session = _run_new_ingest(
        ingestion_service,
        tmp_path,
        frontmatter_tags=["Work"],
        body="Body text with #work inline",
    )

    docs = _added_documents(session)
    assert docs
    tags = docs[0].tags
    assert tags == ["work"]
    assert tags.count("work") == 1


def test_ingest_single_file_task_merging_inherits_body_tags(
    ingestion_service: IngestionService,
    tmp_path: Path,
) -> None:
    """Tasks inherit body tags merged into Document.tags via _create_tasks."""
    session = _run_new_ingest(
        ingestion_service,
        tmp_path,
        frontmatter_tags=["work"],
        body="Intro text with #project tag\n- [ ] do thing",
        task=_make_task("personal"),
    )

    tasks = _added_tasks(session)
    assert tasks
    assert tasks[0].tags == ["work", "project", "personal"]


def test_ingest_single_file_update_path_merges_body_tags(
    ingestion_service: IngestionService,
    tmp_path: Path,
) -> None:
    """The update path also merges body tags into the existing document."""
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
    body = "Body content with #inline tag"
    file_info = _make_file_info(tmp_path, body)

    with patch("obsidian_rag.services.ingestion.parse_frontmatter") as mock_fm:
        mock_fm.return_value = (["updated"], {}, body)
        with patch("obsidian_rag.services.ingestion.parse_tasks_from_content") as mock_pt:
            mock_pt.return_value = [(1, _make_task("inline"))]
            with patch("obsidian_rag.services.ingestion.should_chunk_document") as mock_sc:
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


def test_ingest_single_file_empty_document(
    ingestion_service: IngestionService,
    tmp_path: Path,
) -> None:
    """An empty document yields None tags."""
    session = _run_new_ingest(
        ingestion_service,
        tmp_path,
        frontmatter_tags=None,
        body="",
    )

    docs = _added_documents(session)
    assert docs
    assert docs[0].tags is None


def test_ingest_single_file_all_numeric_excluded(
    ingestion_service: IngestionService,
    tmp_path: Path,
) -> None:
    """An all-numeric token like '#1984' is not treated as a tag."""
    session = _run_new_ingest(
        ingestion_service,
        tmp_path,
        frontmatter_tags=["work"],
        body="Read #1984 today and #real notes",
    )

    docs = _added_documents(session)
    assert docs
    tags = docs[0].tags
    assert tags == ["work", "real"]
    assert "1984" not in tags
    assert "real" in tags
