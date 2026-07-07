"""Tests that handler/tool model_dump(mode='json') returns str ids and datetimes.

These tests exercise the real Pydantic serialization path to catch regressions
where model_dump() might return UUID or datetime objects instead of strings.
"""

import uuid
from datetime import UTC, date, datetime
from unittest.mock import MagicMock, patch

from obsidian_rag.mcp_server.handlers import (
    GetDocumentHandlerParams,
    ListDocumentsHandlerParams,
    _delete_vault_handler,
    _get_all_tags_handler,
    _get_document_handler,
    _get_documents_by_tag_handler,
    _get_tasks_handler,
    _get_vault_handler,
    _list_documents_handler,
    _list_vaults_handler,
    _update_vault_handler,
)
from obsidian_rag.mcp_server.models import (
    DocumentListResponse,
    DocumentResponse,
    TagListResponse,
    TaskListResponse,
    TaskResponse,
    VaultListResponse,
    VaultResponse,
)
from obsidian_rag.mcp_server.tool_definitions import query_documents_tool
from obsidian_rag.mcp_server.tools.tasks_params import GetTasksRequest
from obsidian_rag.mcp_server.tools.vaults_params import VaultUpdateParams


def _mock_db_manager() -> MagicMock:
    """Create a mock DB manager with session context manager."""
    db_manager = MagicMock()
    session = MagicMock()
    db_manager.get_session.return_value.__enter__.return_value = session
    db_manager.get_session.return_value.__exit__.return_value = False
    return db_manager


def test_get_tasks_handler_returns_str_ids() -> None:
    """REQ-001: _get_tasks_handler returns str ids via model_dump(mode='json')."""
    original_uuid_1 = uuid.uuid4()
    original_uuid_2 = uuid.uuid4()
    task_list = TaskListResponse(
        results=[
            TaskResponse(
                id=original_uuid_1,
                raw_text="task 1",
                status="not_completed",
                description="desc 1",
                due=date(2026, 1, 1),
                priority="normal",
                tags=["work"],
                document_path="doc1.md",
                document_name="doc1",
            ),
            TaskResponse(
                id=original_uuid_2,
                raw_text="task 2",
                status="completed",
                description="desc 2",
                due=None,
                priority="high",
                tags=["personal"],
                document_path="doc2.md",
                document_name="doc2",
            ),
        ],
        total_count=2,
        has_more=False,
        next_offset=None,
    )
    db_manager = _mock_db_manager()
    with patch(
        "obsidian_rag.mcp_server.handlers.get_tasks_tool",
        return_value=task_list,
    ):
        request = GetTasksRequest()
        result = _get_tasks_handler(db_manager, request)
    assert isinstance(result, dict)
    assert isinstance(result["results"][0]["id"], str)
    assert result["results"][0]["id"] == str(original_uuid_1)
    assert isinstance(result["results"][1]["id"], str)
    assert result["results"][1]["id"] == str(original_uuid_2)


def test_get_document_handler_returns_str_id() -> None:
    """REQ-001: _get_document_handler by path returns str id and datetime."""
    original_uuid = uuid.uuid4()
    original_created = datetime.now(UTC)
    doc_response = DocumentResponse(
        id=original_uuid,
        vault_name="Personal",
        file_path="notes/test.md",
        relative_path="notes/test.md",
        file_name="test.md",
        content="hello",
        kind="note",
        tags=["work"],
        similarity_score=0.0,
        created_at_fs=original_created,
        modified_at_fs=original_created,
        obsidian_uri="obsidian://open?vault=Personal&file=notes%2Ftest.md",
    )
    db_manager = _mock_db_manager()
    with patch(
        "obsidian_rag.mcp_server.tools.documents.get_document",
        return_value=doc_response,
    ):
        params = GetDocumentHandlerParams(
            db_manager=db_manager,
            vault_name="Personal",
            file_path="notes/test.md",
        )
        result = _get_document_handler(params)
    assert isinstance(result, dict)
    assert isinstance(result["id"], str)
    assert result["id"] == str(original_uuid)
    assert isinstance(result["created_at_fs"], str)


def test_get_document_handler_returns_str_id_by_uuid() -> None:
    """REQ-001: _get_document_handler by document_id returns str id."""
    original_uuid = uuid.uuid4()
    original_created = datetime.now(UTC)
    doc_response = DocumentResponse(
        id=original_uuid,
        vault_name="Personal",
        file_path="notes/test.md",
        relative_path="notes/test.md",
        file_name="test.md",
        content="hello",
        kind="note",
        tags=["work"],
        similarity_score=0.0,
        created_at_fs=original_created,
        modified_at_fs=original_created,
        obsidian_uri="obsidian://open?vault=Personal&file=notes%2Ftest.md",
    )
    db_manager = _mock_db_manager()
    with patch(
        "obsidian_rag.mcp_server.tools.documents.get_document",
        return_value=doc_response,
    ):
        params = GetDocumentHandlerParams(
            db_manager=db_manager,
            document_id=str(original_uuid),
        )
        result = _get_document_handler(params)
    assert isinstance(result, dict)
    assert isinstance(result["id"], str)
    assert result["id"] == str(original_uuid)
    assert isinstance(result["created_at_fs"], str)


def test_list_documents_handler_returns_str_ids() -> None:
    """REQ-001: _list_documents_handler returns str ids via model_dump(mode='json')."""
    original_uuid_1 = uuid.uuid4()
    original_uuid_2 = uuid.uuid4()
    original_created = datetime.now(UTC)
    doc_list = DocumentListResponse(
        results=[
            DocumentResponse(
                id=original_uuid_1,
                vault_name="Personal",
                file_path="notes/a.md",
                relative_path="notes/a.md",
                file_name="a.md",
                content="a",
                kind="note",
                tags=["work"],
                similarity_score=0.0,
                created_at_fs=original_created,
                modified_at_fs=original_created,
                obsidian_uri="obsidian://open?vault=Personal&file=notes%2Fa.md",
            ),
            DocumentResponse(
                id=original_uuid_2,
                vault_name="Work",
                file_path="notes/b.md",
                relative_path="notes/b.md",
                file_name="b.md",
                content="b",
                kind="note",
                tags=["personal"],
                similarity_score=0.0,
                created_at_fs=original_created,
                modified_at_fs=original_created,
                obsidian_uri="obsidian://open?vault=Work&file=notes%2Fb.md",
            ),
        ],
        total_count=2,
        has_more=False,
        next_offset=None,
    )
    db_manager = _mock_db_manager()
    with patch(
        "obsidian_rag.mcp_server.tools.documents.list_documents",
        return_value=doc_list,
    ):
        params = ListDocumentsHandlerParams(
            db_manager=db_manager,
            file_name="a.md",
        )
        result = _list_documents_handler(params)
    assert isinstance(result, dict)
    assert isinstance(result["results"][0]["id"], str)
    assert result["results"][0]["id"] == str(original_uuid_1)
    assert isinstance(result["results"][1]["id"], str)
    assert result["results"][1]["id"] == str(original_uuid_2)
    assert isinstance(result["results"][0]["created_at_fs"], str)


def test_get_documents_by_tag_handler_returns_str_ids() -> None:
    """REQ-001: _get_documents_by_tag_handler returns str ids via model_dump(mode='json')."""
    original_uuid = uuid.uuid4()
    original_created = datetime.now(UTC)
    doc_list = DocumentListResponse(
        results=[
            DocumentResponse(
                id=original_uuid,
                vault_name="Personal",
                file_path="notes/test.md",
                relative_path="notes/test.md",
                file_name="test.md",
                content="hello",
                kind="note",
                tags=["work"],
                similarity_score=0.0,
                created_at_fs=original_created,
                modified_at_fs=original_created,
                obsidian_uri="obsidian://open?vault=Personal&file=notes%2Ftest.md",
            ),
        ],
        total_count=1,
        has_more=False,
        next_offset=None,
    )
    db_manager = _mock_db_manager()
    with patch(
        "obsidian_rag.mcp_server.handlers.get_documents_by_tag_tool",
        return_value=doc_list,
    ):
        result = _get_documents_by_tag_handler(
            db_manager,
            {
                "include_tags": ["work"],
                "exclude_tags": [],
                "match_mode": "all",
                "vault_name": None,
                "limit": 20,
                "offset": 0,
                "include_content": True,
            },
        )
    assert isinstance(result, dict)
    assert isinstance(result["results"][0]["id"], str)
    assert result["results"][0]["id"] == str(original_uuid)
    assert isinstance(result["results"][0]["created_at_fs"], str)


def test_get_documents_by_property_returns_str_ids() -> None:
    """REQ-001: get_documents_by_property returns str ids via model_dump(mode='json').

    Note: _get_documents_by_property_handler does not exist in handlers.py;
    the get_documents_by_property tool wrapper lives in server.py and calls
    model_dump(mode='json') directly. This test verifies that path.
    """
    from obsidian_rag.mcp_server.server import get_documents_by_property
    from obsidian_rag.mcp_server.tool_definitions import (
        MCPToolRegistry,
        _set_registry,
    )

    original_uuid = uuid.uuid4()
    original_created = datetime.now(UTC)
    doc_list = DocumentListResponse(
        results=[
            DocumentResponse(
                id=original_uuid,
                vault_name="Personal",
                file_path="notes/test.md",
                relative_path="notes/test.md",
                file_name="test.md",
                content="hello",
                kind="note",
                tags=["work"],
                similarity_score=0.0,
                created_at_fs=original_created,
                modified_at_fs=original_created,
                obsidian_uri="obsidian://open?vault=Personal&file=notes%2Ftest.md",
            ),
        ],
        total_count=1,
        has_more=False,
        next_offset=None,
    )
    db_manager = _mock_db_manager()
    registry = MCPToolRegistry(
        db_manager=db_manager,
        embedding_provider=None,
        settings=MagicMock(),
    )
    _set_registry(registry)
    try:
        with patch(
            "obsidian_rag.mcp_server.tools.documents.get_documents_by_property",
            return_value=doc_list,
        ):
            result = get_documents_by_property()
        assert isinstance(result, dict)
        assert isinstance(result["results"][0]["id"], str)
        assert result["results"][0]["id"] == str(original_uuid)
        assert isinstance(result["results"][0]["created_at_fs"], str)
    finally:
        _set_registry(None)


def test_get_all_tags_handler_returns_str_tags() -> None:
    """REQ-001: _get_all_tags_handler returns tag list with no UUID leaks."""
    tag_list = TagListResponse(
        tags=["work", "personal"],
        total_count=2,
        has_more=False,
        next_offset=None,
    )
    db_manager = _mock_db_manager()
    with patch(
        "obsidian_rag.mcp_server.handlers.get_all_tags_tool",
        return_value=tag_list,
    ):
        result = _get_all_tags_handler(db_manager, None, 20, 0)
    assert isinstance(result, dict)
    assert isinstance(result["tags"], list)
    assert all(isinstance(tag, str) for tag in result["tags"])
    assert result["tags"] == ["work", "personal"]


def test_list_vaults_handler_returns_str_ids() -> None:
    """REQ-001: _list_vaults_handler returns str ids and datetimes via model_dump(mode='json')."""
    original_uuid = uuid.uuid4()
    original_created = datetime.now(UTC)
    vault_list = VaultListResponse(
        results=[
            VaultResponse(
                id=original_uuid,
                name="Personal",
                description="Personal vault",
                container_path="/data/personal",
                host_path="/home/user/personal",
                document_count=5,
                created_at=original_created,
            ),
        ],
        total_count=1,
        has_more=False,
        next_offset=None,
    )
    db_manager = _mock_db_manager()
    with patch(
        "obsidian_rag.mcp_server.handlers.list_vaults_tool",
        return_value=vault_list,
    ):
        result = _list_vaults_handler(db_manager, 20, 0)
    assert isinstance(result, dict)
    assert isinstance(result["results"][0]["id"], str)
    assert result["results"][0]["id"] == str(original_uuid)
    assert isinstance(result["results"][0]["created_at"], str)


def test_get_vault_handler_returns_str_id() -> None:
    """REQ-001: _get_vault_handler by name returns str id and datetime."""
    original_uuid = uuid.uuid4()
    original_created = datetime.now(UTC)
    vault_response = VaultResponse(
        id=original_uuid,
        name="Personal",
        description="Personal vault",
        container_path="/data/personal",
        host_path="/home/user/personal",
        document_count=5,
        created_at=original_created,
    )
    db_manager = _mock_db_manager()
    with patch(
        "obsidian_rag.mcp_server.handlers.get_vault",
        return_value=vault_response,
    ):
        result = _get_vault_handler(db_manager, name="Personal")
    assert isinstance(result, dict)
    assert isinstance(result["id"], str)
    assert not isinstance(result["id"], uuid.UUID)
    assert result["id"] == str(original_uuid)
    assert isinstance(result["created_at"], str)


def test_get_vault_handler_by_uuid_returns_str_id() -> None:
    """REQ-001: _get_vault_handler by vault_id returns str id."""
    original_uuid = uuid.uuid4()
    original_created = datetime.now(UTC)
    vault_response = VaultResponse(
        id=original_uuid,
        name="Work",
        description="Work vault",
        container_path="/data/work",
        host_path="/home/user/work",
        document_count=10,
        created_at=original_created,
    )
    db_manager = _mock_db_manager()
    with patch(
        "obsidian_rag.mcp_server.handlers.get_vault",
        return_value=vault_response,
    ):
        result = _get_vault_handler(db_manager, vault_id=str(original_uuid))
    assert isinstance(result, dict)
    assert isinstance(result["id"], str)
    assert result["id"] == str(original_uuid)
    assert isinstance(result["created_at"], str)


def test_update_vault_handler_returns_str_id() -> None:
    """REQ-001: _update_vault_handler returns str id via model_dump(mode='json')."""
    original_uuid = uuid.uuid4()
    original_created = datetime.now(UTC)
    vault_response = VaultResponse(
        id=original_uuid,
        name="Personal",
        description="Updated description",
        container_path="/data/personal",
        host_path="/home/user/personal",
        document_count=5,
        created_at=original_created,
    )
    db_manager = _mock_db_manager()
    with patch(
        "obsidian_rag.mcp_server.handlers.update_vault",
        return_value=vault_response,
    ):
        params = VaultUpdateParams(name="Personal", description="Updated")
        result = _update_vault_handler(db_manager, params)
    assert isinstance(result, dict)
    assert isinstance(result["id"], str)
    assert result["id"] == str(original_uuid)
    assert isinstance(result["created_at"], str)


def test_delete_vault_handler_returns_str_id() -> None:
    """REQ-001: _delete_vault_handler returns str id in success dict."""
    original_uuid = uuid.uuid4()
    db_manager = _mock_db_manager()
    with patch(
        "obsidian_rag.mcp_server.handlers.delete_vault",
        return_value={
            "success": True,
            "name": "Personal",
            "id": str(original_uuid),
            "documents_deleted": 5,
            "tasks_deleted": 3,
            "chunks_deleted": 10,
            "warning": "Vault config entry still exists.",
        },
    ):
        result = _delete_vault_handler(db_manager, name="Personal", confirm=True)
    assert isinstance(result, dict)
    assert isinstance(result["id"], str)
    assert result["id"] == str(original_uuid)


def test_query_documents_tool_returns_str_ids() -> None:
    """REQ-001: query_documents_tool returns str ids via model_dump(mode='json')."""
    original_uuid = uuid.uuid4()
    original_created = datetime.now(UTC)
    doc_list = DocumentListResponse(
        results=[
            DocumentResponse(
                id=original_uuid,
                vault_name="Personal",
                file_path="notes/test.md",
                relative_path="notes/test.md",
                file_name="test.md",
                content="hello",
                kind="note",
                tags=["work"],
                similarity_score=0.5,
                created_at_fs=original_created,
                modified_at_fs=original_created,
                obsidian_uri="obsidian://open?vault=Personal&file=notes%2Ftest.md",
            ),
        ],
        total_count=1,
        has_more=False,
        next_offset=None,
    )
    db_manager = _mock_db_manager()
    embedding_provider = MagicMock()
    embedding_provider.generate_embedding.return_value = [0.1] * 1536
    with patch(
        "obsidian_rag.mcp_server.tools.documents.query_documents",
        return_value=doc_list,
    ):
        result = query_documents_tool(
            db_manager=db_manager,
            embedding_provider=embedding_provider,
            query="test",
        )
    assert isinstance(result, dict)
    assert isinstance(result["results"][0]["id"], str)
    assert result["results"][0]["id"] == str(original_uuid)
    assert isinstance(result["results"][0]["created_at_fs"], str)
