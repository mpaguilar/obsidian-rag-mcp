"""Unit tests for MCP server document handlers."""

from unittest.mock import MagicMock, patch


class TestDocumentHandlers:
    """Tests for document tool handlers."""

    def test_get_documents_by_tag_handler_full_flow(self):
        """Test _get_documents_by_tag_handler full flow."""
        from obsidian_rag.mcp_server.handlers import _get_documents_by_tag_handler

        mock_db_manager = MagicMock()
        mock_session = MagicMock()
        mock_db_manager.get_session.return_value.__enter__ = MagicMock(
            return_value=mock_session
        )
        mock_db_manager.get_session.return_value.__exit__ = MagicMock(
            return_value=False
        )

        with patch(
            "obsidian_rag.mcp_server.handlers.get_documents_by_tag_tool"
        ) as mock_tool:
            mock_result = MagicMock()
            mock_result.model_dump.return_value = {
                "results": [],
                "total_count": 0,
                "has_more": False,
            }
            mock_tool.return_value = mock_result

            # Create params as a dict-like object
            params = {
                "include_tags": ["work"],
                "exclude_tags": [],
                "match_mode": "all",
                "vault_root": None,
                "limit": 20,
                "offset": 0,
            }
            result = _get_documents_by_tag_handler(mock_db_manager, params)  # type: ignore[arg-type]

            assert result == {"results": [], "total_count": 0, "has_more": False}
            mock_tool.assert_called_once()

    def test_get_all_tags_handler_full_flow(self):
        """Test _get_all_tags_handler full flow."""
        from obsidian_rag.mcp_server.handlers import _get_all_tags_handler

        mock_db_manager = MagicMock()
        mock_session = MagicMock()
        mock_db_manager.get_session.return_value.__enter__ = MagicMock(
            return_value=mock_session
        )
        mock_db_manager.get_session.return_value.__exit__ = MagicMock(
            return_value=False
        )

        with patch("obsidian_rag.mcp_server.handlers.get_all_tags_tool") as mock_tool:
            mock_result = MagicMock()
            mock_result.model_dump.return_value = {
                "tags": [],
                "total_count": 0,
                "has_more": False,
            }
            mock_tool.return_value = mock_result

            result = _get_all_tags_handler(mock_db_manager, "work*", 20, 0)

            assert result == {"tags": [], "total_count": 0, "has_more": False}
            mock_tool.assert_called_once_with(
                session=mock_session, pattern="work*", limit=20, offset=0
            )

    def test_convert_property_filters_with_valid_filters(self):
        """Test _convert_property_filters with valid filters."""
        from obsidian_rag.mcp_server.handlers import _convert_property_filters

        properties = [
            {"path": "status", "operator": "equals", "value": "draft"},
            {"path": "priority", "operator": "equals", "value": "high"},
        ]

        result = _convert_property_filters(properties)

        assert result is not None
        assert len(result) == len(properties)
        assert result[0].path == "status"
        assert result[0].operator == "equals"
        assert result[0].value == "draft"

    def test_create_tag_filter_with_none_filters(self):
        """Test _create_tag_filter with None filters."""
        from obsidian_rag.mcp_server.handlers import _create_tag_filter

        result = _create_tag_filter(None)

        assert result is None

    def test_create_tag_filter_with_empty_tags(self):
        """Test _create_tag_filter with empty tags."""
        from obsidian_rag.mcp_server.handlers import (
            _create_tag_filter,
            QueryFilterParams,
        )

        filters = QueryFilterParams(
            include_properties=None,
            exclude_properties=None,
            include_tags=None,
            exclude_tags=None,
            match_mode="all",
        )
        result = _create_tag_filter(filters)

        assert result is None


class TestConvertPropertyFilters:
    """Tests for _convert_property_filters."""

    def test_convert_property_filters_empty(self):
        """Test _convert_property_filters with empty properties."""
        from obsidian_rag.mcp_server.handlers import _convert_property_filters

        result = _convert_property_filters([])
        assert result is None

    def test_convert_property_filters_none(self):
        """Test _convert_property_filters with None."""
        from obsidian_rag.mcp_server.handlers import _convert_property_filters

        result = _convert_property_filters(None)
        assert result is None


class TestCreateTagFilter:
    """Tests for _create_tag_filter."""

    def test_create_tag_filter_invalid_match_mode(self):
        """Test _create_tag_filter with invalid match_mode defaults to 'all'."""
        from obsidian_rag.mcp_server.handlers import (
            _create_tag_filter,
            QueryFilterParams,
        )

        filters = QueryFilterParams(
            include_properties=None,
            exclude_properties=None,
            include_tags=["tag1"],
            exclude_tags=None,
            match_mode="invalid",  # type: ignore[arg-type]
        )
        result = _create_tag_filter(filters)

        assert result is not None
        assert result.match_mode == "all"
        assert result.include_tags == ["tag1"]


class TestGetDocumentHandlerParams:
    """Tests for GetDocumentHandlerParams dataclass."""

    def test_get_document_handler_params_dataclass(self):
        """Fields populated correctly."""
        from obsidian_rag.mcp_server.handlers import GetDocumentHandlerParams

        db_manager = MagicMock()
        params = GetDocumentHandlerParams(
            db_manager=db_manager,
            vault_name="Personal",
            file_path="notes.md",
            document_id="abc-123",
        )
        assert params.db_manager is db_manager
        assert params.vault_name == "Personal"
        assert params.file_path == "notes.md"
        assert params.document_id == "abc-123"

    def test_get_document_handler_params_defaults(self):
        """Default values are correct."""
        from obsidian_rag.mcp_server.handlers import GetDocumentHandlerParams

        params = GetDocumentHandlerParams(db_manager=MagicMock())
        assert params.vault_name is None
        assert params.file_path is None
        assert params.document_id is None


class TestListDocumentsHandlerParams:
    """Tests for ListDocumentsHandlerParams dataclass."""

    def test_list_documents_handler_params_dataclass(self):
        """Fields populated correctly."""
        from obsidian_rag.mcp_server.handlers import ListDocumentsHandlerParams

        db_manager = MagicMock()
        limit = 10
        offset = 5
        params = ListDocumentsHandlerParams(
            db_manager=db_manager,
            file_name="notes.md",
            vault_name="Personal",
            limit=limit,
            offset=offset,
        )
        assert params.db_manager is db_manager
        assert params.file_name == "notes.md"
        assert params.vault_name == "Personal"
        assert params.limit == limit
        assert params.offset == offset

    def test_list_documents_handler_params_defaults(self):
        """Default values are correct."""
        from obsidian_rag.mcp_server.handlers import ListDocumentsHandlerParams

        default_limit = 20
        params = ListDocumentsHandlerParams(db_manager=MagicMock())
        assert params.file_name is None
        assert params.vault_name is None
        assert params.limit == default_limit
        assert params.offset == 0


class TestGetDocumentHandler:
    """Tests for _get_document_handler."""

    @patch(
        "obsidian_rag.mcp_server.tools.documents.get_document",
        create=True,
    )
    def test_get_document_handler_success(self, mock_get_document):
        """Test successful document retrieval."""
        from obsidian_rag.mcp_server.handlers import (
            _get_document_handler,
            GetDocumentHandlerParams,
        )

        mock_db_manager = MagicMock()
        mock_session = MagicMock()
        mock_db_manager.get_session.return_value.__enter__ = MagicMock(
            return_value=mock_session
        )
        mock_db_manager.get_session.return_value.__exit__ = MagicMock(
            return_value=False
        )

        mock_result = MagicMock()
        mock_result.model_dump.return_value = {
            "id": "doc-1",
            "file_path": "notes.md",
        }
        mock_get_document.return_value = mock_result

        params = GetDocumentHandlerParams(
            db_manager=mock_db_manager,
            vault_name="Personal",
            file_path="notes.md",
        )
        result = _get_document_handler(params)

        assert result == {"id": "doc-1", "file_path": "notes.md"}
        mock_get_document.assert_called_once_with(
            session=mock_session,
            vault_name="Personal",
            file_path="notes.md",
            document_id=None,
        )

    @patch(
        "obsidian_rag.mcp_server.tools.documents.get_document",
        create=True,
    )
    def test_get_document_handler_by_id_success(self, mock_get_document):
        """Test successful document retrieval by document_id."""
        from obsidian_rag.mcp_server.handlers import (
            _get_document_handler,
            GetDocumentHandlerParams,
        )

        mock_db_manager = MagicMock()
        mock_session = MagicMock()
        mock_db_manager.get_session.return_value.__enter__ = MagicMock(
            return_value=mock_session
        )
        mock_db_manager.get_session.return_value.__exit__ = MagicMock(
            return_value=False
        )

        mock_result = MagicMock()
        mock_result.model_dump.return_value = {
            "id": "doc-1",
            "file_path": "notes.md",
        }
        mock_get_document.return_value = mock_result

        params = GetDocumentHandlerParams(
            db_manager=mock_db_manager,
            document_id="abc-123",
        )
        result = _get_document_handler(params)

        assert result == {"id": "doc-1", "file_path": "notes.md"}
        mock_get_document.assert_called_once_with(
            session=mock_session,
            vault_name=None,
            file_path=None,
            document_id="abc-123",
        )

    @patch(
        "obsidian_rag.mcp_server.tools.documents.get_document",
        create=True,
    )
    def test_get_document_handler_by_vault_path_success(self, mock_get_document):
        """Test successful document retrieval by vault_name and file_path."""
        from obsidian_rag.mcp_server.handlers import (
            _get_document_handler,
            GetDocumentHandlerParams,
        )

        mock_db_manager = MagicMock()
        mock_session = MagicMock()
        mock_db_manager.get_session.return_value.__enter__ = MagicMock(
            return_value=mock_session
        )
        mock_db_manager.get_session.return_value.__exit__ = MagicMock(
            return_value=False
        )

        mock_result = MagicMock()
        mock_result.model_dump.return_value = {
            "id": "doc-1",
            "file_path": "notes.md",
            "vault_name": "Personal",
        }
        mock_get_document.return_value = mock_result

        params = GetDocumentHandlerParams(
            db_manager=mock_db_manager,
            vault_name="Personal",
            file_path="notes.md",
        )
        result = _get_document_handler(params)

        assert result == {
            "id": "doc-1",
            "file_path": "notes.md",
            "vault_name": "Personal",
        }
        mock_get_document.assert_called_once_with(
            session=mock_session,
            vault_name="Personal",
            file_path="notes.md",
            document_id=None,
        )

    @patch(
        "obsidian_rag.mcp_server.tools.documents.get_document",
        create=True,
    )
    def test_get_document_handler_no_params(self, mock_get_document):
        """Test error when no lookup parameters provided."""
        from obsidian_rag.mcp_server.handlers import (
            _get_document_handler,
            GetDocumentHandlerParams,
        )

        mock_db_manager = MagicMock()
        mock_session = MagicMock()
        mock_db_manager.get_session.return_value.__enter__ = MagicMock(
            return_value=mock_session
        )
        mock_db_manager.get_session.return_value.__exit__ = MagicMock(
            return_value=False
        )

        mock_get_document.side_effect = ValueError(
            "Must provide either document_id, or vault_name and file_path"
        )

        params = GetDocumentHandlerParams(
            db_manager=mock_db_manager,
        )
        result = _get_document_handler(params)

        assert result == {
            "success": False,
            "error": "Must provide either document_id, or vault_name and file_path",
        }

    @patch(
        "obsidian_rag.mcp_server.tools.documents.get_document",
        create=True,
    )
    def test_get_document_handler_path_without_vault(self, mock_get_document):
        """Test error when file_path provided without vault_name."""
        from obsidian_rag.mcp_server.handlers import (
            _get_document_handler,
            GetDocumentHandlerParams,
        )

        mock_db_manager = MagicMock()
        mock_session = MagicMock()
        mock_db_manager.get_session.return_value.__enter__ = MagicMock(
            return_value=mock_session
        )
        mock_db_manager.get_session.return_value.__exit__ = MagicMock(
            return_value=False
        )

        mock_get_document.side_effect = ValueError(
            "vault_name is required when using file_path "
            "(file_path is only unique per vault)"
        )

        params = GetDocumentHandlerParams(
            db_manager=mock_db_manager,
            file_path="notes.md",
        )
        result = _get_document_handler(params)

        assert result == {
            "success": False,
            "error": "vault_name is required when using file_path "
            "(file_path is only unique per vault)",
        }

    @patch(
        "obsidian_rag.mcp_server.tools.documents.get_document",
        create=True,
    )
    def test_get_document_handler_not_found(self, mock_get_document):
        """Test error when document not found."""
        from obsidian_rag.mcp_server.handlers import (
            _get_document_handler,
            GetDocumentHandlerParams,
        )

        mock_db_manager = MagicMock()
        mock_session = MagicMock()
        mock_db_manager.get_session.return_value.__enter__ = MagicMock(
            return_value=mock_session
        )
        mock_db_manager.get_session.return_value.__exit__ = MagicMock(
            return_value=False
        )

        mock_get_document.side_effect = ValueError("Document not found")

        params = GetDocumentHandlerParams(
            db_manager=mock_db_manager,
            vault_name="Personal",
            file_path="missing.md",
        )
        result = _get_document_handler(params)

        assert result == {"success": False, "error": "Document not found"}

    @patch(
        "obsidian_rag.mcp_server.tools.documents.get_document",
        create=True,
    )
    def test_get_document_handler_invalid_uuid(self, mock_get_document):
        """Test error when document_id is invalid UUID."""
        from obsidian_rag.mcp_server.handlers import (
            _get_document_handler,
            GetDocumentHandlerParams,
        )

        mock_db_manager = MagicMock()
        mock_session = MagicMock()
        mock_db_manager.get_session.return_value.__enter__ = MagicMock(
            return_value=mock_session
        )
        mock_db_manager.get_session.return_value.__exit__ = MagicMock(
            return_value=False
        )

        mock_get_document.side_effect = ValueError(
            "Invalid document_id UUID format: 'invalid-uuid'"
        )

        params = GetDocumentHandlerParams(
            db_manager=mock_db_manager,
            document_id="invalid-uuid",
        )
        result = _get_document_handler(params)

        assert result == {
            "success": False,
            "error": "Invalid document_id UUID format: 'invalid-uuid'",
        }


class TestListDocumentsHandler:
    """Tests for _list_documents_handler."""

    @patch(
        "obsidian_rag.mcp_server.tools.documents.list_documents",
        create=True,
    )
    def test_list_documents_handler_success(self, mock_list_documents):
        """Test successful document list retrieval."""
        from obsidian_rag.mcp_server.handlers import (
            _list_documents_handler,
            ListDocumentsHandlerParams,
        )

        mock_db_manager = MagicMock()
        mock_session = MagicMock()
        mock_db_manager.get_session.return_value.__enter__ = MagicMock(
            return_value=mock_session
        )
        mock_db_manager.get_session.return_value.__exit__ = MagicMock(
            return_value=False
        )

        mock_result = MagicMock()
        mock_result.model_dump.return_value = {
            "results": [],
            "total_count": 0,
            "has_more": False,
        }
        mock_list_documents.return_value = mock_result

        params = ListDocumentsHandlerParams(
            db_manager=mock_db_manager,
            file_name="notes.md",
            vault_name="Personal",
            limit=10,
            offset=5,
        )
        result = _list_documents_handler(params)

        assert result == {"results": [], "total_count": 0, "has_more": False}
        mock_list_documents.assert_called_once_with(
            session=mock_session,
            file_name="notes.md",
            vault_name="Personal",
            limit=10,
            offset=5,
        )

    @patch(
        "obsidian_rag.mcp_server.tools.documents.list_documents",
        create=True,
    )
    def test_list_documents_handler_value_error(self, mock_list_documents):
        """Test error handling for invalid parameters."""
        from obsidian_rag.mcp_server.handlers import (
            _list_documents_handler,
            ListDocumentsHandlerParams,
        )

        mock_db_manager = MagicMock()
        mock_session = MagicMock()
        mock_db_manager.get_session.return_value.__enter__ = MagicMock(
            return_value=mock_session
        )
        mock_db_manager.get_session.return_value.__exit__ = MagicMock(
            return_value=False
        )

        mock_list_documents.side_effect = ValueError("Invalid vault")

        params = ListDocumentsHandlerParams(db_manager=mock_db_manager)
        result = _list_documents_handler(params)

        assert result == {"success": False, "error": "Invalid vault"}

    @patch(
        "obsidian_rag.mcp_server.tools.documents.list_documents",
        create=True,
    )
    def test_list_documents_handler_empty_results(self, mock_list_documents):
        """Test empty results return empty list dict, not error."""
        from obsidian_rag.mcp_server.handlers import (
            _list_documents_handler,
            ListDocumentsHandlerParams,
        )

        mock_db_manager = MagicMock()
        mock_session = MagicMock()
        mock_db_manager.get_session.return_value.__enter__ = MagicMock(
            return_value=mock_session
        )
        mock_db_manager.get_session.return_value.__exit__ = MagicMock(
            return_value=False
        )

        mock_result = MagicMock()
        mock_result.model_dump.return_value = {
            "results": [],
            "total_count": 0,
            "has_more": False,
        }
        mock_list_documents.return_value = mock_result

        params = ListDocumentsHandlerParams(
            db_manager=mock_db_manager,
            file_name="nonexistent.md",
            limit=10,
            offset=0,
        )
        result = _list_documents_handler(params)

        assert result == {"results": [], "total_count": 0, "has_more": False}
        mock_list_documents.assert_called_once_with(
            session=mock_session,
            file_name="nonexistent.md",
            vault_name=None,
            limit=10,
            offset=0,
        )

    @patch(
        "obsidian_rag.mcp_server.tools.documents.list_documents",
        create=True,
    )
    def test_list_documents_handler_no_file_name(self, mock_list_documents):
        """Test error when file_name is not provided."""
        from obsidian_rag.mcp_server.handlers import (
            _list_documents_handler,
            ListDocumentsHandlerParams,
        )

        mock_db_manager = MagicMock()
        mock_session = MagicMock()
        mock_db_manager.get_session.return_value.__enter__ = MagicMock(
            return_value=mock_session
        )
        mock_db_manager.get_session.return_value.__exit__ = MagicMock(
            return_value=False
        )

        mock_list_documents.side_effect = ValueError("Must provide at least file_name")

        params = ListDocumentsHandlerParams(db_manager=mock_db_manager)
        result = _list_documents_handler(params)

        assert result == {
            "success": False,
            "error": "Must provide at least file_name",
        }

    @patch(
        "obsidian_rag.mcp_server.tools.documents.list_documents",
        create=True,
    )
    def test_list_documents_handler_vault_not_found(self, mock_list_documents):
        """Test error when vault_name is not found."""
        from obsidian_rag.mcp_server.handlers import (
            _list_documents_handler,
            ListDocumentsHandlerParams,
        )

        mock_db_manager = MagicMock()
        mock_session = MagicMock()
        mock_db_manager.get_session.return_value.__enter__ = MagicMock(
            return_value=mock_session
        )
        mock_db_manager.get_session.return_value.__exit__ = MagicMock(
            return_value=False
        )

        mock_list_documents.side_effect = ValueError(
            "Vault 'Missing' not found. Available: Personal, Work"
        )

        params = ListDocumentsHandlerParams(
            db_manager=mock_db_manager,
            file_name="notes.md",
            vault_name="Missing",
        )
        result = _list_documents_handler(params)

        assert result == {
            "success": False,
            "error": "Vault 'Missing' not found. Available: Personal, Work",
        }
