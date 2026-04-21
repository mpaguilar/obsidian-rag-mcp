"""Tests for CLI chunk query options."""

from unittest.mock import MagicMock, Mock, patch

from click.testing import CliRunner

from obsidian_rag.cli import cli


class TestCLIChunkQuery:
    """Test cases for CLI chunk query options."""

    @patch("obsidian_rag.cli_commands._get_embedding_provider")
    @patch("obsidian_rag.cli_commands.DatabaseManager")
    def test_query_with_chunks_flag(self, mock_db_class, mock_get_provider):
        """Test query command with --chunks flag."""
        runner = CliRunner()

        # Setup mock for context manager
        mock_session = MagicMock()
        mock_session.__enter__ = Mock(return_value=mock_session)
        mock_session.__exit__ = Mock(return_value=False)

        mock_db = Mock()
        mock_db.get_session.return_value = mock_session
        mock_db_class.return_value = mock_db

        mock_provider = Mock()
        mock_provider.generate_embedding.return_value = [0.1] * 1536
        mock_get_provider.return_value = mock_provider

        with patch("obsidian_rag.cli_commands.query_chunks") as mock_query:
            mock_query.return_value = []
            result = runner.invoke(
                cli,
                [
                    "--verbose",
                    "query",
                    "test query",
                    "--chunks",
                ],
            )

            assert result.exit_code == 0
            mock_query.assert_called_once()

    @patch("obsidian_rag.cli_commands._get_embedding_provider")
    @patch("obsidian_rag.cli_commands.DatabaseManager")
    def test_query_with_rerank_flag(self, mock_db_class, mock_get_provider):
        """Test query command with --rerank flag."""
        runner = CliRunner()

        # Setup mock for context manager
        mock_session = MagicMock()
        mock_session.__enter__ = Mock(return_value=mock_session)
        mock_session.__exit__ = Mock(return_value=False)

        mock_db = Mock()
        mock_db.get_session.return_value = mock_session
        mock_db_class.return_value = mock_db

        mock_provider = Mock()
        mock_provider.generate_embedding.return_value = [0.1] * 1536
        mock_get_provider.return_value = mock_provider

        with patch("obsidian_rag.cli_commands.query_chunks") as mock_query:
            with patch("obsidian_rag.cli_commands.rerank_chunk_results") as mock_rerank:
                mock_query.return_value = []
                mock_rerank.return_value = []

                result = runner.invoke(
                    cli,
                    [
                        "query",
                        "test query",
                        "--chunks",
                        "--rerank",
                    ],
                )

                assert result.exit_code == 0

    @patch("obsidian_rag.cli_commands._get_embedding_provider")
    @patch("obsidian_rag.cli_commands.DatabaseManager")
    def test_query_chunks_with_vault_filter(self, mock_db_class, mock_get_provider):
        """Test query command with --chunks and --vault flags."""
        runner = CliRunner()

        # Setup mock for context manager
        mock_session = MagicMock()
        mock_session.__enter__ = Mock(return_value=mock_session)
        mock_session.__exit__ = Mock(return_value=False)

        mock_db = Mock()
        mock_db.get_session.return_value = mock_session
        mock_db_class.return_value = mock_db

        mock_provider = Mock()
        mock_provider.generate_embedding.return_value = [0.1] * 1536
        mock_get_provider.return_value = mock_provider

        with patch("obsidian_rag.cli_commands.query_chunks") as mock_query:
            mock_query.return_value = []
            result = runner.invoke(
                cli,
                [
                    "query",
                    "test query",
                    "--chunks",
                    "--vault",
                    "Test Vault",
                ],
            )

            assert result.exit_code == 0
            mock_query.assert_called_once()

    @patch("obsidian_rag.cli_commands._get_embedding_provider")
    @patch("obsidian_rag.cli_commands.DatabaseManager")
    def test_query_chunks_with_limit(self, mock_db_class, mock_get_provider):
        """Test query command with --chunks and --limit flags."""
        runner = CliRunner()

        # Setup mock for context manager
        mock_session = MagicMock()
        mock_session.__enter__ = Mock(return_value=mock_session)
        mock_session.__exit__ = Mock(return_value=False)

        mock_db = Mock()
        mock_db.get_session.return_value = mock_session
        mock_db_class.return_value = mock_db

        mock_provider = Mock()
        mock_provider.generate_embedding.return_value = [0.1] * 1536
        mock_get_provider.return_value = mock_provider

        with patch("obsidian_rag.cli_commands.query_chunks") as mock_query:
            mock_query.return_value = []
            result = runner.invoke(
                cli,
                [
                    "query",
                    "test query",
                    "--chunks",
                    "--limit",
                    "5",
                ],
            )

            assert result.exit_code == 0

    @patch("obsidian_rag.cli_commands._get_embedding_provider")
    @patch("obsidian_rag.cli_commands.DatabaseManager")
    def test_query_chunks_with_json_format(self, mock_db_class, mock_get_provider):
        """Test query command with --chunks and --format json flags."""
        runner = CliRunner()

        # Setup mock for context manager
        mock_session = MagicMock()
        mock_session.__enter__ = Mock(return_value=mock_session)
        mock_session.__exit__ = Mock(return_value=False)

        mock_db = Mock()
        mock_db.get_session.return_value = mock_session
        mock_db_class.return_value = mock_db

        mock_provider = Mock()
        mock_provider.generate_embedding.return_value = [0.1] * 1536
        mock_get_provider.return_value = mock_provider

        # Create a mock chunk result
        mock_result = Mock()
        mock_result.chunk_id = "chunk-1"
        mock_result.content = "Test chunk content"
        mock_result.document_name = "test.md"
        mock_result.document_path = "path/to/test.md"
        mock_result.vault_name = "Test Vault"
        mock_result.chunk_index = 0
        mock_result.total_chunks = 3
        mock_result.token_count = 512
        mock_result.chunk_type = "content"
        mock_result.similarity_score = 0.85
        mock_result.rerank_score = None

        with patch("obsidian_rag.cli_commands.query_chunks") as mock_query:
            mock_query.return_value = [mock_result]
            result = runner.invoke(
                cli,
                [
                    "query",
                    "test query",
                    "--chunks",
                    "--format",
                    "json",
                ],
            )

            assert result.exit_code == 0
            assert "chunk_id" in result.output

    @patch("obsidian_rag.cli_commands._get_embedding_provider")
    @patch("obsidian_rag.cli_commands.DatabaseManager")
    def test_query_chunks_no_results(self, mock_db_class, mock_get_provider):
        """Test query command with --chunks when no results found."""
        runner = CliRunner()

        # Setup mock for context manager
        mock_session = MagicMock()
        mock_session.__enter__ = Mock(return_value=mock_session)
        mock_session.__exit__ = Mock(return_value=False)

        mock_db = Mock()
        mock_db.get_session.return_value = mock_session
        mock_db_class.return_value = mock_db

        mock_provider = Mock()
        mock_provider.generate_embedding.return_value = [0.1] * 1536
        mock_get_provider.return_value = mock_provider

        with patch("obsidian_rag.cli_commands.query_chunks") as mock_query:
            mock_query.return_value = []
            result = runner.invoke(
                cli,
                [
                    "query",
                    "test query",
                    "--chunks",
                ],
            )

            assert result.exit_code == 0
            assert "No matching chunks found" in result.output

    @patch("obsidian_rag.cli_commands._get_embedding_provider")
    @patch("obsidian_rag.cli_commands.DatabaseManager")
    def test_query_chunks_with_rerank_results(self, mock_db_class, mock_get_provider):
        """Test query command with --chunks and --rerank with results."""
        runner = CliRunner()

        # Setup mock for context manager
        mock_session = MagicMock()
        mock_session.__enter__ = Mock(return_value=mock_session)
        mock_session.__exit__ = Mock(return_value=False)

        mock_db = Mock()
        mock_db.get_session.return_value = mock_session
        mock_db_class.return_value = mock_db

        mock_provider = Mock()
        mock_provider.generate_embedding.return_value = [0.1] * 1536
        mock_get_provider.return_value = mock_provider

        # Create mock chunk results
        mock_result1 = Mock()
        mock_result1.chunk_id = "chunk-1"
        mock_result1.content = "First chunk"
        mock_result1.document_name = "test.md"
        mock_result1.document_path = "path/to/test.md"
        mock_result1.vault_name = "Test Vault"
        mock_result1.chunk_index = 0
        mock_result1.total_chunks = 2
        mock_result1.token_count = 512
        mock_result1.chunk_type = "content"
        mock_result1.similarity_score = 0.75
        mock_result1.rerank_score = 0.95

        mock_result2 = Mock()
        mock_result2.chunk_id = "chunk-2"
        mock_result2.content = "Second chunk"
        mock_result2.document_name = "test.md"
        mock_result2.document_path = "path/to/test.md"
        mock_result2.vault_name = "Test Vault"
        mock_result2.chunk_index = 1
        mock_result2.total_chunks = 2
        mock_result2.token_count = 400
        mock_result2.chunk_type = "content"
        mock_result2.similarity_score = 0.80
        mock_result2.rerank_score = 0.90

        with patch("obsidian_rag.cli_commands.query_chunks") as mock_query:
            with patch("obsidian_rag.cli_commands.rerank_chunk_results") as mock_rerank:
                mock_query.return_value = [mock_result1, mock_result2]
                mock_rerank.return_value = [mock_result1, mock_result2]

                result = runner.invoke(
                    cli,
                    [
                        "query",
                        "test query",
                        "--chunks",
                        "--rerank",
                    ],
                )

                assert result.exit_code == 0
                mock_rerank.assert_called_once()

    @patch("obsidian_rag.cli_commands._get_embedding_provider")
    @patch("obsidian_rag.cli_commands.DatabaseManager")
    def test_query_chunks_table_format_with_all_fields(
        self, mock_db_class, mock_get_provider
    ):
        """Test query command with --chunks in table format including token_count and rerank_score."""
        runner = CliRunner()

        # Setup mock for context manager
        mock_session = MagicMock()
        mock_session.__enter__ = Mock(return_value=mock_session)
        mock_session.__exit__ = Mock(return_value=False)

        mock_db = Mock()
        mock_db.get_session.return_value = mock_session
        mock_db_class.return_value = mock_db

        mock_provider = Mock()
        mock_provider.generate_embedding.return_value = [0.1] * 1536
        mock_get_provider.return_value = mock_provider

        # Create mock chunk result with all optional fields present
        mock_result = Mock()
        mock_result.chunk_id = "chunk-1"
        mock_result.content = "Test chunk content"
        mock_result.document_name = "test.md"
        mock_result.document_path = "path/to/test.md"
        mock_result.vault_name = "Test Vault"
        mock_result.chunk_index = 0
        mock_result.total_chunks = 3
        mock_result.token_count = 512  # Present - should trigger branch 504->506
        mock_result.chunk_type = "content"
        mock_result.similarity_score = 0.85
        mock_result.rerank_score = 0.92  # Present - should trigger branch 507->509

        with patch("obsidian_rag.cli_commands.query_chunks") as mock_query:
            mock_query.return_value = [mock_result]
            result = runner.invoke(
                cli,
                [
                    "query",
                    "test query",
                    "--chunks",
                ],
            )

            assert result.exit_code == 0
            # Verify table format output includes optional fields
            assert "Tokens: 512" in result.output
            assert "Rerank: 0.920" in result.output


class TestFormatChunkResultsTable:
    """Direct unit tests for _format_chunk_results_table function."""

    def test_format_table_with_token_count_and_rerank_score(self):
        """Test table formatting includes token_count and rerank_score when present."""
        from obsidian_rag.cli_commands import _format_chunk_results_table

        # Create mock result with all optional fields
        mock_result = Mock()
        mock_result.chunk_id = "chunk-1"
        mock_result.content = "Test content"
        mock_result.document_name = "test.md"
        mock_result.document_path = "path/to/test.md"
        mock_result.vault_name = "Test Vault"
        mock_result.chunk_index = 0
        mock_result.total_chunks = 2
        mock_result.token_count = 512  # Should trigger line 504->506
        mock_result.chunk_type = "content"
        mock_result.similarity_score = 0.85
        mock_result.rerank_score = 0.92  # Should trigger line 507->509

        result = _format_chunk_results_table([mock_result])

        assert "Tokens: 512" in result
        assert "Rerank: 0.920" in result

    def test_format_table_without_optional_fields(self):
        """Test table formatting excludes optional fields when None."""
        from obsidian_rag.cli_commands import _format_chunk_results_table

        mock_result = Mock()
        mock_result.chunk_id = "chunk-1"
        mock_result.content = "Test content"
        mock_result.document_name = "test.md"
        mock_result.document_path = "path/to/test.md"
        mock_result.vault_name = "Test Vault"
        mock_result.chunk_index = 0
        mock_result.total_chunks = 2
        mock_result.token_count = None  # Should skip line 504->506
        mock_result.chunk_type = "content"
        mock_result.similarity_score = 0.85
        mock_result.rerank_score = None  # Should skip line 507->509

        result = _format_chunk_results_table([mock_result])

        assert "Tokens:" not in result
        assert "Rerank:" not in result
