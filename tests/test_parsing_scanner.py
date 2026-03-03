"""Tests for parsing scanner module."""

import hashlib
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from obsidian_rag.parsing.scanner import (
    FileInfo,
    FileScanError,
    calculate_checksum,
    process_files_in_batches,
    read_file_with_metadata,
    scan_markdown_files,
    _check_file_size,
    _is_hidden_path,
    _read_file_content,
    _validate_vault_path,
)


class TestIsHiddenPath:
    """Test cases for _is_hidden_path function."""

    def test_detects_hidden_file(self):
        """Test detecting hidden file."""
        path = Path("/home/user/.hidden.md")
        assert _is_hidden_path(path) is True

    def test_detects_hidden_directory(self):
        """Test detecting file in hidden directory."""
        path = Path("/home/user/.obsidian/notes.md")
        assert _is_hidden_path(path) is True

    def test_allows_visible_file(self):
        """Test allowing visible file."""
        path = Path("/home/user/documents/notes.md")
        assert _is_hidden_path(path) is False


class TestValidateVaultPath:
    """Test cases for _validate_vault_path function."""

    def test_valid_directory(self, tmp_path):
        """Test validating existing directory."""
        # Should not raise
        _validate_vault_path(tmp_path)

    def test_nonexistent_path(self, tmp_path):
        """Test validating non-existent path."""
        nonexistent = tmp_path / "does_not_exist"

        with pytest.raises(FileScanError, match="does not exist"):
            _validate_vault_path(nonexistent)

    def test_file_instead_of_directory(self, tmp_path):
        """Test validating a file instead of directory."""
        file_path = tmp_path / "file.md"
        file_path.write_text("content")

        with pytest.raises(FileScanError, match="not a directory"):
            _validate_vault_path(file_path)


class TestCalculateChecksum:
    """Test cases for calculate_checksum function."""

    def test_calculates_md5(self):
        """Test calculating MD5 checksum."""
        content = "Hello, World!"
        result = calculate_checksum(content)

        expected = hashlib.md5(content.encode("utf-8")).hexdigest()
        assert result == expected
        assert len(result) == 32  # MD5 hex is 32 chars

    def test_different_content_different_checksum(self):
        """Test that different content produces different checksums."""
        checksum1 = calculate_checksum("content1")
        checksum2 = calculate_checksum("content2")

        assert checksum1 != checksum2


class TestCheckFileSize:
    """Test cases for _check_file_size function."""

    def test_allows_small_file(self, tmp_path):
        """Test allowing file under size limit."""
        file_path = tmp_path / "small.md"
        file_path.write_text("Small content")

        assert _check_file_size(file_path) is True

    def test_rejects_large_file(self, tmp_path):
        """Test rejecting file over size limit."""
        file_path = tmp_path / "large.md"
        # Create 11MB file
        file_path.write_bytes(b"x" * (11 * 1024 * 1024))

        assert _check_file_size(file_path) is False

    def test_handles_os_error(self, tmp_path):
        """Test handling OS error when checking file size."""
        nonexistent = tmp_path / "does_not_exist"

        assert _check_file_size(nonexistent) is False


class TestReadFileContent:
    """Test cases for _read_file_content function."""

    def test_reads_utf8_file(self, tmp_path):
        """Test reading UTF-8 encoded file."""
        file_path = tmp_path / "test.md"
        file_path.write_text("Hello, World!", encoding="utf-8")

        result = _read_file_content(file_path)
        assert result == "Hello, World!"

    def test_handles_permission_error(self, tmp_path):
        """Test handling permission error."""
        file_path = tmp_path / "test.md"
        file_path.write_text("content")

        with patch("pathlib.Path.read_text", side_effect=PermissionError("No access")):
            result = _read_file_content(file_path)
            assert result is None

    def test_handles_unicode_decode_error(self, tmp_path):
        """Test handling unicode decode error."""
        file_path = tmp_path / "test.md"
        file_path.write_bytes(b"\xff\xfe")  # Invalid UTF-8

        result = _read_file_content(file_path)
        assert result is None

    def test_handles_generic_exception(self, tmp_path):
        """Test handling generic exception when reading file."""
        file_path = tmp_path / "test.md"
        file_path.write_text("content")

        with patch("pathlib.Path.read_text", side_effect=OSError("Generic OS error")):
            result = _read_file_content(file_path)
            assert result is None


class TestReadFileWithMetadata:
    """Test cases for read_file_with_metadata function."""

    def test_reads_file_successfully(self, tmp_path):
        """Test successfully reading file with metadata."""
        file_path = tmp_path / "test.md"
        file_path.write_text("Test content")

        result = read_file_with_metadata(file_path)

        assert result is not None
        assert result.name == "test.md"
        assert result.content == "Test content"
        assert result.checksum == calculate_checksum("Test content")
        assert isinstance(result.created_at, datetime)
        assert isinstance(result.modified_at, datetime)

    def test_returns_none_for_large_file(self, tmp_path):
        """Test returning None for file over size limit."""
        file_path = tmp_path / "large.md"
        file_path.write_bytes(b"x" * (11 * 1024 * 1024))

        result = read_file_with_metadata(file_path)
        assert result is None

    def test_returns_none_for_unreadable_file(self, tmp_path):
        """Test returning None for unreadable file."""
        file_path = tmp_path / "test.md"
        file_path.write_bytes(b"\xff\xfe")  # Invalid UTF-8

        result = read_file_with_metadata(file_path)
        assert result is None


class TestScanMarkdownFiles:
    """Test cases for scan_markdown_files function."""

    def test_finds_markdown_files(self, tmp_path):
        """Test finding markdown files in directory."""
        # Create some files
        (tmp_path / "file1.md").write_text("content")
        (tmp_path / "file2.md").write_text("content")
        (tmp_path / "not_md.txt").write_text("content")

        result = scan_markdown_files(tmp_path)

        assert len(result) == 2
        assert all(f.suffix == ".md" for f in result)

    def test_skips_hidden_files(self, tmp_path):
        """Test skipping hidden files and directories."""
        # Create visible and hidden files
        (tmp_path / "visible.md").write_text("content")
        (tmp_path / ".hidden.md").write_text("content")

        result = scan_markdown_files(tmp_path)

        assert len(result) == 1
        assert result[0].name == "visible.md"

    def test_scans_subdirectories(self, tmp_path):
        """Test scanning subdirectories recursively."""
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        (subdir / "nested.md").write_text("content")
        (tmp_path / "root.md").write_text("content")

        result = scan_markdown_files(tmp_path)

        assert len(result) == 2

    def test_calls_progress_callback(self, tmp_path):
        """Test calling progress callback."""
        (tmp_path / "file.md").write_text("content")

        callback = Mock()
        scan_markdown_files(tmp_path, progress_callback=callback)

        callback.assert_called_once_with(1, 1)

    def test_raises_error_for_nonexistent_path(self, tmp_path):
        """Test raising error for non-existent path."""
        nonexistent = tmp_path / "does_not_exist"

        with pytest.raises(FileScanError):
            scan_markdown_files(nonexistent)


class TestProcessFilesInBatches:
    """Test cases for process_files_in_batches function."""

    def test_processes_all_files(self, tmp_path):
        """Test processing all files."""
        # Create test files
        file1 = tmp_path / "file1.md"
        file1.write_text("content1")
        file2 = tmp_path / "file2.md"
        file2.write_text("content2")

        files = [file1, file2]
        result = process_files_in_batches(files)

        assert len(result) == 2
        assert result[0].content == "content1"
        assert result[1].content == "content2"

    def test_skips_unreadable_files(self, tmp_path):
        """Test skipping files that can't be read."""
        file1 = tmp_path / "file1.md"
        file1.write_text("content")
        file2 = tmp_path / "file2.md"
        file2.write_bytes(b"\xff\xfe")  # Invalid UTF-8

        files = [file1, file2]
        result = process_files_in_batches(files)

        assert len(result) == 1
        assert result[0].name == "file1.md"

    def test_calls_progress_callback(self, tmp_path):
        """Test calling progress callback."""
        file1 = tmp_path / "file1.md"
        file1.write_text("content1")
        file2 = tmp_path / "file2.md"
        file2.write_text("content2")

        files = [file1, file2]
        callback = Mock()
        process_files_in_batches(files, progress_interval=1, progress_callback=callback)

        # Callback should be called for each file
        assert callback.call_count >= 2
