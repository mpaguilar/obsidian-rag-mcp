"""File system scanner for discovering Obsidian markdown files."""

import hashlib
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

log = logging.getLogger(__name__)

# Maximum file size: 10MB
MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024

# Default batch size
DEFAULT_BATCH_SIZE = 100


class FileScanError(Exception):
    """Exception raised when file scanning fails."""

    pass


@dataclass
class FileInfo:
    """Information about a scanned file.

    Attributes:
        path: Absolute path to the file.
        name: Name of the file.
        content: File content as string.
        checksum: MD5 checksum of the content.
        created_at: Filesystem creation time.
        modified_at: Filesystem modification time.

    """

    path: Path
    name: str
    content: str
    checksum: str
    created_at: datetime
    modified_at: datetime


def calculate_checksum(content: str) -> str:
    """Calculate MD5 checksum of content.

    Args:
        content: The content to hash.

    Returns:
        MD5 checksum as a 32-character hex string.

    """
    _msg = "Calculating MD5 checksum"
    log.debug(_msg)
    return hashlib.md5(content.encode("utf-8")).hexdigest()  # noqa: S324


def _is_hidden_path(path: Path) -> bool:
    """Check if a path contains hidden components.

    Args:
        path: The path to check.

    Returns:
        True if any part of the path starts with '.'.

    """
    return any(part.startswith(".") for part in path.parts)


def _validate_vault_path(vault: Path) -> None:
    """Validate that the vault path exists and is a directory.

    Args:
        vault: Path to validate.

    Raises:
        FileScanError: If path doesn't exist or isn't a directory.

    """
    if not vault.exists():
        _msg = f"Vault path does not exist: {vault}"
        log.error(_msg)
        raise FileScanError(_msg)

    if not vault.is_dir():
        _msg = f"Vault path is not a directory: {vault}"
        log.error(_msg)
        raise FileScanError(_msg)


def scan_markdown_files(
    vault_path: str | Path,
    progress_callback: Callable[[int, int], None] | None = None,
) -> list[Path]:
    """Scan a directory recursively for markdown files.

    Args:
        vault_path: Path to the Obsidian vault/directory.
        progress_callback: Optional callback(current, total) for progress updates.

    Returns:
        List of paths to markdown files (.md extension).

    Raises:
        FileScanError: If the vault path doesn't exist or isn't a directory.

    Notes:
        Skips hidden files and directories (starting with .).
        Only includes files with .md extension.

    """
    _msg = f"Scanning for markdown files in: {vault_path}"
    log.debug(_msg)

    vault = Path(vault_path)
    _validate_vault_path(vault)

    markdown_files = [path for path in vault.rglob("*.md") if not _is_hidden_path(path)]

    _msg = f"Found {len(markdown_files)} markdown files"
    log.debug(_msg)

    # Report progress if callback provided
    if progress_callback:
        progress_callback(len(markdown_files), len(markdown_files))

    return markdown_files


def _check_file_size(path: Path) -> bool:
    """Check if file size is within limits.

    Args:
        path: Path to the file.

    Returns:
        True if file size is acceptable, False otherwise.

    """
    try:
        size = path.stat().st_size
        if size > MAX_FILE_SIZE_BYTES:
            _msg = f"File exceeds max size ({MAX_FILE_SIZE_BYTES} bytes): {path}"
            log.warning(_msg)
            return False
        return True
    except OSError as e:
        _msg = f"Cannot stat file {path}: {e}"
        log.warning(_msg)
        return False


def _read_file_content(path: Path) -> str | None:
    """Read file content with error handling.

    Args:
        path: Path to the file.

    Returns:
        File content or None if reading fails.

    """
    try:
        return path.read_text(encoding="utf-8")
    except PermissionError as e:
        _msg = f"Permission denied reading file {path}: {e}"
        log.warning(_msg)
    except UnicodeDecodeError as e:
        _msg = f"Cannot decode file {path}: {e}"
        log.warning(_msg)
    except Exception as e:
        _msg = f"Error reading file {path}: {e}"
        log.warning(_msg)
    return None


def read_file_with_metadata(file_path: str | Path) -> FileInfo | None:
    """Read a file and extract metadata including checksum.

    Args:
        file_path: Path to the file to read.

    Returns:
        FileInfo object or None if file cannot be read.

    Notes:
        Files larger than 10MB are skipped with a warning.
        Permission errors are logged and the file is skipped.

    """
    _msg = f"Reading file: {file_path}"
    log.debug(_msg)

    path = Path(file_path)

    # Check file size
    if not _check_file_size(path):
        return None

    # Read file content
    content = _read_file_content(path)
    if content is None:
        return None

    # Calculate checksum
    checksum = calculate_checksum(content)

    # Get file timestamps
    stat = path.stat()
    created_at = datetime.fromtimestamp(stat.st_ctime)  # noqa: DTZ006
    modified_at = datetime.fromtimestamp(stat.st_mtime)  # noqa: DTZ006

    file_info = FileInfo(
        path=path,
        name=path.name,
        content=content,
        checksum=checksum,
        created_at=created_at,
        modified_at=modified_at,
    )

    _msg = f"Successfully read file: {path.name}, checksum={checksum[:8]}..."
    log.debug(_msg)
    return file_info


def process_files_in_batches(
    files: list[Path],
    batch_size: int = DEFAULT_BATCH_SIZE,
    progress_interval: int = 10,
    progress_callback: Callable[[int, int, int, int], None] | None = None,
) -> list[FileInfo]:
    """Process files in batches with progress reporting.

    Args:
        files: List of file paths to process.
        batch_size: Number of files to process in each batch.
        progress_interval: Report progress every N files.
        progress_callback: Optional callback(current, total, successes, errors).

    Returns:
        List of successfully read FileInfo objects.

    Notes:
        Errors for individual files are logged but don't stop processing.

    """
    _msg = f"Processing {len(files)} files in batches of {batch_size}"
    log.debug(_msg)

    results = []
    errors = 0

    for i, file_path in enumerate(files):
        # Process file
        file_info = read_file_with_metadata(file_path)
        if file_info:
            results.append(file_info)
        else:
            errors += 1

        # Report progress
        if progress_callback and (i + 1) % progress_interval == 0:
            progress_callback(i + 1, len(files), len(results), errors)

    # Final progress report
    if progress_callback:
        progress_callback(len(files), len(files), len(results), errors)

    _msg = f"Processed {len(files)} files: {len(results)} successful, {errors} errors"
    log.debug(_msg)
    return results
