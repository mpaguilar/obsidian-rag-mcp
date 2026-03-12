"""Tests for cli_dates module."""

from datetime import date
from unittest.mock import MagicMock, patch

import pytest

from obsidian_rag.cli_dates import parse_cli_date


class TestParseCliDate:
    """Tests for parse_cli_date function."""

    def test_valid_date(self):
        """Test parsing a valid date string."""
        result = parse_cli_date("2026-03-15")

        assert result == date(2026, 3, 15)

    def test_none_input(self):
        """Test that None returns None."""
        result = parse_cli_date(None)

        assert result is None

    def test_invalid_format_exits(self):
        """Test that invalid format exits with code 1."""
        with patch("obsidian_rag.cli_dates.click.echo") as mock_echo:
            with pytest.raises(SystemExit) as exc_info:
                parse_cli_date("03/15/2026")

        assert exc_info.value.code == 1
        mock_echo.assert_called_once()
        assert "Invalid date format" in mock_echo.call_args[0][0]

    def test_invalid_date_value_exits(self):
        """Test that invalid date value exits with code 1."""
        with patch("obsidian_rag.cli_dates.click.echo") as mock_echo:
            with pytest.raises(SystemExit) as exc_info:
                parse_cli_date("2026-02-30")  # Feb 30 doesn't exist

        assert exc_info.value.code == 1

    def test_empty_string_exits(self):
        """Test that empty string exits with code 1."""
        with patch("obsidian_rag.cli_dates.click.echo"):
            with pytest.raises(SystemExit) as exc_info:
                parse_cli_date("")

        assert exc_info.value.code == 1
