"""Tests for tasks_dates module."""

from datetime import date
from unittest.mock import patch

import pytest

from obsidian_rag.mcp_server.tools.tasks_dates import parse_iso_date


class TestParseIsoDate:
    """Tests for parse_iso_date function."""

    def test_valid_date(self):
        """Test parsing a valid ISO date string."""
        result = parse_iso_date("2026-03-15")

        assert result == date(2026, 3, 15)

    def test_none_input(self):
        """Test that None returns None."""
        result = parse_iso_date(None)

        assert result is None

    def test_invalid_format(self):
        """Test that invalid format returns None and logs warning."""
        with patch("obsidian_rag.mcp_server.tools.tasks_dates.log") as mock_log:
            result = parse_iso_date("03/15/2026")

        assert result is None
        mock_log.warning.assert_called_once()
        assert "Invalid date format" in mock_log.warning.call_args[0][0]

    def test_invalid_date_value(self):
        """Test that invalid date value returns None."""
        with patch("obsidian_rag.mcp_server.tools.tasks_dates.log") as mock_log:
            result = parse_iso_date("2026-02-30")  # Feb 30 doesn't exist

        assert result is None
        mock_log.warning.assert_called_once()

    def test_empty_string(self):
        """Test that empty string returns None."""
        with patch("obsidian_rag.mcp_server.tools.tasks_dates.log") as mock_log:
            result = parse_iso_date("")

        assert result is None
        mock_log.warning.assert_called_once()
