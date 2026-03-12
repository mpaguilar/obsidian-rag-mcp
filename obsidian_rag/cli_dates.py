"""Date validation utilities for CLI commands."""

import logging
import sys
from datetime import UTC, date, datetime

import click

log = logging.getLogger(__name__)


def parse_cli_date(date_str: str | None) -> date | None:
    """Parse a CLI date string in YYYY-MM-DD format.

    Args:
        date_str: Date string from CLI option, or None.

    Returns:
        Parsed date object, or None if input is None.

    Raises:
        SystemExit: If date format is invalid (exits with code 1).

    Notes:
        Prints error message to stderr and exits on invalid format.
        Uses datetime.strptime with "%Y-%m-%d" format per CONVENTIONS.md.

    """
    _msg = "parse_cli_date starting"
    log.debug(_msg)

    if date_str is None:
        _msg = "parse_cli_date returning: None (input was None)"
        log.debug(_msg)
        return None

    try:
        parsed = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=UTC)
        result = parsed.date()
        _msg = f"parse_cli_date returning: {result}"
        log.debug(_msg)
        return result
    except ValueError:
        _msg = f"Error: Invalid date format '{date_str}'. Use YYYY-MM-DD."
        log.error(_msg)
        click.echo(_msg, err=True)
        sys.exit(1)
