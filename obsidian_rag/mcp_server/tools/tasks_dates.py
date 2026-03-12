"""Date parsing utilities for task query tools."""

import logging
from datetime import date

log = logging.getLogger(__name__)


def parse_iso_date(date_str: str | None) -> date | None:
    """Parse an ISO 8601 date string (YYYY-MM-DD).

    Args:
        date_str: Date string in YYYY-MM-DD format, or None.

    Returns:
        Parsed date object, or None if input is None or invalid.

    Notes:
        Logs a warning if the date string is invalid.

    """
    _msg = "parse_iso_date starting"
    log.debug(_msg)

    if date_str is None:
        _msg = "parse_iso_date returning: None (input was None)"
        log.debug(_msg)
        return None

    try:
        result = date.fromisoformat(date_str)
        _msg = f"parse_iso_date returning: {result}"
        log.debug(_msg)
        return result
    except ValueError:
        _msg = f"Invalid date format: {date_str}. Expected YYYY-MM-DD."
        log.warning(_msg)
        _msg = "parse_iso_date returning: None (invalid format)"
        log.debug(_msg)
        return None
