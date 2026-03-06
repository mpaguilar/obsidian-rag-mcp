"""Task parsing from markdown content."""

import logging
import re
from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import Any

from dateutil.rrule import rrulestr

from obsidian_rag.database.models import TaskPriority, TaskStatus

log = logging.getLogger(__name__)


def _serialize_custom_metadata(
    metadata: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Serialize custom metadata values for JSON storage.

    Converts date and datetime objects to ISO format strings.

    Args:
        metadata: Dictionary of custom metadata values.

    Returns:
        Serialized metadata dictionary or None if input is None.

    """
    _msg = "_serialize_custom_metadata starting"
    log.debug(_msg)
    if metadata is None:
        _msg = "_serialize_custom_metadata returning"
        log.debug(_msg)
        return None

    result: dict[str, Any] = {}
    for key, value in metadata.items():
        if isinstance(value, (date, datetime)):
            result[key] = value.isoformat()
        else:
            result[key] = value
    _msg = "_serialize_custom_metadata returning"
    log.debug(_msg)
    return result


# Pattern to match task lines: - [ ] task text or - [x] task text
TASK_PATTERN = re.compile(
    r"^(\s*)- \[([ x/-])\]\s*(.*?)$",
    re.MULTILINE,
)

# Pattern to match inline tags: #tag (alphanumeric and hyphens/underscores)
TAG_PATTERN = re.compile(r"#([a-zA-Z0-9_-]+)")

# Pattern to match key:: value metadata
METADATA_PATTERN = re.compile(
    r"\[?([a-zA-Z_]+)::\s*([^\]]+)\]?",
)


@dataclass
class ParsedTask:
    """Represents a parsed task with all metadata.

    Attributes:
        status: Task status (not_completed, completed, in_progress, cancelled).
        description: Clean task description without metadata.
        tags: List of tags extracted from the task.
        repeat: Recurrence pattern string.
        scheduled: Scheduled date for the task.
        due: Due date for the task.
        completion: Completion date for the task.
        priority: Task priority level.
        custom_metadata: Additional key-value metadata.
        raw_text: Original task line.

    """

    status: str
    description: str
    tags: list[str] | None = None
    repeat: str | None = None
    scheduled: datetime | None = None
    due: datetime | None = None
    completion: datetime | None = None
    priority: str = TaskPriority.NORMAL.value
    custom_metadata: dict[str, Any] | None = None
    raw_text: str = ""


def _map_checkbox_status(checkbox: str) -> str:
    """Map checkbox character to task status.

    Args:
        checkbox: The checkbox character ([ ], [x], [/], [-]).

    Returns:
        The corresponding TaskStatus value.

    """
    _msg = "_map_checkbox_status starting"
    log.debug(_msg)
    status_map = {
        " ": TaskStatus.NOT_COMPLETED.value,
        "x": TaskStatus.COMPLETED.value,
        "/": TaskStatus.IN_PROGRESS.value,
        "-": TaskStatus.CANCELLED.value,
    }
    result = status_map.get(checkbox, TaskStatus.NOT_COMPLETED.value)
    _msg = "_map_checkbox_status returning"
    log.debug(_msg)
    return result


def _map_priority(value: str) -> str:
    """Map priority string to TaskPriority value.

    Args:
        value: The priority string from the task.

    Returns:
        The corresponding TaskPriority value.

    """
    _msg = "_map_priority starting"
    log.debug(_msg)
    priority_map = {
        "highest": TaskPriority.HIGHEST.value,
        "high": TaskPriority.HIGH.value,
        "normal": TaskPriority.NORMAL.value,
        "low": TaskPriority.LOW.value,
        "lowest": TaskPriority.LOWEST.value,
    }
    normalized = value.lower().strip()
    result = priority_map.get(normalized, TaskPriority.NORMAL.value)
    _msg = "_map_priority returning"
    log.debug(_msg)
    return result


def _parse_date(date_str: str) -> datetime | None:
    """Parse a date string into a datetime object.

    Args:
        date_str: The date string to parse.

    Returns:
        datetime object or None if parsing fails.

    Notes:
        Tries multiple date formats:
        - YYYY-MM-DD
        - YYYY/MM/DD
        - DD-MM-YYYY
        - DD/MM/YYYY

    """
    _msg = "_parse_date starting"
    log.debug(_msg)
    date_str = date_str.strip()
    formats = [
        "%Y-%m-%d",
        "%Y/%m/%d",
        "%d-%m-%Y",
        "%d/%m/%Y",
    ]

    for fmt in formats:
        try:
            result = datetime.strptime(date_str, fmt).replace(tzinfo=UTC)
        except ValueError:
            continue
        else:
            _msg = "_parse_date returning"
            log.debug(_msg)
            return result

    _msg = "_parse_date returning"
    log.debug(_msg)
    return None


def _obsidian_to_rrule(pattern: str) -> str | None:
    """Convert Obsidian recurrence pattern to rrule format.

    Args:
        pattern: The recurrence pattern (e.g., "every day", "every 2 weeks").

    Returns:
        rrule string or None if conversion fails.

    Examples:
        >>> _obsidian_to_rrule("every day")
        'FREQ=DAILY'
        >>> _obsidian_to_rrule("every 2 weeks")
        'FREQ=WEEKLY;INTERVAL=2'

    """
    _msg = "_obsidian_to_rrule starting"
    log.debug(_msg)
    pattern = pattern.lower().strip()
    result = None

    # Simple mappings for common patterns
    mappings = {
        "every day": "FREQ=DAILY",
        "daily": "FREQ=DAILY",
        "every week": "FREQ=WEEKLY",
        "weekly": "FREQ=WEEKLY",
        "every month": "FREQ=MONTHLY",
        "monthly": "FREQ=MONTHLY",
        "every year": "FREQ=YEARLY",
        "yearly": "FREQ=YEARLY",
        "annually": "FREQ=YEARLY",
    }

    if pattern in mappings:
        result = mappings[pattern]
    else:
        # Try to parse "every N units" patterns
        match = re.match(r"every\s+(\d+)\s*(day|week|month|year)s?", pattern)
        if match:
            count = match.group(1)
            unit = match.group(2).upper()
            freq_map = {
                "DAY": "DAILY",
                "WEEK": "WEEKLY",
                "MONTH": "MONTHLY",
                "YEAR": "YEARLY",
            }
            freq = freq_map.get(unit, unit + "LY")
            result = f"FREQ={freq};INTERVAL={count}"

    _msg = "_obsidian_to_rrule returning"
    log.debug(_msg)
    return result


def _validate_rrule(pattern: str) -> bool:
    """Validate if a string is a valid rrule.

    Args:
        pattern: The recurrence pattern to validate.

    Returns:
        True if valid, False otherwise.

    """
    _msg = "_validate_rrule starting"
    log.debug(_msg)
    result = False
    try:
        rrulestr(pattern)
        result = True
    except ValueError:
        result = False
    _msg = "_validate_rrule returning"
    log.debug(_msg)
    return result


def _process_repeat_value(value: str, standard_fields: dict[str, Any]) -> None:
    """Process repeat metadata value."""
    _msg = "_process_repeat_value starting"
    log.debug(_msg)
    rrule_pattern = _obsidian_to_rrule(value)
    if rrule_pattern and _validate_rrule(rrule_pattern):
        standard_fields["repeat"] = rrule_pattern
    else:
        standard_fields["repeat"] = value
    _msg = "_process_repeat_value returning"
    log.debug(_msg)


def _process_standard_field(
    key_lower: str,
    value: str,
    standard_fields: dict[str, Any],
) -> bool:
    """Process standard metadata fields. Returns True if processed."""
    _msg = "_process_standard_field starting"
    log.debug(_msg)
    date_fields = {"scheduled", "due", "completion"}
    result = False

    if key_lower in date_fields:
        standard_fields[key_lower] = _parse_date(value)
        result = True
    elif key_lower == "priority":
        standard_fields["priority"] = _map_priority(value)
        result = True
    elif key_lower == "repeat":
        _process_repeat_value(value, standard_fields)
        result = True
    _msg = "_process_standard_field returning"
    log.debug(_msg)
    return result


def _process_metadata_key(
    key: str,
    value: str,
    standard_fields: dict[str, Any],
    custom_metadata: dict[str, Any],
) -> None:
    """Process a single metadata key-value pair."""
    _msg = "_process_metadata_key starting"
    log.debug(_msg)
    key_lower = key.lower()

    if not _process_standard_field(key_lower, value, standard_fields):
        custom_metadata[key] = value
    _msg = "_process_metadata_key returning"
    log.debug(_msg)


def _extract_task_metadata(task_text: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Extract metadata from task text.

    Args:
        task_text: The raw task text with metadata.

    Returns:
        Tuple of (standard_fields, custom_metadata).
        standard_fields contains: scheduled, due, completion, priority, repeat

    """
    _msg = "_extract_task_metadata starting"
    log.debug(_msg)
    custom_metadata: dict[str, Any] = {}
    standard_fields: dict[str, Any] = {
        "scheduled": None,
        "due": None,
        "completion": None,
        "priority": TaskPriority.NORMAL.value,
        "repeat": None,
    }

    for key, value in METADATA_PATTERN.findall(task_text):
        _process_metadata_key(
            key.strip(),
            value.strip(),
            standard_fields,
            custom_metadata,
        )

    _msg = "_extract_task_metadata returning"
    log.debug(_msg)
    return standard_fields, custom_metadata


def _clean_task_description(task_text: str) -> str:
    """Remove metadata and tags from task text.

    Args:
        task_text: The raw task text.

    Returns:
        Clean description without metadata.

    """
    _msg = "_clean_task_description starting"
    log.debug(_msg)
    description = METADATA_PATTERN.sub("", task_text)
    description = TAG_PATTERN.sub("", description)
    result = " ".join(description.split())
    _msg = "_clean_task_description returning"
    log.debug(_msg)
    return result


def parse_task_line(line: str) -> ParsedTask | None:
    """Parse a single task line and extract metadata.

    Args:
        line: The task line from markdown.

    Returns:
        ParsedTask object or None if not a valid task line.

    """
    _msg = f"Parsing task line: {line[:50]}"
    log.debug(_msg)

    match = TASK_PATTERN.match(line)
    if not match:
        return None

    checkbox = match.group(2)
    task_text = match.group(3)

    status = _map_checkbox_status(checkbox)

    # Extract inline tags
    tags_match = TAG_PATTERN.findall(task_text)
    tags: list[str] | None = list(tags_match) if tags_match else None

    # Extract metadata
    standard_fields, custom_metadata = _extract_task_metadata(task_text)

    # Clean description
    description = _clean_task_description(task_text)

    parsed = ParsedTask(
        status=status,
        description=description,
        tags=tags,
        repeat=standard_fields["repeat"],
        scheduled=standard_fields["scheduled"],
        due=standard_fields["due"],
        completion=standard_fields["completion"],
        priority=standard_fields["priority"],
        custom_metadata=_serialize_custom_metadata(custom_metadata),
        raw_text=line.strip(),
    )

    _msg = f"Successfully parsed task: status={status}, priority={standard_fields['priority']}"
    log.debug(_msg)
    return parsed


def parse_tasks_from_content(content: str) -> list[tuple[int, ParsedTask]]:
    """Parse all tasks from markdown content.

    Args:
        content: The markdown content to parse.

    Returns:
        List of tuples (line_number, ParsedTask).

    Notes:
        Maximum of 10,000 tasks per document. Additional tasks are skipped.

    """
    _msg = "Parsing tasks from content"
    log.debug(_msg)

    tasks: list[tuple[int, ParsedTask]] = []
    lines = content.split("\n")
    max_tasks = 10000

    for line_number, line in enumerate(lines, start=1):
        if len(tasks) >= max_tasks:
            _msg = f"Reached maximum task limit ({max_tasks}), skipping remaining tasks"
            log.warning(_msg)
            break

        task = parse_task_line(line)
        if task:
            tasks.append((line_number, task))

    _msg = f"Found {len(tasks)} tasks in content"
    log.debug(_msg)
    return tasks
