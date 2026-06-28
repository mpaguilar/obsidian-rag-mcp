"""FrontMatter extraction from markdown documents."""

import logging
import re
from datetime import date, datetime
from typing import Any

import yaml

log = logging.getLogger(__name__)


def _serialize_for_json(obj: object) -> object:
    """Serialize an object for JSON storage.

    Converts date and datetime objects to ISO format strings.
    Recursively handles dictionaries and lists.

    Args:
        obj: The object to serialize.

    Returns:
        JSON-serializable version of the object.

    """
    if isinstance(obj, (date, datetime)):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {k: _serialize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_serialize_for_json(item) for item in obj]
    return obj


def _serialize_dict_for_json(obj: dict[str, Any]) -> dict[str, Any]:
    """Serialize a dictionary for JSON storage.

    Converts date and datetime objects to ISO format strings.
    Recursively handles dictionaries and lists.

    Args:
        obj: The dictionary to serialize.

    Returns:
        JSON-serializable dictionary.

    """
    result: dict[str, Any] = {}
    for k, v in obj.items():
        if isinstance(v, (date, datetime)):
            result[k] = v.isoformat()
        elif isinstance(v, dict):
            result[k] = _serialize_dict_for_json(v)
        elif isinstance(v, list):
            result[k] = [_serialize_for_json(item) for item in v]
        else:
            result[k] = v
    return result


# Pattern to match YAML frontmatter: --- at start of file
FRONTMATTER_PATTERN = re.compile(
    r"^---\s*\n(.*?)\n---\s*\n",
    re.DOTALL,
)


def _normalize_indentation_tabs(yaml_content: str) -> str:
    """Convert indentation-position tab characters to spaces in YAML frontmatter.

    Only tabs in the leading whitespace run (indentation) are converted.
    Tabs inside quoted string values or mid-line positions are untouched.
    Each indentation tab is replaced with 2 spaces, matching the project convention.

    Args:
        yaml_content: The raw YAML frontmatter text to normalize.

    Returns:
        The normalized YAML string with indentation tabs replaced by spaces.
        Returns the input unchanged if it contains no indentation tabs.

    """
    _msg = "_normalize_indentation_tabs starting"
    log.debug(_msg)
    if not yaml_content:
        _msg = "_normalize_indentation_tabs returning empty input unchanged"
        log.debug(_msg)
        return yaml_content
    lines = yaml_content.split("\n")
    normalized_lines: list[str] = []
    for line in lines:
        leading_len = len(line) - len(line.lstrip())
        if leading_len == 0:
            normalized_lines.append(line)
            continue
        leading_whitespace = line[:leading_len]
        rest_of_line = line[leading_len:]
        new_leading = leading_whitespace.replace("\t", "  ")
        normalized_lines.append(new_leading + rest_of_line)
    result = "\n".join(normalized_lines)
    _msg = "_normalize_indentation_tabs returning"
    log.debug(_msg)
    return result


def _parse_yaml_frontmatter(yaml_content: str) -> dict[str, Any]:
    """Parse YAML frontmatter content into a dictionary.

    Args:
        yaml_content: The raw YAML frontmatter text.

    Returns:
        Parsed dictionary, or empty dict if parsing fails.

    Notes:
        If YAML is corrupted or cannot be parsed, an empty dict
        is returned and a warning is logged.

    """
    try:
        loaded = yaml.safe_load(yaml_content)
        if loaded is None:
            return {}
        if not isinstance(loaded, dict):
            _msg = f"FrontMatter is not a dictionary, got {type(loaded)}"
            log.warning(_msg)
            return {}
        return loaded
    except yaml.YAMLError as e:
        _msg = f"Failed to parse FrontMatter YAML: {e}"
        log.warning(_msg)
        return {}


def extract_frontmatter(content: str) -> tuple[dict[str, Any], str]:
    """Extract YAML frontmatter from markdown content.

    Args:
        content: The full markdown document content.

    Returns:
        Tuple of (frontmatter_dict, remaining_content).
        frontmatter_dict is empty if no frontmatter found.

    Notes:
        If FrontMatter is corrupted or cannot be parsed, an empty dict
        is returned and a warning is logged.

    """
    _msg = "Extracting FrontMatter from content"
    log.debug(_msg)

    match = FRONTMATTER_PATTERN.match(content)
    if not match:
        _msg = "No FrontMatter found in content"
        log.debug(_msg)
        return {}, content

    yaml_content = match.group(1)
    remaining_content = content[match.end() :]

    normalized = _normalize_indentation_tabs(yaml_content)
    if normalized != yaml_content:
        _msg = "Frontmatter indentation tabs normalized to spaces for YAML parsing"
        log.debug(_msg)
    yaml_content = normalized

    frontmatter = _parse_yaml_frontmatter(yaml_content)

    _msg = f"Successfully extracted FrontMatter with {len(frontmatter)} keys"
    log.debug(_msg)
    return frontmatter, remaining_content


def _deduplicate_tags(tag_list: list[str]) -> list[str]:
    """Remove empty tags and duplicates while preserving order.

    Args:
        tag_list: List of tag strings.

    Returns:
        Deduplicated list of tags.

    """
    seen = set()
    normalized = []
    for tag in tag_list:
        if tag and tag not in seen:
            seen.add(tag)
            normalized.append(tag)
    return normalized


def normalize_tags(tags: object) -> list[str] | None:
    """Normalize tags to a list of strings.

    Args:
        tags: Tags as string, list of strings, or None.

    Returns:
        List of normalized tags or None if input is None.

    Examples:
        >>> normalize_tags("tag1, tag2")
        ['tag1', 'tag2']
        >>> normalize_tags(["tag1", "tag2"])
        ['tag1', 'tag2']
        >>> normalize_tags("tag1")
        ['tag1']

    """
    _msg = "Normalizing tags"
    log.debug(_msg)

    if tags is None:
        return None

    if isinstance(tags, str):
        # Split by comma or space
        tag_list = [t.strip() for t in tags.replace(",", " ").split()]
    elif isinstance(tags, list):
        tag_list = [str(t).strip() for t in tags]
    else:
        tag_list = [str(tags).strip()]

    normalized = _deduplicate_tags(tag_list)

    _msg = f"Normalized {len(tag_list)} tags to {len(normalized)} unique tags"
    log.debug(_msg)
    return normalized if normalized else None


def parse_frontmatter(
    content: str,
) -> tuple[list[str] | None, dict[str, Any], str]:
    """Parse FrontMatter and extract structured data.

    Args:
        content: The full markdown document content.

    Returns:
        Tuple of (tags, metadata, remaining_content).
        - tags: Normalized list of tags (or None)
        - metadata: All FrontMatter fields as dict (includes 'kind' if present)
        - remaining_content: Content without FrontMatter

    Notes:
        The 'kind' field is now included in the metadata dict.
        The 'tags' field is normalized to a list and deduplicated.
        All other fields are stored in metadata.

    """
    _msg = "Parsing FrontMatter content"
    log.debug(_msg)

    frontmatter, remaining = extract_frontmatter(content)

    if not frontmatter:
        _msg = "No FrontMatter found, returning empty values"
        log.debug(_msg)
        return None, {}, remaining

    # Extract and normalize tags (removed from frontmatter)
    tags = normalize_tags(frontmatter.get("tags"))

    # Everything else goes into metadata (now includes 'kind')
    metadata = {k: v for k, v in frontmatter.items() if k != "tags"}

    # Serialize metadata to handle date/datetime objects from YAML parsing
    metadata = _serialize_dict_for_json(metadata)

    _msg = f"Parsed FrontMatter: tags_count={len(tags) if tags else 0}, metadata_keys={len(metadata)}"
    log.debug(_msg)
    return tags, metadata, remaining
