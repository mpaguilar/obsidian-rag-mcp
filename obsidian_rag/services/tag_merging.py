"""Tag merging utilities for ingestion."""

import logging

log = logging.getLogger(__name__)


def _filter_tags(tags: list[str] | None) -> list[str] | None:
    """Filter out empty strings from a tag list.

    Args:
        tags: Raw tag list, potentially containing empty strings.

    Returns:
        List with empty strings removed, or None if input is None or empty.

    """
    if not tags:
        return None
    filtered = [t for t in tags if t]
    return filtered if filtered else None


def _add_unique_tags(
    source: list[str] | None,
    result: list[str],
    seen: set[str],
    *,
    strip_hash: bool = False,
) -> None:
    """Add unique lowercased tags from source to result.

    Args:
        source: Tag list to process.
        result: Accumulator list for ordered results.
        seen: Set of already-seen normalized tags.
        strip_hash: If True, strip leading '#' before processing.

    """
    for tag in source or []:
        clean = tag.lstrip("#") if strip_hash else tag
        if not clean:
            continue
        normalized = clean.lower()
        if normalized not in seen:
            result.append(normalized)
            seen.add(normalized)


def _merge_tags(
    doc_tags: list[str] | None,
    task_tags: list[str] | None,
) -> list[str] | None:
    """Merge document-level and task-level tags with case-insensitive dedup.

    Args:
        doc_tags: Document-level tags from frontmatter (original casing).
        task_tags: Task-level inline tags from #tag patterns (text casing).

    Returns:
        Lowercased, deduplicated tag list with doc tags first.
        Returns None if both inputs are None or empty after filtering.

    """
    filtered_doc = _filter_tags(doc_tags)
    filtered_task = _filter_tags(task_tags)

    if not filtered_doc and not filtered_task:
        return None

    result: list[str] = []
    seen: set[str] = set()

    _add_unique_tags(filtered_doc, result, seen)
    _add_unique_tags(filtered_task, result, seen, strip_hash=True)

    return result if result else None
