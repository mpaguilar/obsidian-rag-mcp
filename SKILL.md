---
name: obsidian
description: Read, search, create, and edit notes in an Obsidian vault via the Obsidian MCP server and filesystem tools.
---

# Obsidian Vault

Use this skill for Obsidian vault work via two complementary approaches: the **Obsidian MCP server** (semantic search, tag/property queries, tasks, ingestion) and **filesystem tools** (direct file read/write/edit for content changes the MCP can't make).

## Configuration

The Obsidian MCP server indexes one or more vault directories. Each vault is configured with a **name** (used in MCP tool calls) and a **container path** (the filesystem path the server scans). The MCP server's container path is internal to its runtime — your filesystem tool paths may differ depending on how you mount the vault directory.

---

## MCP Server Tools (read-only queries + ingestion)

The Obsidian MCP server provides indexed, queryable access to ingested vault content. **All MCP tools are read-only except `ingest`** — they cannot create, edit, or delete notes. Use filesystem tools for writes.

### query_documents (semantic search)

```
query_documents(query, filters?, limit?, offset?, use_chunks?, rerank?, vault_name?)
```

- **query**: natural-language search string (embedding-based similarity)
- **filters**: `{include_tags, exclude_tags, include_properties, exclude_properties, match_mode}`. Can pass as dict or JSON string.
- **limit** / **offset**: pagination (default 20, max 100)
- **use_chunks**: `true` → search at chunk level (more precise for large docs); returns best matching chunk per document
- **rerank**: `true` → apply flashrank re-ranking to chunk results (only with `use_chunks=true`)
- **vault_name**: scope to a specific vault

Returns per-result: `id`, `vault_name`, `file_path`, `relative_path`, `file_name`, `content`, `kind`, `tags`, `similarity_score`, `matching_chunk`, `created_at_fs`, `modified_at_fs`, `obsidian_uri`.

### get_documents_by_tag

```
get_documents_by_tag(filters?, vault_name?, limit?, offset?)
```

- **filters**: `{include_tags, exclude_tags, match_mode}`. Tags do NOT include `#` prefix — use `"projects/tech"` not `"#projects/tech"`.
- **vault_name**: filter to a single vault
- **match_mode**: `"all"` (AND — doc must match every include tag) or `"any"` (OR)

### get_documents_by_property

```
get_documents_by_property(filters?, vault_name?, limit?, offset?)
```

- **filters**: `{include_properties, exclude_properties, include_tags, exclude_tags, match_mode}`
- Same tag rules as above. Properties match frontmatter fields.

### get_tasks

```
get_tasks(status?, priority?, date_filters?, tag_filters?, inline_filters?, limit?, offset?, vault_name?)
```

- **status**: array of status strings (e.g. `["not_completed"]`, `["completed"]`, `["in_progress"]`, `["cancelled"]`)
- **priority**: array of priority strings (`"highest"`, `"high"`, `"normal"`, `"low"`, `"lowest"`)
- **date_filters**: `{due_after, due_before, scheduled_after, scheduled_before, completion_after, completion_before, match_mode}`
- **tag_filters**: `{include_tags, exclude_tags, match_mode}`
- **inline_filters**: array of property filters for Dataview inline fields (`[key:: value]`). Same operators as document property filtering.

### get_all_tags

```
get_all_tags(pattern?, limit?, offset?, vault_name?)
```

- **pattern**: glob filter (e.g. `"projects/*"`). Supports `*`, `?`, `[abc]`.

### get_document

```
get_document(file_path?, document_id?, vault_name?)
```

- Retrieve a single document by vault+path or by UUID.

### list_documents

```
list_documents(file_name, vault_name?, limit?, offset?)
```

- Find documents by exact filename. No globs — for fuzzy lookup, use `query_documents`.

### Vault management

- **list_vaults**: `list_vaults(limit?, offset?)` — list configured vaults with document counts
- **get_vault**: `get_vault(vault_name? | vault_id?)` — get single vault details
- **update_vault**: `update_vault(vault_name, ...)` — update vault properties. Changing `container_path` requires `force=true` and deletes all documents/tasks/chunks.
- **delete_vault**: `delete_vault(vault_name, confirm=true)` — irreversible, cascades all data
- **ingest**: `ingest(vault_name, path?, no_delete?, force?)` — sync vault directory into the database

### Ingestion

Ingestion scans `.md` files in the vault directory and indexes them into the database. It extracts frontmatter properties, tags, tasks (checkboxes), inline fields, and generates embeddings for semantic search.

| Flag | Default | Effect |
|------|---------|--------|
| `force` | `false` | Re-ingest everything, ignoring checksums. **Expensive** — re-embeds all content. |
| `no_delete` | `false` | When `true`, skip orphan cleanup (documents in the DB but not on disk are kept). |
| `path` | vault root | Subdirectory to scan. Must be a directory, not a single file. |

**⚠️ Pitfall**: The `path` parameter only scopes the **scan** (which files are checked for new/updated content). Orphan cleanup still runs **vault-wide** regardless of `path`. If you ingest a subdirectory, it will delete database entries for files in other parts of the vault that were previously ingested but whose on-disk files no longer exist. To avoid unintended deletions, either:
- Pass `no_delete=true` to suppress orphan cleanup entirely, or
- Ingest the root vault path instead, which gives you a full sync with safe cleanup.

**After creating or editing notes via filesystem tools**, changes won't appear in MCP query results until the next ingestion. Plan your workflow accordingly.

---

## Result Fields

All document queries return objects with these fields:

| Field | Description |
|-------|-------------|
| `id` | Internal UUID |
| `vault_name` | Configured vault name |
| `file_path` | Path within vault |
| `relative_path` | Same as file_path |
| `file_name` | Filename with extension |
| `content` | Full document text |
| `kind` | Inferred doc type (e.g. `"requirements"`, `"testing"`) |
| `tags` | Array of tag strings (from frontmatter and inline body tags) |
| `similarity_score` | 0–1 relevance (semantic search only) |
| `matching_chunk` | Best chunk text (when `use_chunks=true`) |
| `obsidian_uri` | Deep link: `obsidian://open?vault=...&file=...` |
| `created_at_fs` / `modified_at_fs` | Filesystem timestamps |

### Task Fields

| Field | Description |
|-------|-------------|
| `id` | Task UUID |
| `raw_text` | Original markdown line |
| `status` | `not_completed`, `completed`, `in_progress`, or `cancelled` |
| `description` | Parsed description (text with metadata and tags removed) |
| `due` | Due date (ISO, from `[due:: YYYY-MM-DD]`) |
| `priority` | `highest`, `high`, `normal`, `low`, or `lowest` |
| `tags` | Task-specific tags merged with parent document tags |
| `document_path` | Source file path |
| `properties` | Parent document's parsed frontmatter |
| `inline_fields` | All `[key:: value]` inline fields as dict |

---

## Pagination

Most tools support `limit` (default 20, max 100) and `offset`. Results include `total_count`, `has_more`, and `next_offset`. For large result sets, loop with increasing `offset` until `has_more` is `false`.

---

## Filesystem Integration

The MCP server cannot write notes. Use your environment's filesystem tools for creating, reading, and editing markdown files directly in the vault directory.

### Dataview inline fields

Obsidian Dataview plugin enables structured properties inline via `[property:: value]` syntax (e.g. `[scheduled:: 2026-01-01]`). These are indexed as `inline_fields` and queryable via `get_tasks` with `inline_filters`.

**Always use ISO 8601 `yyyy-mm-dd` format** for date values in inline fields (e.g. `2026-01-01`, not `Jan 1`). Non-ISO dates break Dataview date comparisons and sorting.

### Wikilinks

Obsidian links notes with `[[Note Name]]` syntax. When creating notes, use these to link related content.

### Frontmatter

Notes can include YAML frontmatter. The MCP server parses this into the `properties` field. Tags in frontmatter (under a `tags:` key) are merged with inline `#tags` found in the body.

---

## When to Use Which Approach

| Goal | Use | Why |
|------|-----|-----|
| Semantic / conceptual search | `query_documents` | Embedding-based, finds by meaning not exact text |
| Find docs by tag or frontmatter | `get_documents_by_tag` / `get_documents_by_property` | Structured filter on indexed metadata |
| Find tasks by status, date, or inline field | `get_tasks` | Queryable task index with inline field filtering |
| Discover available tags | `get_all_tags` | Tag inventory with glob filtering |
| Full-text / regex search | Filesystem tools | Regex on raw files; MCP only does semantic |
| Read a specific note | Filesystem tools or `get_document` | Simpler, no indexing lag |
| Create or edit a note | Filesystem tools | MCP is read-only |
| Check if vault is indexed | `list_vaults` or `get_vault` | See `document_count` |
| Sync vault to database | `ingest` | Required after filesystem changes |