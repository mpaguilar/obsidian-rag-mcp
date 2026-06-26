# Project: Obsidian RAG

A Python library for ingesting and querying Obsidian markdown documents with vector embeddings and task extraction capabilities.

## Overview

This project provides a CLI tool and library for:
- Ingesting Obsidian markdown documents into PostgreSQL with pg_vector support
- Extracting and managing tasks from markdown content
- Performing semantic search on documents using vector embeddings
- Supporting configurable LLM providers (OpenAI, OpenRouter, HuggingFace) for embeddings and analysis

## Project Structure

```
obsidian_rag/                    # Main package
├── __init__.py
├── cli.py                       # CLI entry point (Click command definitions only)
├── cli_commands.py              # CLI business logic (extracted from cli.py)
├── cli_dates.py                 # CLI date parsing utilities
├── cli_ingest.py                # CLI ingest path resolution helpers
├── cli_query_exact.py           # CLI exact document query implementation
├── cli_vault_commands.py        # CLI vault command implementations
├── config.py                    # Configuration management (entry point; re-exports split modules)
├── config_env.py                # Environment-variable interpolation (T TypeVar, _interpolate_env_vars)
├── config_models.py             # Pydantic config model classes (EndpointConfig..MCPConfig)
├── config_validators.py         # Standalone validation helpers used by Settings validators
├── database/                    # Database layer
│   ├── __init__.py
│   ├── engine.py                # Database connection
│   └── models.py                # SQLAlchemy models
├── llm/                         # LLM provider layer
│   ├── __init__.py
│   ├── base.py                  # Base provider classes
│   └── providers.py             # Provider implementations
├── mcp_server/                  # MCP server layer
│   ├── __init__.py
│   ├── __main__.py              # Server entry point
│   ├── document_tools.py       # Document retrieval MCP tool wrappers
│   ├── handlers.py              # Request handlers for tools
│   ├── ingest_helpers.py        # Ingest helper functions (request ID, dedup)
│   ├── ingest_tracker.py        # Request tracking for ingest tool deduplication
│   ├── middleware.py            # HTTP request/response logging middleware
│   ├── models.py                # Pydantic request/response models
│   ├── server.py                # FastMCP server setup and tool wrappers
│   ├── session_manager.py       # Session lifecycle and metrics tracking
│   ├── tool_definitions.py      # Tool implementations and MCPToolRegistry
│   ├── vault_tools.py           # Vault MCP tool wrappers
│   └── tools/                   # MCP tools
│       ├── __init__.py
│       ├── documents.py         # Document query tools (public API)
│       ├── documents_filters.py # Property filter implementations
│       ├── documents_params.py  # Filter parameter dataclasses
│       ├── documents_postgres.py # PostgreSQL-specific queries
│       ├── documents_tags.py    # Tag filtering logic
│       ├── tasks.py             # Task query tools
│       ├── tasks_dates.py       # Date parsing for task filters
│       ├── tasks_params.py      # Task filter parameter dataclasses
│       ├── vaults.py            # Vault query tools (list, get, update, delete)
│       └── vaults_params.py     # Vault update parameter dataclass
├── parsing/                     # Document parsing
│   ├── __init__.py
│   ├── body_tags.py             # Inline body tag extraction (Obsidian rules)
│   ├── frontmatter.py           # FrontMatter extraction
│   ├── scanner.py               # File scanning
│   └── tasks.py                 # Task parsing
└── services/                    # Service layer
    ├── __init__.py
    ├── ingestion.py             # Document ingestion service
    ├── ingestion_chunks.py      # Chunk creation service
    ├── ingestion_cleanup.py     # Document deletion operations
    ├── ingestion_integrity.py   # IntegrityError recovery for document ingestion
    └── tag_merging.py           # Document/task tag merging utilities
```

## Requirements

- Python 3.12+
- PostgreSQL with pg_vector extension
- See `pyproject.toml` for Python dependencies

## CLI Commands

- `obsidian-rag [--log-level LEVEL] <command>` - Global options include `--log-level` (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- `obsidian-rag ingest --vault <name> [PATH]` - Ingest documents from vault path. PATH is optional when vault has `container_path` configured; if provided, PATH overrides the configured container_path
- `obsidian-rag query "search"` - Semantic search documents
- `obsidian-rag query --exact --vault NAME --path PATH` - Exact document lookup by vault and file path
- `obsidian-rag query --exact --name FILENAME` - Exact document lookup by file name
- `obsidian-rag query --exact --id UUID` - Exact document lookup by document UUID
- `obsidian-rag tasks [options]` - Query tasks
- `obsidian-rag vault list [--format json|table] [--limit N] [--offset N]` - List all vaults
- `obsidian-rag vault get --name NAME [--id UUID]` - Get vault details by name or UUID
- `obsidian-rag vault update --name NAME [--description DESC] [--host-path PATH] [--container-path PATH] [--force]` - Update vault properties
- `obsidian-rag vault delete --name NAME --confirm` - Delete vault and cascade-delete all associated data

### Vault Support

The CLI supports multiple vaults via the `--vault` parameter. Vaults must be configured in the config file:

```yaml
vaults:
  "Personal":
    container_path: "/data/personal"
    host_path: "/home/user/obsidian/personal"
    description: "Personal knowledge base"
  "Work":
    container_path: "/data/work"
    host_path: "/home/user/obsidian/work"
    description: "Work notes"
```

Vault names must contain only alphanumeric characters, spaces, hyphens, and underscores.

## Configuration

Configuration sources (precedence: highest to lowest):
1. CLI flags (e.g., `--embedding-provider openai`)
2. Environment variables (e.g., `OBSIDIAN_RAG_EMBEDDING_PROVIDER=openai`)
3. Config file (YAML format)
4. Default values

Config file locations (searched in order):
1. `$PWD/.obsidian-rag.yaml` - Project-specific config
2. `$XDG_CONFIG_HOME/obsidian-rag/config.yaml` - User config

### Key Configuration Options

- `database.vector_dimension`: Vector embedding dimension (default: 1536, max: 2000)
  - Must match the output dimension of your embedding model
  - pgvector HNSW index has a 2000 dimension limit
  - Compatible models: text-embedding-3-small (1536), all-MiniLM-L6-v2 (384), etc.
- `endpoints.embedding.provider`: LLM provider for embeddings ('openai', 'openrouter', 'huggingface')
- `endpoints.chat.provider`: LLM provider for chat/analysis ('openai', 'openrouter')

### Vault Support

The CLI supports multiple vaults via the `--vault` parameter. Vaults must be configured in the config file:

```yaml
vaults:
  "Personal":
    container_path: "/data/personal"
    host_path: "/home/user/obsidian/personal"
    description: "Personal knowledge base"
  "Work":
    container_path: "/data/work"
    host_path: "/home/user/obsidian/work"
    description: "Work notes"
```

Vault names must contain only alphanumeric characters, spaces, hyphens, and underscores.

## Development Standards

- All code must pass ruff linting
- 100% test coverage is required, including branch coverage (see CONVENTIONS.md)
- All functions require type hints and docstrings
- McCabe complexity max: 5
- All source files under 1000 lines

## Testing

Run tests with coverage:
```bash
python -m pytest tests/ --cov=obsidian_rag --cov-branch --cov-report=term-missing
```

Run ruff checks:
```bash
ruff check obsidian_rag/ tests/
```

> **Technical Implementation Details**: For architecture patterns, component details, and data flow, see [ARCHITECTURE.md](./ARCHITECTURE.md). For coding conventions and standards, see [CONVENTIONS.md](./CONVENTIONS.md).

## Checkpoint History

### 040.cleanup (Completed 2026-06-26)

**Objective:** Clean up pre-existing tech debt: resolve 9 bandit warnings (B101/B324/B104/B105), remove 7 inline `# noqa` pragmas (6 PIE790 + 1 ANN401), audit/reduce 15 `# pragma: no cover` (down to 5 documented-legitimate), fix 70 test-file lint violations (51 F401 + 19 F841), exclude tests from mypy via `pyproject.toml`, split 10 oversized test files (with growth headroom ~600-750 lines each), split `config.py` from 1223 to 587 lines via `config_env.py`/`config_models.py`/`config_validators.py` extraction, and fix the Alembic `downgrade base` blocker (test data pollution). Zero behavioral change to CLI/MCP tools.

**Changes Made:**
- **Group A — Bandit fixes (REQ-001, REQ-008):** Added `usedforsecurity=False` to `hashlib.md5()` in `ingest_helpers.py` (B324). Replaced 5 `assert X is not None` with `if X is None: log.error(_msg); raise RuntimeError(_msg)` in `cli.py`, `documents.py`, `vaults.py` (B101). Added `[tool.bandit]` config in `pyproject.toml` with `skips = ["B104", "B105"]` and `exclude_dirs = ["tests", ".venv"]`. New covering test files: `test_cli_type_guards.py`, `test_documents_type_guards.py`, `test_vaults_type_guards.py`.
- **Group B — Remove inline noqa (REQ-002):** Deleted `pass`/`...` placeholders from 6 locations (`llm/base.py` 3 abstract methods, `reranking.py` 2 Protocol/exception, `tokenizer.py` 1 exception), leaving docstring-only bodies (PIE790 passes without noqa). Removed `# noqa: ANN401` from `handlers.py:parse_json_str` by adding `per-file-ignores` entry in `pyproject.toml` and documenting the accepted exception in `CONVENTIONS.md`.
- **Group C — Pragma audit (REQ-003):** Removed 10 of 15 `# pragma: no cover` — 3 via TASK-004 placeholder deletion, 7 config.py pragmas covered by new tests or removed as unreachable. 5 remaining documented-legitimate (server.py JSON fallbacks, documents_chunks.py defensive, tokenizer.py unreachable, chunking.py defensive).
- **Group D — Test lint hygiene (REQ-004, REQ-009):** Fixed 51 F401 unused-import + 19 F841 unused-variable violations across test files. Expanded ruff `include` to `tests/**/*.py` with extended `per-file-ignores` enforcing only F401/F841 on tests (suppressing PLR2004, ARG002, C4, C901, PIE790, TC002/006, SIM110, etc.).
- **Group E — mypy exclusion (REQ-005):** Added `exclude = "^tests/"` to `[tool.mypy]` in `pyproject.toml`. Confirmed `--mypy` not in pytest addopts.
- **Group F — Test file splits (REQ-006):** Split 10 oversized test files into ~42 smaller files with growth headroom (~600-750 lines each): `test_cli.py` (3206→6 files), `test_services_ingestion.py` (2870→7 files), `test_server.py` (2851→6 files), `test_tools_vaults.py` (1564→2 files), `test_llm_providers.py` (1454→2 files), `test_server_handlers.py` (1313→2 files), `test_tools_documents.py` (1306→2 files), `test_config.py` (1245→2 files), `test_cli_vault.py` (1190→2 files), `test_handlers.py` (1141→2 files).
- **Group G — config.py split (REQ-007):** Extracted `config_env.py` (102 lines, env-var interpolation + T TypeVar), `config_models.py` (354 lines, Pydantic model classes), `config_validators.py` (292 lines, validation helpers). `config.py` re-exports all symbols — public API unchanged. `config.py` now 587 lines (was 1223).
- **Group H — Alembic blocker (REQ-010):** Fixed test data pollution (duplicate `file_path` values) via transaction-rollback fixtures. Corrected documentation: failing migration is 003, not 002.
- **Group I — Final verification:** All 11 completion criteria verified.

**Key Design Decisions:**
- **`assert`→`RuntimeError`** (not `cast()`): User chose explicit raise — survives `python -O`, covers the branch with a test, no pragma needed.
- **Docstring-only bodies** for PIE790: Empirically validated — ruff accepts a docstring as a sufficient body for abstract methods, Protocol methods, and exception classes.
- **`per-file-ignores` over inline noqa**: Config-level suppression + CONVENTIONS.md documentation is the accepted alternative to inline `# noqa` (CONVENTIONS policy).
- **F401/F841-only on tests**: User-confirmed policy — PLR2004/ARG002/etc. are normal in test files.
- **Growth headroom**: Test splits target ~600-750 lines per file, leaving ~250+ lines of room under 1000 so future additions don't immediately re-exceed.
- **config.py re-export pattern**: Follows `ingestion.py`→`ingestion_*.py` precedent. `from obsidian_rag.config import X` still works — no caller/test breakage.

**No breaking changes:** Zero behavioral change to CLI or MCP tools. All changes are code quality, test structure, and dev-tooling config. No schema changes, no API changes.

**Verification:**
- All 1861 tests pass (1 skipped pre-existing)
- 100% code coverage (5116 statements, 980 branches)
- `bandit -r obsidian_rag/ -q` → 0 violations
- `ruff check obsidian_rag/ tests/` → All checks passed
- `ruff format --check` → 173 files formatted
- `mypy obsidian_rag/` → Success: no issues found in 55 source files
- `mypy tests/` → Excluded (0 .py[i] files)
- `grep -rn "# noqa" obsidian_rag/` → 0 matches
- `grep -rn "# pragma: no cover" obsidian_rag/` → 5 (documented-legitimate)
- All source files under 1000 lines (max: ingestion.py at 999)
- All test files under 1000 lines (max: test_tools_documents_postgres_complete.py at 919)
- config.py at 587 lines (was 1223)
- Code review: 4/4 reviewers APPROVED, no CHANGES REQUESTED

### 039.ingestion-path (Completed 2026-06-25)

**Objective:** Auto-force `no_delete=True` when an MCP client passes an incremental path (a subdirectory of the vault's `container_path`) without explicitly specifying `no_delete`, preventing accidental deletion of documents outside the scanned subdirectory. The MCP `ingest` tool's `no_delete` parameter changed from `bool = False` to `bool | None = None` so the system can distinguish "client did not specify" (None, eligible for auto-force) from "client explicitly chose False" (honored as-is). CLI behavior is unchanged (CLI rejects non-matching paths via `_validate_path_matches_vault`).

**Changes Made:**
- **Updated `obsidian_rag/mcp_server/handlers.py`** (lines 127, 302-375, 416-420): Changed `IngestHandlerParams.no_delete` from `bool = False` to `bool | None = None` (TASK-004). Added `_is_incremental_path(path, container_path)` helper (TASK-001) that resolves both paths with `Path.resolve()` and returns True only when the candidate is a strict subdirectory of `container_path` (via `relative_to`), False for None/equal/outside. Added `_resolve_no_delete(path_override, container_path, *, no_delete)` helper that honors explicit True/False as-is, auto-forces True (with INFO log) when None + incremental, and returns False when None + not incremental. Wired `_resolve_no_delete()` into `_ingest_handler()` so a concrete `bool` reaches `IngestVaultOptions` (TASK-002, TASK-007). `_resolve_no_delete` uses keyword-only `no_delete` (after `*`) to comply with FBT001 (a minor improvement over the plan's positional signature). The INFO log (REQ-003) fires ONLY on the auto-force branch, using the pre-formatted `_msg` variable pattern per CONVENTIONS.
- **Updated `obsidian_rag/mcp_server/ingest_helpers.py`** (lines 18, 26, 151, 161): Changed `no_delete` parameter type hint from `bool` to `bool | None` in `_generate_request_id()` (TASK-005) and `_check_and_handle_duplicate()` (TASK-006). Pure type-signature change — NO logic changes (`json.dumps` serializes None as `null`, dict storage handles None natively). Docstrings updated to note "None if unspecified by client".
- **Updated `obsidian_rag/mcp_server/server.py`** (lines 330, 340-346): Changed MCP `ingest()` tool `no_delete` parameter from `bool = False` to `bool | None = None` (TASK-003). Updated docstring (TASK-008/REQ-004) to document the auto-force behavior. NO call-site logic changes — None flows through unchanged to `_generate_request_id()`, `_check_and_handle_duplicate()`, and `IngestHandlerParams()`; the handler does the resolution.
- **Created `tests/mcp_server/test_ingest_helpers.py`** (142 lines, NEW): Dedicated test file for `ingest_helpers.py` (no dedicated file existed before). 10 tests covering `_generate_request_id` with None (distinct from True/False, null serialization, deterministic), `_check_and_handle_duplicate` with None (params pass-through, cached return, None cached result), plus pre-existing helpers (`_create_vault_error_response`, `_is_vault_not_found_error`, `_handle_vault_not_found`).
- **Updated `tests/mcp_server/test_server_handlers.py`** (lines 946-1312): Added `TestIsIncrementalPath` (7 tests: None, equals, subdirectory, outside, trailing slash, nested, symlink — TASK-009), `TestResolveNoDelete` (6 tests: None+incremental→True, None+full→False, True honored, False honored, None+path_none→False, auto-force INFO log), `TestIngestHandlerNoDeleteResolution` (5 integration tests through `_ingest_handler`: None+incremental→True, None+full→False, True honored, False honored, auto-force log — TASK-010). Updated existing `test_ingest_handler_with_path_override` to use a real `container_path` string (the plan flagged this would break with a MagicMock container_path passed to `Path.resolve()` — TASK-013).
- **Updated `tests/mcp_server/test_server.py`** (lines 2690-2851): Added `TestIngestNoDeleteNone` class (6 tests: None propagates to IngestHandlerParams, request ID, duplicate check; omitting no_delete defaults to None; explicit False still works; explicit True still works — TASK-011, TASK-012). Existing `TestIngestRequestTracking` and `TestVaultErrorHandling` tests using `no_delete=False` remain valid (False is valid for `bool | None`).
- **Created `tests/mcp_server/test_ingest_incremental_integration.py`** (119 lines, NEW): End-to-end integration test (TASK-014) with 4 tests proving the safety guarantee: incremental path + no_delete=None passes None to handler; full vault + no_delete=None passes None; explicit False with incremental path preserved; TRUE end-to-end auto-force INFO log emitted across server→handler boundary using real temp dirs (`runner.isolated_filesystem()`).

**Key Design Decisions:**
- **`bool | None` boundary at MCP layer only**: The `no_delete` type changes to `bool | None = None` on the MCP-facing boundary (`server.py` ingest param, `handlers.py:IngestHandlerParams`, `ingest_helpers.py` type hints). `IngestVaultOptions.no_delete` stays `bool` — the handler is the documented boundary where the API-facing optional type resolves to the concrete internal type via `_resolve_no_delete()`. By the time the ingestion service runs, `no_delete` is always a concrete `bool`. No service-layer changes needed.
- **`_is_incremental_path()` via `Path.resolve()` + `relative_to()`**: Mirrors the existing `_validate_path_matches_vault()` normalization pattern in `cli_ingest.py`. `relative_to()` raises `ValueError` when the path is outside `container_path`, cleanly distinguishing "inside" from "outside." `resolve()` follows symlinks and strips trailing slashes.
- **`_resolve_no_delete()` extracted as a pure function**: Adding the None-resolution if/else directly into `_ingest_handler()` would push its McCabe complexity above the max of 5. Extracting a pure function keeps `_ingest_handler()` simple and makes the resolution independently unit-testable.
- **INFO log only on auto-force**: REQ-003 — operators must distinguish system-enforced vs client-requested. The log fires ONLY in the auto-force branch (None → True for incremental), never when the client explicitly set a value.
- **Request ID uses RAW input `no_delete` (None), not resolved value**: A call with `no_delete=None` + incremental path and `no_delete=True` + same path produce different request IDs (None vs True in JSON hash). Requirements explicitly accept this (Section 4, Detail 2). No change to `_generate_request_id()` logic.
- **CLI unchanged (intentional)**: CLI rejects any path not matching `container_path` exactly via `_validate_path_matches_vault` → never encounters incremental paths → no auto-force needed. MCP allows path overrides → can receive incremental paths → auto-force applies as a safety net. (TASK-015, 016, 017 cancelled.)

**No breaking changes:** Default behavior unchanged for full-vault ingestion (None resolves to False, identical to prior `False` default). Existing MCP clients passing `no_delete=False` still work (False is valid for `bool | None`). No schema changes, no Alembic migration, no config file changes. Only NEW behavior: incremental paths with unspecified `no_delete` now auto-force True (safety improvement).

**Verification:**
- All 1810 tests pass (1 skipped pre-existing)
- 100% code coverage (5054 statements, 962 branches)
- All ruff checks pass (canonical `ruff check obsidian_rag/ tests/`); ruff format clean
- mypy clean on source code (52 source files, no issues)
- All source files under 1000 lines (config.py at 1223 is pre-existing exception, untouched)
- Code review: 4/4 reviewers APPROVED, no CHANGES REQUESTED
- Verification easy fix: removed unused `import pytest` and 2 unused `result` variable assignments in the new `test_ingest_incremental_integration.py` (F401/F841); applied `ruff format`.
- Pre-existing conditions noted (not introduced by this checkpoint): bandit reports 9 pre-existing violations across the codebase (B101 assert_used, B104 bind 0.0.0.0, B105 password default, B324 MD5 in `ingest_helpers.py` from checkpoint 033) — all accepted across prior 20+ checkpoints, none introduced here; one pre-existing `# noqa: ANN401` on `parse_json_str` (from checkpoint 026) in handlers.py; pre-existing unused imports/variables in `test_server.py` lines 6-7, 184, 275, 310, 1058, 1086 (test files excluded from ruff by `include = ["obsidian_rag/**/*.py"]` config).

### 038.limit-output (Completed 2026-06-25)

**Objective:** Increase the pagination limit maximum from 100 to 10,000 so MCP tools and CLI vault commands respect client-provided `limit` values instead of silently capping at 100. Align all docstrings, tests, and documentation to the new maximum.

**Changes Made:**
- **Updated `obsidian_rag/mcp_server/models.py`** (lines 14, 267, 279): Changed `MAX_PAGINATION_LIMIT` constant from 100 to 10000. Updated `_validate_limit()` docstring to "clamped between 1 and 10000". **Critical bug fix:** the `limit > MAX_PAGINATION_LIMIT` branch had a hardcoded `return 100` instead of `return MAX_PAGINATION_LIMIT` — without this fix, changing the constant alone would NOT have changed behavior (values above 100 would still be clamped to 100). Changed to `return MAX_PAGINATION_LIMIT`.
- **Updated `obsidian_rag/cli_vault_commands.py`** (lines 30, 40): Changed `MAX_VAULT_LIMIT` constant from 100 to 10000. Updated `_validate_limit()` docstring to "max 10000". (This file already correctly used `return MAX_VAULT_LIMIT`, so the constant change alone fixed the CLI path.)
- **Updated tool wrapper/implementation docstrings** ("max: 100" → "max: 10000"): `mcp_server/server.py` (5 wrappers: query_documents, get_documents_by_tag, get_documents_by_property, get_all_tags, list_vaults), `mcp_server/document_tools.py` (list_documents), `mcp_server/tool_definitions.py` (get_all_tags_tool, list_vaults_tool — preserved the no-colon "max 10000" style for list_vaults_tool), `mcp_server/tools/documents.py` (get_documents_by_tag, get_all_tags, list_documents), `mcp_server/tools/tasks_params.py` (GetTasksFilterParams.limit), `mcp_server/tools/vaults.py` (list_vaults).
- **Updated tests**: `tests/mcp_server/test_models.py` `TestValidateLimit` — expanded `test_valid_limit` to include 500 and 10000 (previously capped); changed `test_limit_above_maximum` from `101→100`/`1000→100` to `10001→10000`/`100000→10000`; updated docstrings. `tests/test_cli_vault.py` `test_vault_list_limit_validation` — changed `--limit 200` to `--limit 20000` with `mock_query.offset.return_value.limit.assert_called_with(10000)`.
- **Updated documentation**: `ARCHITECTURE.md` Pagination Pattern "maximum 100" → "maximum 10000"; `DOMAIN_GLOSSARY.md` Pagination Limits "capped at 100" → "capped at 10000".
- **Verification easy fix**: `tests/mcp_server/test_tools_vaults.py::TestListVaults::test_list_vaults_limit_validation` had stale docstring/comment referencing the old max 100 and used `limit=200` (no longer clamped under the new max). Updated to `limit=20000`, docstring "clamped to max 10000", comment "# Should be clamped to 10000", and added `mock_query.limit.assert_called_with(10000)` to actually verify the clamp.

**Key Design Decisions:**
- **10,000 ceiling** balances legitimate client needs (e.g., "give me all tasks in this vault") against resource exhaustion. PostgreSQL `.limit(N)` is efficient via query planning; 10,000 full-document results use acceptable memory (~10–50MB) for legitimate use cases while preventing pathological requests (e.g., 1M results).
- **Root-cause bug fix** in `models.py:_validate_limit()` (hardcoded `return 100`) was the actual blocker — the constant change alone was insufficient. `cli_vault_commands.py` already used the constant correctly.
- **No schema changes, no config changes, no migration needed.** Pure code/docstring change. Backward compatible — clients passing `limit <= 100` see identical behavior; only clients requesting `limit > 100` see the change (more results returned).

**No breaking changes:** Default behavior unchanged for `limit <= 100`. No schema, config, or API changes.

**Verification:**
- All 1772 tests pass (1 skipped)
- 100% code coverage (5015 statements, 954 branches)
- All ruff checks pass (canonical `ruff check obsidian_rag/ tests/`); ruff format clean
- mypy clean on source code (52 source files, no issues)
- All source files under 1000 lines (config.py at 1223 is pre-existing exception, untouched)
- Code review: 4/4 reviewers APPROVED, no CHANGES REQUESTED

### 036.ingestion-bug (Completed 2026-06-17)

**Objective:** Fix `IntegrityError` (UniqueViolation on `uq_document_vault_path`) in `_ingest_single_file()` by catching it and retrying as an UPDATE. Update `ingested_at` timestamp in `_update_document()` when a document is re-ingested. Add diagnostic logging before the document lookup query.

**Changes Made:**
- **Created `obsidian_rag/services/ingestion_integrity.py`** (165 lines): New module containing `ingest_new_document()` and `handle_integrity_error()` functions. `ingest_new_document()` wraps the INSERT path with try/except for `IntegrityError`, delegating recovery to `handle_integrity_error()`. `handle_integrity_error()` checks for `"uq_document_vault_path"` in the error message, rolls back the session, re-queries for the existing document, and follows the UPDATE path. Non-matching IntegrityErrors and re-query-returns-None are re-raised.
- **Updated `obsidian_rag/services/ingestion.py`** (lines 8, 28, 607-610, 637-653, 854): Added `datetime` import and `ingest_new_document` import. Refactored the `else` branch of `_ingest_single_file()` to delegate to `ingest_new_document()` instead of inline document creation. Added diagnostic debug log before the `filter_by` query (REQ-003). Added `document.ingested_at = datetime.now(UTC)` in `_update_document()` (REQ-002).
- **Created `tests/test_services_ingestion_integrity.py`** (744 lines): 14 test cases covering: IntegrityError recovery to update, non-unique constraint re-raise, re-query-None re-raise, rollback verification, warning logging, end-to-end recovery through `_ingest_single_file()`, non-unique IntegrityError propagation, file counted as "updated" not "error" in stats, `ingested_at` update via `_update_document()`, UTC timezone verification, `ingested_at` update via update path, `ingested_at` update via IntegrityError recovery, diagnostic log presence, and exact value verification in log messages.

**Key Design Decisions:**
- **Extraction to `ingestion_integrity.py`**: Kept `ingestion.py` at 996 lines (under 1000 limit) by extracting the IntegrityError handling logic. Follows the established pattern of `ingestion_cleanup.py` and `ingestion_chunks.py`.
- **Constraint detection**: Checks for `"uq_document_vault_path"` in the error message string, matching the existing pattern in `vaults.py:_handle_flush_with_integrity_check()`.
- **Session rollback**: Explicit `session.rollback()` before re-query since SQLAlchemy session is in invalid state after IntegrityError.
- **Recovery result status**: Returns `"updated"` — matches user's mental model (the document was updated).
- **Defensive re-raise**: If re-query returns None after IntegrityError (should not happen), the original IntegrityError is re-raised.

**No breaking changes:** Default behavior unchanged for the happy path. Only the error recovery path is new.

**Verification:**
- All 1692 tests pass (1 skipped)
- 100% code coverage (4968 statements, 944 branches)
- All ruff checks pass
- All mypy type checks pass
- All source files under 1000 lines (config.py at 1223 is pre-existing exception)

### 033.full-reset-ingest (Completed 2026-05-31)

**Objective:** Add a `force` flag to the ingestion pipeline that re-ingests all documents regardless of checksums, exposed via CLI `--force` and MCP `force=True` parameter.

**Changes Made:**
- **Updated `obsidian_rag/services/ingestion.py`** (lines 46, 369, 483, 549, 615, 621-625): Added `force: bool = False` field to `IngestVaultOptions` dataclass. Added `force` parameter to `_process_files_with_stats()`, `_ingest_single_file()`, and `ingest_vault()`. The checksum comparison in `_ingest_single_file` uses `if not force and existing.checksum_md5 == file_info.checksum` — when `force=True`, the check is bypassed and all documents take the `_update_document` path with debug log "Force re-ingestion enabled - updating document regardless of checksum".
- **Updated `obsidian_rag/cli.py`** (lines 100-104, 112, 140): Added `--force` Click option (`is_flag=True`) to the `ingest` command with help text "Re-ingest all documents regardless of checksums."
- **Updated `obsidian_rag/cli_commands.py`** (lines 96, 222, 253-255, 324, 948, 970): Added `force: bool = False` to `IngestOptions` dataclass, `_run_ingest_command()`, `_report_ingest_results()`. Added force confirmation message "Force re-ingestion enabled: all documents will be re-processed regardless of checksums" in `_run_ingestion()`. Added force-aware result reporting in `_report_ingest_results()` with message "Force re-ingestion: all documents re-processed (unchanged count expected to be 0)".
- **Updated `obsidian_rag/mcp_server/handlers.py`** (lines 128, 344): Added `force: bool = False` to `IngestHandlerParams` dataclass and `_ingest_handler()` passes it to `IngestVaultOptions`.
- **Updated `obsidian_rag/mcp_server/server.py`** (lines 327, 338-339, 366, 376, 390): Added `force: bool = False` keyword parameter to `ingest()` MCP tool. Refactored `_generate_request_id`, `_check_and_handle_duplicate`, and `_create_vault_error_response` to new `ingest_helpers.py` module to keep server.py under 1000 lines.
- **Created `obsidian_rag/mcp_server/ingest_helpers.py`**: Extracted ingest helper functions from server.py (37 lines, 100% coverage). `_generate_request_id()` includes `force` in params dict. `_check_and_handle_duplicate()` accepts `force` parameter and includes it in the request tracking dict.
- **Created `tests/test_services_ingestion_force.py`**: 10 integration tests for force flag behavior in ingestion service.
- **Updated `tests/test_services_ingestion.py`**: 9 force-related unit tests for IngestVaultOptions, _ingest_single_file, _process_files_with_stats, ingest_vault.
- **Updated `tests/test_cli.py`**: 17 CLI force tests (help text, flag propagation, confirmation messages, result reporting).
- **Updated `tests/mcp_server/test_server.py`**: 7 MCP force tests (request ID differentiation, parameter propagation, cache isolation).
- **Updated `tests/mcp_server/test_server_handlers.py`**: 2 handler force tests (dataclass field, IngestVaultOptions propagation).

**Key Design Decisions:**
- **`no_delete` pattern mirroring**: `force` propagates through the exact same chain as `no_delete` — CLI flag → `IngestOptions` → `IngestVaultOptions` → `_process_files_with_stats` → `_ingest_single_file`.
- **Checksum skip**: `if not force and existing.checksum_md5 == file_info.checksum` — minimal change, force=True bypasses check and falls through to `_update_document`.
- **MCP request deduplication**: `force` is included in the MD5 hash params dict, ensuring force and non-force requests produce different IDs (REQ-004).
- **Debug logging**: Force path logs "Force re-ingestion enabled - updating document regardless of checksum" vs normal "Updating existing document".
- **Keyword-only parameters**: All boolean `force` parameters use keyword-only syntax (after `*`) to comply with FBT001.
- **No schema changes**: No Alembic migration needed, no config file changes, no breaking changes (default `force=False`).

**No breaking changes:** Default `force=False` preserves all existing behavior. All existing parameters and signatures remain unchanged.

**Verification:**
- All 1563 tests pass (1 skipped)
- 100% code coverage (4690 statements, 896 branches)
- All ruff checks pass
- All mypy type checks pass on source code
- All source files under 1000 lines (config.py at 1223 is pre-existing exception)

### 030.sql-error (Completed 2026-05-09)

**Objective:** Fix incorrect use of PostgreSQL `unnest()` set-returning function in `_extract_tags_postgresql()` that could cause SQL errors in some PostgreSQL configurations. PostgreSQL does not allow SRFs in SELECT or WHERE scalar positions — they must appear in FROM.

**Changes Made:**
- **Updated `obsidian_rag/mcp_server/tools/documents.py`** (lines 396-410): Rewrote `_extract_tags_postgresql()` to use `func.unnest(Document.tags).table_valued(column("tag"))` instead of `func.unnest(Document.tags)` in SELECT and WHERE clauses. Added `column` import from `sqlalchemy`. Changed SELECT from `func.distinct(func.unnest(Document.tags)).label("tag")` to `func.distinct(tag_tbl.c.tag).label("tag")`. Changed WHERE pattern filter from `func.lower(func.unnest(Document.tags)).ilike(...)` to `func.lower(tag_tbl.c.tag).ilike(...)`.

**SQL Comparison:**
- **Before (broken):** `unnest()` in SELECT and WHERE — SRF in scalar position
- **After (correct):** `unnest()` in FROM via `table_valued()` — generates proper cross-join lateral: `FROM documents, unnest(documents.tags) AS anon_1(tag)`

**Key Design Decision:**
Used `table_valued(column("tag"))` pattern (available in SQLAlchemy >= 1.4) to generate a proper FROM-clause unnest. The alias column (`tag_tbl.c.tag`) is then referenced in SELECT and WHERE instead of the bare `func.unnest()`. This is the standard PostgreSQL pattern for array unnesting in a query and eliminates the SRF execution error.

**No breaking changes:** The function returns identical results to callers. No schema changes, no API changes, no database migration needed.

**Verification:**
- All 1464 tests pass (1 skipped)
- 100% code coverage (4642 statements, 876 branches)
- All ruff checks pass
- All mypy type checks pass
- All source files under 1000 lines (config.py at 1223 is pre-existing exception)

### 031.sql-errors (Completed 2026-05-17)

**Objective:** Fix three SQL generation bugs in PostgreSQL document tag extraction and filtering: `table_valued()` column name rendering failure causing `UndefinedColumn`, empty `or_()` crash, and fragile text-concatenation NOT expression.

**Changes Made:**
- **Updated `obsidian_rag/mcp_server/tools/documents.py`** (lines 384-416): Replaced `func.unnest(Document.tags).table_valued(column("tag"))` pattern in `_extract_tags_postgresql()` with a subquery approach using `sa_select(func.unnest(Document.tags).label("tag")).select_from(Document).subquery("tag_subq")`. The `table_valued()` method does not render column names for `unnest()`, producing `AS anon_1` instead of `AS anon_1(tag)`, causing `UndefinedColumn: column anon_1.tag does not exist`. The subquery wraps `unnest()` in a SELECT that properly labels the column as "tag", making it accessible as `tag_subq.c.tag`.
- **Updated `obsidian_rag/mcp_server/tools/documents_tags.py`** (lines 301-308): Two fixes to `apply_postgresql_exclude_tags()`: (1) Added guard `if not exclude_lower: return query` after `_strip_tag_list()` to handle edge case where all exclude tags become empty after `#` prefix stripping, and (2) Replaced fragile text concatenation `text("NOT (") + or_() + text(")")` with clean `~or_(*exclude_conditions)` SQLAlchemy negation operator.
- **Created `tests/mcp_server/test_tools_documents_select_from.py`**: Rewritten for subquery approach with `TestExtractTagsPostgresqlSubquery` class (5 tests) verifying: happy path returns tags, filters out None tags, pattern filter adds second `.filter()` call, outer query does NOT call `.select_from()` (key behavioral change), and empty results.
- **Created `tests/mcp_server/test_tools_documents_subquery_sql.py`**: SQL compilation tests (4 tests) verifying the subquery generates correct SQL structure: `unnest` in subquery with `AS tag` alias, `ILIKE` pattern filtering through `tag_subq.c.tag`, column accessibility, and no `table_valued` artifacts.
- **Updated `tests/mcp_server/test_tools_documents_tags.py`**: Added `TestApplyPostgresqlExcludeTagsEdgeCases` class with 3 tests covering all-hash exclude_tags, mixed valid/hash tags, and `~or_()` pattern verification.
- **Updated `tests/mcp_server/test_tools_documents.py`**: Added `TestGetAllTagsSQLGeneration` class with 2 integration tests for `get_all_tags` through the PostgreSQL path.
- **Updated `tests/mcp_server/test_tools_documents_postgres.py`**: Added `TestQueryDocumentsExcludeTagsIntegration` class with 2 integration tests for `query_documents` with exclude_tags through the PostgreSQL path.
- **Refactored `tests/mcp_server/test_tools_documents_postgres_complete.py`**: Extracted `TestExtractTagsPostgresqlSelectFrom` class and `_create_vault_name_mock_query` helper to keep file under 1000 lines and reduce complexity.

**Key Design Decisions:**
- **Subquery approach** replaces `table_valued()` because `func.unnest(...).table_valued(column("tag"))` is buggy — SQLAlchemy does not render `(tag)` column name for the `unnest()` function, producing `AS anon_1` without a column alias, causing `UndefinedColumn` when referencing `anon_1.tag`. The subquery `SELECT unnest(documents.tags) AS tag FROM documents` properly labels the column.
- **Filter change**: Uses `tag_subq.c.tag.isnot(None)` instead of `Document.tags.isnot(None)` — filters NULL tags from the unnested result, not the array column itself.
- Empty `exclude_lower` guard prevents `or_()` with zero arguments, which produces invalid SQL `NOT ()`
- `~or_()` operator is SQLAlchemy's standard negation pattern, producing clean `NOT (a OR b)` SQL
- Test approach follows existing mocking patterns with MagicMock-based query chains, PostgreSQL dialect mocking, and SQL compilation tests for real SQL structure verification

**SQL Comparison (tag extraction):**
- **Before (broken):** `table_valued(column("tag"))` → `unnest(documents.tags) AS anon_1` (no column name) → `anon_1.tag` fails with UndefinedColumn
- **After (correct):** `sa_select(func.unnest(...).label("tag")).subquery("tag_subq")` → `(SELECT unnest(documents.tags) AS tag FROM documents) AS tag_subq` → `tag_subq.tag` works correctly

**No breaking changes:** No schema changes, no migration needed, no API changes. Same behavior for valid inputs; fixes only address edge cases and defensive code paths.

**Verification:**
- All 1480 tests pass (1 skipped)
- 100% code coverage (4645 statements, 878 branches)
- All ruff checks pass
- All mypy type checks pass on source code
- All source files under 1000 lines (config.py at 1223 is pre-existing exception)

### 027.task-tags (Completed 2026-04-15)

**Objective:** Implement defensive tag prefix stripping and remove legacy `tags` parameter from MCP task tools. Tags are stored in the database without the `#` prefix, but LLM clients may include it when passing filter values. This checkpoint ensures tag filters work correctly regardless of whether the `#` prefix is included.

**Changes Made:**
- **Updated `obsidian_rag/mcp_server/tools/tasks.py`**:
  - Added `_strip_tag_prefix()` helper function to strip leading `#` characters from tag strings
  - Added `_strip_tag_list()` helper to strip prefixes from all tags in a list
  - Applied prefix stripping in `_apply_include_tags_any()`, `_apply_include_tags_all()`, and `_apply_exclude_tags()`
  - Removed `_apply_legacy_tags()` function (legacy `tags` parameter removed per REQ-003)
  - Updated `_apply_tag_filters()` to remove legacy tags support
- **Updated `obsidian_rag/mcp_server/tools/documents_tags.py`**:
  - Imported `_strip_tag_list` from tasks module
  - Applied prefix stripping in `apply_postgresql_include_tags()` and `apply_postgresql_exclude_tags()`
- **Updated `obsidian_rag/cli.py`**:
  - Imported `_strip_tag_list` from tasks module
  - Replaced legacy `--tag` option with `--include-tags` and `--exclude-tags` options
  - Updated `TaskFilterOptions` dataclass to use `include_tags` and `exclude_tags`
  - Applied prefix stripping in `_apply_include_tags_cli()` and `_apply_exclude_tags_cli()`
- **Updated `obsidian_rag/mcp_server/tools/tasks_params.py`**:
  - Removed legacy `tags` field from `GetTasksFilterParams` dataclass
  - Removed legacy `tags` field from `GetTasksRequest` dataclass
  - Updated docstrings to remove references to legacy parameter
- **Updated `obsidian_rag/mcp_server/handlers.py`**:
  - Removed legacy `tags` field from `GetTasksToolInput` dataclass
  - Updated `_get_tasks_handler()` to remove legacy tags support
  - Updated `TagFilterStrings` docstring to explicitly state tags should NOT include `#` prefix
- **Updated `obsidian_rag/mcp_server/server.py`**:
  - Updated `get_tasks()` docstring with explicit note: "Tags should NOT include the '#' prefix"
  - Removed `tags=params.tags` from `GetTasksRequest` creation
- **Updated test files**:
  - Updated tests to use `include_tags` instead of legacy `tags` parameter
  - Removed tests for legacy `tags` parameter functionality

**Key Design Decisions:**
- Defensive prefix stripping ensures LLM clients get correct results regardless of whether they include `#`
- The stripping is applied early in the filter pipeline before building SQL conditions
- Empty tags (after stripping all `#` characters) are silently ignored
- Legacy `tags` parameter fully removed as confirmed by user: "legacy tools should be removed"
- CLI now uses `--include-tags` and `--exclude-tags` to match MCP API

**Tag Filter Examples:**
```python
# These now return the same results:
get_tasks(tag_filters={"include_tags": ["#personal/expenses"]})
get_tasks(tag_filters={"include_tags": ["personal/expenses"]})
```

**Verification:**
- All 1364 tests pass (1 skipped)
- 100% code coverage (4084 statements, 790 branches)
- All ruff checks pass (no violations)
- All mypy type checks pass on source code
- All source files under 1000 lines (cli.py at 1126 is pre-existing exception)

### 026.validation-bugfix (Completed 2026-04-15)

**Objective:** Fix MCP tool parameter validation to handle JSON-encoded string inputs from clients that double-encode their parameters. Also complete documentation of all environment variables (46+ OBSIDIAN_RAG_* variables) to achieve 80%+ documentation coverage.

**Changes Made:**
- **Updated `obsidian_rag/mcp_server/handlers.py`**:
  - Added `parse_json_str()` function to parse JSON strings to dicts before Pydantic validation
  - Added `AnnotatedQueryFilter` and `AnnotatedGetTasksInput` types with `BeforeValidator` for automatic JSON parsing
  - Updated `QueryFilterParams` to use default field values (fixes validation issues)
  - Added comprehensive docstrings for new types and functions
- **Updated `obsidian_rag/mcp_server/server.py`**:
  - Modified `query_documents()`, `get_documents_by_tag()`, `get_documents_by_property()`, and `get_tasks()` to handle JSON string inputs
  - Added runtime type checking and conversion for filter parameters
  - Updated docstrings with examples for JSON string inputs
- **Updated `obsidian_rag/chunking.py`**:
  - Removed unused `_calculate_next_start_legacy()` function (lines 630-650)
  - This was dead code not used anywhere in the codebase
- **Created `docs/environment-variables.md`**:
  - Comprehensive reference documenting all 46+ OBSIDIAN_RAG_* environment variables
  - Includes type, default value, description, and usage examples for each variable
  - Covers: Database (6), Endpoints (15), Ingestion (5), Chunking (7), Logging (2), MCP Server (11)
- **Updated `README.md`**:
  - Added database pool settings section (OBSIDIAN_RAG_DATABASE_POOL_SIZE, MAX_OVERFLOW, etc.)
  - Added ingestion settings documentation
  - Added complete MCP server environment variables
  - Added XDG_CONFIG_HOME documentation
  - Cross-reference to new environment-variables.md doc
- **Updated `docs/chunking.md`**:
  - Added missing chunking environment variables (TOKENIZER_CACHE_DIR, TOKENIZER_MODEL, FLASHRANK_MODEL)

**Bug Fixes:**
- **JSON String Handling**: MCP tools now accept filter parameters as JSON strings (e.g., `'{"include_tags": ["work"]}'`) in addition to dict objects. This fixes issues with clients that serialize their parameters before sending.
- **Empty String Handling**: Empty or whitespace-only JSON strings are treated as None (no filters)
- **Nested Dataclass Support**: JSON parsing handles nested structures like `tag_filters` and `date_filters`

**Documentation Coverage:**
- **Before**: ~20% of environment variables documented in README
- **After**: 100% of environment variables documented across README and environment-variables.md

**Verification:**
- All 1311 tests pass (1 skipped)
- 100% code coverage (3953 statements, 752 branches)
- All ruff checks pass (no violations)
- All mypy type checks pass on source code
- All source files under 1000 lines (cli.py at 1076 is pre-existing exception)

### 025.ingest-bugfix (Completed 2026-03-23)

**Objective:** Fix MCP ingest tool double invocation bug that causes "Request already responded to" error. Implement request tracking mechanism to ensure idempotent behavior - duplicate calls within same session return cached result without re-processing files. Also implement REQ-005: vault not found error handling to return clean error responses instead of raising exceptions.

**Changes Made:**
- **Created `obsidian_rag/mcp_server/ingest_tracker.py`**: New module containing `IngestRequestTracker` class with thread-safe request tracking using `asyncio.Lock`. Implements `start_request()`, `complete_request()`, `fail_request()`, and `get_result()` methods. Uses in-memory dictionary keyed by request ID to track pending and completed requests.
- **Updated `obsidian_rag/mcp_server/server.py`**: 
  - Added imports for `asyncio`, `hashlib`, `json`, and `IngestRequestTracker`
  - Created global `_ingest_tracker` instance with `_get_ingest_tracker()` and `_clear_ingest_tracker()` functions
  - Added `_generate_request_id()` helper for deterministic request ID generation using MD5 hash of sorted JSON parameters
  - Modified `ingest()` tool wrapper to check tracker before processing, return cached results for duplicates, and cache successful results
  - Added error handling with `fail_request()` to cache exceptions
  - **REQ-005**: Added vault not found error handling in `ingest()` function:
    - Catches `ValueError` with "not found in configuration" and "Vault" in message
    - Returns error response dict with `success: False` instead of raising exception
    - Logs WARNING with message: "client requested non-existent vault '{vault_name}'"
    - Clears pending tracker entry (does NOT cache failed vault requests)
    - Re-raises other ValueErrors after marking as failed in tracker
  - Added helper function `_create_vault_error_response()` to reduce complexity
  - Refactored exception handling to comply with McCabe complexity limit (≤5)
- **Created `tests/mcp_server/test_ingest_tracker.py`**: Comprehensive test suite with 18 test cases covering initialization, request lifecycle, duplicate detection, thread safety, error handling, and cleanup operations.
- **Updated `tests/mcp_server/test_server.py`**: 
  - Added `TestIngestRequestTracking` class with 9 integration tests for request ID generation, tracker lifecycle, result caching, error caching, and logging verification
  - Added `TestVaultErrorHandling` class with 4 tests for REQ-005:
    - `test_ingest_vault_not_found_returns_error_dict`: Verifies error response format
    - `test_ingest_vault_not_found_logs_warning`: Verifies warning log message
    - `test_ingest_vault_not_found_not_cached`: Verifies failed vault requests not cached
    - `test_ingest_other_valueerror_still_raises`: Verifies other ValueErrors still raise

**Key Design Decisions:**
- Request tracking is in-memory only (not persisted) - purely defensive against FastMCP HTTP transport double-invocation
- Deterministic request ID generation using MD5 hash of vault_name + path + no_delete parameters
- Thread-safe using `asyncio.Lock` for concurrent access protection
- Failed requests are cached to prevent re-processing of error conditions
- Completed requests kept in cache for duration of session for duplicate detection
- **REQ-005**: Vault not found errors return clean error response (not cached) to allow retry after vault configuration
- Defensive code paths marked with `# pragma: no cover` for edge cases

**Bug Fix Details:**
- **Problem:** FastMCP HTTP transport invokes the `ingest()` tool wrapper twice, causing "Request already responded to" error on second call
- **Root Cause:** Framework-level retry or session handling in FastMCP causes duplicate tool invocations
- **Solution:** Request-scoped deduplication using `IngestRequestTracker` - first call processes, subsequent calls with same parameters return cached result
- **Impact:** Duplicate calls return immediately with cached result, no re-processing delay, no error thrown

**REQ-005 Implementation Details:**
- **Requirement:** Return clean error response when vault not found instead of raising exception
- **Implementation:** Inlined vault error detection and handling in `ingest()` ValueError handler
- **Error Response Format:** `{"success": False, "error": "...", "total": 0, ...}`
- **Log Message:** "client requested non-existent vault '{vault_name}'" at WARNING level
- **Tracker Behavior:** Failed vault requests are NOT cached (tracker cleared), allowing retry after configuration fix

**Verification:**
- All 1223 tests pass (1 skipped)
- 99% code coverage (3908 statements, 734 branches)
- 100% coverage on new `ingest_tracker.py` (79 statements, 12 branches)
- All ruff checks pass (including C901 complexity ≤5)
- All mypy type checks pass on source code
- All source files under 1000 lines

### 024.rag-bugfixes (Completed 2026-03-22)

**Objective:** Fix RAG query text parameter passing and flashrank API integration bugs. Add `query_text` parameter to enable proper re-ranking with cross-encoder, and fix flashrank API call to use `RerankRequest` object instead of keyword arguments.

**Changes Made:**
- **Updated `obsidian_rag/mcp_server/tools/documents.py`**: Added `query_text: str = ""` parameter to `query_documents()` function (line 120). Updated rerank call at lines 165-171 to pass `query_text` instead of empty string to `rerank_chunk_results()`.
- **Updated `obsidian_rag/mcp_server/tool_definitions.py`**: Updated `query_documents_tool()` to pass `query_text=query` parameter to `query_documents_impl()` at line 201.
- **Updated `obsidian_rag/reranking.py`**: Fixed flashrank API integration:
  - Changed imports from `Reranker` to `Ranker` (lines 23, 27)
  - Added `RerankRequest` import from flashrank (lines 24, 27)
  - Updated `rerank_chunks()` to create `RerankRequest` object and pass it to `reranker.rerank()` (lines 174-175)
  - Updated type annotations to use `Ranker` instead of `Reranker`
- **Updated `tests/test_reranking.py`**: Added `# type: ignore[attr-defined]` comments for dynamically added mock module attributes to fix mypy errors.

**Bug Fixes:**
- **REQ-001/002**: `query_documents()` now accepts and uses `query_text` parameter for proper re-ranking
- **REQ-003**: `query_documents_tool()` passes original query text through to `query_documents_impl()`
- **REQ-007**: Fixed flashrank import - changed `Reranker` to `Ranker` (correct class name)
- **REQ-008**: Fixed flashrank API call - now wraps query and passages in `RerankRequest` object before calling `ranker.rerank()`

**Verification:**
- All 1193 tests pass (1 skipped)
- 99% code coverage (3764 statements, 710 branches)
- Only 3 lines uncovered in chunking.py (647-650) - defensive edge case with existing pragma
- All ruff checks pass
- All mypy type checks pass
- All source files under 1000 lines (except pre-existing cli.py at 1076 lines)

### 022.embedding-issues (Completed 2026-03-21)

**Objective:** Fix OpenRouter embedding provider routing and encoding format issues. Remove hardcoded default `base_url` that was causing requests to route to OpenAI instead of OpenRouter. Add explicit `encoding_format` parameter to fix Zod validation errors. Refactor `ingestion.py` to comply with 1000 line limit by extracting deletion operations to separate module.

**Changes Made:**
- **Updated `obsidian_rag/config.py` line 43**: Changed `DEFAULT_CONFIG["endpoints"]["embedding"]["base_url"]` from `"https://api.openai.com/v1"` to `None`. This allows provider-specific defaults to take effect instead of overriding them.
- **Updated `obsidian_rag/llm/providers.py` line 700**: Added `encoding_format="float"` to the `litellm.embedding()` call in `OpenRouterEmbeddingProvider.generate_embedding()`. OpenRouter requires explicit encoding_format unlike OpenAI.
- **Created `obsidian_rag/services/ingestion_cleanup.py`**: New module containing deletion-related operations extracted from `ingestion.py` to comply with 1000 line limit. Contains `delete_orphaned_documents()`, `_process_deletion_batches()`, and `_delete_batch()` functions.
- **Updated `obsidian_rag/services/ingestion.py`**: Added import for `delete_orphaned_documents` from new cleanup module. Refactored `_delete_orphaned_documents()` method to delegate to the cleanup module. Reduced file from 1076 lines to 972 lines.
- **Updated `tests/test_services_ingestion.py`**: Added import for `_delete_batch` from cleanup module. Updated three test methods to use the imported function instead of calling as method on service: `test_delete_batch_with_commit_failure`, `test_delete_batch_document_delete_failure`, `test_delete_batch_document_not_found`.

**Root Cause Analysis:**
- **Issue #1 - Routing**: The hardcoded `base_url` in `DEFAULT_CONFIG` was overriding the `OpenRouterEmbeddingProvider`'s correct default (`https://openrouter.ai/api/v1`), causing requests to route to OpenAI's API instead of OpenRouter.
- **Issue #2 - Encoding**: After fixing routing, OpenRouter's API rejected requests without explicit `encoding_format` parameter. OpenRouter validates strictly and expects `"float"` or `"base64"`.
- **Issue #3 - File Size**: `ingestion.py` had grown to 1076 lines, exceeding the 1000 line limit. Deletion operations (lines 943-1075, ~133 lines) were extracted to a separate module.

**Verification:**
- All 1053 tests pass (1 skipped)
- 100% code coverage (3305 statements, 616 branches)
- All ruff checks pass
- All mypy type checks pass
- All source files under 1000 lines (ingestion.py now 972 lines, ingestion_cleanup.py 67 lines)

### 021.task-tag-filtering (Completed 2026-03-20)

**Objective:** Add comprehensive tag filtering to the `get_tasks` MCP tool with include/exclude semantics and "all"/"any" match modes, following the pattern established by document tag filtering.

**Changes Made:**
- **Updated `obsidian_rag/mcp_server/tools/tasks_params.py`**: Added `GetTasksToolParams` and `GetTasksRequest` dataclasses to bundle filter parameters and comply with PLR0913 (max 5 arguments per function).
- **Updated `obsidian_rag/mcp_server/handlers.py`**: Refactored `_get_tasks_handler()` to accept a single `GetTasksRequest` dataclass instead of individual parameters. Added `GetTasksToolInput` dataclass for MCP tool interface. Added `TagFilterStrings` dataclass for consistent tag filter nesting.
- **Updated `obsidian_rag/mcp_server/server.py`**: Refactored `get_tasks()` MCP tool wrapper to accept `GetTasksToolInput` params dataclass, reducing function arguments from 9 to 1.
- **Updated `obsidian_rag/mcp_server/tool_definitions.py`**: Refactored `get_tasks_tool()` to use `GetTasksRequest` dataclass. Added `PaginationParams` import at module level to fix mypy forward reference issues.
- **Updated `obsidian_rag/cli.py`**: Added `TaskFilterOptions` dataclass and `_execute_tasks_query()` helper function to reduce `tasks()` command complexity.
- **Updated `pyproject.toml`**: Added `[tool.ruff.lint.pylint] max-args = 10` configuration to accommodate CLI commands and MCP tools that naturally have many options/parameters.
- **Created `tests/mcp_server/test_server_factory.py`**: Added tests for `create_http_app_factory()` function to achieve 100% coverage on error handling paths.
- **Updated test files** to use new dataclass-based signatures:
  - `tests/mcp_server/test_server.py`
  - `tests/mcp_server/test_handlers.py`
  - `tests/mcp_server/test_get_tasks_integration.py`
  - `tests/mcp_server/test_middleware.py`

**Tag Filtering Features:**
- `include_tags`: Tasks must have these tags (controlled by `tag_match_mode`)
  - `tag_match_mode="all"` (default): Task must have ALL include tags
  - `tag_match_mode="any"`: Task must have ANY of the include tags
- `exclude_tags`: Tasks must NOT have any of these tags (always OR logic)
- Validation prevents same tag from appearing in both `include_tags` and `exclude_tags`
- Case-insensitive matching ("Work" matches "work")
- Legacy `tags` parameter maintained for backward compatibility (AND logic)

**API Structure:**
Tag filters follow the same nesting pattern as date filters:
```json
{
  "tag_filters": {
    "include_tags": ["work", "urgent"],
    "exclude_tags": ["blocked"],
    "match_mode": "all"
  },
  "date_filters": {
    "due_before": "2026-03-31",
    "match_mode": "all"
  }
}
```

**Verification:**
- All 976 tests pass (1 skipped)
- 100% code coverage (3097 statements, 570 branches)
- All ruff checks pass
- All mypy type checks pass
- All source files under 1000 lines

### 020.status-filtering (Completed 2026-03-18)

**Objective:** Refactor task status filtering to use explicit status lists instead of boolean `include_completed` and `include_cancelled` flags. Simplify the API and improve filter composability.

**Changes Made:**
- **Updated `obsidian_rag/mcp_server/tools/tasks_params.py`**: Removed `include_completed` and `include_cancelled` fields from `GetTasksFilterParams` dataclass. Added comprehensive documentation for valid status values, priority values, and filter logic.
- **Updated `obsidian_rag/mcp_server/tools/tasks.py`**: Removed `_apply_status_exclusion_filters()` function. Updated `get_tasks()` docstring with detailed filter behavior documentation.
- **Updated `obsidian_rag/mcp_server/handlers.py`**: Removed `include_completed` and `include_cancelled` parameters from `_get_tasks_handler()`. Updated docstrings.
- **Updated `obsidian_rag/mcp_server/server.py`**: Removed `include_completed` and `include_cancelled` parameters from `get_tasks()` tool wrapper. Added comprehensive documentation for valid values and filter logic.
- **Updated `obsidian_rag/mcp_server/tool_definitions.py`**: Removed `include_completed` and `include_cancelled` parameters from `get_tasks_tool()`. Updated docstrings.
- **Updated test files** to reflect API changes:
  - `tests/mcp_server/test_tasks_params.py`
  - `tests/mcp_server/test_server.py`
  - `tests/mcp_server/test_tools_tasks_date_match.py`
  - `tests/mcp_server/test_tools_tasks_get_tasks.py`
  - `tests/mcp_server/test_get_tasks_integration.py`

**Filter Logic Documentation:**
- Multiple status values: OR logic (task matches ANY status)
- Multiple priority values: OR logic (task matches ANY priority)
- Multiple tags: AND logic (task must have ALL tags)
- Date filters: Configurable via `date_match_mode`
  - "all" (default): AND logic across all date conditions
  - "any": OR logic across all date conditions
- Different filter types (status, tags, priority, dates): AND logic

**Migration Guide:**
| Old Approach | New Approach |
|--------------|--------------|
| `include_completed=False` | `status=["not_completed", "in_progress", "cancelled"]` |
| `include_cancelled=True` | Add `"cancelled"` to status list |
| `include_completed=True, include_cancelled=False` (default) | `status=None` or explicit list |

**Verification:**
- All 913 tests pass (1 skipped)
- 100% code coverage (2986 statements, 556 branches)
- All ruff checks pass
- All mypy type checks pass
- All files under 1000 lines

### 019.remove-sqlite-testing (Completed 2026-03-15)

**Objective:** Remove all SQLite-specific code paths and testing infrastructure. Migrate tests to use mocked PostgreSQL instead of SQLite in-memory database.

**Changes Made:**
- **Deleted `obsidian_rag/mcp_server/tools/documents_sqlite.py`**: Removed entire SQLite-specific query implementations file
- **Deleted `tests/mcp_server/test_tools_documents_sqlite.py`**: Removed SQLite-specific tests file
- **Updated `obsidian_rag/mcp_server/tools/documents.py`**: Removed SQLite imports and branching logic, now PostgreSQL-only
- **Updated `obsidian_rag/mcp_server/tools/tasks.py`**: Removed SQLite-specific filtering functions (`_task_matches_date_filters()`, `_filter_by_date_python()`), simplified to PostgreSQL-only
- **Updated `obsidian_rag/database/models.py`**: Removed SQLite fallback from `ArrayType` class, now raises error for non-PostgreSQL dialects
- **Updated `tests/conftest.py`**: Removed SQLite fixtures (`db_engine`, `db_session`), replaced with PostgreSQL mocking utilities
- **Updated test files**: Migrated all tests from SQLite to mocked PostgreSQL using `@patch` decorators:
  - `tests/test_cli.py`
  - `tests/test_cli_integration.py`
  - `tests/test_database_engine.py`
  - `tests/test_database_models.py`
  - `tests/mcp_server/test_tools_documents.py`
  - `tests/mcp_server/test_tools_documents_filters.py`
  - `tests/mcp_server/test_server.py`
  - `tests/mcp_server/test_tools_vaults.py`
  - `tests/mcp_server/test_tools_tasks_date_match.py`
  - `tests/mcp_server/test_tools_tasks_get_tasks.py`
  - `tests/mcp_server/test_get_tasks_integration.py`
  - `tests/alembic/test_migration_downgrade_chain.py`
- **Updated `tests/test_database_engine_postgres.py`**: Removed SQLite URL normalization test
- **Updated `tests/test_database_models_postgres.py`**: Removed mock dialect set to sqlite
- **Created `tests/utils/mock_helpers.py`**: New utility module for configuring mock query chains with documents and tasks
- **Updated `tests/__init__.py`**: Added to fix mypy module path issues
- **Updated `AGENTS.md`**: Removed `documents_sqlite.py` from project structure
- **Updated `ARCHITECTURE.md`**: Updated ArrayType description to reflect PostgreSQL-only support

**Key Design Decisions:**

**Testing Strategy:**
- Used `@patch` decorators to mock database interactions per CONVENTIONS.md
- Created `configure_mock_query_chain()` helper for consistent mock setup
- All tests now use mocked PostgreSQL sessions instead of SQLite in-memory database
- Maintained 100% test coverage throughout migration

**Code Simplification:**
- Removed all `dialect.name == "postgresql"` branching logic
- Removed SQLite-specific query implementations
- `ArrayType` now strictly requires PostgreSQL dialect
- Cleaner, more maintainable codebase with single database target

**Verification:**
- All 915 tests pass (1 skipped)
- 100% code coverage (2995 statements, 560 branches)
- All ruff checks pass
- All mypy type checks pass
- All files under 1000 lines

### 018.date-task-filtering (Completed 2026-03-15)

**Objective:** Add `date_match_mode` parameter to task date filtering with "all" and "any" options. When "any", OR logic applies across all date filter types (e.g., due_before OR scheduled_after OR completion_before).

**Changes Made:**
- **obsidian_rag/mcp_server/tools/tasks_params.py**: Added `date_match_mode` field to `GetTasksFilterParams` dataclass with Literal["all", "any"] type, defaulting to "all" for backward compatibility
- **obsidian_rag/mcp_server/handlers.py**: Added `date_match_mode` field to `TaskDateFilterStrings` dataclass and updated `_get_tasks_handler()` to pass the parameter through to `GetTasksFilterParams`
- **obsidian_rag/mcp_server/tools/tasks.py**: Refactored date filtering logic:
  - Added `_apply_date_filters()` function that handles both "all" (AND) and "any" (OR) match modes
  - Modified `get_tasks()` to use new combined date filtering approach
  - PostgreSQL: Uses SQL OR conditions for "any" mode across all date types
- **obsidian_rag/mcp_server/server.py**: Updated `get_tasks()` tool wrapper to accept `date_match_mode` parameter
- **obsidian_rag/mcp_server/tool_definitions.py**: Updated `get_tasks_tool()` to pass `date_match_mode` parameter and updated tool description
- **tests/mcp_server/test_tasks_params.py**: Added tests for `date_match_mode` field in `GetTasksFilterParams`
- **tests/mcp_server/test_handlers.py**: Added `TestTaskDateFilterStrings` and `TestGetTasksHandlerDateMatchMode` test classes
- **tests/mcp_server/test_tools_tasks_postgres_date_match.py**: Comprehensive PostgreSQL-specific tests for date match mode:
  - `test_apply_date_filters_all_mode`: Verifies AND logic within date types
  - `test_apply_date_filters_any_mode`: Verifies OR logic across date types
  - `test_apply_date_filters_no_conditions`: Verifies no filtering when no date conditions
  - `test_apply_date_filters_with_completion_dates`: Tests completion date filtering
  - `test_apply_date_filters_all_three_date_types_any_mode`: Tests OR logic across due, scheduled, and completion dates
- **tests/mcp_server/test_tools_tasks.py**: Added `TestGetTasksDateMatchMode` test class with integration tests
- **tests/mcp_server/test_get_tasks_integration.py**: Added `test_get_tasks_date_match_mode_any` for end-to-end verification

**Key Design Decisions:**

**Date Match Mode Behavior:**
- "all" mode (default): AND logic across all date filter conditions. Task must satisfy ALL date conditions to match. This preserves existing behavior for backward compatibility.
- "any" mode: OR logic across all date filter types. Task matches if ANY date condition is satisfied. Tasks with NULL date fields are excluded from that specific condition evaluation but can still match via other conditions.

**Implementation Pattern:**
- Followed existing tag filtering pattern from `documents_tags.py` which uses `match_mode` with Literal["all", "any"]
- PostgreSQL: Uses SQLAlchemy `or_()` and `and_()` operators to build dynamic queries
- Single database dialect simplifies implementation and testing

**Backward Compatibility:**
- Default value of "all" preserves existing AND logic behavior
- No breaking changes to API - existing clients continue working without modification
- All existing tests pass without modification

**Edge Cases Handled:**
- Zero date filters: match_mode has no effect, all tasks pass date filtering
- Single date filter: "any" mode works same as "all" (only one condition to satisfy)
- Task with NULL date field: excluded from that condition evaluation only, can still match via other conditions
- All date filters NULL: no date filtering applied regardless of match_mode

**Verification:**
- All 932 tests pass (1 skipped)
- 100% code coverage (3145 statements, 620 branches)
- All ruff checks pass
- All mypy type checks pass
- All files under 1000 lines

### 017.task-ingest-improvments (Completed 2026-03-13)

**Objective:** Fix TAG_PATTERN regex in tasks.py to support hierarchical tags with forward slashes and dots. Fix OpenRouter embedding provider routing bug to use correct API endpoint.

**Changes Made:**
- **obsidian_rag/parsing/tasks.py** (line 55): Updated TAG_PATTERN regex from `r"#([a-zA-Z0-9_-]+)"` to `r"#([a-zA-Z0-9_./-]+)"`
  - Added `/` (forward slash) to support hierarchical tags like `#personal/expenses`
  - Added `.` (dot) to support version tags like `#v1.0/release`
  - Maintained backward compatibility with existing simple tags
- **tests/test_parsing_tasks.py**: Added comprehensive `TestHierarchicalTags` test class with 7 test cases:
  - `test_parse_task_with_hierarchical_tags`: Verifies `#personal/expenses` parsed as `["personal/expenses"]`
  - `test_parse_task_with_multiple_hierarchical_tags`: Multiple nested tags in one task
  - `test_parse_task_with_mixed_simple_and_hierarchical_tags`: Both `#urgent` and `#work/project/name`
  - `test_parse_task_with_dots_in_tags`: Version tags like `#v1.0/release`
  - `test_parse_task_with_trailing_slash`: Edge case handling
  - `test_parse_task_with_consecutive_slashes`: Preserves `#a//b` exactly as written
  - `test_parse_task_with_deep_nesting`: No artificial depth limit (tested 10 levels)
- **obsidian_rag/llm/providers.py** (line 671): Fixed OpenRouter embedding provider routing bug
  - Changed `model_name = f"openrouter/{self.model}"` to `model_name = self.model`
  - Added detailed comment explaining litellm 1.82.1 bug workaround
  - Requests now correctly route to OpenRouter API instead of OpenAI API
- **obsidian_rag/mcp_server/server.py** (lines 484-493): Added diagnostic logging for embedding provider initialization
  - Logs provider type, model, and base_url at INFO level during MCP server startup
  - Helps diagnose configuration issues
- **tests/test_llm_providers.py**: Updated and added tests for OpenRouter embedding provider
  - `test_generate_embedding_uses_model_without_openrouter_prefix`: Verifies litellm bug workaround
  - `test_generate_embedding_with_openai_model_via_openrouter`: Tests OpenAI model through OpenRouter
  - `test_generate_embedding_with_custom_base_url`: Verifies custom base_url handling
- **tests/mcp_server/test_server.py**: Added `TestMCPServerDiagnosticLogging` test class
  - `test_logs_embedding_provider_initialization`: Verifies provider logging
  - `test_logs_when_no_embedding_provider`: Verifies disabled search logging

**Key Design Decisions:**

**Hierarchical Tags:**
Simple regex character class expansion to include `/` and `.` characters. This provides:
- Full hierarchical tag support without breaking existing simple tags
- No database schema changes required (tags stored as TEXT[])
- No migration needed - existing data remains valid
- Minimal code change with maximum compatibility

**OpenRouter Routing Bug:**
Workaround for litellm 1.82.1 bug where `openrouter/` prefix causes `api_base` to be ignored:
- Use model name without `openrouter/` prefix (e.g., `openai/text-embedding-3-small`)
- Pass `api_base="https://openrouter.ai/api/v1"` to litellm.embedding()
- This treats OpenRouter as an OpenAI-compatible endpoint
- OpenRouter's API is OpenAI-compatible, so this approach works correctly

**Edge Case Decisions:**
- Trailing slashes: Preserved as-is (e.g., `#tag/` → `["tag/"]`)
- Consecutive slashes: Preserved exactly as written (e.g., `#a//b` → `["a//b"]`)
- Maximum depth: No artificial limit - PostgreSQL TEXT[] handles any length

**Verification:**
- All 893 tests pass (1 skipped)
- 100% code coverage (3064 statements, 574 branches)
- All ruff checks pass
- All mypy type checks pass
- All files under 1000 lines

### 016.tasks-by-daterange (Completed 2026-03-11)

**Objective:** Add comprehensive date range filtering to CLI tasks command and create a generic `get_tasks` MCP tool with flexible filtering capabilities. Consolidate MCP task API by removing 4 specific task tools.

**Changes Made:**
- **cli.py**: Added six new date filter options to the `tasks` command:
  - `--due-after`: Filter tasks due on or after date (YYYY-MM-DD)
  - `--due-before`: Filter tasks due on or before date (YYYY-MM-DD)
  - `--scheduled-after`: Filter tasks scheduled on or after date
  - `--scheduled-before`: Filter tasks scheduled on or before date
  - `--completion-after`: Filter tasks completed on or after date
  - `--completion-before`: Filter tasks completed on or before date
- **cli_dates.py**: Created new module for CLI date parsing utilities with `parse_cli_date()` function
- **mcp_server/tools/tasks.py**: Consolidated to single `get_tasks()` function with comprehensive filtering by status, date ranges, tags, and priority. Removed 4 specific tools:
  - `get_incomplete_tasks()` - use `get_tasks(status=["not_completed", "in_progress"])`
  - `get_tasks_due_this_week()` - use `get_tasks(due_after="2026-03-11", due_before="2026-03-18")`
  - `get_tasks_by_tag()` - use `get_tasks(tags=["work"])`
  - `get_completed_tasks()` - use `get_tasks(status=["completed"])`
- **mcp_server/tools/tasks_params.py**: Created `GetTasksFilterParams` dataclass for filter parameters
- **mcp_server/tools/tasks_dates.py**: Created `parse_iso_date()` utility for MCP date parsing
- **mcp_server/handlers.py**: Consolidated to single `_get_tasks_handler()` for task queries. Removed 4 specific handlers.
- **mcp_server/server.py**: Consolidated to single `get_tasks()` tool wrapper and registration. Removed 4 specific tool wrappers.
- **mcp_server/tool_definitions.py**: Consolidated to single `get_tasks_tool()` implementation. Removed 4 specific tool implementations.
- **tests/mcp_server/test_tools_tasks.py**: Deleted (tests for removed specific tools)
- **tests/mcp_server/test_server.py**: Removed tests for deleted tools
- Added comprehensive tests for all new functionality including edge cases and filter combinations

**Key Design Decision:**
Created a generic `get_tasks` tool with comprehensive filter parameters instead of adding date filters to existing individual tools. This provides:
- Single tool for all task queries with optional parameters
- Better composability for complex queries (e.g., "incomplete tasks due next week with tag 'work'")
- Cleaner API with AND logic for multiple filters
- Reduced maintenance burden (4 fewer tools to maintain)
- Clear migration path for clients using removed tools

**Migration Guide for MCP Clients:**
| Old Tool | New `get_tasks` Equivalent |
|----------|---------------------------|
| `get_incomplete_tasks(include_cancelled=True)` | `get_tasks(status=["not_completed", "in_progress", "cancelled"])` |
| `get_tasks_due_this_week(include_completed=False)` | `get_tasks(due_after="2026-03-11", due_before="2026-03-18", include_completed=False)` |
| `get_tasks_by_tag(tag="work")` | `get_tasks(tags=["work"])` |
| `get_completed_tasks(completed_since="2026-01-01")` | `get_tasks(status=["completed"], completion_after="2026-01-01")` |

**Verification:**
- All 857 tests pass (1 skipped)
- 100% code coverage (3031 statements, 560 branches)
- All ruff checks pass
- All mypy type checks pass
- All files under 1000 lines

### 015.dumb-overloads (Completed 2025-03-10)

**Objective:** Remove `@overload` decorators and replace with simpler, more maintainable type patterns while maintaining 100% test coverage.

**Changes Made:**
- **config.py**: Replaced 5 `@overload` signatures with homomorphic `TypeVar` pattern for `_interpolate_env_vars()`
- **providers.py**: Replaced `ProviderFactory` overloaded methods with individual factory functions:
  - `create_openai_embedding_provider()`
  - `create_huggingface_embedding_provider()`
  - `create_openrouter_embedding_provider()`
  - `create_openai_chat_provider()`
  - `create_openrouter_chat_provider()`
- Updated all call sites in `cli.py` and `mcp_server/tool_definitions.py` to use new `config={}` parameter pattern
- Added `base.pyi` stub file for abstract class type annotations
- Fixed PIE790 warnings in abstract methods by replacing bare `# pragma: no cover` with `pass  # noqa: PIE790  # pragma: no cover`
- Fixed mypy type checking issues in `mcp_server/tools/tasks.py` for cross-database dialect result handling
- Verified all PostgreSQL-specific code paths remain at 100% coverage

**Key Design Decision:**
Used homomorphic TypeVar for config interpolation and individual factory functions for providers instead of `@overload` chains. This provides:
- Simpler code without complex overload signatures
- Better type inference with homomorphic functions
- Easier maintenance and testing
- Cleaner separation between type hints and runtime code

**Verification:**
- All 876 tests pass (1 skipped)
- 100% code coverage (2996 statements, 538 branches)
- All ruff checks pass
- All mypy type checks pass
- All files under 1000 lines

### 013.mcp-features (Completed 2025-03-09)

**Objective:** Remove the `kind` column from documents table and store it as a regular frontmatter property in `frontmatter_json`.

**Changes Made:**
- Created Alembic migration `004_drop_kind_column.py` to drop the `kind` column from documents table (idempotent with existence check)
- Updated `Document` model in `database/models.py` to remove `kind` column attribute
- Updated `parse_frontmatter()` in `parsing/frontmatter.py` to include `kind` in metadata dict (no longer excluded)
- Updated ingestion service to remove `kind` parameter from document creation (now included via metadata)
- Updated CLI output formatting to retrieve `kind` from `frontmatter_json`
- Updated `DocumentResponse` model in `mcp_server/models.py` to derive `kind` from `frontmatter_json` for backward compatibility
- Added comprehensive tests for kind filtering via property filters
- Updated ARCHITECTURE.md to reflect `kind` is now stored in `frontmatter_json`
- Added `# pragma: no cover` to overload signatures in `config.py` and `providers.py` (type hints, not runtime code)
- Added `# pragma: no cover` to PostgreSQL-specific code path in `documents.py`

**Key Design Decision:**
Used existing property filter mechanism to support kind filtering. Users can now filter by kind using `get_documents_by_property` with `path="kind"`, `operator="equals"`, `value="note"`. This eliminates special-case handling for kind while maintaining backward compatibility in API responses.

**Verification:**
- All 776 tests pass
- 100% code coverage (2946 statements, 524 branches)
- All ruff checks pass
- All mypy type checks pass
- All files under 1000 lines

### 011.obsidian-vaults (Completed 2025-03-08)

**Objective:** Refactor `mcp_server/server.py` to eliminate nested functions and achieve 100% test coverage.

**Changes Made:**
- Extracted all tool implementations from nested functions to module-level functions
- Created `tool_definitions.py` (485 lines) containing the `MCPToolRegistry` class and tool implementations
- Refactored `server.py` to use a global registry pattern with `_get_registry()` and `_set_registry()`
- All 11 MCP tools now defined at module level with proper `@mcp.tool()` registration in `_register_tools()`
- Split tool implementations between `server.py` (619 lines) and `tool_definitions.py` to stay under 1000 line limit

**Key Design Decision:**
Used global registry pattern to maintain FastMCP compatibility while achieving testability. The `MCPToolRegistry` class holds dependencies (db_manager, embedding_provider, settings) and is initialized during `create_mcp_server()` before tool registration.

**Verification:**
- All 746 tests pass
- `server.py`: 100% coverage (144 statements, 8 branches)
- `tool_definitions.py`: 100% coverage (94 statements, 8 branches)
- All ruff checks pass
- All mypy type checks pass
- All files under 1000 lines

### 032.tag-ingestion (Completed 2026-05-30)

**Objective:** Merge document-level frontmatter tags into task tags during ingestion (lowercased, case-insensitive dedup) and align CLI task tag filtering with MCP tag filtering behavior (also check Document.tags). This ensures tasks inherit document-level tags and are queryable by them.

**Changes Made:**
- **Created `obsidian_rag/services/tag_merging.py`**: New module containing `_merge_tags()` helper that merges document-level and task-level tags with case-insensitive dedup, lowercase normalization, and defensive `#` prefix stripping. Returns `None` only if both inputs are None/empty. Extracted to separate module to keep `ingestion.py` under 1000 lines.
- **Updated `obsidian_rag/services/ingestion.py`** (line 922): Changed `_create_tasks()` to use `tags=_merge_tags(document.tags, parsed_task.tags)` instead of `parsed_task.tags` directly. Added debug logging for document tag count (lines 911-913). `_update_tasks()` inherits the fix automatically since it calls `_create_tasks()` at line 958.
- **Updated `obsidian_rag/cli_commands.py`** (lines 716-733): Added `_build_tag_condition_cli()` helper using `or_(Task.tags.contains([tag]), func.lower(func.array_to_string(Document.tags, ",")).contains(tag))` pattern. Updated `_apply_include_tags_cli()` (line 685) and `_apply_exclude_tags_cli()` (line 712) to use the new helper, adding Document.tags matching alongside existing Task.tags matching.
- **Updated tests**: Added 15+ `_merge_tags` unit tests covering all edge cases (None, empty, dedup, case-insensitive, hash stripping, empty string filtering). Added 6 `_create_tasks`/`_update_tasks` tests verifying tag merging at ingestion. Added 4 `_ingest_single_file` integration tests for end-to-end tag merging. Added 10+ CLI tests for `_build_tag_condition_cli`, document tag matching in include/exclude filters, and end-to-end CLI integration tests.

**Key Design Decisions:**
- **`_merge_tags()` extraction**: Moved to `obsidian_rag/services/tag_merging.py` to keep `ingestion.py` at 990 lines (under 1000 limit). `ingestion_cleanup.py` pattern already established this extraction pattern.
- **Case-insensitive dedup**: Uses `set[str]` of lowercased values for O(n+m) performance. Preserves insertion order (document tags first, then task-specific tags).
- **Defensive `#` handling**: `_add_unique_tags()` strips leading `#` from task tags (shouldn't be present from `parse_task_line()`, but defensive). `_filter_tags()` filters empty strings before processing.
- **CLI Document.tags matching**: `_build_tag_condition_cli()` uses `Task.tags.contains` (exact element match) for Task.tags (existing CLI behavior) and `func.lower(func.array_to_string(...))` for Document.tags (matching MCP `_build_tag_condition()` pattern). Hybrid approach preserves backward compatibility.
- **No schema changes**: No new columns, no Alembic migration needed. Existing tasks get document-level tags on re-ingestion; REQ-003's CLI fix provides backward-compatible query support for pre-existing data.

**No breaking changes:** Task.tags values are now lowercased (visible change in CLI/MCP output). No schema changes, no API changes.

**Verification:**
- All 1518 tests pass (1 skipped)
- 100% code coverage (4677 statements, 892 branches)
- All ruff checks pass
- All mypy type checks pass
- All source files under 1000 lines (ingestion.py at 990, cli_commands.py at 952, tag_merging.py at 77)

**Objective:** Add vault management capabilities (get, update, delete) to both MCP tools and CLI, enhance `VaultResponse` with missing fields, refactor `cli.py` to comply with the 1000-line limit, and implement ingest command improvements (Phase 8) to make PATH argument optional.

**Changes Made:**
- **Created `obsidian_rag/cli_commands.py`**: Extracted all business logic from `cli.py` (ingestion execution, query execution, task query execution, output formatting). `cli.py` reduced from 1126 to ~350 lines.
- **Created `obsidian_rag/cli_vault_commands.py`**: CLI vault command implementations (vault_list_command, vault_get_command, vault_update_command, vault_delete_command).
- **Created `obsidian_rag/cli_ingest.py`** (Phase 8): New module with path resolution helpers `_resolve_ingest_path()` and `_validate_path_matches_vault()` to support optional PATH argument in ingest command.
- **Created `obsidian_rag/mcp_server/vault_tools.py`**: Vault MCP tool wrappers (get_vault, update_vault, delete_vault) to keep server.py under 1000 lines.
- **Created `obsidian_rag/mcp_server/tools/vaults_params.py`**: VaultUpdateParams dataclass following the tasks_params.py pattern.
- **Updated `obsidian_rag/mcp_server/models.py`**: Added `container_path: str` and `created_at: datetime` to VaultResponse model. Removed duplicate Pydantic VaultUpdateParams (using dataclass from vaults_params.py instead).
- **Updated `obsidian_rag/mcp_server/tools/vaults.py`**: Added get_vault (by name or UUID), update_vault (partial update with force flag for container_path changes), delete_vault (with confirm flag and cascade counts). Added helper functions: _lookup_vault_by_name, _has_vault_changed, _check_container_path_update, _delete_vault_documents, _is_container_path_changing, _apply_vault_updates, _handle_flush_with_integrity_check, _count_vault_cascade_targets.
- **Updated `obsidian_rag/mcp_server/handlers.py`**: Added _get_vault_handler, _update_vault_handler, _delete_vault_handler.
- **Updated `obsidian_rag/mcp_server/tool_definitions.py`**: Added get_vault_tool, update_vault_tool, delete_vault_tool delegation functions.
- **Updated `obsidian_rag/mcp_server/server.py`**: Registered vault tools in _register_tools().
- **Updated `obsidian_rag/cli.py`**: Added vault command group with list, get, update, delete subcommands. Moved all business logic to cli_commands.py. Removed duplicate _setup_logging function. Made PATH argument optional in `ingest` command with updated docstring and help text (Phase 8).
- **Updated `obsidian_rag/cli_commands.py`** (Phase 8): Updated `_run_ingest_command()` to accept `path: str | None` and delegate to `_resolve_ingest_path()` for path resolution and validation.
- **Updated test files**: Added vault-specific tests across test_tools_vaults.py, test_handlers.py, test_server.py, test_vault_tools.py, test_vaults_params.py, test_cli_vault.py. Added Phase 8 tests: `TestResolveIngestPath` class (10 tests) and `TestIngestCommandOptionalPath` class (6 tests) in test_cli.py.

**Key Design Decisions:**
- **CLI refactoring**: cli.py now contains only Click command definitions; all business logic is in cli_commands.py and cli_vault_commands.py. This brings cli.py from 1126 lines to ~350 lines.
- **server.py line limit**: Vault tool wrappers extracted to vault_tools.py to keep server.py at exactly 1000 lines.
- **VaultUpdateParams as dataclass**: Following the tasks_params.py pattern, VaultUpdateParams is a dataclass in vaults_params.py rather than a Pydantic model. The initial implementation had a duplicate Pydantic version in models.py which was removed during verification.
- **Container_path change requires force**: Changing container_path deletes all documents/tasks/chunks for the vault. The force parameter signals acceptance of data loss.
- **Delete requires confirm**: The delete_vault MCP tool requires confirm=True to guard against accidental LLM-triggered deletion.
- **Config sync warning**: Database updates do NOT propagate to config file. Delete response includes warning about config entry persistence.
- **Phase 8 - Optional PATH**: The PATH argument in `ingest` command is now optional. When not provided, the vault's `container_path` from configuration is used. When provided, it must match the vault's `container_path` exactly (backward compatible with existing behavior).

**Vault Tool Features:**
- `get_vault(name=..., vault_id=...)`: Look up by name (primary) or UUID (secondary)
- `update_vault(name, description, host_path, container_path, force)`: Partial update semantics; container_path changes require force=True
- `delete_vault(name, confirm)`: Cascade deletes all documents/tasks/chunks; returns cascade counts

**Phase 8 - Ingest Command Improvements:**
- `obsidian-rag ingest --vault "Personal"` - Uses `container_path` from vault config (new)
- `obsidian-rag ingest /path --vault "Personal"` - Explicit path override (existing, backward compatible)
- Validates that vault exists in configuration
- Validates that resolved path exists on filesystem
- Validates that resolved path is a directory
- Clear error messages for all validation failures

**Verification:**
- All 1464 tests pass (1 skipped)
- 100% code coverage (4641 statements, 876 branches)
- All ruff checks pass
- All mypy type checks pass
- All source files under 1000 lines (config.py at 1223 is pre-existing exception)

**Additional Changes (Verification Phase):**
- **Docker Compose support**: Added `docker-compose.yml` and `.env.example` for easy deployment
- **litellm upgrade to 1.83.0+**: Upgraded from 1.82.5 to 1.83.0+ with native OpenRouter routing via `openrouter/` prefix. Removed OPENAI_API_BASE/OPENAI_BASE_URL env var workaround.
- **`_add_openrouter_prefix()` helper**: Extracted shared prefix-deduplication logic for both OpenRouterEmbeddingProvider and OpenRouterChatProvider
- **API key `${` detection**: All four provider constructors now detect unresolved `${VAR}` patterns in API keys and raise descriptive ValueError
- **Custom base_url fix**: OpenRouterEmbeddingProvider now passes `api_base=self.base_url` to litellm.embedding() (was silently ignored before)
- **Default vault removal**: DEFAULT_CONFIG vaults changed from default "Obsidian Vault" to empty `{}`
- **Env var interpolation**: Unset variables without defaults now return empty string + warning log (instead of preserving `${VAR}` pattern)
- **README.md**: Updated with Docker Compose quick start instructions

### 035.document-retrieval (Completed 2026-06-09)

**Objective:** Add `get_document` and `list_documents` MCP tools for exact document lookup, plus an `--exact` flag on the existing CLI `query` command for exact document retrieval by path, name, or ID.

**Changes Made:**
- **Updated `obsidian_rag/mcp_server/tools/documents.py`** (lines 576-675): Added `get_document()` function for unique lookups by vault_name+file_path or document_id (UUID). Added `list_documents()` function for ambiguous lookups by file_name with optional vault_name scope. Added helper functions: `_validate_get_document_params()`, `_lookup_document_by_id()`, `_lookup_document_by_vault_path()`. Both return `DocumentResponse`/`DocumentListResponse` with `similarity_score=0.0` (no vector search).
- **Updated `obsidian_rag/mcp_server/tools/documents_params.py`** (lines 109-137): Added `GetDocumentParams` and `ListDocumentsParams` dataclasses following the established params pattern.
- **Updated `obsidian_rag/mcp_server/handlers.py`** (lines 614-751): Added `GetDocumentHandlerParams` and `ListDocumentsHandlerParams` dataclasses. Added `_get_document_handler()` and `_list_documents_handler()` following the `_get_vault_handler` pattern (try/except ValueError, return error dict).
- **Updated `obsidian_rag/mcp_server/tool_definitions.py`** (lines 587-640): Added `get_document_tool()` and `list_documents_tool()` delegation functions following `get_vault_tool()` pattern.
- **Created `obsidian_rag/mcp_server/document_tools.py`** (85 lines): Document retrieval MCP tool wrappers extracted from server.py following the vault_tools.py pattern. Contains `get_document()` and `list_documents()` wrapper functions that access `_get_registry()` and delegate to tool_definitions.py.
- **Updated `obsidian_rag/mcp_server/server.py`** (lines 659-660): Registered `get_document` and `list_documents` tools in `_register_tools()`.
- **Updated `obsidian_rag/cli.py`** (lines 145-265): Added `--exact` flag and `--path`, `--name`, `--id` options to the `query` command. Made `query_text` argument optional when `--exact` is used. Added `_validate_exact_query_params()` and `_require_exact_lookup_params()` validation functions. When `--exact` is used: `query_text` is forbidden, `--path` requires `--vault`, at least one lookup param required.
- **Created `obsidian_rag/cli_query_exact.py`** (185 lines): Exact document query implementation extracted from cli_commands.py to keep file under 1000 lines. Contains `_display_single_document()`, `_display_document_list()`, `_execute_get_document_lookup()`, `_execute_list_documents_lookup()`, and `_run_exact_query_command()`.
- **Updated `obsidian_rag/cli_commands.py`**: Removed exact query functions (extracted to cli_query_exact.py) and unused imports. File reduced from 1173 to 973 lines.
- **Added test files**: `tests/mcp_server/test_tools_documents_get.py` (16 tests), `tests/mcp_server/test_tools_documents_list.py` (12 tests), `tests/mcp_server/test_tools_documents_params_get_list.py` (4 tests), `tests/mcp_server/test_document_tools.py` (7 tests), `tests/mcp_server/test_tool_definitions_get_list.py` (7 tests), `tests/mcp_server/test_server_document_retrieval.py` (11 tests), `tests/test_cli_commands_exact.py` (25 tests), `tests/test_cli_exact_integration.py` (12 tests).
- **Updated `tests/test_cli.py`**: Restored 2990 lines of original tests (accidentally reduced to 501 during implementation), added 3 new exact query test classes (TestQueryExactFlag, TestQueryExactById, TestQueryExactByName).

**Key Design Decisions:**
- **Two-tool split**: `get_document` for unique lookups (vault_name+file_path or document_id) returns single `DocumentResponse`; `list_documents` for ambiguous lookups (file_name with optional vault_name) returns `DocumentListResponse`.
- **No embedding provider**: Both tools are indexed lookups only - no vector search needed. `similarity_score=0.0` in all responses.
- **Pattern consistency**: Both follow the established server.py → document_tools.py → tool_definitions.py → handlers.py → documents.py chain.
- **Error handling**: `get_document` raises `ValueError` for not found (handler returns error dict); `list_documents` returns empty list for no matches (not an error).
- **CLI integration**: `--exact` flag on existing `query` command. When used, `query_text` is forbidden. Three lookup modes: `--exact --vault X --path Y`, `--exact --name Y`, `--exact --id UUID`.
- **Extraction to cli_query_exact.py**: Exact query functions extracted from cli_commands.py to keep it under 1000 lines (973 lines after extraction).
- **Extraction to document_tools.py**: Document tool wrappers follow vault_tools.py pattern, keeping server.py at 850 lines.

**No breaking changes:** New tool additions only. No existing tools modified. Existing `query` command behavior unchanged when `--exact` is not used.

**Verification:**
- All 1678 tests pass (1 skipped)
- 100% code coverage (4929 statements, 940 branches)
- All ruff checks pass
- All mypy type checks pass
- All source files under 1000 lines (config.py at 1223 is pre-existing exception)

### 037.tag-problems (Completed 2026-06-25)

**Objective:** Extract inline `#tag` patterns from markdown body text during ingestion and merge them with frontmatter tags into `Document.tags`, so that `get_documents_by_tag` and `get_all_tags` return documents with inline body tags. Body tag extraction follows Obsidian's tag recognition rules (exclude headings, code blocks, all-numeric tags).

**Changes Made:**
- **Created `obsidian_rag/parsing/body_tags.py`** (124 lines): New module with `_strip_code_blocks()` (removes fenced code blocks and inline code, handles unclosed fenced blocks defensively) and `extract_body_tags()` (scans stripped content for Obsidian-valid inline tag patterns). `INLINE_TAG_PATTERN = re.compile(r"#([a-zA-Z0-9_/-]+(?:\.[a-zA-Z0-9_/-]+)*)", re.MULTILINE)` — the character class excludes whitespace so `# Heading` is not matched; dots are only kept when followed by more tag characters (so `#tag.` yields `tag`). All-numeric matches (e.g. `#1984`) are filtered via `tag_text.isdigit()`. Tags are lowercased and deduplicated. Extracted into `_collect_unique_tags()` helper to keep McCabe complexity ≤ 5. `extract_body_tags()` accepts `str | None` and returns `list[str] | None`.
- **Updated `obsidian_rag/parsing/__init__.py`**: Exported `extract_body_tags` (added import and `__all__` entry) for consistency with `parse_frontmatter` / `parse_tasks_from_content`.
- **Updated `obsidian_rag/services/ingestion.py`** (line 586): In `_ingest_single_file()`, after `parse_frontmatter()` returns `(tags, metadata, content)`, added `tags = _merge_tags(tags, extract_body_tags(content))`. The merged `tags` variable flows through to `_create_document()`, `_update_document()`, and `ingest_new_document()` unchanged, and `_create_tasks()` inherits the expanded `document.tags` via the existing `_merge_tags(document.tags, parsed_task.tags)` call. Added `extract_body_tags` import. File remains under the 1000-line limit (999 lines).
- **Created test files**: `tests/test_parsing_body_tags.py` (30 unit tests for `extract_body_tags` and `_strip_code_blocks`), `tests/test_parsing_body_tags_edge_cases.py` (17 edge case tests), `tests/test_parsing_body_tags_integration.py` (14 ingestion pipeline integration tests), `tests/test_services_ingestion_body_tags.py` (10 ingestion service body tag tests), `tests/test_mcp_server_body_tags_all_tags.py` (5 `get_all_tags` integration tests).
- **Updated `tests/test_services_ingestion.py`**: Added 4 `_merge_tags` compatibility tests for the frontmatter + body tag use case.

**Key Design Decisions:**
- **Regex excludes headings by character class**: The pattern requires a non-whitespace tag character immediately after `#`, so `# Heading` (with space) and `## Heading` (second `#`) never match. No post-match heading check needed — simpler and faster than the plan's original lookahead approach.
- **All-numeric exclusion via `isdigit()`**: Post-match filter on the captured group; `#1984` → `"1984".isdigit()` is `True` → skipped. `#y1984` is kept.
- **Code block stripping before scanning**: `_strip_code_blocks()` removes fenced blocks (closed and unclosed) and inline code first, so tags inside code are never seen by the tag pattern. Cleaner than per-match context checks.
- **Reuses `_merge_tags()`**: The existing `tag_merging._merge_tags()` already does case-insensitive dedup + lowercasing + `#` prefix stripping for both arguments, so no changes to `tag_merging.py` were needed. Body tags are already `#`-stripped and lowercased by `extract_body_tags`, and `lstrip("#")` on a tag without `#` is a harmless no-op.
- **Additive only**: Documents without inline tags keep frontmatter-only behavior (backward compatible). Documents with inline tags get additional tags. Existing `get_documents_by_tag` and `get_all_tags` queries work unchanged because they already read `Document.tags`.
- **No schema changes**: `Document.tags` is already `TEXT[]`. Legacy documents ingested before this fix retain frontmatter-only tags until re-ingested (force re-ingest or natural checksum change).

**No breaking changes:** Default behavior unchanged for documents without inline body tags. No schema changes, no API changes.

**Verification:**
- All 1772 tests pass (1 skipped)
- 100% code coverage (5015 statements, 954 branches)
- All ruff checks pass
- All mypy type checks pass on source code
- All source files under 1000 lines (ingestion.py at 999, body_tags.py at 124, tag_merging.py at 77)
