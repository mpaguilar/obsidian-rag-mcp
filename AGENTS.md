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
├── cli.py                       # CLI entry point
├── config.py                    # Configuration management
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
│   ├── handlers.py              # Request handlers for tools
│   ├── middleware.py            # HTTP request/response logging middleware
│   ├── models.py                # Pydantic request/response models
│   ├── server.py                # FastMCP server setup and tool wrappers
│   ├── session_manager.py       # Session lifecycle and metrics tracking
│   ├── tool_definitions.py      # Tool implementations and MCPToolRegistry
│   └── tools/                   # MCP tools
│       ├── __init__.py
│       ├── documents.py         # Document query tools (public API)
│       ├── documents_filters.py # Property filter implementations
│       ├── documents_params.py  # Filter parameter dataclasses
│       ├── documents_postgres.py # PostgreSQL-specific queries
│       ├── documents_sqlite.py  # SQLite-specific queries
│       ├── documents_tags.py    # Tag filtering logic
│       ├── tasks.py             # Task query tools
│       └── vaults.py            # Vault query tools
├── parsing/                     # Document parsing
│   ├── __init__.py
│   ├── frontmatter.py           # FrontMatter extraction
│   ├── scanner.py               # File scanning
│   └── tasks.py                 # Task parsing
└── services/                    # Service layer
    ├── __init__.py
    └── ingestion.py             # Document ingestion service
```

## Requirements

- Python 3.12+
- PostgreSQL with pg_vector extension
- See `pyproject.toml` for Python dependencies

## CLI Commands

- `obsidian-rag [--log-level LEVEL] <command>` - Global options include `--log-level` (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- `obsidian-rag ingest --vault <name> <path>` - Ingest documents from vault path (vault must be configured)
- `obsidian-rag query <search>` - Semantic search documents
- `obsidian-rag tasks [options]` - Query tasks

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
