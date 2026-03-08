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
