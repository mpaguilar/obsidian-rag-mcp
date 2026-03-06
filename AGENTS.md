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
│   ├── middleware.py            # HTTP request/response logging middleware
│   ├── models.py                # Pydantic request/response models
│   ├── server.py                # FastMCP server setup
│   ├── session_manager.py       # Session lifecycle and metrics tracking
│   └── tools/                   # MCP tools
│       ├── __init__.py
│       ├── documents.py         # Document query tools (public API)
│       ├── documents_filters.py # Property filter implementations
│       ├── documents_params.py  # Filter parameter dataclasses
│       ├── documents_postgres.py # PostgreSQL-specific queries
│       ├── documents_sqlite.py  # SQLite-specific queries
│       ├── documents_tags.py    # Tag filtering logic
│       └── tasks.py             # Task query tools
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
- `obsidian-rag ingest <path>` - Ingest documents from vault path
- `obsidian-rag query <search>` - Semantic search documents
- `obsidian-rag tasks [options]` - Query tasks

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

### Coverage Status

| Module | Coverage | Notes |
|--------|----------|-------|
| `config.py` | 97% | Environment variable interpolation branches |
| `parsing/` | 100% | All parsing modules fully covered |
| `database/engine.py` | 100% | Complete coverage |
| `database/models.py` | 100% | Complete coverage |
| `llm/base.py` | 100% | Complete coverage |
| `llm/providers.py` | 100% | Complete coverage |
| `services/ingestion.py` | 100% | Complete coverage |
| `cli.py` | 95% | Error handling and edge cases |
| `mcp_server/__main__.py` | 100% | Complete coverage |
| `mcp_server/server.py` | 72% | Tool registration and logging functions |
| `mcp_server/middleware.py` | 100% | Complete coverage |
| `mcp_server/models.py` | 100% | Complete coverage |
| `mcp_server/session_manager.py` | 98% | Defensive timing branches |
| `mcp_server/tools/documents.py` | 94% | PostgreSQL-specific tag filtering requires integration testing |
| `mcp_server/tools/documents_filters.py` | 99% | Single defensive branch |
| `mcp_server/tools/documents_postgres.py` | 93% | SQLite-specific defensive branches |
| `mcp_server/tools/documents_sqlite.py` | 100% | Complete coverage |
| `mcp_server/tools/documents_tags.py` | 100% | Complete coverage |
| `mcp_server/tools/documents_params.py` | 100% | Complete coverage |
| `mcp_server/tools/tasks.py` | 100% | Complete coverage |

> **Technical Implementation Details**: For architecture patterns, component details, and data flow, see [ARCHITECTURE.md](./ARCHITECTURE.md). For coding conventions and standards, see [CONVENTIONS.md](./CONVENTIONS.md).
