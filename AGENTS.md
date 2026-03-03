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
└── parsing/                     # Document parsing
    ├── __init__.py
    ├── frontmatter.py           # FrontMatter extraction
    ├── scanner.py               # File scanning
    └── tasks.py                 # Task parsing
```

## Requirements

- Python 3.12+
- PostgreSQL with pg_vector extension
- See `pyproject.toml` for Python dependencies

## CLI Commands

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

## Development

- All code must pass ruff linting
- 100% test coverage on core modules (config, parsing, database engine)
- 95%+ coverage on modules with platform-specific code (database models, llm providers)
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

> **Technical Implementation Details**: For architecture patterns, HTMX/SSE implementation specifics, and testing patterns, see [ARCHITECTURE.md](./ARCHITECTURE.md). For coding conventions and standards, see [CONVENTIONS.md](./CONVENTIONS.md).
