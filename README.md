# Obsidian RAG

CLI tool for ingesting Obsidian markdown documents into PostgreSQL with vector embeddings and semantic search capabilities.

## Overview

Obsidian RAG provides a complete pipeline for:

- **Ingesting** Obsidian markdown documents into a PostgreSQL database with pg_vector support
- **Extracting** tasks from markdown content with metadata parsing
- **Searching** documents using semantic similarity with vector embeddings
- **Querying** tasks with flexible filtering

## Installation

### Requirements

- Python 3.12+
- PostgreSQL 14+ with pg_vector extension

### Install from Source

```bash
# Clone the repository
git clone <repository-url>
cd obsidian-rag

# Install with pip
pip install -e .

# Or install with optional dependencies for specific providers
pip install -e ".[openai]"      # For OpenAI embeddings
pip install -e ".[local]"       # For local HuggingFace embeddings
pip install -e ".[all]"         # All optional dependencies
```

### Database Setup

1. Create a PostgreSQL database:

```bash
createdb obsidian_rag
```

2. Enable the pg_vector extension:

```sql
psql -d obsidian_rag -c "CREATE EXTENSION IF NOT EXISTS pgvector;"
```

## Usage

### Ingest Documents

Scan and ingest markdown files from an Obsidian vault:

```bash
# Basic ingestion
obsidian-rag ingest /path/to/vault

# Dry run to preview changes without writing to database
obsidian-rag ingest /path/to/vault --dry-run

# Verbose output for detailed progress
obsidian-rag ingest /path/to/vault --verbose
```

**Output:**
- Total files processed
- New documents added
- Documents updated (content changed)
- Documents unchanged (already up-to-date)
- Errors (if any)

### Semantic Search

Query documents using natural language:

```bash
# Basic search
obsidian-rag query "project planning methodologies"

# Limit results
obsidian-rag query "meeting notes" --limit 5

# JSON output for programmatic use
obsidian-rag query "architecture decisions" --format json
```

**Output formats:**
- `table` (default): Human-readable format with file names, paths, and similarity scores
- `json`: Machine-readable JSON array with document metadata

### Task Queries

Filter and display tasks extracted from your documents:

```bash
# List all incomplete tasks
obsidian-rag tasks

# Filter by status
obsidian-rag tasks --status not_completed
obsidian-rag tasks --status completed
obsidian-rag tasks --status in_progress
obsidian-rag tasks --status cancelled

# Filter tasks due before a date
obsidian-rag tasks --due-before 2024-12-31

# Filter by tag
obsidian-rag tasks --tag urgent

# Combine filters
obsidian-rag tasks --status not_completed --due-before 2024-12-31 --limit 50
```

**Supported task syntax:**

```markdown
- [ ] Regular task
- [x] Completed task
- [/] In progress task
- [-] Cancelled task
- [ ] Task with priority [priority:: high]
- [ ] Task with due date [due:: 2024-12-31]
- [ ] Task with tag #urgent
- [ ] Task with repeat [repeat:: every day]
- [ ] Task with scheduled date [scheduled:: 2024-12-31]
- [ ] Task with completion date [completion:: 2024-12-31]
```

## Configuration

Configuration is layered with the following precedence (highest to lowest):

1. **CLI flags** (e.g., `--embedding-provider openai`)
2. **Environment variables** (e.g., `OBSIDIAN_RAG_EMBEDDING_PROVIDER=openai`)
3. **Config files** (YAML format)
4. **Default values**

### Config File Locations

Config files are searched in order:

1. `$PWD/.obsidian-rag.yaml` - Project-specific config
2. `$XDG_CONFIG_HOME/obsidian-rag/config.yaml` (or `~/.config/obsidian-rag/config.yaml`) - User config

### Example Configuration

```yaml
# .obsidian-rag.yaml
endpoints:
  embedding:
    provider: openai
    model: text-embedding-3-small
    api_key: ${OPENAI_API_KEY}
    base_url: https://api.openai.com/v1
  
  analysis:
    provider: openai
    model: gpt-4
    api_key: ${OPENAI_API_KEY}
    temperature: 0.7
    max_tokens: 2000
  
  chat:
    provider: openai
    model: gpt-4
    api_key: ${OPENAI_API_KEY}
    temperature: 0.8

database:
  url: postgresql://localhost/obsidian_rag

ingestion:
  batch_size: 100
  max_file_size_mb: 10
  progress_interval: 10

logging:
  level: INFO
  format: text
```

### Environment Variable Interpolation

Config files support environment variable interpolation:

```yaml
# Basic syntax
api_key: ${OPENAI_API_KEY}

# With default value
api_key: ${OPENAI_API_KEY:-default_key}
```

### Environment Variables

All settings can be configured via environment variables using the prefix `OBSIDIAN_RAG_`:

```bash
# Database URL
export OBSIDIAN_RAG_DATABASE_URL="postgresql://user:pass@localhost/obsidian_rag"

# Embedding provider settings
export OBSIDIAN_RAG_ENDPOINTS_EMBEDDING_PROVIDER="openai"
export OBSIDIAN_RAG_ENDPOINTS_EMBEDDING_MODEL="text-embedding-3-small"
export OBSIDIAN_RAG_ENDPOINTS_EMBEDDING_API_KEY="sk-..."

# Logging
export OBSIDIAN_RAG_LOGGING_LEVEL="DEBUG"
```

## LLM Providers

### OpenAI (Cloud)

Requires `litellm` package:

```bash
pip install litellm
```

Configuration:

```yaml
endpoints:
  embedding:
    provider: openai
    model: text-embedding-3-small
    api_key: ${OPENAI_API_KEY}
```

### HuggingFace (Local)

Requires `langchain` packages:

```bash
pip install langchain langchain-huggingface
```

Configuration:

```yaml
endpoints:
  embedding:
    provider: huggingface
    model: sentence-transformers/all-MiniLM-L6-v2
```

## Document Features

### Frontmatter Support

Documents can include YAML frontmatter:

```markdown
---
kind: note
tags: [project, planning]
priority: high
---

# My Document

Content here...
```

**Reserved frontmatter fields:**
- `kind`: Document type/category
- `tags`: List of tags (string or list format)

All other frontmatter is stored as JSON in the `frontmatter_json` column.

### Task Parsing

Tasks are automatically extracted from document content using checkbox syntax:

- `[ ]` - Not completed
- `[x]` - Completed
- `[/]` - In progress
- `[-]` - Cancelled

**Task metadata extracted:**
- Tags (`#tag`)
- Due dates (`[due:: YYYY-MM-DD]`)
- Scheduled dates (`[scheduled:: YYYY-MM-DD]`)
- Completion dates (`[completion:: YYYY-MM-DD]`)
- Priority (`[priority:: highest|high|normal|low|lowest]`)
- Recurrence (`[repeat:: every day|week|month|year]`)
- Custom metadata (`[key:: value]`)

## Database Schema

### Documents Table

| Column | Type | Description |
|--------|------|-------------|
| id | UUID | Primary key |
| file_path | TEXT | Unique file path (indexed) |
| file_name | TEXT | File name (indexed) |
| content | TEXT | Document content |
| content_vector | VECTOR | Vector embedding (configurable dimension) |
| checksum_md5 | CHAR(32) | MD5 checksum for change detection |
| created_at_fs | TIMESTAMP | Filesystem creation date |
| modified_at_fs | TIMESTAMP | Filesystem modification date |
| ingested_at | TIMESTAMP | Last ingestion timestamp |
| kind | TEXT | Document type from frontmatter |
| tags | TEXT[] | Tags from frontmatter |
| frontmatter_json | JSONB | Additional frontmatter properties |

### Tasks Table

| Column | Type | Description |
|--------|------|-------------|
| id | UUID | Primary key |
| document_id | UUID | Foreign key to documents |
| line_number | INTEGER | Position in document |
| raw_text | TEXT | Full task line |
| status | ENUM | Task status |
| description | TEXT | Task text without metadata |
| tags | TEXT[] | Extracted tags |
| repeat | TEXT | Recurrence pattern |
| scheduled | DATE | Scheduled date |
| due | DATE | Due date |
| completion | DATE | Completion date |
| priority | ENUM | Priority level |
| custom_metadata | JSONB | Other metadata |

## Development

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=obsidian_rag --cov-branch --cov-report=term-missing

# Run specific test file
pytest tests/test_cli.py
```

### Code Quality

```bash
# Run linting
ruff check

# Fix auto-fixable issues
ruff check --fix
```

### Project Structure

```
obsidian_rag/
├── __init__.py
├── cli.py              # CLI entry point
├── config.py           # Configuration management
├── database/
│   ├── __init__.py
│   ├── engine.py       # Database connection
│   └── models.py       # SQLAlchemy models
├── llm/
│   ├── __init__.py
│   ├── base.py         # Base provider classes
│   └── providers.py    # Provider implementations
└── parsing/
    ├── __init__.py
    ├── frontmatter.py  # Frontmatter extraction
    ├── scanner.py      # File scanning
    └── tasks.py        # Task parsing
```

## License

[Add your license information here]

## Contributing

[Add contribution guidelines here]
