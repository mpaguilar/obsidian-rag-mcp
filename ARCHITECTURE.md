# Architecture: Obsidian RAG

## System Overview

The Obsidian RAG system is a Python library designed to ingest, store, and query Obsidian markdown documents with support for vector embeddings and task management.

## Core Components

### 1. Database Layer (`database/`)

#### ArrayType TypeDecorator

The `ArrayType` class in `models.py` provides cross-database compatibility for PostgreSQL arrays:
- Uses `postgresql.ARRAY` type when connected to PostgreSQL (production)
- Falls back to `JSON` type for other databases (e.g., SQLite for testing)
- Used for `tags` columns in both Document and Task models
- Ensures schema-model alignment regardless of database backend

#### Vector Index

The `content_vector` column uses HNSW (Hierarchical Navigable Small World) index for fast similarity search:
- **Index type**: HNSW with `vector_cosine_ops` operator class
- **Dimension limit**: Both HNSW and IVFFLAT have a 2000 dimension limit in pgvector
- **Configuration**: `database.vector_dimension` setting (default: 1536)
- **Validation**: Configuration validates that `vector_dimension ≤ 2000`

#### Models (`models.py`)

**documents table:**
- `id` (UUID, PK)
- `file_path` (TEXT, UNIQUE, indexed)
- `file_name` (TEXT, indexed)
- `content` (TEXT)
- `content_vector` (VECTOR(N) - configurable dimension, default 1536)
- `checksum_md5` (CHAR(32))
- `created_at_fs` (TIMESTAMP) - filesystem creation date
- `modified_at_fs` (TIMESTAMP) - filesystem modification date
- `ingested_at` (TIMESTAMP) - when we last ingested
- `kind` (TEXT, nullable) - from FrontMatter
- `tags` (TEXT[], nullable) - from FrontMatter, deduplicated
- `frontmatter_json` (JSONB) - all other FrontMatter properties

**tasks table:**
- `id` (UUID, PK)
- `document_id` (UUID, FK → documents.id, ON DELETE CASCADE)
- `line_number` (INTEGER) - position in document
- `raw_text` (TEXT) - full task line
- `status` (ENUM: 'not_completed', 'completed', 'in_progress', 'cancelled')
- `description` (TEXT) - task text without metadata
- `tags` (TEXT[]) - extracted #tags
- `repeat` (TEXT, nullable) - recurrence pattern
- `scheduled` (DATE, nullable)
- `due` (DATE, nullable)
- `completion` (DATE, nullable)
- `priority` (ENUM: 'highest', 'high', 'normal', 'low', 'lowest', default 'normal')
- `custom_metadata` (JSONB) - other [key:: value] pairs

#### Engine (`engine.py`)

Database connection management using SQLAlchemy with:
- Connection pooling
- Session context managers
- Table creation/migration support

### 2. Parsing Layer (`parsing/`)

#### FrontMatter Extraction (`frontmatter.py`)

- Extracts YAML frontmatter from markdown documents
- Supports `kind` (string), `tags` (string or list), and arbitrary metadata
- Handles corrupted frontmatter gracefully (logs warning, continues)

#### Task Parsing (`tasks.py`)

- Identifies tasks via checkbox prefix patterns: `[ ]`, `[x]`, `[/]`, `[-]`
- Extracts metadata: tags (`#tag`), `repeat`, `scheduled`, `due`, `completion`, `priority`
- Supports dateutil.rrule for recurrence patterns
- Maximum 10,000 tasks per document

#### File Scanner (`scanner.py`)

- Discovers `.md` files in specified directories
- Calculates MD5 checksums for change detection
- Skips files > 10MB
- Handles permission errors gracefully

### 3. LLM Provider Layer (`llm/`)

#### Base Classes (`base.py`)

- Provider-agnostic interface for LLM operations
- Three endpoint types: embedding, analysis, chat
- Factory pattern for provider instantiation

#### Provider Implementations (`providers.py`)

**Supported Providers:**
- OpenAI (embeddings, analysis, chat) - uses `litellm`
- OpenRouter (embeddings, chat) - uses `litellm` with `openrouter/` prefix
- HuggingFace (local embeddings) - uses `langchain.embeddings.HuggingFaceEmbeddings`
- Extensible design for additional providers

**Library Usage:**
- `litellm`: Provider-agnostic LLM connectivity for OpenAI and OpenRouter endpoints
  - `OpenAIEmbeddingProvider`: Uses `litellm.embedding()`
  - `OpenAIChatProvider`: Uses `litellm.completion()`
  - `OpenRouterEmbeddingProvider`: Uses `litellm.embedding()` with `openrouter/` prefix
  - `OpenRouterChatProvider`: Uses `litellm.completion()` with `openrouter/` prefix
- `langchain`: Local embedding models via HuggingFace
  - `HuggingFaceEmbeddingProvider`: Uses `HuggingFaceEmbeddings`

**Configuration per endpoint:**
- `provider`: Provider type ('openai' or 'huggingface')
- `model`: Model name
- `api_key`: Authentication (via env var interpolation)
- `base_url`: Optional custom endpoint
- `temperature`: Generation temperature
- `max_tokens`: Maximum response tokens

### 4. Configuration (`config.py`)

Layered configuration system:

**Sources (highest to lowest precedence):**
1. CLI flags: `--<section>-<key>` format
2. Environment variables: `OBSIDIAN_RAG_<SECTION>_<KEY>` format
3. Config files: YAML format
4. Default values

**Config File Locations:**
1. `$PWD/.obsidian-rag.yaml`
2. `$XDG_CONFIG_HOME/obsidian-rag/config.yaml`

**Features:**
- Environment variable interpolation: `${VAR}` or `${VAR:-default}`
- Nested configuration merging
- Pydantic validation
- **Vector dimension validation**: Enforces maximum of 2000 dimensions (pgvector limit)
- **Cross-validation**: Validates embedding provider dimension matches `database.vector_dimension`

### 5. CLI Layer (`cli.py`)

Entry point for all user interactions:

**Commands:**
- `ingest <path>`: Scan and ingest documents
  - Options: `--dry-run`, `--verbose`
  - Progress reporting: total, new, updated, skipped, errors
  - Batch processing with configurable size
  
- `query <search>`: Semantic search
  - Options: `--limit N`, `--format json/table`
  - Vector similarity search
  
- `tasks [options]`: Task queries
  - Options: `--status`, `--due-before DATE`, `--tag TAG`, `--limit N`
  - Flexible filtering

## Data Flow

### Ingestion Flow

```
File System → Scanner → FrontMatter Parser → Task Parser → Database
                ↓              ↓                    ↓
            Checksum    Document Model        Task Models
            Check       + Embeddings          (via LLM Provider)
```

1. Scanner discovers `.md` files
2. MD5 checksum calculated and compared with database
3. If changed or new:
   - FrontMatter extracted and parsed
   - Tasks extracted from content
   - Vector embeddings generated via LLM provider
   - Document and tasks stored in PostgreSQL
4. If deleted from filesystem: hard delete from database

### Query Flow

```
CLI Query → Config → LLM Provider → Vector Generation → Database Search → Results
```

1. CLI command parsed with configuration
2. For semantic search: query text converted to vector via LLM provider
3. Database query executed with vector similarity
4. Results formatted and returned

## Design Decisions

| Decision | Selection | Rationale |
|----------|-----------|-----------|
| Database | PostgreSQL + pg_vector | Industry standard, vector similarity support |
| Embedding Provider | Configurable (OpenAI or HuggingFace) | Flexibility for cloud vs local use |
| LLM Connectivity | litellm | Provider-agnostic, per CONVENTIONS.md |
| Local Embeddings | langchain.embeddings.HuggingFaceEmbeddings | Standardized interface, per CONVENTIONS.md |
| HTTP Client | httpx | Per CONVENTIONS.md preferred library |
| Recurrence Library | dateutil.rrule with shims | RFC-compliant, maintain translation |
| Checksum Algorithm | MD5 | Sufficient for change detection, fast |
| Deleted Files | Hard delete | Files are source of truth, backed up separately |
| Max File Size | 10MB | Balance capability with memory constraints |
| Configuration Format | YAML | Human-readable, supports comments |
| Config Precedence | CLI > Env > Config > Defaults | Industry standard, maximum flexibility |

## Testing Architecture

- pytest with branch coverage
- Tests in top-level `tests/` directory mirroring source structure
- Mock-based unit tests for database and LLM operations
- All provider classes tested with mocked dependencies

### Coverage Status

| Module | Coverage | Notes |
|--------|----------|-------|
| `config.py` | 100% | Complete coverage |
| `parsing/` | 100% | All parsing modules fully covered |
| `database/engine.py` | 100% | Complete coverage |
| `database/models.py` | 95% | PostgreSQL-specific code paths |
| `llm/base.py` | 95% | Abstract methods (by design) |
| `llm/providers.py` | 99% | Defensive branches |
| `cli.py` | 79% | Integration tests require database setup |

### Running Tests

```bash
# Run all tests with coverage
python -m pytest tests/ --cov=obsidian_rag --cov-branch --cov-report=term-missing

# Run ruff checks
ruff check obsidian_rag/ tests/
```

## Dependencies

**Core Dependencies:**
- `sqlalchemy` + `alembic` + `psycopg` - Database ORM and migrations
- `pgvector` - PostgreSQL vector extension support
- `click` - CLI framework
- `pyyaml` - YAML parsing for config and FrontMatter
- `python-dateutil` - Date parsing and recurrence rules
- `pydantic` / `pydantic-settings` - Configuration validation
- `httpx` - HTTP client for web-related calls

**Optional Dependencies:**
- `litellm` - Provider-agnostic LLM connectivity (for OpenAI support)
- `langchain` - Local embedding models (for HuggingFace support)

## Performance Targets

- Ingestion rate: 10 docs/sec (local), 2 docs/sec (API embeddings)
- Query response: < 2 seconds (semantic), < 500ms (task queries)
- Max vault size: 100,000 documents
- Batch size: 100 files per batch

## Migration & Deployment

- Alembic for database migrations
- Package installable via pip
- Self-contained library for external application use
