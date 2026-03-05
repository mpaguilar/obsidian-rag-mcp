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
- `vault_root` (TEXT, nullable) - root path of vault during ingestion

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

### 6. Service Layer (`services/`)

#### IngestionService (`services/ingestion.py`)

Centralized service for document ingestion, shared by CLI and MCP server:

**Key Features:**
- **Shared Logic**: Same ingestion logic used by both CLI and MCP server
- **Per-File Transactions**: Each file processed in independent transaction
- **Error Isolation**: Individual file failures don't stop overall ingestion
- **Progress Callbacks**: Optional callback for progress reporting
- **Dry-Run Support**: Can simulate ingestion without database writes

**Class Structure:**
```python
class IngestionService:
    def __init__(db_manager, embedding_provider, settings)
    def ingest_vault(vault_path, dry_run=False, progress_callback=None, file_infos=None) -> IngestionResult
    def _ingest_single_file(file_info, dry_run=False) -> str
    def _create_document(file_info, parsed_data) -> Document
    def _update_document(document, file_info, parsed_data)
    def _create_tasks(session, document, parsed_tasks)
    def _update_tasks(session, document, parsed_tasks)
```

**IngestionResult Dataclass:**
- `total`: Total files processed
- `new`: New documents created
- `updated`: Existing documents updated
- `unchanged`: Unchanged documents (same checksum)
- `errors`: Files that failed processing
- `processing_time_seconds`: Time taken
- `message`: Human-readable summary

**Usage Patterns:**
- **CLI**: Uses service with progress callbacks and verbose output
- **MCP**: Uses service with path override support, returns structured results

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
| `config.py` | 97% | Environment variable interpolation branches |
| `parsing/` | 100% | All parsing modules fully covered |
| `database/engine.py` | 100% | Complete coverage |
| `database/models.py` | 100% | Complete coverage |
| `llm/base.py` | 100% | Complete coverage |
| `llm/providers.py` | 100% | Complete coverage |
| `services/ingestion.py` | 100% | Complete coverage |
| `cli.py` | 95% | Error handling and edge cases |
| `mcp_server/__main__.py` | 100% | Complete coverage |
| `mcp_server/server.py` | 71% | Tool registration and logging functions |
| `mcp_server/tools/documents.py` | 94% | PostgreSQL-specific tag filtering requires integration testing |
| `mcp_server/tools/documents_filters.py` | 99% | Single defensive branch |
| `mcp_server/tools/documents_postgres.py` | 100% | Complete coverage |
| `mcp_server/tools/documents_sqlite.py` | 100% | Complete coverage |
| `mcp_server/tools/documents_tags.py` | 100% | Complete coverage |
| `mcp_server/tools/documents_params.py` | 100% | Complete coverage |
| `mcp_server/tools/tasks.py` | 100% | Complete coverage |

### Running Tests

```bash
# Run all tests with coverage
python -m pytest tests/ --cov=obsidian_rag --cov-branch --cov-report=term-missing

# Run ruff checks
ruff check obsidian_rag/ tests/
```

### 6. MCP Server Layer (`mcp_server/`)

The MCP (Model Context Protocol) server provides remote access to Obsidian RAG functionality via HTTP transport.

#### Server (`server.py`)

FastMCP server configuration with:
- HTTP transport for remote access
- Bearer token authentication via `BearerTokenAuth`
- CORS middleware for browser-based clients
- Health check endpoint at `/health`
- Seven read-only tools for querying tasks, documents, and ingestion status

**Server Creation:**
```python
mcp = create_mcp_server(settings)
app = create_http_app(settings)  # ASGI app with CORS
```

**Running the Server:**
```bash
python -m obsidian_rag.mcp_server
# or
uvicorn obsidian_rag.mcp_server.server:create_http_app --factory
```

#### Tools (`tools/`)

All tools are read-only and use SQLAlchemy `select()` operations only:

**Task Tools:**
- `get_incomplete_tasks`: Query tasks with status not_completed, in_progress, optionally cancelled
- `get_tasks_due_this_week`: Query tasks due within next 7 days
- `get_tasks_by_tag`: Query tasks by tag (matches task or document level, case-insensitive)
- `get_completed_tasks`: Query completed tasks with optional date filter

**Document Tools:**
- `query_documents`: Semantic search using vector similarity (cosine distance) with optional property and tag filters
- `get_documents_by_tag`: Query documents by tags with include/exclude lists and match_mode ("all" or "any")
- `get_documents_by_property`: Query documents by frontmatter properties with include/exclude filters
- `get_all_tags`: Query all unique document tags with optional glob pattern filtering

**Property Filter Operators:**
- `equals`: Exact match (case-insensitive)
- `contains`: Substring match (case-insensitive)
- `exists`: Property key exists (no value required)
- `in`: Value is in provided list
- `starts_with`: String starts with pattern (case-insensitive)
- `regex`: Regular expression match

**Tag Filter Features:**
- `include_tags`: Documents must have ALL (match_mode="all") or ANY (match_mode="any") of these tags
- `exclude_tags`: Documents must NOT have any of these tags
- Tag matching is case-insensitive substring match
- Conflicting tags (in both include and exclude) are rejected with validation error

**Filter Combinations:**
- Property filters and tag filters can be combined (AND logic between filter types)
- Multiple property filters within include list use AND logic
- Multiple property filters within exclude list use OR logic (any match excludes)

**Ingest Tools:**
- `ingest`: Ingest markdown files and return processing statistics

**Pagination Pattern:**
- `limit`: Default 20, maximum 100
- `offset`: Starting position for results
- Response includes: `total_count`, `has_more`, `next_offset`

#### Models (`models.py`)

Pydantic models for request/response validation:

**Task Models:**
- `TaskResponse`: Single task with document info
- `TaskListResponse`: Paginated task list

**Document Models:**
- `DocumentResponse`: Single document with similarity score
- `DocumentListResponse`: Paginated document list

**Tag Models:**
- `TagListResponse`: Paginated list of unique tags

**Filter Models:**
- `PropertyFilter`: Filter for document frontmatter properties with path, operator, and value
- `TagFilter`: Filter for document tags with include/exclude lists and match_mode
- `QueryFilterParams`: Combined filter parameters for property and tag filtering

**Health Model:**
- `HealthResponse`: Health check status

#### Authentication

Bearer token authentication using `BearerTokenAuth`:
- Token configured via `OBSIDIAN_RAG_MCP_TOKEN` env var or `mcp.token` config
- All endpoints require valid Bearer token (401 for invalid/missing)
- No per-tool permission granularity (all-or-nothing access)

#### Configuration

MCP-specific configuration (`MCPConfig`):
- `host`: Bind address (default: "0.0.0.0")
- `port`: HTTP port (default: 8000)
- `token`: Bearer token for authentication (required)
- `cors_origins`: Allowed CORS origins (default: ["*"])
- `enable_health_check`: Enable `/health` endpoint (default: true)
- `stateless_http`: Stateless mode for horizontal scaling (default: false)

#### Docker Support

Dockerfile with configurable build scope:
- `INSTALL_MODE=full`: Install all dependencies including optional
- `INSTALL_MODE=mcp-only`: Install only MCP-related dependencies (default)

```bash
docker build -t obsidian-rag-mcp .
docker run -p 8000:8000 -e OBSIDIAN_RAG_MCP_TOKEN=secret obsidian-rag-mcp
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
- `fastmcp` - MCP server framework
- `starlette` - ASGI middleware (CORS)

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
