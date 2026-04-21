# Architecture: Obsidian RAG

This document describes the architecture of Obsidian RAG for developers who want to understand or modify the codebase.

## System Overview

Obsidian RAG is a Python library designed to ingest, store, and query Obsidian markdown documents with support for vector embeddings and task management.

## Core Components

### 1. Database Layer (`database/`)

#### ArrayType TypeDecorator

The `ArrayType` class in `models.py` provides PostgreSQL array support:
- Uses `postgresql.ARRAY` type for PostgreSQL arrays
- Used for `tags` columns in both Document and Task models
- Raises an error if used with non-PostgreSQL dialects

#### Vector Index

The `content_vector` column uses HNSW (Hierarchical Navigable Small World) index for fast similarity search:
- **Index type**: HNSW with `vector_cosine_ops` operator class
- **Dimension limit**: Both HNSW and IVFFLAT have a 2000 dimension limit in pgvector
- **Configuration**: `database.vector_dimension` setting (default: 1536)
- **Validation**: Configuration validates that `vector_dimension ≤ 2000`

#### Models (`models.py`)

**vaults table:**
- `id` (UUID, PK)
- `name` (VARCHAR(100), UNIQUE, indexed)
- `description` (TEXT, nullable)
- `container_path` (TEXT) - path inside container/Docker
- `host_path` (TEXT) - path on host system
- `created_at` (TIMESTAMP)

**documents table:**
- `id` (UUID, PK)
- `vault_id` (UUID, FK → vaults.id, ON DELETE CASCADE)
- `file_path` (TEXT, indexed) - relative to vault root
- `file_name` (TEXT, indexed)
- `content` (TEXT)
- `content_vector` (VECTOR(N) - configurable dimension, default 1536)
- `checksum_md5` (CHAR(32))
- `created_at_fs` (TIMESTAMP) - filesystem creation date
- `modified_at_fs` (TIMESTAMP) - filesystem modification date
- `ingested_at` (TIMESTAMP) - when we last ingested
- `tags` (TEXT[], nullable) - from FrontMatter, deduplicated
- `frontmatter_json` (JSONB) - all FrontMatter properties including `kind`

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

**document_chunks table:**
- `id` (UUID, PK)
- `document_id` (UUID, FK → documents.id, ON DELETE CASCADE)
- `chunk_index` (INTEGER) - position within document chunks (0-based)
- `content` (TEXT) - chunk text content
- `chunk_vector` (VECTOR(N) - configurable dimension, default 1536)
- `token_count` (INTEGER) - number of tokens in chunk
- `chunk_type` (ENUM: 'content', 'task') - type of chunk for analytics
- `created_at` (TIMESTAMP)

The `document_chunks` table stores token-based chunks of document content for semantic search. Each chunk has its own vector embedding and can be searched independently. The HNSW index on `chunk_vector` enables fast similarity search across chunks.

#### Engine (`engine.py`)

Database connection management using SQLAlchemy with:
- Connection pooling
- Session context managers
- Table creation/migration support

### 2. Parsing Layer (`parsing/`)

#### FrontMatter Extraction (`frontmatter.py`)

- Extracts YAML frontmatter from markdown documents
- `tags` are extracted separately and normalized (string or list)
- All other properties including `kind` are stored in `frontmatter_json`
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

### 3. Chunking Layer (`chunking/`, `tokenizer.py`, `reranking.py`)

#### Token-based Chunking (`chunking.py`)

The chunking system splits large documents into smaller, semantically coherent segments for more precise vector search:

- **Target size**: 512 tokens per chunk (configurable, range 64-2048)
- **Overlap**: 50 tokens between chunks (configurable, preserves context)
- **Boundary strategy**: Paragraph boundaries preferred, fallback to sentence boundaries, then arbitrary token boundaries
- **Task preservation**: Individual task lines become separate chunks when possible for better task discoverability

**Key Functions:**
- `create_chunks()`: Main entry point for chunk creation from document content
- `should_chunk_document()`: Determines if document exceeds chunk size threshold
- `_find_split_point()`: Locates optimal boundary for chunk splitting
- `_calculate_next_start()`: Calculates start position for next chunk with overlap

#### Tokenization (`tokenizer.py`)

Token counting for accurate chunk sizing:

- **Primary**: HuggingFace Tokenizers (fast, Rust-based)
- **Fallback**: Character-based estimation (~4 characters per token) when tokenizer unavailable
- **Caching**: Tokenizer models cached in configurable directory (default: `~/.cache/obsidian-rag/tokenizers`)
- **Model mapping**: Auto-detects tokenizer based on embedding provider configuration

**Key Functions:**
- `count_tokens()`: Returns token count for text using configured tokenizer
- `get_tokenizer()`: Returns cached tokenizer instance or initializes new one
- `clear_tokenizer_cache()`: Clears tokenizer cache (primarily for testing)

#### Re-ranking (`reranking.py`)

Optional flashrank integration for improving chunk relevance:

- **Model**: `ms-marco-MiniLM-L-12-v2` (default, ~100-200MB RAM)
- **Input**: Top 20 chunks from vector similarity search
- **Output**: Top 5-10 re-ranked chunks with cross-encoder scores
- **Truncation**: Flashrank truncates 512-token chunks to 128 tokens internally
- **Fallback**: Returns unranked chunks if flashrank unavailable or errors

**Key Functions:**
- `rerank_chunks()`: Re-ranks chunks using flashrank cross-encoder
- `get_reranker()`: Returns cached reranker instance or initializes new one
- `clear_reranker_cache()`: Clears reranker cache (primarily for testing)

### 4. LLM Provider Layer (`llm/`)

#### Base Classes (`base.py`)

- Provider-agnostic interface for LLM operations
- Three endpoint types: embedding, analysis, chat
- Factory pattern for provider instantiation
- **Type Stubs (`base.pyi`)**: Abstract method signatures moved to `.pyi` stub file for cleaner type checking without coverage requirements on abstract definitions

#### Provider Implementations (`providers.py`)

**Supported Providers:**
- OpenAI (embeddings, analysis, chat) - uses `litellm`
- OpenRouter (embeddings, chat) - uses `litellm` with `openrouter/` prefix for native routing
- HuggingFace (local embeddings) - uses `langchain.embeddings.HuggingFaceEmbeddings`
- Extensible design for additional providers

**Library Usage:**
- `litellm`: Provider-agnostic LLM connectivity for OpenAI and OpenRouter endpoints
  - `OpenAIEmbeddingProvider`: Uses `litellm.embedding()`
  - `OpenAIChatProvider`: Uses `litellm.completion()`
  - `OpenRouterEmbeddingProvider`: Uses `litellm.embedding()` with `openrouter/` prefix for native routing via litellm 1.83+. Custom base_url passed as `api_base` parameter.
  - `OpenRouterChatProvider`: Uses `litellm.completion()` with `openrouter/` prefix
- `langchain`: Local embedding models via HuggingFace
  - `HuggingFaceEmbeddingProvider`: Uses `HuggingFaceEmbeddings`

**Factory Functions:**
Instead of overloaded class methods, individual factory functions provide type-safe provider creation:
- `create_openai_embedding_provider()`: Creates OpenAIEmbeddingProvider instances
- `create_huggingface_embedding_provider()`: Creates HuggingFaceEmbeddingProvider instances
- `create_openrouter_embedding_provider()`: Creates OpenRouterEmbeddingProvider instances
- `create_openai_chat_provider()`: Creates OpenAIChatProvider instances
- `create_openrouter_chat_provider()`: Creates OpenRouterChatProvider instances

This approach eliminates `@overload` decorators while maintaining precise return types.

**Configuration per endpoint:**
- `provider`: Provider type ('openai' or 'huggingface')
- `model`: Model name
- `api_key`: Authentication (via env var interpolation)
- `base_url`: Optional custom endpoint
- `temperature`: Generation temperature
- `max_tokens`: Maximum response tokens

### 5. Configuration (`config.py`)

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
  - Uses homomorphic TypeVar pattern for type-safe interpolation (preserves input type)
- Nested configuration merging
- Pydantic validation
- **Vector dimension validation**: Enforces maximum of 2000 dimensions (pgvector limit)
- **Cross-validation**: Validates embedding provider dimension matches `database.vector_dimension`

### 6. CLI Layer (`cli.py`)

Entry point for all user interactions:

**Commands:**
- `ingest <path>`: Scan and ingest documents
  - Options: `--dry-run`, `--verbose`, `--no-delete`
  - Progress reporting: total, new, updated, skipped, errors, deleted
  - Batch processing with configurable size
  - Orphaned document deletion (disabled with `--no-delete`)
  
- `query <search>`: Semantic search
  - Options: `--limit N`, `--format json/table`
  - Vector similarity search
  
- `tasks [options]`: Task queries
  - Options: `--status`, `--due-before DATE`, `--tag TAG`, `--limit N`
  - Flexible filtering

### 7. Service Layer (`services/`)

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
    def ingest_vault(options: IngestVaultOptions) -> IngestionResult
    def _resolve_vault_config(vault: VaultConfig | str) -> VaultConfig
    def _compute_relative_path(file_path: Path, container_path: str) -> str
    def _validate_files_in_vault(file_infos: list[FileInfo], container_path: str) -> None
    def _get_or_create_vault(session, vault_config: VaultConfig) -> Vault
    def _ingest_single_file(file_info, vault_id: UUID, dry_run=False) -> str
    def _create_document(file_info, vault_id: UUID, parsed_data) -> Document
    def _update_document(document, file_info, parsed_data)
    def _create_tasks(session, document, parsed_tasks)
    def _update_tasks(session, document, parsed_tasks)
    def _delete_orphaned_documents(filesystem_paths, dry_run=False) -> tuple[int, int]
    def _process_deletion_batches(orphaned_documents) -> tuple[int, int]
    def _delete_batch(session, batch) -> tuple[int, int]
```

**IngestionResult Dataclass:**
- `total`: Total files processed
- `new`: New documents created
- `updated`: Existing documents updated
- `unchanged`: Unchanged documents (same checksum)
- `errors`: Files that failed processing
- `deleted`: Number of orphaned documents deleted from database
- `processing_time_seconds`: Time taken
- `message`: Human-readable summary

**Usage Patterns:**
- **CLI**: Uses service with progress callbacks and verbose output
- **MCP**: Uses service with path override support, returns structured results

### 8. MCP Server Layer (`mcp_server/`)

The MCP (Model Context Protocol) server provides remote access to Obsidian RAG functionality via HTTP transport.

#### Server (`server.py` and `tool_definitions.py`)

FastMCP server configuration with:
- HTTP transport for remote access
- Bearer token authentication via `BearerTokenAuth`
- CORS middleware for browser-based clients
- Health check endpoint at `/health` with session metrics
- Session management with lifecycle logging and metrics
- Rate limiting to prevent resource exhaustion
- HTTP request/response logging middleware
- Eleven read-only tools for querying tasks, documents, vaults, and ingestion status

**Architecture Pattern - Global Registry:**
To achieve 100% test coverage while maintaining FastMCP compatibility, the server uses a global registry pattern:

- `MCPToolRegistry` class (`tool_definitions.py`): Holds dependencies (db_manager, embedding_provider, settings)
- Module-level `_tool_registry` variable: Initialized during `create_mcp_server()` before tool registration
- `_get_registry()` / `_set_registry()` functions: Access and mutate the global registry
- Tool wrappers (`server.py`): Module-level functions that call `_get_registry()` to access dependencies
- Tool implementations (`tool_definitions.py`): Module-level functions containing the actual business logic

This pattern eliminates nested functions (which are untestable) while allowing tools to access runtime-created dependencies.

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
- `get_tasks`: Generic task query with comprehensive filtering by status, date ranges (due, scheduled, completion), tags, and priority. All filters are optional and combined with AND logic.

**Task Filter Features (get_tasks):**
- `status`: List of statuses to filter by (e.g., ['not_completed', 'in_progress'])
- `due_after`/`due_before`: Date range filtering for due dates (inclusive)
- `scheduled_after`/`scheduled_before`: Date range filtering for scheduled dates (inclusive)
- `completion_after`/`completion_before`: Date range filtering for completion dates (inclusive)
- `date_match_mode`: How to combine date filters - "all" (AND logic, default) or "any" (OR logic)
  - "all": Task must satisfy ALL date conditions to match (backward compatible)
  - "any": Task matches if ANY date condition is satisfied
- `include_tags`: List of tags tasks must have (controlled by `tag_match_mode`)
  - `tag_match_mode="all"` (default): Task must have ALL include tags
  - `tag_match_mode="any"`: Task must have ANY of the include tags
- `exclude_tags`: List of tags tasks must NOT have (always OR logic - any excluded tag disqualifies)
- `tag_match_mode`: How to combine include_tags - "all" (AND logic, default) or "any" (OR logic)
- `priority`: List of priorities to filter by (e.g., ['high', 'highest'])
- All filters are optional and combined with AND logic
- Date comparisons are inclusive (>= for after, <= for before)
- Tasks without dates are excluded from date filter comparisons
- Conflicting tags (same tag in both include_tags and exclude_tags) are rejected with validation error

**Tag Prefix Handling:**
- Tags should NOT include the `#` prefix in filter values
- The system defensively strips leading `#` characters from tag filter values
- Both `include_tags=["#personal/expenses"]` and `include_tags=["personal/expenses"]` return the same results
- Empty tags (after stripping all `#` characters) are silently ignored

**Migration from removed tools:**
| Old Tool | New `get_tasks` Equivalent |
|----------|---------------------------|
| `get_incomplete_tasks(include_cancelled=True)` | `get_tasks(status=["not_completed", "in_progress", "cancelled"])` |
| `get_tasks_due_this_week(include_completed=False)` | `get_tasks(due_after="2026-03-11", due_before="2026-03-18")` |
| `get_tasks_by_tag(tag="work")` | `get_tasks(include_tags=["work"])` |
| `get_completed_tasks(completed_since="2026-01-01")` | `get_tasks(status=["completed"], completion_after="2026-01-01")` |

**Tag Filtering Examples:**
- Find tasks with tag 'work' OR 'personal': `include_tags=["work", "personal"], tag_match_mode="any"`
- Find tasks with both 'urgent' AND 'work' tags: `include_tags=["urgent", "work"], tag_match_mode="all"`
- Find tasks with 'work' but NOT 'blocked': `include_tags=["work"], exclude_tags=["blocked"]`

**Document Tools:**
- `query_documents`: Semantic search using vector similarity (cosine distance) with optional property and tag filters
- `get_documents_by_tag`: Query documents by tags with include/exclude lists and match_mode ("all" or "any")
- `get_documents_by_property`: Query documents by frontmatter properties with include/exclude filters
- `get_all_tags`: Query all unique document tags with optional glob pattern filtering

**Vault Tools:**
- `list_vaults`: List all vaults with document counts and metadata
- `get_vault`: Get a single vault by name or UUID with document count
- `update_vault`: Update vault properties (description, host_path, container_path); container_path changes require `force=True` and delete associated data
- `delete_vault`: Delete vault with cascade deletion of documents/tasks/chunks; requires `confirm=True`

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

**Vault Models:**
- `VaultResponse`: Single vault with metadata, document count, container_path, and created_at
- `VaultListResponse`: Paginated vault list

**Filter Models:**
- `PropertyFilter`: Filter for document frontmatter properties with path, operator, and value
- `TagFilter`: Filter for document tags with include/exclude lists and match_mode
- `QueryFilterParams`: Combined filter parameters for property and tag filtering

**Health Model:**
- `HealthResponse`: Health check status with session metrics
- `SessionMetrics`: Session tracking metrics (total created/destroyed, active count, connection rate)

#### Authentication

Bearer token authentication using `BearerTokenAuth`:
- Token configured via `OBSIDIAN_RAG_MCP_TOKEN` env var or `mcp.token` config
- All endpoints require valid Bearer token (401 for invalid/missing)
- No per-tool permission granularity (all-or-nothing access)

#### Session Management (`session_manager.py`)

Session lifecycle tracking and connection protection:
- **Session tracking**: Records session creation/destruction with duration metrics
- **Rate limiting**: Per-IP connection rate limiting (default: 10/sec)
- **Concurrent limits**: Maximum concurrent sessions (default: 100)
- **Session timeout**: Automatic cleanup of inactive sessions (default: 300s)
- **Metrics**: Total created/destroyed, active count, peak concurrent, connection rate

Logged events:
- Session creation: `Session created: <id> from <ip> (active: <count>)`
- Session destruction: `Session destroyed: <id> (duration: <seconds>s, requests: <count>)`
- Rate limit exceeded: `Rate limit exceeded for <ip>: <rate>/sec over <window>s`

#### Middleware (`middleware.py`)

HTTP request/response logging for debugging:
- `SessionLoggingMiddleware`: Logs all HTTP requests/responses at DEBUG level
- Records method, path, status code, and duration
- Helps diagnose client connection behavior

#### Configuration

MCP-specific configuration (`MCPConfig`):
- `host`: Bind address (default: "0.0.0.0")
- `port`: HTTP port (default: 8000)
- `token`: Bearer token for authentication (required)
- `cors_origins`: Allowed CORS origins (default: ["*"])
- `enable_health_check`: Enable `/health` endpoint (default: true)
- `stateless_http`: Stateless mode for horizontal scaling (default: false)
- `max_concurrent_sessions`: Maximum concurrent sessions (default: 100)
- `session_timeout_seconds`: Session timeout in seconds (default: 300)
- `rate_limit_per_second`: Max connections per second per IP (default: 10.0)
- `rate_limit_window`: Rate limit window in seconds (default: 60)
- `enable_request_logging`: Enable HTTP request/response logging (default: true)

#### Docker Support

Dockerfile with configurable build scope:
- `INSTALL_MODE=full`: Install all dependencies including optional
- `INSTALL_MODE=mcp-only`: Install only MCP-related dependencies (default)

```bash
docker build -t obsidian-rag-mcp .
docker run -p 8000:8000 -e OBSIDIAN_RAG_MCP_TOKEN=secret obsidian-rag-mcp
```

#### LibreChat Integration

Documentation for LibreChat MCP client configuration and known issues:
- See `docs/librechat-mcp-client.md` for complete configuration guide
- Documents known SSE stream disconnection errors (client-side issue)
- Provides recommended configuration for LibreChat MCP servers
- Explains session metrics and health check usage for troubleshooting

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

### Database Migrations

This project uses Alembic for database schema migrations.

```bash
# Create a new migration (after modifying models.py)
alembic revision --autogenerate -m "Description of changes"

# Apply all pending migrations
alembic upgrade head

# Apply specific migration
alembic upgrade +1

# Downgrade one migration
alembic downgrade -1

# View current migration version
alembic current
```

**Note:** Always review auto-generated migrations before applying them.

### Code Quality

```bash
# Run linting
ruff check

# Fix auto-fixable issues
ruff check --fix
```
