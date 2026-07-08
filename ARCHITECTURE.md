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
- `inline_fields` (JSONB) - all inline metadata fields including well-known fields and custom [key:: value] pairs

**Well-known field duplication:** During ingestion, well-known fields (`scheduled`, `due`, `completion`, `priority`, `repeat`) are stored in both their dedicated typed columns AND the `inline_fields` JSONB dict. This ensures typed columns provide efficient filtering while `inline_fields` preserves the complete inline metadata context. Custom fields (any `[key:: value]` not in the well-known set) are stored only in `inline_fields`.

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
- **Tab normalization:**
  - `_normalize_indentation_tabs()` helper converts indentation-position tab characters to 2 spaces before `yaml.safe_load()`
  - Only the leading whitespace run of each line is transformed; tabs in quoted string values and mid-line positions are untouched
  - Normalization runs before `yaml.safe_load()` — existing `YAMLError` handling remains unchanged as graceful fallback for genuinely malformed YAML
  - DEBUG log emitted when normalization alters content (frontmatter contained tabs)

#### Body Tag Extraction (`body_tags.py`)

- Extracts inline `#tag` patterns from markdown body text (post-frontmatter-removal)
- Follows Obsidian's tag recognition rules:
  - `#` immediately followed by tag characters (no space) is a tag
  - `#` followed by a space is a heading (NOT extracted)
  - All-numeric tags like `#1984` are NOT valid (must contain non-numerical character)
  - Tags inside fenced code blocks and inline code are NOT extracted (stripped first)
  - Hierarchical tags (`personal/expenses`) and dotted tags (`v1.0/release`) ARE extracted
  - Tags in blockquotes and callouts ARE extracted
- Tags are lowercased and deduplicated
- Reuses `_merge_tags()` from `tag_merging.py` to combine frontmatter + body tags into `Document.tags`

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

Layered configuration system. The implementation is split across several modules to keep individual files under the 1000-line limit, while `config.py` remains the public entry point that re-exports all classes and helpers.

**Module decomposition:**
- `config_env.py`: Environment variable interpolation utilities (`_interpolate_env_vars`, homomorphic `T` TypeVar)
- `config_models.py`: Pydantic model classes for each configuration section (`EndpointConfig`, `DatabaseConfig`, `ChunkingConfig`, `IngestionConfig`, `LoggingConfig`, `MCPConfig`, `VaultConfig`)
- `config_validators.py`: Standalone validation helpers used by `Settings` and model validators (dimension limits, vault name validation, endpoint merging)
- `config.py`: Public entry point; re-exports all model classes and helpers, defines `DEFAULT_CONFIG`, and implements the `Settings` class with layered precedence

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
- `query --exact --vault NAME --path PATH`: Exact document lookup by vault and file path
- `query --exact --name FILENAME`: Exact document lookup by file name (returns list)
- `query --exact --id UUID`: Exact document lookup by document UUID
  
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
        # Updates ingested_at = datetime.now(UTC) on each call
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

#### IntegrityError Recovery (`services/ingestion_integrity.py`)

Handles `UniqueViolation` on `uq_document_vault_path` during document insertion:

- **`ingest_new_document()`**: Wraps the INSERT path with try/except for `IntegrityError`. On successful INSERT, returns `("new", chunks_created)`. On `IntegrityError`, delegates to `handle_integrity_error()`.
- **`handle_integrity_error()`**: Checks if the error is on `uq_document_vault_path`. If so, rolls back the session, re-queries for the existing document, and follows the UPDATE path via `service._update_document()` and `service._update_tasks()`. Non-matching IntegrityErrors and re-query-returns-None are re-raised.

**Recovery Flow:**
1. `_ingest_single_file()` → INSERT path → `session.flush()` → `IntegrityError` raised
2. `ingest_new_document()` catches `IntegrityError` → delegates to `handle_integrity_error()`
3. `handle_integrity_error()` checks constraint name → `session.rollback()` → re-query → `_update_document()` + `_update_tasks()` → returns `("updated", chunks_created)`
4. Back in `_ingest_single_file()`, the result tuple is set and method returns normally

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
- `get_tasks`: Generic task query with comprehensive filtering. All filters are optional and combined with AND logic. Returns `TaskResponse` with `properties` (parent document's frontmatter key-value pairs excluding tags) and `include_content` support.

**Parameters (flat keyword schema):**
- `status`: List of statuses to filter by (e.g., ['not_completed', 'in_progress'])
- `tag_filters`: Tag filter specification. Accepts `str` (JSON string), `dict`, or `GetTasksFilterParams` dataclass. Parsed manually before validation.
  - `include_tags`: List of tags tasks must have
  - `exclude_tags`: List of tags tasks must NOT have
  - `tag_match_mode`: "all" (AND logic, default) or "any" (OR logic)
- `date_filters`: Date filter specification. Accepts `str` (JSON string), `dict`, or dataclass. Parsed manually before validation.
  - `due_after`/`due_before`: Date range filtering for due dates (inclusive)
  - `scheduled_after`/`scheduled_before`: Date range filtering for scheduled dates (inclusive)
  - `completion_after`/`completion_before`: Date range filtering for completion dates (inclusive)
  - `date_match_mode`: "all" (AND logic, default) or "any" (OR logic)
- `priority`: List of priorities to filter by (e.g., ['high', 'highest'])
- `inline_filters`: Inline field filter specification. Accepts `str` (JSON string), `dict`, or dataclass. Parsed manually before validation.
  - Uses `PropertyFilter` operators (`equals`, `contains`, `exists`, `in`, `starts_with`, `regex`) applied to task `inline_fields`
  - `path`: Dot-separated path to the field within `inline_fields` (e.g., `"repeat"`, `"custom_key"`)
  - `operator`: Filter operator (same as property filters)
  - `value`: Value to compare against (not required for `exists` operator)
  - Multiple inline filters are combined with AND logic
- `include_content`: Whether to include task content in response (default: True)
- `limit`: Maximum results (default: 20, max: 10000)
- `offset`: Result offset for pagination (default: 0)

**Handler-level bundling:**
- `GetTasksRequest` remains the internal bundling type used by the handler layer to group parameters before passing to the service layer.

**Filter behavior:**
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
- `query_documents`: Semantic search using vector similarity (cosine distance) with optional property and tag filters. Returns `DocumentResponse` with `properties` (frontmatter key-value pairs excluding tags) and `include_content` support.
- `get_documents_by_tag`: Query documents by tags with include/exclude lists and match_mode ("all" or "any")
- `get_documents_by_property`: Query documents by frontmatter properties with include/exclude filters
- `get_all_tags`: Query all unique document tags with optional glob pattern filtering

**Document Retrieval Tools:**
- `get_document`: Get a single document by vault_name+file_path or document_id (UUID). Returns `DocumentResponse` with `similarity_score=0.0` (no vector search). Raises `ValueError` if not found or invalid params. Supports `include_content=False` to omit the document body.
- `list_documents`: List documents by file_name with optional vault_name scope. Returns `DocumentListResponse` with paginated results. Returns empty list (not error) when no matches. Supports `include_content=False` to omit document bodies.

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
- Property filters, tag filters, and inline_filters can be combined (AND logic between filter types)
- Multiple property filters within include list use AND logic
- Multiple property filters within exclude list use OR logic (any match excludes)
- Multiple inline filters are combined with AND logic

**Ingest Tools:**
- `ingest`: Ingest markdown files and return processing statistics
  - `no_delete` parameter type is `bool | None = None`:
    - `None` (default, not specified): the system auto-sets `True` when `path` is a subdirectory of `container_path` (incremental ingestion, preventing deletion of documents outside the scanned subdirectory), and `False` for full-vault ingestion.
    - `True`: skip deletion of orphaned documents (honored as-is).
    - `False`: deletion of orphaned documents proceeds (honored as-is — the client explicitly accepts the risk, even with an incremental path).
  - The handler (`_ingest_handler`) is the boundary where the API-facing `bool | None` resolves to a concrete `bool` via `_resolve_no_delete()` before reaching `IngestVaultOptions` (which stays `bool`). An INFO-level log is emitted only when `no_delete` is auto-forced to `True` (None → True for incremental paths).
  - CLI behavior is unchanged: the CLI rejects paths not matching `container_path` exactly, so it never encounters incremental paths and the auto-force does not apply.

**Pagination Pattern:**
- `limit`: Default 20, maximum 10000
- `offset`: Starting position for results
- Response includes: `total_count`, `has_more`, `next_offset`

**Output File Parameter:**
Seven query tools support an optional `output_file` parameter that writes the full JSON result to an external destination and returns a compact summary instead of the payload:

**Applicable tools:**
- `query_documents` (server.py)
- `get_documents_by_tag` (server.py)
- `get_documents_by_property` (server.py)
- `get_all_tags` (server.py)
- `get_tasks` (server.py)
- `get_document` (document_tools.py)
- `list_documents` (document_tools.py)

**Non-applicable tools:**
- `ingest` (returns structured statistics, not a large result set)
- `list_vaults`, `get_vault`, `update_vault`, `delete_vault` (vault tools return small, fixed-size metadata)

**Response format when `output_file` is provided:**
```json
{"output_file": {"type": "local", "path": "/tmp/results.json", "bytes": 24580, "item_count": 50}}
```

**Local mode:**
- Writes atomically via a temporary file followed by `os.replace()` to avoid partial reads
- Restricted to `/tmp/` directory and its subdirectories for security
- Parent directories are created automatically if they do not exist
- On success, returns `OutputFileResult` with `type="local"`, the resolved `path`, `bytes` written, and `item_count`
- On write failure, returns `{"success": False, "error": "..."}` with the error message

**S3 mode:**
- Uses `boto3` `PutObject` to upload the serialized JSON payload
- Credentials (`access_key_id`, `secret_access_key`) are provided by the caller in `OutputFileConfig`; no AWS credential caching or logging is performed
- The boto3 client is constructed with `Config(connect_timeout=10, read_timeout=30)` to fail fast on unreachable S3 endpoints
- The `addressing_style` from `OutputFileConfig` is passed to boto3 via `Config(s3={"addressing_style": ...})`; the default `"virtual"` works with AWS S3, while `"path"` is required for Garage/MinIO
- The `endpoint` URL is optional and passed through to boto3 for S3-compatible services (e.g., MinIO)
- On success, returns `OutputFileResult` with `type="s3"`, `bucket`, `key`, `bytes`, and `item_count`
- On upload failure, returns `{"success": False, "error": "..."}` with the error message

**Validation behavior:**
- Missing or malformed `output_file` fields raise `ValueError` with a descriptive message
- Invalid local paths (outside `/tmp/`) raise `ValueError`
- Invalid S3 configurations (missing bucket, key, or credentials) raise `ValueError`

#### Models (`models.py`)

Pydantic models for request/response validation:

**Task Models:**
- `TaskResponse`: Single task with document info, `properties` (parent document's frontmatter key-value pairs excluding tags), `inline_fields` (all inline metadata from the task's JSONB column), and `include_content` support
- `TaskListResponse`: Paginated task list

**Document Models:**
- `DocumentResponse`: Single document with similarity score, `properties` (frontmatter key-value pairs excluding tags), and `include_content` support
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

**Output File Models:**
- `OutputFileConfig`: Configuration for writing tool results to an external file. Fields:
  - `type`: `"local"` or `"s3"`
  - `path`: Absolute local file path (local mode)
  - `endpoint`: Optional S3-compatible endpoint URL (s3 mode)
  - `bucket`: S3 bucket name (s3 mode)
  - `key`: S3 object key (s3 mode)
  - `access_key_id`: AWS access key (s3 mode, from caller)
  - `secret_access_key`: AWS secret key (s3 mode, from caller)
  - `addressing_style`: S3 addressing style. `"virtual"` (default) for AWS S3
    (bucket in hostname: `bucket.endpoint/key`); `"path"` for non-AWS
    S3-compatible services such as Garage and MinIO (bucket in URL path:
    `endpoint/bucket/key`). Clients targeting Garage/MinIO MUST set
    `addressing_style="path"`.
- `OutputFileResult`: Summary returned when `output_file` is used instead of the full payload. Fields:
  - `type`: `"local"` or `"s3"`
  - `path` / `bucket` / `key`: Location of the written file
  - `bytes`: Size of the written JSON payload
  - `item_count`: Number of result items written

**Health Model:**
- `HealthResponse`: Health check status with session metrics
- `SessionMetrics`: Session tracking metrics (total created/destroyed, active count, connection rate)

#### Output File (`output_file.py`)

Core write dispatcher for tool results to external destinations, used by MCP query tools when `output_file` is provided:

- **`write_output_file()`**: Main dispatcher that routes to local or S3 writer based on `OutputFileConfig.type`. Both branches call `json.dumps(result, default=str)` — the `default=str` is defense-in-depth: handlers/tools pre-serialize `uuid.UUID`/`datetime`/`date` to strings via `model_dump(mode="json")` (see Handler/Tool serialization below), and `default=str` catches any future regression where a non-serializable object leaks into the result dict.
- **`_write_local()`**: Writes JSON payload to a local file atomically using a temp file + `os.replace()`; restricted to `/tmp/` and its subdirectories; auto-creates parent directories
- **`_write_s3()`**: Uploads JSON payload via `boto3` `PutObject`; credentials come from the caller config; optional `endpoint` passed through for S3-compatible services. The boto3 client is constructed with `Config(connect_timeout=10, read_timeout=30)` to fail fast on unreachable S3 endpoints. The `addressing_style` from `OutputFileConfig` is passed to boto3 via `Config(s3={"addressing_style": ...})`; the default `"virtual"` works with AWS S3, while `"path"` is required for Garage/MinIO.
- **`_validate_local_path()`**: Validates that the local path is within `/tmp/` and has a valid parent directory
- **`_validate_s3_config()`**: Validates that bucket, key, and both credentials are present
- **`build_output_file_summary()`**: Builds the compact `OutputFileResult` dict returned to the caller
- **`_count_items()`**: Counts the number of items in a result payload for the `item_count` field

#### Handler/Tool Serialization

All MCP tool handlers (`handlers.py`) and tool implementations (`tool_definitions.py`, `server.py` wrappers) serialize their Pydantic response models via `model_dump(mode="json")` before returning the dict to the MCP framework or to `write_output_file()`. The `mode="json"` argument performs source-level type coercion:

- `uuid.UUID` → canonical hex string (e.g. `"550e8400-e29b-41d4-a716-446655440000"`)
- `datetime` → ISO 8601 string (e.g. `"2026-07-07T12:34:56Z"`)
- `date` → ISO 8601 string (e.g. `"2026-07-07"`)

This guarantees the returned dict is JSON-serializable through plain `json.dumps()` without a custom encoder, which is required for the `output_file` code path and for external tool wrappers (e.g. `mcp_s3_call`) that re-serialize the result. The `default=str` fallback in `write_output_file()` is defense-in-depth against future regressions where a non-serializable object leaks past the source fix.

`OutputFileResult.model_dump()` (without `mode="json"`) in `output_file.py` is intentionally left as-is because that model contains only `str`, `int`, and `None` fields — no UUID/datetime possible.

#### Authentication

Bearer token authentication using `BearerTokenAuth`:
- Token configured via `OBSIDIAN_RAG_MCP_TOKEN` env var or `mcp.token` config
- All endpoints require valid Bearer token (401 for invalid/missing)
- No per-tool permission granularity (all-or-nothing access)

#### Host Origin Protection (Disabled)

FastMCP 3.4+ installs `HostOriginGuardMiddleware` by default (`host_origin_protection=True`) with a localhost-only allowlist (`127.0.0.1`, `localhost`, `::1`). This rejects external clients with HTTP 421 "Misdirected Request" when the server is bound to `0.0.0.0` or deployed behind a reverse proxy.

The Obsidian RAG MCP server explicitly disables this guard via `host_origin_protection=False` in the `mcp.http_app()` call (`server.py`). Rationale:
- The server is already protected by Bearer token auth (`StaticTokenVerifier`), so the DNS-rebinding threat model (unauthenticated localhost browser exploitation, CVE-2025-66416) does not apply
- The guard prevents legitimate external access when deployed in Docker or behind a reverse proxy

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
   - FrontMatter extracted and parsed (indentation tabs normalized to spaces before YAML parsing)
   - Tasks extracted from content
    - **Document-level tags merged into task tags** (case-insensitive dedup, lowercased) via `tag_merging.py`
    - **Well-known fields** (`scheduled`, `due`, `completion`, `priority`, `repeat`) stored in both typed columns AND `inline_fields` JSONB dict
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

When `output_file` is provided on an MCP query tool:
- Handler result → JSON serialization → write to target destination → compact `OutputFileResult` summary returned
- When absent: unchanged flow (full result payload returned directly)

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
