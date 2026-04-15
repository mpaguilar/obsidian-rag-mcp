# Obsidian RAG/MCP

A production-ready **Retrieval-Augmented Generation (RAG)** pipeline with **Model Context Protocol (MCP)** server integration for semantic search over Obsidian markdown knowledge bases.

## Overview

**Obsidian RAG/MCP** is a high-performance document ingestion and semantic search system built for knowledge management at scale. It combines vector embeddings, PostgreSQL with pg_vector, and MCP protocol support to enable intelligent document retrieval for LLM-powered applications.

### Key Capabilities

- **Vector Embedding Pipeline**: Ingests Obsidian markdown documents into PostgreSQL with pg_vector, generating semantic embeddings via configurable LLM providers (OpenAI, OpenRouter, HuggingFace)
- **Token-Based Document Chunking**: Splits large documents into semantically coherent chunks for improved search precision, with optional cross-encoder re-ranking via FlashRank
- **MCP Server Architecture**: Full Model Context Protocol implementation with streamable HTTP transport, enabling seamless integration with MCP-compatible clients (Claude Desktop, LibreChat, etc.)
- **Semantic Search**: Cosine similarity search over document embeddings with sub-second query latency
- **Task Extraction Engine**: Parses markdown task syntax with metadata extraction (due dates, priorities, recurrence patterns, custom fields)
- **Multi-Vault Support**: Manages multiple isolated knowledge bases with per-vault access control and metadata
- **Horizontal Scalability**: Stateless HTTP transport design supports containerized deployment and load balancing

### Architecture Highlights

- **Database**: PostgreSQL 14+ with pg_vector extension for high-dimensional vector indexing (HNSW)
- **ORM**: SQLAlchemy with Alembic migrations for schema management
- **LLM Integration**: Provider-agnostic via litellm (OpenAI, OpenRouter) and langchain (HuggingFace)
- **API Layer**: FastMCP server with Bearer token authentication, CORS middleware, and rate limiting
- **Configuration**: Layered config system (CLI flags → Environment variables → YAML → Defaults) with Pydantic validation

## Quick Start

```bash
# 1. Install
pip install obsidian-rag

# 2. Configure (create .obsidian-rag.yaml)
echo "database:
  url: postgresql://localhost/obsidian_rag
endpoints:
  embedding:
    provider: openai
    model: text-embedding-3-small
    api_key: \${OPENAI_API_KEY}" > .obsidian-rag.yaml

# 3. Setup database
createdb obsidian_rag
psql -d obsidian_rag -c "CREATE EXTENSION IF NOT EXISTS vector;"

# 4. Ingest your vault
obsidian-rag ingest /path/to/obsidian/vault

# 5. Search
obsidian-rag query "project planning ideas"
```

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
psql -d obsidian_rag -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

3. Run database migrations (create tables):

```bash
alembic upgrade head
```

## Usage

### Ingest Documents

Scan and ingest markdown files from an Obsidian vault:

```bash
# Basic ingestion (uses default "Obsidian Vault")
obsidian-rag ingest /path/to/vault

# Ingest to a specific vault
obsidian-rag ingest /path/to/vault --vault "My Vault"

# Dry run to preview changes without writing to database
obsidian-rag ingest /path/to/vault --vault "My Vault" --dry-run

# Verbose output for detailed progress
obsidian-rag ingest /path/to/vault --vault "My Vault" --verbose
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

### Chunk-Level Search (Optional)

For improved search precision on large documents, enable token-based chunking with optional cross-encoder re-ranking:

```bash
# Search at chunk level
obsidian-rag query "project planning" --chunks --limit 20

# Search with re-ranking for better result quality
obsidian-rag query "architecture patterns" --chunks --rerank
```

**Benefits:**
- More precise matching on specific document sections
- Better context relevance for multi-topic documents
- Cross-encoder re-ranking improves result ordering

See [docs/chunking.md](./docs/chunking.md) for detailed configuration options, including chunk size, overlap settings, and FlashRank re-ranking parameters.

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
- [ ] Task with hierarchical tag #work/project/alpha
- [ ] Task with version tag #v1.0/release
- [ ] Task with repeat [repeat:: every day]
- [ ] Task with scheduled date [scheduled:: 2024-12-31]
- [ ] Task with completion date [completion:: 2024-12-31]
```

## MCP Server

The **Model Context Protocol (MCP)** server exposes all RAG functionality via HTTP transport, enabling LLM clients to query your knowledge base programmatically.

### Why MCP?

- **Standardized Protocol**: MCP is an open protocol for integrating external data sources with LLM applications
- **Universal Compatibility**: Works with any MCP-compatible client (Claude Desktop, LibreChat, custom implementations)
- **Streamable HTTP**: Stateless transport supports horizontal scaling and containerized deployments
- **Type-Safe API**: Pydantic models ensure request/response validation

### Running the MCP Server

```bash
# Run directly
python -m obsidian_rag.mcp_server

# Or with uvicorn
uvicorn obsidian_rag.mcp_server.server:create_http_app --factory
```

### MCP Server Configuration

Add MCP settings to your config file:

```yaml
mcp:
  host: "0.0.0.0"
  port: 8000
  token: ${OBSIDIAN_RAG_MCP_TOKEN}  # Required for authentication
  cors_origins: ["*"]
  enable_health_check: true
  stateless_http: false
  max_concurrent_sessions: 100
  session_timeout_seconds: 300
  rate_limit_per_second: 10.0
  rate_limit_window: 60
  enable_request_logging: true
```

Or via environment variables:

```bash
# Required
export OBSIDIAN_RAG_MCP_TOKEN="your-secret-token"

# Server binding and ports
export OBSIDIAN_RAG_MCP_HOST="0.0.0.0"
export OBSIDIAN_RAG_MCP_PORT=8000

# CORS configuration
export OBSIDIAN_RAG_MCP_CORS_ORIGINS='["*"]'  # JSON array format

# Health check endpoint
export OBSIDIAN_RAG_MCP_ENABLE_HEALTH_CHECK=true

# Stateless mode for horizontal scaling
export OBSIDIAN_RAG_MCP_STATELESS_HTTP=false

# Connection limits and rate limiting
export OBSIDIAN_RAG_MCP_MAX_CONCURRENT_SESSIONS=100
export OBSIDIAN_RAG_MCP_SESSION_TIMEOUT_SECONDS=300
export OBSIDIAN_RAG_MCP_RATE_LIMIT_PER_SECOND=10.0
export OBSIDIAN_RAG_MCP_RATE_LIMIT_WINDOW=60

# Request logging
export OBSIDIAN_RAG_MCP_ENABLE_REQUEST_LOGGING=true
```

### MCP Endpoint and Transport

The MCP server uses **HTTP transport** (Streamable HTTP) on the following endpoints:

| Endpoint | Purpose |
|----------|---------|
| `http://localhost:8000/` | **MCP Protocol endpoint** - Main MCP communication |
| `http://localhost:8000/health` | Health check endpoint with session metrics |

**Important:** The MCP protocol runs on the **root path** (`/`), not a sub-path like `/mcp` or `/sse`.

**Authentication:** All requests must include a Bearer token:
```
Authorization: Bearer your-token-here
```

### Available MCP Tools

The MCP server provides read-only tools for querying tasks, documents, and vaults:

**Vault Tools:**
- `list_vaults`: Query all vaults with document counts and metadata

**Task Tools:**
- `get_tasks`: Query tasks with comprehensive filtering by status, date ranges, tags, and priority

**Document Tools:**
- `query_documents`: Semantic search using vector similarity with optional chunk-level search and re-ranking
- `get_documents_by_tag`: Query documents by tags with optional `vault_name` filter
- `get_documents_by_property`: Query documents by frontmatter properties with optional `vault_name` filter
- `get_all_tags`: Query all unique document tags
- `ingest`: Ingest documents from a vault (requires `vault_name` parameter)

### Connecting to the MCP Server

Connect using any MCP client that supports HTTP transport:

```python
from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport

# Using StreamableHttpTransport
transport = StreamableHttpTransport(
    url="http://localhost:8000/",
    headers={"Authorization": "Bearer your-token-here"}
)
client = Client(transport)

# Or using the convenience method with BearerAuth
from fastmcp.client.auth import BearerAuth
client = Client(
    "http://localhost:8000/",
    auth=BearerAuth("your-token-here")
)

# Use the client
async with client:
    # Query incomplete tasks
    tasks = await client.call_tool(
        "get_tasks",
        {"status": ["not_completed", "in_progress"], "limit": 10}
    )

    # Search documents
    docs = await client.call_tool(
        "query_documents",
        {"query": "project planning", "limit": 5}
    )
```

### Chunk Search with `query_documents`

The `query_documents` tool supports chunk-level semantic search for more precise retrieval from large documents. When chunk search is enabled, documents are split into smaller segments (chunks) and the search returns the best matching chunk per document.

**Chunk Search Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_chunks` | boolean | `false` | Enable chunk-level search instead of document-level search |
| `rerank` | boolean | `false` | Apply flashrank re-ranking to chunk results (only when `use_chunks` is true) |

**When to use chunk search:**

- **Large documents**: When documents exceed 512 tokens, chunk search provides more granular matching
- **Specific queries**: When searching for specific concepts that may appear in small sections of long documents
- **Better precision**: Chunk embeddings often yield more accurate semantic matches than full-document embeddings

**Example with chunk search:**

```python
# Search with chunk-level matching
chunk_results = await client.call_tool(
    "query_documents",
    {
        "query": "authentication middleware configuration",
        "use_chunks": True,
        "limit": 10
    }
)

# Search with chunk search and re-ranking
reranked_results = await client.call_tool(
    "query_documents",
    {
        "query": "database connection pooling",
        "use_chunks": True,
        "rerank": True,
        "limit": 5
    }
)
```

**Response format with chunk search:**

When `use_chunks` is enabled, the response includes:
- `content`: The matching chunk text (not the full document)
- `matching_chunk`: The chunk content (same as `content` for chunk search)
- `similarity_score`: The similarity score for the matching chunk
- All other document metadata (file_path, vault_name, obsidian_uri, etc.)

**Note:** Chunk search requires documents to have been ingested with chunking enabled (default behavior). Documents ingested before chunking was implemented will not have chunks available.

### MCP Server Data Access

**Important:** The MCP server only requires access to the **PostgreSQL database**, not the original vault files.

| Requirement | Purpose | Notes |
|-------------|---------|-------|
| PostgreSQL database | Store/query all data | Must be same DB used during ingestion |
| Bearer token | Authentication | Required for all requests |
| Embedding provider | Semantic search | Only needed for `query_documents` tool |

**Not Required:**
- Original Obsidian vault files
- File system access to markdown files

All content is stored in the database during ingestion:
- Document content → `documents.content` (TEXT)
- Vector embeddings → `documents.content_vector` (VECTOR)
- Task data → `tasks` table
- File metadata → `documents.file_path`, `tags`, etc.

This means you can run the MCP server on a completely different machine from where the files were ingested, or even delete the original vault after ingestion.

### Embedding Provider Configuration

The MCP server can start successfully even if the embedding provider is not configured or is missing required credentials (e.g., API key). In this case:

- The server logs a warning about the failed embedding provider initialization
- Tools that don't require embeddings (`get_tasks`, `list_vaults`, `get_all_tags`, etc.) continue to work normally
- The `query_documents` tool will return an error: "Embedding provider not configured"

**To enable semantic search**, ensure your embedding provider is properly configured with one of these options:

1. **OpenAI** (cloud-based, requires API key):
   ```yaml
   endpoints:
     embedding:
       provider: openai
       model: text-embedding-3-small
       api_key: ${OPENAI_API_KEY}
   ```

2. **HuggingFace** (local, no API key needed):
   ```yaml
   endpoints:
     embedding:
       provider: huggingface
       model: sentence-transformers/all-MiniLM-L6-v2
   ```
   Requires: `pip install langchain langchain-huggingface`

3. **OpenRouter** (cloud-based, requires API key):
   ```yaml
   endpoints:
     embedding:
       provider: openrouter
       model: qwen/qwen3-embedding-8b
       api_key: ${OPENROUTER_API_KEY}
   ```
   Requires: `pip install litellm`

See the [LLM Providers](#llm-providers) section for detailed configuration of each provider.

### Docker Support

Build and run the MCP server with Docker:

```bash
# Build image
docker build -t obsidian-rag-mcp .

# Run with MCP-only dependencies
docker run -p 8000:8000 \
  -e OBSIDIAN_RAG_MCP_TOKEN=secret \
  -e OBSIDIAN_RAG_DATABASE_URL=postgresql://host.docker.internal/obsidian_rag \
  obsidian-rag-mcp
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

**XDG_CONFIG_HOME:**

The `XDG_CONFIG_HOME` environment variable specifies the base directory for user-specific configuration files. If not set, it defaults to `~/.config`.

```bash
# Use custom config directory
export XDG_CONFIG_HOME="/path/to/custom/config"

# Config will be loaded from:
# /path/to/custom/config/obsidian-rag/config.yaml
```

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

# Vault configuration (optional - default vault created automatically)
vaults:
  "Obsidian Vault":
    container_path: "/data"
    host_path: "/data"
    description: "Default vault"
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

All settings can be configured via environment variables using the prefix `OBSIDIAN_RAG_`. Create a `.env` file or set variables directly:

```bash
# =============================================================================
# DATABASE CONFIGURATION
# =============================================================================

# PostgreSQL connection URL (required)
# Format: postgresql+psycopg://user:password@host:port/database
# Default: postgresql+psycopg://localhost/obsidian_rag
OBSIDIAN_RAG_DATABASE_URL=postgresql://user:password@localhost/obsidian_rag

# Vector embedding dimension - must match your embedding model output
# Valid range: 1-2000 (pgvector HNSW index limit)
# Default: 1536 (matches text-embedding-3-small)
# Other common values: 384 (all-MiniLM-L6-v2), 768 (text-embedding-3-large)
OBSIDIAN_RAG_DATABASE_VECTOR_DIMENSION=1536

# Connection pool settings for production deployments
# Number of persistent connections to maintain
OBSIDIAN_RAG_DATABASE_POOL_SIZE=10

# Maximum temporary connections beyond pool_size during bursts
OBSIDIAN_RAG_DATABASE_MAX_OVERFLOW=20

# Seconds to wait for a connection from the pool before timeout
OBSIDIAN_RAG_DATABASE_POOL_TIMEOUT=30

# Seconds after which to recycle connections (prevent stale connections)
OBSIDIAN_RAG_DATABASE_POOL_RECYCLE=3600


# =============================================================================
# EMBEDDING ENDPOINT (for vector search)
# =============================================================================

# Provider: openai, openrouter, or huggingface
OBSIDIAN_RAG_ENDPOINTS_EMBEDDING_PROVIDER=openai

# Model name - must match provider
# OpenAI: text-embedding-3-small, text-embedding-3-large
# OpenRouter: openai/text-embedding-3-small
# HuggingFace: all-MiniLM-L6-v2, sentence-transformers/all-mpnet-base-v2
OBSIDIAN_RAG_ENDPOINTS_EMBEDDING_MODEL=text-embedding-3-small

# API key for the embedding provider (required for openai/openrouter)
# For HuggingFace, this is optional (uses local models)
OBSIDIAN_RAG_ENDPOINTS_EMBEDDING_API_KEY=sk-your-api-key-here

# Optional: Custom base URL for the embedding API
# Leave empty to use provider defaults
OBSIDIAN_RAG_ENDPOINTS_EMBEDDING_BASE_URL=


# =============================================================================
# ANALYSIS ENDPOINT (for document analysis tasks)
# =============================================================================

# Provider: openai or openrouter
OBSIDIAN_RAG_ENDPOINTS_ANALYSIS_PROVIDER=openai

# Model for analysis tasks (e.g., summarization, extraction)
OBSIDIAN_RAG_ENDPOINTS_ANALYSIS_MODEL=gpt-4

# API key for analysis provider
OBSIDIAN_RAG_ENDPOINTS_ANALYSIS_API_KEY=sk-your-api-key-here

# Base URL for analysis API
OBSIDIAN_RAG_ENDPOINTS_ANALYSIS_BASE_URL=https://api.openai.com/v1

# Temperature for generation (0.0-2.0, lower is more deterministic)
OBSIDIAN_RAG_ENDPOINTS_ANALYSIS_TEMPERATURE=0.7

# Maximum tokens for analysis responses
OBSIDIAN_RAG_ENDPOINTS_ANALYSIS_MAX_TOKENS=2000


# =============================================================================
# CHAT ENDPOINT (for conversational queries)
# =============================================================================

# Provider: openai or openrouter
OBSIDIAN_RAG_ENDPOINTS_CHAT_PROVIDER=openai

# Model for chat/completion tasks
OBSIDIAN_RAG_ENDPOINTS_CHAT_MODEL=gpt-4

# API key for chat provider
OBSIDIAN_RAG_ENDPOINTS_CHAT_API_KEY=sk-your-api-key-here

# Base URL for chat API
OBSIDIAN_RAG_ENDPOINTS_CHAT_BASE_URL=https://api.openai.com/v1

# Temperature for chat responses (higher = more creative)
OBSIDIAN_RAG_ENDPOINTS_CHAT_TEMPERATURE=0.8


# =============================================================================
# INGESTION SETTINGS
# =============================================================================

# Number of files to process in a single batch
OBSIDIAN_RAG_INGESTION_BATCH_SIZE=100

# Maximum file size in MB (files larger than this are skipped)
OBSIDIAN_RAG_INGESTION_MAX_FILE_SIZE_MB=10

# Log progress every N files during ingestion
OBSIDIAN_RAG_INGESTION_PROGRESS_INTERVAL=10

# Legacy: Maximum characters per chunk (use chunking settings below instead)
OBSIDIAN_RAG_INGESTION_MAX_CHUNK_CHARS=24000

# Legacy: Character overlap between chunks
OBSIDIAN_RAG_INGESTION_CHUNK_OVERLAP_CHARS=800


# =============================================================================
# CHUNKING CONFIGURATION (token-based document splitting)
# =============================================================================

# Target chunk size in tokens (determines how documents are split)
# Valid range: 64-2048
# Smaller chunks = more precise search, more storage
# Larger chunks = broader context, less storage
OBSIDIAN_RAG_CHUNKING_CHUNK_SIZE=512

# Token overlap between chunks (preserves context across boundaries)
OBSIDIAN_RAG_CHUNKING_CHUNK_OVERLAP=50

# Directory to cache tokenizer models (created automatically)
OBSIDIAN_RAG_CHUNKING_TOKENIZER_CACHE_DIR=~/.cache/obsidian-rag/tokenizers

# Tokenizer model to use for counting tokens
# gpt2 is a good general-purpose tokenizer
OBSIDIAN_RAG_CHUNKING_TOKENIZER_MODEL=gpt2

# Enable flashrank re-ranking for improved search relevance
# Requires flashrank package: pip install flashrank
OBSIDIAN_RAG_CHUNKING_FLASHRANK_ENABLED=true

# Flashrank model for re-ranking (cross-encoder)
# ms-marco-MiniLM-L-12-v2 is ~100-200MB RAM
OBSIDIAN_RAG_CHUNKING_FLASHRANK_MODEL=ms-marco-MiniLM-L-12-v2

# Number of top results to re-rank after vector search
OBSIDIAN_RAG_CHUNKING_FLASHRANK_TOP_K=10


# =============================================================================
# MCP SERVER CONFIGURATION
# =============================================================================

# Bind address for the HTTP server (0.0.0.0 = all interfaces)
OBSIDIAN_RAG_MCP_HOST=0.0.0.0

# HTTP port for the MCP server
OBSIDIAN_RAG_MCP_PORT=8000

# Bearer token for authentication (REQUIRED - no default)
# Clients must include: Authorization: Bearer <token>
OBSIDIAN_RAG_MCP_TOKEN=your-secret-token-here

# CORS origins (JSON array format)
# Use '["*"]' to allow all origins (development only)
# Use '["https://app.example.com"]' for specific domains
OBSIDIAN_RAG_MCP_CORS_ORIGINS=["*"]

# Enable health check endpoint at /health
OBSIDIAN_RAG_MCP_ENABLE_HEALTH_CHECK=true

# Stateless mode for horizontal scaling (disables session affinity)
OBSIDIAN_RAG_MCP_STATELESS_HTTP=false

# Maximum concurrent client sessions
OBSIDIAN_RAG_MCP_MAX_CONCURRENT_SESSIONS=100

# Session timeout in seconds (inactive sessions are cleaned up)
OBSIDIAN_RAG_MCP_SESSION_TIMEOUT_SECONDS=300

# Rate limit: maximum connections per second per IP
OBSIDIAN_RAG_MCP_RATE_LIMIT_PER_SECOND=10.0

# Rate limit window in seconds
OBSIDIAN_RAG_MCP_RATE_LIMIT_WINDOW=60

# Enable HTTP request/response logging (can be verbose)
OBSIDIAN_RAG_MCP_ENABLE_REQUEST_LOGGING=true


# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

# Log level: DEBUG, INFO, WARNING, ERROR, CRITICAL
OBSIDIAN_RAG_LOGGING_LEVEL=INFO

# Log format: text or json
# text = human-readable, json = structured for log aggregation
OBSIDIAN_RAG_LOGGING_FORMAT=text
```

**Configuration Precedence** (highest to lowest):
1. CLI flags (e.g., `--embedding-provider openai`)
2. Environment variables (shown above)
3. Config file (`.obsidian-rag.yaml` or `~/.config/obsidian-rag/config.yaml`)
4. Default values

**See Also:**
- [Complete Environment Variables Reference](./docs/environment-variables.md) - Detailed reference with validation rules and examples

## Multi-Vault Configuration

Obsidian RAG supports managing multiple vaults, each with its own documents and metadata. This is useful for separating personal notes from work documents, or managing different projects.

### Vault Configuration Schema

Each vault requires:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `container_path` | string | Yes | Path inside container/Docker for file operations |
| `host_path` | string | No | Path on host system (defaults to `container_path`) |
| `description` | string | No | Human-readable description of the vault |

### Multi-Vault Example

```yaml
# .obsidian-rag.yaml
vaults:
  "Personal":
    container_path: "/data/personal"
    host_path: "/home/user/obsidian/personal"
    description: "Personal knowledge base"

  "Work":
    container_path: "/data/work"
    host_path: "/home/user/obsidian/work"
    description: "Work notes and projects"

  "Projects":
    container_path: "/data/projects"
    description: "Side projects and ideas"
```

### Vault Name Validation

Vault names must follow these rules:

- Must start with an alphanumeric character (a-z, A-Z, 0-9)
- Can contain letters, numbers, spaces, hyphens (`-`), and underscores (`_`)
- Maximum length: 100 characters
- Names are case-sensitive

**Valid vault names:**
- `Personal`
- `Work Vault`
- `my-vault`
- `notes_2024`

**Invalid vault names:**
- `Vault.Name` (contains period)
- `Vault@Home` (contains special character)
- ` My Vault` (starts with space)
- (empty string)

### Docker Path Mapping

When running in Docker, paths inside the container may differ from host paths. Use `container_path` for the path inside Docker and `host_path` for the actual host filesystem path:

```yaml
vaults:
  "Personal":
    # Path inside Docker container where files are mounted
    container_path: "/vaults/personal"
    # Actual path on host system (used for Obsidian URIs)
    host_path: "/home/user/Documents/Obsidian/Personal"
    description: "Personal notes"
```

**Docker run example:**

```bash
docker run -p 8000:8000 \
  -v /home/user/Documents/Obsidian/Personal:/vaults/personal \
  -v /home/user/Documents/Obsidian/Work:/vaults/work \
  -e OBSIDIAN_RAG_MCP_TOKEN=secret \
  obsidian-rag-mcp
```

### Default Vault

If no vaults are configured, a default vault named "Obsidian Vault" is automatically created with `container_path: "/data"`. To use the default vault, simply run:

```bash
obsidian-rag ingest /data
```

### MCP Vault Tools

The MCP server provides vault management tools:

- `list_vaults`: Query all vaults with document counts
- Document tools support optional `vault_name` filtering
- `ingest` tool requires a `vault_name` parameter

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

### OpenRouter (Cloud - Multiple Providers)

Requires `litellm` package:

```bash
pip install litellm
```

OpenRouter provides access to models from multiple providers through a single API.

**Important:** When using `qwen/qwen3-embedding-8b`, you must set `database.vector_dimension: 4096` in your config.

**Note on OpenAI models via OpenRouter:** Due to a litellm 1.82.4 bug, model names with `openai/` prefix (e.g., `openai/text-embedding-3-small`) are automatically handled. The prefix is stripped internally to ensure requests route to OpenRouter's API instead of OpenAI's API. You can use either format:
- `openai/text-embedding-3-small` (OpenRouter format - prefix stripped automatically)
- `text-embedding-3-small` (stripped format - works directly)

Configuration:

```yaml
# Example with qwen/qwen3-embedding-8b (4096 dimensions)
database:
  url: postgresql://localhost/obsidian_rag
  vector_dimension: 4096  # Required for qwen3-embedding-8b

endpoints:
  embedding:
    provider: openrouter
    model: qwen/qwen3-embedding-8b
    api_key: ${OPENROUTER_API_KEY}
    base_url: https://openrouter.ai/api/v1

  chat:
    provider: openrouter
    model: anthropic/claude-3-opus  # or openai/gpt-4, google/gemini-pro, etc.
    api_key: ${OPENROUTER_API_KEY}
    base_url: https://openrouter.ai/api/v1
    temperature: 0.7
```

**Model Format:** OpenRouter uses `provider/model` format (e.g., `qwen/qwen3-embedding-8b`, `anthropic/claude-3-opus`). For OpenAI models, the `openai/` prefix is automatically stripped to work around a litellm routing bug.

## Vector Dimension Limits

PostgreSQL's pgvector extension has a **2000 dimension limit** for indexed vector columns (both HNSW and IVFFLAT indexes). This affects which embedding models you can use.

### Compatible Models (≤ 2000 dimensions)

These models work with pgvector indexes and provide fast similarity search:

| Provider | Model | Dimensions | Compatible |
|----------|-------|------------|------------|
| OpenAI | text-embedding-3-small | 1536 | ✓ |
| OpenAI | text-embedding-ada-002 | 1536 | ✓ |
| HuggingFace | all-MiniLM-L6-v2 | 384 | ✓ |
| HuggingFace | all-MiniLM-L12-v2 | 384 | ✓ |
| HuggingFace | all-mpnet-base-v2 | 768 | ✓ |
| HuggingFace | paraphrase-multilingual-MiniLM-L12-v2 | 384 | ✓ |

### Incompatible Models (> 2000 dimensions)

These models **cannot** be used with pgvector indexes:

| Provider | Model | Dimensions | Issue |
|----------|-------|------------|-------|
| OpenAI | text-embedding-3-large | 3072 | Exceeds limit |
| OpenRouter | qwen/qwen3-embedding-8b | 4096 | Exceeds limit |

**Note:** Using an incompatible model will result in a configuration error at startup. You must choose a model with ≤ 2000 dimensions.

### Configuration Validation

The system validates your configuration at startup:

1. **Dimension limit check:** `database.vector_dimension` must be ≤ 2000
2. **Provider-dimension matching:** The embedding model's output dimension must match `database.vector_dimension`

Example error for exceeding dimension limit:
```
ValueError: vector_dimension must be <= 2000 for pgvector index compatibility.
Compatible models: text-embedding-3-small (1536), text-embedding-ada-002 (1536), ...
Got: 4096
```

Example error for dimension mismatch:
```
ValueError: Embedding dimension mismatch: model 'text-embedding-3-small' produces
1536-dimensional embeddings, but database.vector_dimension is set to 768.
Please set database.vector_dimension to 1536 in your configuration.
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
- Tags (`#tag`) - supports hierarchical tags like `#personal/expenses` or `#v1.0/release`
- Due dates (`[due:: YYYY-MM-DD]`)
- Scheduled dates (`[scheduled:: YYYY-MM-DD]`)
- Completion dates (`[completion:: YYYY-MM-DD]`)
- Priority (`[priority:: highest|high|normal|low|lowest]`)
- Recurrence (`[repeat:: every day|week|month|year]`)
- Custom metadata (`[key:: value]`)

## Development

For information about the codebase architecture, development setup, and contributing guidelines, see [ARCHITECTURE.md](./ARCHITECTURE.md).

For coding conventions and standards, see [CONVENTIONS.md](./CONVENTIONS.md).

## License

MIT © 2026 MP Aguilar
