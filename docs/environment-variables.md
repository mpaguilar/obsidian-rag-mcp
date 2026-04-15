# Environment Variables Reference

Complete reference for all environment variables supported by Obsidian RAG.

## Quick Reference

| Category | Count | Prefix |
|----------|-------|--------|
| Database | 6 | `OBSIDIAN_RAG_DATABASE_*` |
| Endpoints | 15 | `OBSIDIAN_RAG_ENDPOINTS_*` |
| Ingestion | 5 | `OBSIDIAN_RAG_INGESTION_*` |
| Chunking | 7 | `OBSIDIAN_RAG_CHUNKING_*` |
| Logging | 2 | `OBSIDIAN_RAG_LOGGING_*` |
| MCP Server | 11 | `OBSIDIAN_RAG_MCP_*` |
| **Total** | **46** | `OBSIDIAN_RAG_*` |

## Configuration Precedence

Configuration sources (highest to lowest precedence):

1. **CLI flags** (e.g., `--embedding-provider openai`)
2. **Environment variables** (e.g., `OBSIDIAN_RAG_EMBEDDING_PROVIDER=openai`)
3. **Config files** (YAML format)
4. **Default values**

## External Environment Variables

These variables are used but not prefixed with `OBSIDIAN_RAG_`:

| Variable | Default | Description |
|----------|---------|-------------|
| `XDG_CONFIG_HOME` | `~/.config` | Base directory for user-specific configuration files |
| `OPENAI_API_KEY` | None | OpenAI API authentication (used via interpolation) |
| `OPENROUTER_API_KEY` | None | OpenRouter API authentication (used via interpolation) |

---

## Database Configuration

### OBSIDIAN_RAG_DATABASE_URL
- **Type:** string
- **Default:** `postgresql+psycopg://localhost/obsidian_rag`
- **Description:** PostgreSQL connection URL using psycopg driver

### OBSIDIAN_RAG_DATABASE_VECTOR_DIMENSION
- **Type:** integer
- **Default:** `1536`
- **Description:** Vector embedding dimension (max 2000 for pgvector compatibility)
- **Valid Range:** 1-2000

### OBSIDIAN_RAG_DATABASE_POOL_SIZE
- **Type:** integer
- **Default:** `10`
- **Description:** Number of persistent connections in the pool
- **Use Case:** Increase for high-concurrency deployments

### OBSIDIAN_RAG_DATABASE_MAX_OVERFLOW
- **Type:** integer
- **Default:** `20`
- **Description:** Maximum temporary connections beyond pool_size

### OBSIDIAN_RAG_DATABASE_POOL_TIMEOUT
- **Type:** integer
- **Default:** `30`
- **Description:** Seconds to wait for a connection from the pool

### OBSIDIAN_RAG_DATABASE_POOL_RECYCLE
- **Type:** integer
- **Default:** `3600`
- **Description:** Seconds after which to recycle connections

---

## Endpoints Configuration

### Embedding Endpoint

#### OBSIDIAN_RAG_ENDPOINTS_EMBEDDING_PROVIDER
- **Type:** string
- **Default:** `openai`
- **Description:** Embedding provider (`openai`, `openrouter`, `huggingface`)

#### OBSIDIAN_RAG_ENDPOINTS_EMBEDDING_MODEL
- **Type:** string
- **Default:** `text-embedding-3-small`
- **Description:** Model name for embeddings

#### OBSIDIAN_RAG_ENDPOINTS_EMBEDDING_API_KEY
- **Type:** string
- **Default:** None
- **Description:** API key for the embedding provider

#### OBSIDIAN_RAG_ENDPOINTS_EMBEDDING_BASE_URL
- **Type:** string
- **Default:** None
- **Description:** Custom base URL for the embedding API

### Analysis Endpoint

#### OBSIDIAN_RAG_ENDPOINTS_ANALYSIS_PROVIDER
- **Type:** string
- **Default:** `openai`
- **Description:** Provider for analysis tasks

#### OBSIDIAN_RAG_ENDPOINTS_ANALYSIS_MODEL
- **Type:** string
- **Default:** `gpt-4`
- **Description:** Model for analysis

#### OBSIDIAN_RAG_ENDPOINTS_ANALYSIS_API_KEY
- **Type:** string
- **Default:** None
- **Description:** API key for analysis

#### OBSIDIAN_RAG_ENDPOINTS_ANALYSIS_BASE_URL
- **Type:** string
- **Default:** `https://api.openai.com/v1`
- **Description:** Base URL for analysis API

#### OBSIDIAN_RAG_ENDPOINTS_ANALYSIS_TEMPERATURE
- **Type:** float
- **Default:** `0.7`
- **Description:** Generation temperature

#### OBSIDIAN_RAG_ENDPOINTS_ANALYSIS_MAX_TOKENS
- **Type:** integer
- **Default:** `2000`
- **Description:** Maximum tokens in response

### Chat Endpoint

#### OBSIDIAN_RAG_ENDPOINTS_CHAT_PROVIDER
- **Type:** string
- **Default:** `openai`
- **Description:** Provider for chat

#### OBSIDIAN_RAG_ENDPOINTS_CHAT_MODEL
- **Type:** string
- **Default:** `gpt-4`
- **Description:** Model for chat

#### OBSIDIAN_RAG_ENDPOINTS_CHAT_API_KEY
- **Type:** string
- **Default:** None
- **Description:** API key for chat

#### OBSIDIAN_RAG_ENDPOINTS_CHAT_BASE_URL
- **Type:** string
- **Default:** `https://api.openai.com/v1`
- **Description:** Base URL for chat API

#### OBSIDIAN_RAG_ENDPOINTS_CHAT_TEMPERATURE
- **Type:** float
- **Default:** `0.8`
- **Description:** Generation temperature

---

## Ingestion Configuration

### OBSIDIAN_RAG_INGESTION_BATCH_SIZE
- **Type:** integer
- **Default:** `100`
- **Description:** Number of files to process per batch

### OBSIDIAN_RAG_INGESTION_MAX_FILE_SIZE_MB
- **Type:** integer
- **Default:** `10`
- **Description:** Maximum file size in MB to process

### OBSIDIAN_RAG_INGESTION_PROGRESS_INTERVAL
- **Type:** integer
- **Default:** `10`
- **Description:** Log progress every N files

### OBSIDIAN_RAG_INGESTION_MAX_CHUNK_CHARS
- **Type:** integer
- **Default:** `24000`
- **Description:** Legacy: Maximum characters per chunk

### OBSIDIAN_RAG_INGESTION_CHUNK_OVERLAP_CHARS
- **Type:** integer
- **Default:** `800`
- **Description:** Legacy: Character overlap between chunks

---

## Chunking Configuration

### OBSIDIAN_RAG_CHUNKING_CHUNK_SIZE
- **Type:** integer
- **Default:** `512`
- **Description:** Target tokens per chunk
- **Valid Range:** 64-2048

### OBSIDIAN_RAG_CHUNKING_CHUNK_OVERLAP
- **Type:** integer
- **Default:** `50`
- **Description:** Tokens to overlap between chunks
- **Valid Range:** 0-256

### OBSIDIAN_RAG_CHUNKING_TOKENIZER_CACHE_DIR
- **Type:** string
- **Default:** `~/.cache/obsidian-rag/tokenizers`
- **Description:** Directory for caching tokenizer models

### OBSIDIAN_RAG_CHUNKING_TOKENIZER_MODEL
- **Type:** string
- **Default:** `gpt2`
- **Description:** HuggingFace tokenizer model name

### OBSIDIAN_RAG_CHUNKING_FLASHRANK_ENABLED
- **Type:** boolean
- **Default:** `true`
- **Description:** Enable FlashRank re-ranking

### OBSIDIAN_RAG_CHUNKING_FLASHRANK_MODEL
- **Type:** string
- **Default:** `ms-marco-MiniLM-L-12-v2`
- **Description:** FlashRank model for cross-encoder re-ranking

### OBSIDIAN_RAG_CHUNKING_FLASHRANK_TOP_K
- **Type:** integer
- **Default:** `10`
- **Description:** Number of top results to re-rank

---

## Logging Configuration

### OBSIDIAN_RAG_LOGGING_LEVEL
- **Type:** string
- **Default:** `INFO`
- **Description:** Log level (`DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`)

### OBSIDIAN_RAG_LOGGING_FORMAT
- **Type:** string
- **Default:** `text`
- **Description:** Log format (`text` or `json`)

---

## MCP Server Configuration

### OBSIDIAN_RAG_MCP_HOST
- **Type:** string
- **Default:** `0.0.0.0`
- **Description:** Bind address for MCP server

### OBSIDIAN_RAG_MCP_PORT
- **Type:** integer
- **Default:** `8000`
- **Description:** HTTP port for MCP server
- **Valid Range:** 1-65535

### OBSIDIAN_RAG_MCP_TOKEN
- **Type:** string
- **Default:** None
- **Description:** Bearer token for authentication (required)

### OBSIDIAN_RAG_MCP_CORS_ORIGINS
- **Type:** JSON array
- **Default:** `["*"]`
- **Description:** Allowed CORS origins
- **Example:** `'["https://example.com", "https://app.example.com"]'`

### OBSIDIAN_RAG_MCP_ENABLE_HEALTH_CHECK
- **Type:** boolean
- **Default:** `true`
- **Description:** Enable `/health` endpoint

### OBSIDIAN_RAG_MCP_STATELESS_HTTP
- **Type:** boolean
- **Default:** `false`
- **Description:** Enable stateless mode for horizontal scaling

### OBSIDIAN_RAG_MCP_MAX_CONCURRENT_SESSIONS
- **Type:** integer
- **Default:** `100`
- **Description:** Maximum concurrent sessions

### OBSIDIAN_RAG_MCP_SESSION_TIMEOUT_SECONDS
- **Type:** integer
- **Default:** `300`
- **Description:** Session timeout in seconds

### OBSIDIAN_RAG_MCP_RATE_LIMIT_PER_SECOND
- **Type:** float
- **Default:** `10.0`
- **Description:** Maximum connections per second per IP

### OBSIDIAN_RAG_MCP_RATE_LIMIT_WINDOW
- **Type:** integer
- **Default:** `60`
- **Description:** Rate limit window in seconds

### OBSIDIAN_RAG_MCP_ENABLE_REQUEST_LOGGING
- **Type:** boolean
- **Default:** `true`
- **Description:** Enable HTTP request/response logging

---

## Environment Variable Interpolation

Config files support environment variable interpolation using `${VAR}` syntax:

```yaml
# Basic syntax
api_key: ${OPENAI_API_KEY}

# With default value
api_key: ${OPENAI_API_KEY:-default_key}
```

This is useful for keeping secrets out of configuration files.

---

## See Also

- [Configuration](../README.md#configuration) - General configuration guide
- [Chunking](./chunking.md) - Chunking-specific configuration
- [MCP Server](../README.md#mcp-server) - MCP server setup
