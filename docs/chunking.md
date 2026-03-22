# Token-Based Document Chunking

Obsidian RAG supports token-based document chunking for improved semantic search precision. This feature splits large documents into smaller, semantically coherent chunks optimized for re-ranking models.

## Overview

Token-based chunking addresses a common limitation in semantic search: large documents often contain multiple topics, causing vector embeddings to become diluted and less precise. By splitting documents into smaller chunks (typically 256-512 tokens), each chunk focuses on a specific topic, resulting in more accurate similarity scores.

### Benefits

- **Improved precision**: Smaller chunks have more focused semantic meaning
- **Better context matching**: Query terms match specific sections rather than entire documents
- **Re-ranking support**: Chunks are sized optimally for cross-encoder re-ranking models
- **Task isolation**: Task lines are automatically detected and preserved as separate chunks

## Configuration

Add chunking settings to your `.obsidian-rag.yaml`:

```yaml
chunking:
  chunk_size: 512              # Target tokens per chunk (64-2048)
  chunk_overlap: 50            # Tokens to overlap between chunks (0-256)
  tokenizer_cache_dir: ~/.cache/obsidian-rag/tokenizers
  tokenizer_model: gpt2        # HuggingFace tokenizer model
  flashrank_enabled: true        # Enable re-ranking
  flashrank_model: ms-marco-MiniLM-L-12-v2
  flashrank_top_k: 10          # Number of results to re-rank
```

### Configuration Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `chunk_size` | integer | 512 | Target tokens per chunk. Range: 64-2048. Smaller chunks (256-512) work best for re-ranking. |
| `chunk_overlap` | integer | 50 | Tokens to overlap between consecutive chunks. Range: 0-256. Helps preserve context at chunk boundaries. |
| `tokenizer_cache_dir` | string | `~/.cache/obsidian-rag/tokenizers` | Directory for caching tokenizer models. |
| `tokenizer_model` | string | `gpt2` | HuggingFace tokenizer model name. Used for accurate token counting. |
| `flashrank_enabled` | boolean | `true` | Enable FlashRank re-ranking for improved result quality. |
| `flashrank_model` | string | `ms-marco-MiniLM-L-12-v2` | FlashRank model for cross-encoder re-ranking. |
| `flashrank_top_k` | integer | 10 | Number of initial results to re-rank. Higher values improve recall but increase latency. |

### Environment Variables

All chunking options can be configured via environment variables:

```bash
export OBSIDIAN_RAG_CHUNKING_CHUNK_SIZE=512
export OBSIDIAN_RAG_CHUNKING_CHUNK_OVERLAP=50
export OBSIDIAN_RAG_CHUNKING_FLASHRANK_ENABLED=true
export OBSIDIAN_RAG_CHUNKING_FLASHRANK_TOP_K=10
```

## CLI Usage

### Search at Chunk Level

Use the `--chunks` flag to search at the chunk level instead of document level:

```bash
# Basic chunk-level search
obsidian-rag query "project planning" --chunks --limit 20

# Search with higher limit for better recall
obsidian-rag query "machine learning concepts" --chunks --limit 50

# Combine with vault filter
obsidian-rag query "architecture decisions" --chunks --vault "Work"
```

### Search with Re-ranking

Use the `--rerank` flag to apply cross-encoder re-ranking to results:

```bash
# Search with re-ranking (implies --chunks)
obsidian-rag query "project planning" --rerank

# Combine both flags explicitly
obsidian-rag query "meeting notes" --chunks --rerank --limit 20

# JSON output for programmatic use
obsidian-rag query "API design patterns" --chunks --rerank --format json
```

### CLI Output

Chunk-level search output includes additional metadata:

```
┌─────────────────────────────────────────────────────────────────────┐
│ File: projects/alpha.md                                             │
│ Path: projects/alpha.md                                             │
│ Score: 0.9234                                                       │
│ Chunk: 3/5 (tokens: 487)                                              │
│ Type: content                                                       │
├─────────────────────────────────────────────────────────────────────┤
│ The architecture follows a microservices pattern with each...       │
└─────────────────────────────────────────────────────────────────────┘
```

**Output fields:**
- `Chunk`: Position within document (e.g., "3/5")
- `Tokens`: Number of tokens in the chunk
- `Type`: Either `content` or `task`

## MCP Tool Usage

### Chunk-Level Search

Use the `use_chunks` parameter in the `query_documents` MCP tool:

```json
{
  "query": "project planning methodologies",
  "use_chunks": true,
  "limit": 20
}
```

### Search with Re-ranking

Use the `rerank` parameter to enable cross-encoder re-ranking:

```json
{
  "query": "architecture decisions",
  "use_chunks": true,
  "rerank": true,
  "limit": 20
}
```

### Python Client Example

```python
from fastmcp import Client
from fastmcp.client.auth import BearerAuth

client = Client(
    "http://localhost:8000/",
    auth=BearerAuth("your-token-here")
)

async with client:
    # Chunk-level search with re-ranking
    results = await client.call_tool(
        "query_documents",
        {
            "query": "machine learning concepts",
            "use_chunks": True,
            "rerank": True,
            "limit": 10
        }
    )
```

### MCP Response Format

Chunk-level search responses include chunk metadata:

```json
{
  "documents": [
    {
      "id": "chunk-uuid-here",
      "file_name": "ml-guide.md",
      "file_path": "guides/ml-guide.md",
      "content": "Neural networks consist of layers of interconnected nodes...",
      "similarity_score": 0.9234,
      "chunk_index": 3,
      "total_chunks": 8,
      "chunk_type": "content",
      "token_count": 487
    }
  ],
  "total_count": 42,
  "has_more": true,
  "next_offset": 10
}
```

## Chunk Types

Documents are split into two types of chunks:

### Content Chunks

Regular document content split by token count with overlap:

- Split at token boundaries (not character boundaries)
- Overlap preserves context between chunks
- Respects paragraph boundaries when possible
- Maximum chunk size enforced strictly

### Task Chunks

Task lines are automatically detected and preserved as separate chunks:

- Task lines are never split across chunks
- Each task becomes its own chunk
- Task chunks include surrounding context (up to `chunk_size` tokens)
- Task metadata (due dates, priority, tags) is preserved

**Task detection patterns:**

```markdown
- [ ] Regular task becomes a task chunk
- [x] Completed task becomes a task chunk
- [/] In-progress task becomes a task chunk
- [-] Cancelled task becomes a task chunk
```

### Chunk Type Identification

Each chunk is tagged with its type:

| Type | Description |
|------|-------------|
| `content` | Regular document content |
| `task` | Task line with optional context |

## Ingestion Statistics

When chunking is enabled, ingestion results include chunk statistics:

```bash
$ obsidian-rag ingest /path/to/vault --verbose

Ingestion complete:
  Total files: 150
  New documents: 12
  Updated: 5
  Unchanged: 133
  Errors: 0
  
Chunk statistics:
  Total chunks: 1,247
  Average tokens per chunk: 498
  Content chunks: 1,180
  Task chunks: 67
```

### Statistics Fields

| Field | Description |
|-------|-------------|
| `total_chunks` | Total number of chunks created across all documents |
| `avg_chunk_tokens` | Average token count per chunk |
| `content_chunk_count` | Number of regular content chunks |
| `task_chunk_count` | Number of task-specific chunks |

### Programmatic Access

When using the `ingest` MCP tool, chunk statistics are included in the response:

```json
{
  "total": 150,
  "new": 12,
  "updated": 5,
  "unchanged": 133,
  "errors": 0,
  "deleted": 0,
  "total_chunks": 1247,
  "avg_chunk_tokens": 498.3,
  "content_chunk_count": 1180,
  "task_chunk_count": 67,
  "processing_time_seconds": 45.2
}
```

## Re-ranking with FlashRank

FlashRank provides cross-encoder re-ranking for improved search quality.

### How Re-ranking Works

1. **Initial retrieval**: Vector similarity search returns top-K chunks
2. **Re-ranking**: Cross-encoder model scores each chunk against the query
3. **Reordering**: Results are reordered by the cross-encoder scores
4. **Return**: Top-N results after re-ranking are returned

### Benefits

- **Better precision**: Cross-encoders capture subtle semantic relationships
- **Query-aware scoring**: Scores depend on the specific query, not just chunk content
- **Improved ranking**: Chunks that are more relevant to the query are boosted

### Performance Considerations

- Re-ranking adds latency (typically 50-200ms for top 10 results)
- Larger `flashrank_top_k` values improve recall but increase latency
- Recommended `flashrank_top_k`: 10-20 for most use cases
- Re-ranking is most beneficial for complex, multi-topic queries

### Disabling Re-ranking

To disable re-ranking while still using chunking:

```yaml
chunking:
  flashrank_enabled: false
```

Or via CLI:

```bash
obsidian-rag query "project planning" --chunks  # No --rerank flag
```

## Best Practices

### Chunk Size Selection

| Use Case | Recommended Size | Rationale |
|----------|------------------|-----------|
| General search | 512 tokens | Good balance of precision and context |
| Re-ranking | 256-384 tokens | Optimal for cross-encoder models |
| Long documents | 512-768 tokens | Reduces total chunk count |
| Task-heavy docs | 256 tokens | Keeps tasks focused |

### Overlap Recommendations

- **0 tokens**: No overlap, maximum efficiency
- **50 tokens** (default): Good balance for most documents
- **100+ tokens**: Better for documents with complex interdependencies

### When to Use Chunking

**Use chunking when:**
- Documents are longer than 1000 tokens
- Documents cover multiple topics
- You need precise search results
- You're using re-ranking for improved quality

**Skip chunking when:**
- All documents are short notes (< 500 tokens)
- Documents are highly focused on single topics
- Latency is critical and precision is less important

### Combining with Other Features

Chunking works seamlessly with other Obsidian RAG features:

- **Multi-vault**: Chunking is applied per-vault during ingestion
- **Tag filtering**: Tag filters apply to chunks (chunks inherit document tags)
- **Property filtering**: Property filters apply at document level before chunking
- **Task queries**: Task chunks are searchable via both `query_documents` and `get_tasks`

## Troubleshooting

### Common Issues

**Issue**: Chunks are too small, creating too many results
- **Solution**: Increase `chunk_size` to 768 or 1024

**Issue**: Context is lost at chunk boundaries
- **Solution**: Increase `chunk_overlap` to 100+ tokens

**Issue**: Re-ranking is slow
- **Solution**: Reduce `flashrank_top_k` to 5-10

**Issue**: Tasks are split across chunks
- **Solution**: This shouldn't happen - tasks are always preserved. If observed, report as bug.

### Validation Errors

**Error**: `chunk_size must be between 64 and 2048`
- **Fix**: Adjust `chunk_size` to valid range

**Error**: `chunk_overlap must be between 0 and 256`
- **Fix**: Adjust `chunk_overlap` to valid range

**Error**: `chunk_overlap cannot exceed chunk_size`
- **Fix**: Reduce `chunk_overlap` or increase `chunk_size`

## Migration Guide

### Enabling Chunking on Existing Data

Chunking is applied during ingestion. To enable chunking on existing documents:

1. Update configuration to enable chunking:
   ```yaml
   chunking:
     chunk_size: 512
     chunk_overlap: 50
   ```

2. Re-ingest your vaults:
   ```bash
   obsidian-rag ingest /path/to/vault --vault "My Vault"
   ```

3. Documents will be re-chunked during ingestion

### Backward Compatibility

- Chunking is opt-in via configuration
- Existing documents without chunks continue to work
- Document-level search remains available via `--chunks=false` or omitting `use_chunks`
- No database migration required - chunks are created during ingestion
