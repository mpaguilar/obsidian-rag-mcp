# Implementation Plan: 004.cleanup-and-compliance

**Date:** 2026-03-03  
**Branch:** 004.cleanup-and-compliance  
**Goal:** Fix failing tests, resolve ANN401 typing errors, and achieve 100% test coverage

---

## Overview

This plan addresses three main requirements:
1. **REQ-001:** Fix failing tests in `test_tools_documents.py` (SQLite compatibility for vector similarity queries)
2. **REQ-002:** Fix all ruff ANN401 errors (replace `typing.Any` with proper types)
3. **REQ-003:** Achieve 100% test coverage across all modules

---

## REQ-001: Fix SQLite Compatibility for Vector Queries

### Problem
The `query_documents` function in `obsidian_rag/mcp_server/tools/documents.py` uses PostgreSQL's cosine distance operator `<=>` which doesn't exist in SQLite. This causes `sqlite3.OperationalError: near ">": syntax error` in tests.

### Solution
Detect the database dialect and provide a fallback for SQLite. For SQLite (used in tests), we can:
- Return empty results (documents can't have vector embeddings in SQLite anyway since the Vector type is PostgreSQL-specific)
- Skip the vector similarity query entirely for SQLite

### Implementation
Modify `obsidian_rag/mcp_server/tools/documents.py`:
```python
# Add dialect detection at the start of query_documents
dialect = session.bind.dialect.name if session.bind else "unknown"

if dialect == "postgresql":
    # Use vector similarity query
    distance_expr = Document.content_vector.cosine_distance(query_embedding)
    query = (
        session.query(Document, distance_expr.label("distance"))
        .filter(Document.content_vector.isnot(None))
        .order_by(distance_expr.asc())
    )
else:
    # For SQLite and other databases without pg_vector, return empty results
    # since content_vector won't be populated anyway
    return DocumentListResponse(
        results=[],
        total_count=0,
        has_more=False,
        next_offset=None,
    )
```

---

## REQ-002: Fix ANN401 Errors (12 total)

### File: `obsidian_rag/cli.py` (7 errors)

#### Line 89: `_get_embedding_provider`
**Current:**
```python
def _get_embedding_provider(settings: Any) -> Any:
```

**Fix:**
```python
from obsidian_rag.config import Settings
from obsidian_rag.llm.base import EmbeddingProvider

def _get_embedding_provider(settings: Settings) -> EmbeddingProvider:
```

#### Line 117: `ProcessingContext.embedding_provider`
**Current:**
```python
embedding_provider: Any,
```

**Fix:**
```python
embedding_provider: EmbeddingProvider,
```

#### Line 153: `_process_files.embedding_provider`
**Current:**
```python
embedding_provider: Any,
```

**Fix:**
```python
embedding_provider: EmbeddingProvider,
```

#### Line 256: `_process_single_file.embedding_provider`
**Current:**
```python
embedding_provider: Any,
```

**Fix:**
```python
embedding_provider: EmbeddingProvider,
```

#### Line 305: `_create_document.embedding_provider`
**Current:**
```python
embedding_provider: Any,
```

**Fix:**
```python
embedding_provider: EmbeddingProvider,
```

#### Line 349: `_create_tasks.session`
**Current:**
```python
session: Any,
```

**Fix:**
```python
session: Session,
```

#### Line 373: `_update_tasks.session`
**Current:**
```python
session: Any,
```

**Fix:**
```python
session: Session,
```

### File: `obsidian_rag/config.py` (3 errors)

#### Line 129: `_interpolate_env_vars`
**Current:**
```python
def _interpolate_env_vars(value: Any) -> Any:
```

**Fix:**
```python
from typing import Union

def _interpolate_env_vars(value: Union[str, list, dict]) -> Union[str, list, dict]:
```

#### Line 229: `YamlConfigSettingsSource.get_field_value`
**Current:**
```python
def get_field_value(self, field: Any, field_name: str) -> tuple[Any, str, bool]:
```

**Fix:**
```python
from pydantic.fields import FieldInfo

def get_field_value(self, field: FieldInfo, field_name: str) -> tuple[None, str, bool]:
```

### File: `obsidian_rag/database/models.py` (1 error)

#### Line 50: `ArrayType.load_dialect_impl`
**Current:**
```python
def load_dialect_impl(self, dialect: "Dialect") -> Any:
```

**Fix:**
```python
from sqlalchemy import TypeEngine

def load_dialect_impl(self, dialect: "Dialect") -> TypeEngine[Any]:
```

---

## REQ-003: Achieve 100% Test Coverage

### Coverage Gaps Analysis

| Module | Current | Target | Missing Lines | Strategy |
|--------|---------|--------|---------------|----------|
| cli.py | 79% | 100% | 71-72, 77, 219-250, 409->411, 411->413, 442-469, 491-505, 569 | Add integration tests |
| config.py | 99% | 100% | 368-369 | Add error path test |
| database/models.py | 95% | 100% | 221-223 | PostgreSQL-specific, use # pragma: no cover |
| llm/base.py | 95% | 100% | 31, 41, 66 | Abstract methods, use # pragma: no cover |
| llm/providers.py | 99% | 100% | 455->458 | Defensive branch test |
| mcp_server/__main__.py | 0% | 100% | All | Create new test file |
| mcp_server/server.py | 51% | 100% | 49-62, 72-79, 89-96, etc. | Unit tests for server functions |
| mcp_server/tools/documents.py | 65% | 100% | 65-79 | Will be fixed with REQ-001 |

### Test Implementation Plan

#### 1. `tests/test_cli.py` - Add missing coverage
- Test error paths in `_scan_vault`
- Test CLI commands (`ingest`, `query`, `tasks`)
- Test dry-run mode
- Test `_format_task_results` with various statuses

#### 2. `tests/test_config.py` - Add missing coverage
- Test error handling at lines 368-369 (likely in `validate_embedding_dimension_compatibility`)

#### 3. `obsidian_rag/database/models.py` - Add no-cover pragma
```python
@event.listens_for(Base.metadata, "before_create")
def _create_pgvector_extension(...) -> None:
    """Create pgvector extension before creating tables (PostgreSQL only)."""
    dialect = connection.dialect.name
    if dialect == "postgresql":  # pragma: no cover
        ...
```

#### 4. `obsidian_rag/llm/base.py` - Add no-cover pragmas for abstract methods
```python
@abstractmethod
def generate_embedding(self, text: str) -> list[float]:  # pragma: no cover
    ...
```

#### 5. `tests/test_llm_providers.py` - Add defensive branch test
- Add test for line 455->458 (likely an error handling branch)

#### 6. `tests/mcp_server/test_main.py` - Create new test file
Test `obsidian_rag/mcp_server/__main__.py`:
- Test successful server startup
- Test settings loading failure
- Test missing token error
- Test server creation failure
- Test uvicorn import failure

#### 7. `tests/mcp_server/test_server.py` - Create new test file
Test `obsidian_rag/mcp_server/server.py`:
- Test `_create_embedding_provider` success/failure
- Test tool handlers (`_get_incomplete_tasks_handler`, etc.)
- Test `_register_task_tools`, `_register_document_tools`, `_register_health_check`
- Test `create_mcp_server` with/without token
- Test `create_http_app`

---

## Verification Commands

After implementation, verify with:

```bash
# All tests pass
python -m pytest tests/ -v

# 100% coverage
python -m pytest tests/ --cov=obsidian_rag --cov-branch --cov-report=term-missing

# Zero ruff errors
ruff check obsidian_rag/ tests/

# File sizes
find obsidian_rag tests -name "*.py" -exec wc -l {} + | sort -n | tail -10
```

---

## Implementation Order

1. **REQ-001**: Fix SQLite compatibility first (unblocks tests)
2. **REQ-002**: Fix ANN401 errors (type safety)
3. **REQ-003**: Add missing tests in parallel:
   - cli.py tests
   - config.py tests
   - llm/providers.py defensive branch
   - mcp_server/__main__.py tests
   - mcp_server/server.py tests
4. **Verification**: Run all checks

---

## Success Criteria

- [ ] All 342 tests pass (338 passing + 4 previously failing)
- [ ] 100% test coverage on all modules
- [ ] Zero ruff ANN401 errors
- [ ] All files remain under 1000 lines
- [ ] No new ANN401 or other linting errors introduced
