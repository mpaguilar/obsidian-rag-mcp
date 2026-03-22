"""Custom type stubs for external libraries without community stubs.

This package provides type stubs for libraries that lack official
or community type stubs, enabling mypy strict mode compliance.

Stubbed libraries:
    - flashrank: Cross-encoder re-ranking
    - pgvector: PostgreSQL vector extension
    - fastmcp: MCP server framework
    - litellm: Provider-agnostic LLM connectivity
    - langchain_huggingface: Local HuggingFace embeddings
    - tiktoken: Fast tokenization for OpenAI models

Usage:
    mypy automatically discovers these stubs via mypy_path configuration
    in pyproject.toml. No explicit imports needed.
"""
