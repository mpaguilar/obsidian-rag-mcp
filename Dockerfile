# Dockerfile for Obsidian RAG MCP Server
# Build with: docker build -t obsidian-rag-mcp .
# Run with: docker run -p 8000:8000 -e OBSIDIAN_RAG_MCP_TOKEN=secret obsidian-rag-mcp

ARG INSTALL_MODE=mcp-only

FROM python:3.12-slim AS base

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy package files
COPY pyproject.toml .
COPY obsidian_rag/ ./obsidian_rag/

# Install Python dependencies
RUN pip install --no-cache-dir -e "."

# Production stage
FROM base AS production

ARG INSTALL_MODE

# Install additional dependencies based on INSTALL_MODE
RUN if [ "$INSTALL_MODE" = "full" ]; then \
        pip install --no-cache-dir -e ".[all]"; \
    fi

# Create non-root user
RUN useradd -m -u 1000 obsidian && chown -R obsidian:obsidian /app
USER obsidian

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Expose port
EXPOSE 8000

# Run MCP server
CMD ["python", "-m", "obsidian_rag.mcp_server"]
