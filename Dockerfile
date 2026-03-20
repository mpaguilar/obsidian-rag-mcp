# Dockerfile for Obsidian RAG MCP Server
# Build with: docker build -t obsidian-rag-mcp .
# Run with: docker run -p 8000:8000 -e OBSIDIAN_RAG_MCP_TOKEN=secret obsidian-rag-mcp
#
# Production deployment uses Gunicorn with Uvicorn workers for:
# - Multiple worker processes for concurrent request handling
# - Graceful shutdown and process management
# - Better resource utilization and stability

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

# Copy health check script
COPY healthcheck.py /app/healthcheck.py
RUN chown obsidian:obsidian /app/healthcheck.py && chmod +x /app/healthcheck.py

USER obsidian

# Environment variables for Gunicorn configuration
# These can be overridden at runtime
ENV GUNICORN_WORKERS=4 \
    GUNICORN_THREADS=2 \
    GUNICORN_TIMEOUT=120 \
    GUNICORN_KEEPALIVE=5 \
    GUNICORN_WORKER_CONNECTIONS=1000 \
    GUNICORN_MAX_REQUESTS=10000 \
    GUNICORN_MAX_REQUESTS_JITTER=1000

# Health check - verifies MCP server is responding
# Uses the health endpoint which checks database connectivity
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD python /app/healthcheck.py || exit 1

# Expose port
EXPOSE 8000

# Run MCP server with Gunicorn and Uvicorn workers
# - Workers: 4 processes (adjust based on CPU cores)
# - Worker class: UvicornWorker for ASGI support
# - Threads: 2 threads per worker for concurrent handling
# - Timeout: 120s for long-running requests
# - Keepalive: 5s to maintain connections
# - Max requests: Restart workers after 10k requests (prevent memory leaks)
# - Access log: Disabled (use structured logging from app)
# - Error log: stderr for container environments
CMD exec gunicorn \
    --bind 0.0.0.0:8000 \
    --workers "${GUNICORN_WORKERS}" \
    --threads "${GUNICORN_THREADS}" \
    --worker-class uvicorn.workers.UvicornWorker \
    --timeout "${GUNICORN_TIMEOUT}" \
    --keep-alive "${GUNICORN_KEEPALIVE}" \
    --worker-connections "${GUNICORN_WORKER_CONNECTIONS}" \
    --max-requests "${GUNICORN_MAX_REQUESTS}" \
    --max-requests-jitter "${GUNICORN_MAX_REQUESTS_JITTER}" \
    --access-logfile - \
    --error-logfile - \
    --capture-output \
    --enable-stdio-inheritance \
    --preload \
    "obsidian_rag.mcp_server.server:create_http_app_factory()"
