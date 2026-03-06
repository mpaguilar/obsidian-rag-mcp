# LibreChat MCP Client Configuration

This document describes known issues and recommended configuration when using obsidian-rag MCP server with LibreChat.

## Overview

The obsidian-rag MCP server supports the Model Context Protocol (MCP) over HTTP transport. While the server functions correctly, users may observe misleading error messages in LibreChat logs that originate from the LibreChat MCP client itself.

## Known Issue: SSE Stream Disconnection Errors

### Symptom

When using LibreChat with the obsidian-rag MCP server configured for `streamable-http` transport, you may see error messages like:

```
error: [MCP][obsidian_rag] SSE stream disconnected: AbortError: This operation was aborted
error: [MCP][obsidian_rag] SSE stream disconnected: TypeError: terminated
error: [MCP][obsidian_rag] Connection error: this may require manual intervention
```

### Root Cause

**IMPORTANT**: This error originates from the LibreChat MCP client, NOT from the obsidian-rag server. The obsidian-rag server is functioning correctly.

The FastMCP "streamable-http" transport internally uses Server-Sent Events (SSE) for streaming responses. LibreChat appears to aggressively close and reopen connections, causing the client-side error messages. This is client-side behavior that the obsidian-rag server cannot control.

### Evidence

Server logs will show successful responses despite the client-side errors:

```
# obsidian-rag server logs (WORKING CORRECTLY):
INFO - Session created: 213cbac8dee445df9db8c0d67d45d5b2 from 172.18.0.8 (active: 1)
INFO: 172.18.0.8:34278 - "POST / HTTP/1.1" 200 OK
INFO: 172.18.0.8:34296 - "POST / HTTP/1.1" 202 Accepted
INFO: 172.18.0.8:34278 - "GET / HTTP/1.1" 200 OK
INFO - Session destroyed: 213cbac8dee445df9db8c0d67d45d5b2 (duration: 2.34s, requests: 3)
```

### Impact

- The error is **benign** - all MCP tool calls succeed
- LibreChat may briefly show "Reconnecting X/3" messages
- MCP functionality works correctly despite the errors
- The misleading "may require manual intervention" message is incorrect

## Recommended Configuration

### LibreChat Configuration

Add to your LibreChat `librechat.yaml` configuration:

```yaml
mcpServers:
  obsidian_rag:
    type: streamable-http
    url: http://obsidian-rag-mcp:8000
    headers:
      Authorization: "Bearer YOUR_MCP_TOKEN"
    timeout: 6000  # Not the cause of errors, but recommended minimum
```

### Docker Compose Setup

```yaml
services:
  obsidian-rag-mcp:
    image: obsidian-rag-mcp:latest
    environment:
      - OBSIDIAN_RAG_MCP_TOKEN=YOUR_MCP_TOKEN
      - OBSIDIAN_RAG_DATABASE_URL=postgresql+psycopg://db:5432/obsidian_rag
      - OBSIDIAN_RAG_MCP_MAX_CONCURRENT_SESSIONS=100
      - OBSIDIAN_RAG_MCP_SESSION_TIMEOUT_SECONDS=300
      - OBSIDIAN_RAG_MCP_RATE_LIMIT_PER_SECOND=10
    ports:
      - "8000:8000"
    networks:
      - librechat-network

  librechat:
    image: librechat:latest
    environment:
      - MCP_TOKEN=YOUR_MCP_TOKEN
    volumes:
      - ./librechat.yaml:/app/librechat.yaml
    networks:
      - librechat-network

networks:
  librechat-network:
    driver: bridge
```

## Server-Side Logging

The obsidian-rag server now includes enhanced session lifecycle logging to help verify server health:

### Session Lifecycle Logs

```
INFO - Session created: <session_id> from <client_ip> (active: <count>)
INFO - Session destroyed: <session_id> (duration: <seconds>s, requests: <count>)
```

### Connection Rate Limiting

When rate limits are exceeded:

```
WARNING - Rate limit exceeded for <client_ip>: 10/sec over 60s
WARNING - Max concurrent sessions reached (100): rejecting session <id>
```

### Health Check Endpoint

Check server health including session metrics:

```bash
curl http://obsidian-rag-mcp:8000/health \
  -H "Authorization: Bearer YOUR_MCP_TOKEN"
```

Response:

```json
{
  "status": "healthy",
  "version": "0.2.3",
  "database": "connected",
  "sessions": {
    "total_created": 42,
    "total_destroyed": 40,
    "active_count": 2,
    "total_requests": 156,
    "peak_concurrent": 5,
    "connection_rate": 0.7,
    "active_sessions_by_ip": {
      "172.18.0.8": 2
    }
  }
}
```

## Troubleshooting

### Verify Server Health

1. Check health endpoint returns `healthy` status
2. Verify session metrics show active connections
3. Confirm database status is `connected`

### Verify Client Connectivity

1. Check LibreChat can successfully call MCP tools
2. Ignore SSE disconnection errors (client-side)
3. Monitor request success rate in server logs

### Adjusting Rate Limits

If you see frequent rate limit warnings:

```yaml
# .obsidian-rag.yaml or environment variables
mcp:
  rate_limit_per_second: 20      # Increase if needed (default: 10)
  rate_limit_window: 60          # Time window in seconds (default: 60)
  max_concurrent_sessions: 100   # Maximum concurrent (default: 100)
  session_timeout_seconds: 300   # Session timeout (default: 300)
```

## Summary

- The "SSE stream disconnected" errors are **client-side** and **benign**
- The obsidian-rag server functions correctly
- Use server-side logs and health endpoint to verify server health
- All MCP tools work correctly despite the misleading error messages
- This is a LibreChat MCP client behavior, not a server issue

For additional support, check the obsidian-rag server logs rather than relying on LibreChat error messages.
