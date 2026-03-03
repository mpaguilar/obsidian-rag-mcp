# Quickstart

Welcome! This guide will help you quickly set up FastMCP, run your first MCP server, and deploy a server to Prefect Horizon.

## Create a FastMCP Server

A FastMCP server is a collection of tools, resources, and other MCP components. To create a server, start by instantiating the `FastMCP` class.

```python
from fastmcp import FastMCP

mcp = FastMCP("My MCP Server")
```

## Add a Tool

To add a tool that returns a simple greeting, write a function and decorate it with `@mcp.tool` to register it with the server:

```python
from fastmcp import FastMCP

mcp = FastMCP("My MCP Server")

@mcp.tool
def greet(name: str) -> str:
    return f"Hello, {name}!"
```

## Run the Server

The simplest way to run your FastMCP server is to call its `run()` method. You can choose between different transports, like `stdio` for local servers, or `http` for remote access:

```python
# HTTP transport for remote access
if __name__ == "__main__":
    mcp.run(transport="http", port=8000)
```

Your server is now accessible at `http://localhost:8000/mcp`.

## Authentication

FastMCP supports Bearer token authentication for HTTP servers:

```python
from fastmcp import FastMCP
from fastmcp.server.auth import BearerTokenAuth

auth = BearerTokenAuth(token="your-secret-token")
mcp = FastMCP("Protected Server", auth=auth)
```
