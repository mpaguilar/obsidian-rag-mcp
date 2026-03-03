> ## Documentation Index
> Fetch the complete documentation index at: https://gofastmcp.com/llms.txt
> Use this file to discover all available pages before exploring further.

# __init__

# `fastmcp.server.openapi`

OpenAPI server implementation for FastMCP.

.. deprecated::
This module is deprecated. Import from fastmcp.server.providers.openapi instead.

The recommended approach is to use OpenAPIProvider with FastMCP:

from fastmcp import FastMCP
from fastmcp.server.providers.openapi import OpenAPIProvider
import httpx

client = httpx.AsyncClient(base\_url="[https://api.example.com](https://api.example.com)")
provider = OpenAPIProvider(openapi\_spec=spec, client=client)

mcp = FastMCP("My API Server")
mcp.add\_provider(provider)

FastMCPOpenAPI is still available but deprecated.
