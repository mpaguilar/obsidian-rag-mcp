> ## Documentation Index
> Fetch the complete documentation index at: https://gofastmcp.com/llms.txt
> Use this file to discover all available pages before exploring further.

# server

# `fastmcp.server.openapi.server`

FastMCPOpenAPI - backwards compatibility wrapper.

This class is deprecated. Use FastMCP with OpenAPIProvider instead:

from fastmcp import FastMCP
from fastmcp.server.providers.openapi import OpenAPIProvider
import httpx

client = httpx.AsyncClient(base\_url="[https://api.example.com](https://api.example.com)")
provider = OpenAPIProvider(openapi\_spec=spec, client=client)
mcp = FastMCP("My API Server", providers=\[provider])

## Classes

### `FastMCPOpenAPI` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/openapi/server.py#L30" target="_blank"><Icon icon="github" style="width: 14px; height: 14px;" /></a></sup>

FastMCP server implementation that creates components from an OpenAPI schema.

.. deprecated::
Use FastMCP with OpenAPIProvider instead. This class will be
removed in a future version.

Example (deprecated):

```python  theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from fastmcp.server.openapi import FastMCPOpenAPI
import httpx

server = FastMCPOpenAPI(
    openapi_spec=spec,
    client=httpx.AsyncClient(),
)
```
