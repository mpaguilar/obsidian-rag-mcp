> ## Documentation Index
> Fetch the complete documentation index at: https://gofastmcp.com/llms.txt
> Use this file to discover all available pages before exploring further.

# memory

# `fastmcp.client.transports.memory`

## Classes

### `FastMCPTransport` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/client/transports/memory.py#L14" target="_blank"><Icon icon="github" style="width: 14px; height: 14px;" /></a></sup>

In-memory transport for FastMCP servers.

This transport connects directly to a FastMCP server instance in the same
Python process. It works with both FastMCP 2.x servers and FastMCP 1.0
servers from the low-level MCP SDK. This is particularly useful for unit
tests or scenarios where client and server run in the same runtime.

**Methods:**

#### `connect_session` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/client/transports/memory.py#L33" target="_blank"><Icon icon="github" style="width: 14px; height: 14px;" /></a></sup>

```python  theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
connect_session(self, **session_kwargs: Unpack[SessionKwargs]) -> AsyncIterator[ClientSession]
```
