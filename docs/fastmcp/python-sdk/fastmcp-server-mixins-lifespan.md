> ## Documentation Index
> Fetch the complete documentation index at: https://gofastmcp.com/llms.txt
> Use this file to discover all available pages before exploring further.

# lifespan

# `fastmcp.server.mixins.lifespan`

Lifespan and Docket task infrastructure for FastMCP Server.

## Classes

### `LifespanMixin` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/mixins/lifespan.py#L24" target="_blank"><Icon icon="github" style="width: 14px; height: 14px;" /></a></sup>

Mixin providing lifespan and Docket task infrastructure for FastMCP.

**Methods:**

#### `docket` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/mixins/lifespan.py#L28" target="_blank"><Icon icon="github" style="width: 14px; height: 14px;" /></a></sup>

```python  theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
docket(self: FastMCP) -> Docket | None
```

Get the Docket instance if Docket support is enabled.

Returns None if Docket is not enabled or server hasn't been started yet.
