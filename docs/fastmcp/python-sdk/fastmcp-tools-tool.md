> ## Documentation Index
> Fetch the complete documentation index at: https://gofastmcp.com/llms.txt
> Use this file to discover all available pages before exploring further.

# tool

# `fastmcp.tools.tool`

## Functions

### `default_serializer` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/tools/tool.py#L64" target="_blank"><Icon icon="github" style="width: 14px; height: 14px;" /></a></sup>

```python  theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
default_serializer(data: Any) -> str
```

## Classes

### `ToolResult` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/tools/tool.py#L68" target="_blank"><Icon icon="github" style="width: 14px; height: 14px;" /></a></sup>

**Methods:**

#### `to_mcp_result` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/tools/tool.py#L121" target="_blank"><Icon icon="github" style="width: 14px; height: 14px;" /></a></sup>

```python  theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
to_mcp_result(self) -> list[ContentBlock] | tuple[list[ContentBlock], dict[str, Any]] | CallToolResult
```

### `Tool` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/tools/tool.py#L137" target="_blank"><Icon icon="github" style="width: 14px; height: 14px;" /></a></sup>

Internal tool registration info.

**Methods:**

#### `to_mcp_tool` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/tools/tool.py#L179" target="_blank"><Icon icon="github" style="width: 14px; height: 14px;" /></a></sup>

```python  theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
to_mcp_tool(self, **overrides: Any) -> MCPTool
```

Convert the FastMCP tool to an MCP tool.

#### `from_function` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/tools/tool.py#L206" target="_blank"><Icon icon="github" style="width: 14px; height: 14px;" /></a></sup>

```python  theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from_function(cls, fn: Callable[..., Any]) -> FunctionTool
```

Create a Tool from a function.

#### `run` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/tools/tool.py#L246" target="_blank"><Icon icon="github" style="width: 14px; height: 14px;" /></a></sup>

```python  theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
run(self, arguments: dict[str, Any]) -> ToolResult
```

Run the tool with arguments.

This method is not implemented in the base Tool class and must be
implemented by subclasses.

`run()` can EITHER return a list of ContentBlocks, or a tuple of
(list of ContentBlocks, dict of structured output).

#### `convert_result` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/tools/tool.py#L258" target="_blank"><Icon icon="github" style="width: 14px; height: 14px;" /></a></sup>

```python  theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
convert_result(self, raw_value: Any) -> ToolResult
```

Convert a raw result to ToolResult.

Handles ToolResult passthrough and converts raw values using the tool's
attributes (serializer, output\_schema) for proper conversion.

#### `register_with_docket` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/tools/tool.py#L356" target="_blank"><Icon icon="github" style="width: 14px; height: 14px;" /></a></sup>

```python  theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
register_with_docket(self, docket: Docket) -> None
```

Register this tool with docket for background execution.

#### `add_to_docket` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/tools/tool.py#L362" target="_blank"><Icon icon="github" style="width: 14px; height: 14px;" /></a></sup>

```python  theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
add_to_docket(self, docket: Docket, arguments: dict[str, Any], **kwargs: Any) -> Execution
```

Schedule this tool for background execution via docket.

**Args:**

* `docket`: The Docket instance
* `arguments`: Tool arguments
* `fn_key`: Function lookup key in Docket registry (defaults to self.key)
* `task_key`: Redis storage key for the result
* `**kwargs`: Additional kwargs passed to docket.add()

#### `from_tool` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/tools/tool.py#L386" target="_blank"><Icon icon="github" style="width: 14px; height: 14px;" /></a></sup>

```python  theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
from_tool(cls, tool: Tool | Callable[..., Any]) -> TransformedTool
```

#### `get_span_attributes` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/tools/tool.py#L434" target="_blank"><Icon icon="github" style="width: 14px; height: 14px;" /></a></sup>

```python  theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
get_span_attributes(self) -> dict[str, Any]
```
