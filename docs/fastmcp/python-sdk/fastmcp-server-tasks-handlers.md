> ## Documentation Index
> Fetch the complete documentation index at: https://gofastmcp.com/llms.txt
> Use this file to discover all available pages before exploring further.

# handlers

# `fastmcp.server.tasks.handlers`

SEP-1686 task execution handlers.

Handles queuing tool/prompt/resource executions to Docket as background tasks.

## Functions

### `submit_to_docket` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/tasks/handlers.py#L34" target="_blank"><Icon icon="github" style="width: 14px; height: 14px;" /></a></sup>

```python  theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
submit_to_docket(task_type: Literal['tool', 'resource', 'template', 'prompt'], key: str, component: Tool | Resource | ResourceTemplate | Prompt, arguments: dict[str, Any] | None = None, task_meta: TaskMeta | None = None) -> mcp.types.CreateTaskResult
```

Submit any component to Docket for background execution (SEP-1686).

Unified handler for all component types. Called by component's internal
methods (\_run, \_read, \_render) when task metadata is present and mode allows.

Queues the component's method to Docket, stores raw return values,
and converts to MCP types on retrieval.

**Args:**

* `task_type`: Component type for task key construction
* `key`: The component key as seen by MCP layer (with namespace prefix)
* `component`: The component instance (Tool, Resource, ResourceTemplate, Prompt)
* `arguments`: Arguments/params (None for Resource which has no args)
* `task_meta`: Task execution metadata. If task\_meta.ttl is provided, it
  overrides the server default (docket.execution\_ttl).

**Returns:**

* Task stub with proper Task object
