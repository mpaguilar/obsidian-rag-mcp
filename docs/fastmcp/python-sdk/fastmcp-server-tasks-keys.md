> ## Documentation Index
> Fetch the complete documentation index at: https://gofastmcp.com/llms.txt
> Use this file to discover all available pages before exploring further.

# keys

# `fastmcp.server.tasks.keys`

Task key management for SEP-1686 background tasks.

Task keys encode security scoping and metadata in the Docket key format:
`{session_id}:{client_task_id}:{task_type}:{component_identifier}`

This format provides:

* Session-based security scoping (prevents cross-session access)
* Task type identification (tool/prompt/resource)
* Component identification (name or URI for result conversion)

## Functions

### `build_task_key` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/tasks/keys.py#L15" target="_blank"><Icon icon="github" style="width: 14px; height: 14px;" /></a></sup>

```python  theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
build_task_key(session_id: str, client_task_id: str, task_type: str, component_identifier: str) -> str
```

Build Docket task key with embedded metadata.

Format: `{session_id}:{client_task_id}:{task_type}:{component_identifier}`

The component\_identifier is URI-encoded to handle special characters (colons, slashes, etc.).

**Args:**

* `session_id`: Session ID for security scoping
* `client_task_id`: Client-provided task ID
* `task_type`: Type of task ("tool", "prompt", "resource")
* `component_identifier`: Tool name, prompt name, or resource URI

**Returns:**

* Encoded task key for Docket

**Examples:**

> > > build\_task\_key("session123", "task456", "tool", "my\_tool")
> > > 'session123:task456:tool:my\_tool'
> > > build\_task\_key("session123", "task456", "resource", "file://data.txt")
> > > 'session123:task456:resource:file%3A%2F%2Fdata.txt'

### `parse_task_key` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/tasks/keys.py#L47" target="_blank"><Icon icon="github" style="width: 14px; height: 14px;" /></a></sup>

```python  theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
parse_task_key(task_key: str) -> dict[str, str]
```

Parse Docket task key to extract metadata.

**Args:**

* `task_key`: Encoded task key from Docket

**Returns:**

* Dict with keys: session\_id, client\_task\_id, task\_type, component\_identifier

**Examples:**

> > > parse\_task\_key("session123:task456:tool:my\_tool")
> > > `{'session_id': 'session123', 'client_task_id': 'task456', 'task_type': 'tool', 'component_identifier': 'my_tool'}`
> > > parse\_task\_key("session123:task456:resource:file%3A%2F%2Fdata.txt")
> > > `{'session_id': 'session123', 'client_task_id': 'task456', 'task_type': 'resource', 'component_identifier': 'file://data.txt'}`

### `get_client_task_id_from_key` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/tasks/keys.py#L78" target="_blank"><Icon icon="github" style="width: 14px; height: 14px;" /></a></sup>

```python  theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
get_client_task_id_from_key(task_key: str) -> str
```

Extract just the client task ID from a task key.

**Args:**

* `task_key`: Full encoded task key

**Returns:**

* Client-provided task ID (second segment)
