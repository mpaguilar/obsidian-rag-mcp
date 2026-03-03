> ## Documentation Index
> Fetch the complete documentation index at: https://gofastmcp.com/llms.txt
> Use this file to discover all available pages before exploring further.

# catalog

# `fastmcp.server.transforms.catalog`

Base class for transforms that need to read the real component catalog.

Some transforms replace `list_tools()` output with synthetic components
(e.g. a search interface) while still needing access to the *real*
(auth-filtered) catalog at call time.  `CatalogTransform` provides the
bypass machinery so subclasses can call `get_tool_catalog()` without
triggering their own replacement logic.

## Re-entrancy problem

When a synthetic tool handler calls `get_tool_catalog()`, that calls
`ctx.fastmcp.list_tools()` which re-enters the transform pipeline —
including *this* transform's `list_tools()`.  If the subclass overrides
`list_tools()` directly, the re-entrant call would hit the subclass's
replacement logic again (returning synthetic tools instead of the real
catalog).  A `super()` call can't prevent this because Python can't
short-circuit a method after `super()` returns.

Solution: `CatalogTransform` owns `list_tools()` and uses a
per-instance `ContextVar` to detect re-entrant calls.  During bypass,
it passes through to the base `Transform.list_tools()` (a no-op).
Otherwise, it delegates to `transform_tools()` — the subclass hook
where replacement logic lives.  Same pattern for resources, prompts,
and resource templates.

This is *not* the same as the `Provider._list_tools()` convention
(which produces raw components with no arguments).  `transform_tools()`
receives the current catalog and returns a transformed version.  The
distinct name avoids confusion between the two patterns.

Usage::

class MyTransform(CatalogTransform):
async def transform\_tools(self, tools):
return \[self.\_make\_search\_tool()]

def \_make\_search\_tool(self):
async def search(ctx: Context = None):
real\_tools = await self.get\_tool\_catalog(ctx)
...
return Tool.from\_function(fn=search, name="search")

## Classes

### `CatalogTransform` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/transforms/catalog.py#L64" target="_blank"><Icon icon="github" style="width: 14px; height: 14px;" /></a></sup>

Transform that needs access to the real component catalog.

Subclasses override `transform_tools()` / `transform_resources()`
/ `transform_prompts()` / `transform_resource_templates()`
instead of the `list_*()` methods.  The base class owns
`list_*()` and handles re-entrant bypass automatically — subclasses
never see re-entrant calls from `get_*_catalog()`.

The `get_*_catalog()` methods fetch the real (auth-filtered) catalog
by temporarily setting a bypass flag so that this transform's
`list_*()` passes through without calling the subclass hook.

**Methods:**

#### `list_tools` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/transforms/catalog.py#L88" target="_blank"><Icon icon="github" style="width: 14px; height: 14px;" /></a></sup>

```python  theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
list_tools(self, tools: Sequence[Tool]) -> Sequence[Tool]
```

#### `list_resources` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/transforms/catalog.py#L93" target="_blank"><Icon icon="github" style="width: 14px; height: 14px;" /></a></sup>

```python  theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
list_resources(self, resources: Sequence[Resource]) -> Sequence[Resource]
```

#### `list_resource_templates` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/transforms/catalog.py#L98" target="_blank"><Icon icon="github" style="width: 14px; height: 14px;" /></a></sup>

```python  theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
list_resource_templates(self, templates: Sequence[ResourceTemplate]) -> Sequence[ResourceTemplate]
```

#### `list_prompts` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/transforms/catalog.py#L105" target="_blank"><Icon icon="github" style="width: 14px; height: 14px;" /></a></sup>

```python  theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
list_prompts(self, prompts: Sequence[Prompt]) -> Sequence[Prompt]
```

#### `transform_tools` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/transforms/catalog.py#L114" target="_blank"><Icon icon="github" style="width: 14px; height: 14px;" /></a></sup>

```python  theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
transform_tools(self, tools: Sequence[Tool]) -> Sequence[Tool]
```

Transform the tool catalog.

Override this method to replace, filter, or augment the tool listing.
The default implementation passes through unchanged.

Do NOT override `list_tools()` directly — the base class uses it
to handle re-entrant bypass when `get_tool_catalog()` reads the
real catalog.

#### `transform_resources` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/transforms/catalog.py#L126" target="_blank"><Icon icon="github" style="width: 14px; height: 14px;" /></a></sup>

```python  theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
transform_resources(self, resources: Sequence[Resource]) -> Sequence[Resource]
```

Transform the resource catalog.

Override this method to replace, filter, or augment the resource listing.
The default implementation passes through unchanged.

Do NOT override `list_resources()` directly — the base class uses it
to handle re-entrant bypass when `get_resource_catalog()` reads the
real catalog.

#### `transform_resource_templates` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/transforms/catalog.py#L140" target="_blank"><Icon icon="github" style="width: 14px; height: 14px;" /></a></sup>

```python  theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
transform_resource_templates(self, templates: Sequence[ResourceTemplate]) -> Sequence[ResourceTemplate]
```

Transform the resource template catalog.

Override this method to replace, filter, or augment the template listing.
The default implementation passes through unchanged.

Do NOT override `list_resource_templates()` directly — the base class
uses it to handle re-entrant bypass when
`get_resource_template_catalog()` reads the real catalog.

#### `transform_prompts` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/transforms/catalog.py#L154" target="_blank"><Icon icon="github" style="width: 14px; height: 14px;" /></a></sup>

```python  theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
transform_prompts(self, prompts: Sequence[Prompt]) -> Sequence[Prompt]
```

Transform the prompt catalog.

Override this method to replace, filter, or augment the prompt listing.
The default implementation passes through unchanged.

Do NOT override `list_prompts()` directly — the base class uses it
to handle re-entrant bypass when `get_prompt_catalog()` reads the
real catalog.

#### `get_tool_catalog` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/transforms/catalog.py#L170" target="_blank"><Icon icon="github" style="width: 14px; height: 14px;" /></a></sup>

```python  theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
get_tool_catalog(self, ctx: Context) -> Sequence[Tool]
```

Fetch the real tool catalog, bypassing this transform.

**Args:**

* `ctx`: The current request context.
* `run_middleware`: Whether to run middleware on the inner call.
  Defaults to True because this is typically called from a
  tool handler where list\_tools middleware has not yet run.

#### `get_resource_catalog` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/transforms/catalog.py#L187" target="_blank"><Icon icon="github" style="width: 14px; height: 14px;" /></a></sup>

```python  theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
get_resource_catalog(self, ctx: Context) -> Sequence[Resource]
```

Fetch the real resource catalog, bypassing this transform.

**Args:**

* `ctx`: The current request context.
* `run_middleware`: Whether to run middleware on the inner call.
  Defaults to True because this is typically called from a
  tool handler where list\_resources middleware has not yet run.

#### `get_prompt_catalog` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/transforms/catalog.py#L204" target="_blank"><Icon icon="github" style="width: 14px; height: 14px;" /></a></sup>

```python  theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
get_prompt_catalog(self, ctx: Context) -> Sequence[Prompt]
```

Fetch the real prompt catalog, bypassing this transform.

**Args:**

* `ctx`: The current request context.
* `run_middleware`: Whether to run middleware on the inner call.
  Defaults to True because this is typically called from a
  tool handler where list\_prompts middleware has not yet run.

#### `get_resource_template_catalog` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/server/transforms/catalog.py#L221" target="_blank"><Icon icon="github" style="width: 14px; height: 14px;" /></a></sup>

```python  theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
get_resource_template_catalog(self, ctx: Context) -> Sequence[ResourceTemplate]
```

Fetch the real resource template catalog, bypassing this transform.

**Args:**

* `ctx`: The current request context.
* `run_middleware`: Whether to run middleware on the inner call.
  Defaults to True because this is typically called from a
  tool handler where list\_resource\_templates middleware has
  not yet run.
