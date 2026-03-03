> ## Documentation Index
> Fetch the complete documentation index at: https://gofastmcp.com/llms.txt
> Use this file to discover all available pages before exploring further.

# async_utils

# `fastmcp.utilities.async_utils`

Async utilities for FastMCP.

## Functions

### `call_sync_fn_in_threadpool` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/utilities/async_utils.py#L13" target="_blank"><Icon icon="github" style="width: 14px; height: 14px;" /></a></sup>

```python  theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
call_sync_fn_in_threadpool(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any
```

Call a sync function in a threadpool to avoid blocking the event loop.

Uses anyio.to\_thread.run\_sync which properly propagates contextvars,
making this safe for functions that depend on context (like dependency injection).

### `gather` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/utilities/async_utils.py#L38" target="_blank"><Icon icon="github" style="width: 14px; height: 14px;" /></a></sup>

```python  theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
gather(*awaitables: Awaitable[T]) -> list[T] | list[T | BaseException]
```

Run awaitables concurrently and return results in order.

Uses anyio TaskGroup for structured concurrency.

**Args:**

* `*awaitables`: Awaitables to run concurrently
* `return_exceptions`: If True, exceptions are returned in results.
  If False, first exception cancels all and raises.

**Returns:**

* List of results in the same order as input awaitables.
