> ## Documentation Index
> Fetch the complete documentation index at: https://gofastmcp.com/llms.txt
> Use this file to discover all available pages before exploring further.

# exceptions

# `fastmcp.utilities.exceptions`

## Functions

### `iter_exc` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/utilities/exceptions.py#L12" target="_blank"><Icon icon="github" style="width: 14px; height: 14px;" /></a></sup>

```python  theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
iter_exc(group: BaseExceptionGroup)
```

### `get_catch_handlers` <sup><a href="https://github.com/PrefectHQ/fastmcp/blob/main/src/fastmcp/utilities/exceptions.py#L42" target="_blank"><Icon icon="github" style="width: 14px; height: 14px;" /></a></sup>

```python  theme={"theme":{"light":"snazzy-light","dark":"dark-plus"}}
get_catch_handlers() -> Mapping[type[BaseException] | Iterable[type[BaseException]], Callable[[BaseExceptionGroup[Any]], Any]]
```
