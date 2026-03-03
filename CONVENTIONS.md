# General
* Imports **always** should be at the top of the file, **never** inline
* The root folder is for project configuration files, only.
* Functions **must** be short, and serve a single purpose. Avoid long functions, create new functions as needed.
* Functions **must** be easily mocked, avoid complexity.
* All text output for questions and answers **must** be in Markdown format
* All assigned variables should be used. 
    * If a variable is unused, remove the variable.

* Do NOT empty files, delete them.

# Git Practices

* **Respect `.gitignore`** - Never force add files that are excluded by `.gitignore`
* Do NOT use `git add -f` or `--force` to add ignored files
* If a file is in `.gitignore`, it should stay out of the repository
* This applies to directories like `_agent/`, `.venv/`, `__pycache__/`, etc.

# Configuration Management

* **DO NOT create configuration files in the worktree.** A properly configured config file already exists in the home directory.
* If changes to configuration are needed, ask first before creating or modifying any config files.
* This ensures consistent configuration across all work sessions and prevents configuration drift.

# Preferred libraries

* `click` for command line parsing.
* `pathlib` for file operations
* `pytest` for unit testing
* `httpx` for web-related calls, unless a specific library is offered
* `litellm` for LLM connectivity
* `langchain` for LLM functions not covered by `litellm`

 # Variable conventions

* **always** use type-hints for all arguments and return values
* Correct type-hints are **critical**
* **`Any` type restrictions**: Only use `Any` for arguments and parameters that can truly accept any type, or when required by a third-party library interface. **Never** use `Any` as a shortcut to avoid proper typing—use specific types, `Union`, `Optional`, or `object` instead.
* Use named arguments when calling functions when possible.

# General formatting
* use double-quotes for strings
* all functions should have a docstring describing:
    * what it does
    * what arguments it takes ("Args:\n")
    * what it returns ("Returns:\n)
* Multi-line docstrings should start at the first line, with no line break.
    * For example: `"""The line should start like this`
* Blank lines **must** be blank, with no unnecessary spaces or tabs.

# Defensive coding
* call functions using named arguments
* Code should be written to be easily mocked and tested
* No secret keys should be in code, not even defaults. **Always** get secrets from an external source.

# Exceptions
* `try` blocks should not `return` from within the `try`
* `try` blocks should use `else` to `return`
* Exceptions should have an error log before being raised.
* General exceptions on FastAPI routes will be handled by FastAPI.

# Logging
* Every source file must have logging setup using the following in it's header:
    ```
    import logging

    log = logging.getLogger(__name__)
    ```
* functions should `log.debug` at the start of the function with a message including the function name and "starting".
* functions should `log.debug` before returning with a message including the function name and "returning".
* Logging should never use f-string or `%s` formatting. Format the message into its own variable, and pass the variable to the log statment.

    For example, this is correct:
    ```
    _msg = f"{component_name} completed processing"
    log.info(_msg)
    ```

    ```
    These are incorrect:
    ```
    log.info(f"{component_name} completed processing")
    log.info("%s completed processing", component_name)
    ```
* When logging from an exception, use `log.exception`

# Deprecated usage
* `typing.List` is deprecated, use `list` instead
* `datetime.datetime.utcnow()` is deprecated and scheduled for removal in a future version. Use timezone-aware objects to represent datetimes in UTC: `datetime.datetime.now(datetime.UTC)`
* IMPORTANT: DeprecationWarning: The `name` is not the first parameter anymore. The first parameter should be the `Request` instance.
  Replace `TemplateResponse(name, {"request": request})` by `TemplateResponse(request, name)`.
* PydanticDeprecatedSince20: Support for class-based `config` is deprecated, use `ConfigDict` instead. Deprecated in Pydantic V2.0 to be removed in V3.0.

# Docstrings
* Every function should have a docstring.
* The purpose of the docstring is to act as a specification for the function
* You will be asked to update functions using the docstring as a specification
* Classes should be documented.
    - Class purpose, if known
    - Members should be listed, with their type if known.
* The docstring should include
    - an `Args:` section, which includes the name, type, and purpose of the function argument.
    - a `Returns:` section, which includes the type and purpose of all possible return values
    - an optional "Raises:" section, which includes any potential exceptions raised
    - an optional `Notes:` section, which documents operations that may affect testing:
        - Network access (external API calls, HTTP requests)
        - Database calls (SQL queries, ORM operations)
        - Disk access (file reads/writes)
        - Sleep calls or delays that may affect timing
        - Any other operations that could impact test reliability or performance
    - **Important:** If a function has no testing-relevant operations, the Notes section can be omitted entirely.
    - Important logic descriptions should be extracted into inline code comments within the function body, not in the Notes section.
    - Include a blank line after the last section. For example:
    ```
    Notes:
        <some notes>

    ```

# HTML considerations
* Prefer putting HTML into templates
* Putting HTML into Python code is discouraged

# Context management

It is **important** to keep the size of individual files manageable.

* Individual files **must not exceed** 1000 lines
* When files approach or exceed this limit, developers must:
  - Create new files to distribute functionality
  - Extract related functions into separate modules
  - Maintain logical organization while splitting
* Create new files and libraries as necessary

# FastAPI dependency injection

When using `Depends` dependency injection, it should use `Annotated`.

When not using `Annotated`, `ruff` will issue a `FAST002` error.

This is an example of correct usage:

```
from typing import Annotated

from fastapi import Depends, FastAPI

app = FastAPI()

async def common_parameters(q: str | None = None, skip: int = 0, limit: int = 100):
    return {"q": q, "skip": skip, "limit": limit}

@app.get("/items/")
async def read_items(commons: Annotated[dict, Depends(common_parameters)]):
    return commons
```

# FastAPI Response Types

Due to some versioning conflicts, using `TemplateResponse` as a return type is currently broken.

Set `response_class=HTMLResponse` and `response_model=None` when returning templates:
```python
@router.get("/dashboard", response_class=HTMLResponse, response_model=None)
async def dashboard(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(request, "dashboard.html", {})
```

# Unit tests
* Unit tests are run with `pytest`.
* Tests should be written as functions, do **not** use test classes.
* Each `*.py` file should have its own test file.
* Tests **must** be placed in a top-level `./tests` directory
    * NEVER create tests inside the source directory (e.g., no `./myapp/tests/`)
    * The `./tests` directory should mirror the structure of the source code
* Unit tests should be run with a logging level of DEBUG
* Unit tests should be written before the code, and they should fail if the code is incorrect.
* **100% test coverage is required, including branch coverage**
    * Run with: `pytest --cov=myapp --cov-branch --cov-report=term-missing`
    * All code paths and branches must be tested
* IMPORTANT: Do not use duplicate file names for tests, even in separate paths. This causes errors.
    * All test files **must** have unique filenames

## Special considerations for FastAPI route dependency injection

To successfully mock FastAPI route calls with dependency injection, `app.dependency_overrides` **must** be used.

For example:
```python
from myapp.app.api.routes.my_module import get_db

@patch("myapp.app.api.routes.my_module.get_current_user")
@patch("myapp.app.api.routes.my_module.get_db")
def test_something():
    app = create_app()
    client = TestClient(app)

    mock_db = Mock()

    # mock the return value
    mock_db.query.return_value.filter.return_value.first.return_value = mock_result

    # These next lines are **very** important
    def get_mock_db():
        yield mock_db

    # note that we are overriding `get_db` here
    app.dependency_overrides[get_db] = get_mock_db

    # perform tests and assertions

    # clear the overrides
    app.dependency_overrides.clear()
```

## Test file name exception

There are cases where test files cannot be named exactly like the source file due to naming collisions. For example:
- `user.py` → `test_user_route.py` (not `test_user.py`)

Keep a mappings file to track these exceptions. ALWAYS update it as new mappings are discovered or created.

# Ruff Linting Rules

The following ruff rules are enforced via configuration in `pyproject.toml`. All rules are checked during quality checks.

## Enabled Rules

* **A002** - Argument shadows a Python built-in (e.g., naming a parameter `id` or `list`)
* **ANN001** - Missing type annotation for function arguments
* **ANN201** - Missing return type annotation on public functions
* **ANN401** - Disallow use of `typing.Any` type
* **ARG001** - Unused function argument
* **ARG002** - Unused method argument (self/cls excluded)
* **F841** - Unused local variable
* **RUF006** - Store a reference to the return value of `asyncio.create_task`
* **TC002** - Move third-party import into a type-checking block
* **FAST** - FastAPI-specific linting errors (e.g., FAST002 for non-Annotated Depends)
* **D417** - Missing argument description in the docstring
* **C** - McCabe complexity check (cyclomatic complexity)
* **PLR0913** - Too many arguments in function definition

## McCabe Complexity

Maximum cyclomatic complexity is set to 5. Functions with complexity above this threshold will be flagged for refactoring.

## Running Quality Checks

Run ruff directly to check code quality:

```bash
# Show all violations
uv run ruff check

# Automatically fix violations where possible
uv run ruff check --fix

# Check specific file or directory
uv run ruff check myapp/app/models.py
```

Always run quality checks before committing to ensure code standards are met.

## noqa Policy

`noqa` comments are **not** a valid workaround for ruff linting violations. All code must pass ruff checks without using noqa suppressions. If a rule violation occurs:

1. Fix the underlying issue
2. Refactor the code to comply with the rule
3. Do not suppress the rule with noqa

# Third-Party Library Documentation

When using third-party libraries with substantial documentation, download the documentation to a local directory for offline reference and version consistency.

## FastMCP Documentation

FastMCP v3.0.0 documentation is downloaded from https://gofastmcp.com/ and stored in:
- Location: `./docs/fastmcp/`
- Version: 3.0.0
- Key files:
  - `quickstart.md` - Quick start guide
  - `changelog.md` - Version changelog
  - `updates.md` - Documentation updates
  - `getting-started/` - Getting started guides
  - `servers/` - Server implementation guides
  - `clients/` - Client implementation guides
  - `deployment/` - Deployment guides
  - `integrations/` - Integration examples
  - `patterns/` - Design patterns
  - `python-sdk/` - Python SDK reference
  - `cli/` - CLI documentation
  - `apps/` - Application examples
  - `development/` - Development guides

This ensures consistent reference documentation even if the upstream site changes.

# Output Formatting Standards

## Task Output

**NEVER use emojis** in task output or status indicators. This applies to:
- CLI output
- MCP tool responses
- API responses
- Log messages

Use plain text representations instead:
- Status: "completed", "not_completed", "in_progress", "cancelled"
- Priority: "highest", "high", "normal", "low", "lowest"
- Indicators: Text labels or ASCII characters only

Rationale: Emojis cause rendering issues in various terminals, logging systems, and client applications. They also create accessibility concerns and inconsistent display across platforms.