"""Environment variable interpolation utilities.

Provides homomorphic environment variable interpolation for configuration
values supporting ${VAR} and ${VAR:-default} syntax.
"""

import logging
import os
import re
from typing import Any, TypeVar

log = logging.getLogger(__name__)

# TypeVar for homomorphic _interpolate_env_vars function
T = TypeVar("T", str, int, float, None, dict, list)


def _replace_env_var(match: re.Match) -> str:
    """Replace a single environment variable match.

    Supports ${VAR} and ${VAR:-default} syntax. When a variable
    is not set and no default is provided, returns an empty string
    and logs a warning.

    Args:
        match: Regex match object containing the variable expression.

    Returns:
        The environment variable value, the default value, or an
        empty string if the variable is not set and no default is
        provided.

    """
    _msg = "_replace_env_var starting"
    log.debug(_msg)
    var_expr = match.group(1)
    if ":-" in var_expr:
        var_name, default = var_expr.split(":-", 1)
        result = os.environ.get(var_name, default)
        _msg = "_replace_env_var returning"
        log.debug(_msg)
        return str(result)
    result = os.environ.get(var_expr)
    if result is None:
        _msg = (
            f"Environment variable '{var_expr}' is not set, "
            f"no default provided; returning empty string"
        )
        log.warning(_msg)
        _msg = "_replace_env_var returning"
        log.debug(_msg)
        return ""
    _msg = "_replace_env_var returning"
    log.debug(_msg)
    return str(result)


def _interpolate_env_vars(
    value: T,
) -> T:
    """Interpolate environment variables in configuration values.

    Supports ${VAR} and ${VAR:-default} syntax. When a variable
    referenced by ${VAR} is not set and no default is provided,
    it is replaced with an empty string and a warning is logged.

    This is a homomorphic function that preserves the input type:
    - str input -> str output (with env var interpolation)
    - dict input -> dict output (recursively interpolate values)
    - list input -> list output (recursively interpolate items)
    - int/float/None input -> same output (unchanged)

    Args:
        value: The value to interpolate (string, dict, list, or primitive).

    Returns:
        The interpolated value with the same type as input.

    """
    _msg = "_interpolate_env_vars starting"
    log.debug(_msg)
    if isinstance(value, str):
        pattern = r"\$\{([^}]+)\}"
        str_result: str = re.sub(pattern, _replace_env_var, value)
        _msg = "_interpolate_env_vars returning"
        log.debug(_msg)
        return str_result  # type: ignore[return-value]
    if isinstance(value, dict):
        dict_result: dict[str, Any] = {
            k: _interpolate_env_vars(v) for k, v in value.items()
        }
        _msg = "_interpolate_env_vars returning"
        log.debug(_msg)
        return dict_result  # type: ignore[return-value]
    if isinstance(value, list):
        list_result: list[Any] = [_interpolate_env_vars(item) for item in value]
        _msg = "_interpolate_env_vars returning"
        log.debug(_msg)
        return list_result  # type: ignore[return-value]
    _msg = "_interpolate_env_vars returning"
    log.debug(_msg)
    return value
