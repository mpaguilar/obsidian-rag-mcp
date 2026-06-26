"""Tests for CLI type guard replacements for assert statements."""

from unittest.mock import patch

from click.testing import CliRunner
import pytest

from obsidian_rag.cli import cli


NO_OP_VALIDATOR_PATH = "obsidian_rag.cli._validate_exact_query_params"


def _noop_validator(*_args: object, **_kwargs: object) -> None:
    """Do nothing, allowing an invalid state to reach the guard."""


def test_query_text_none_raises_runtime_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """B101 fix: None query_text past validation raises RuntimeError."""
    monkeypatch.setattr(NO_OP_VALIDATOR_PATH, _noop_validator)

    runner = CliRunner()
    result = runner.invoke(cli, ["query"])

    assert result.exit_code != 0
    assert isinstance(result.exception, RuntimeError)
    assert "query_text is None" in str(result.exception)


def test_query_text_none_logs_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """B101 fix: log.error is called before raising."""
    monkeypatch.setattr(NO_OP_VALIDATOR_PATH, _noop_validator)

    with patch("obsidian_rag.cli.log.error") as mock_log_error:
        runner = CliRunner()
        result = runner.invoke(cli, ["query"])

    assert result.exit_code != 0
    assert mock_log_error.called
    logged_message = mock_log_error.call_args[0][0]
    assert "query_text is None" in logged_message
