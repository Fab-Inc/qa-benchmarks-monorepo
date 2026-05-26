"""Test cases for the command-line interface."""

import pytest
import typer
from language_knowledge_benchmark.cmd import app, main
from typer.testing import CliRunner


@pytest.fixture
def runner() -> CliRunner:
    """Fixture for invoking command-line interfaces."""
    return CliRunner()


def test_main_succeeds(runner: CliRunner) -> None:
    """It exits with a status code of zero."""
    result = runner.invoke(app)
    assert result.exit_code == 0

    app_custom = typer.Typer()
    app_custom.command()(main)
    result = runner.invoke(app_custom)
    assert result.exit_code == 0

