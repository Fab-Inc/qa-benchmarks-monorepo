"""Tests that the pipeline scripts in ``scripts/`` stay wired to the package.

The scripts sit outside the importable package and ``ruff``'s per-file ignores
allow unused imports there, so a symbol renamed or deleted in
``src/phonics_benchmark/`` can leave a script importing something that no longer
exists — an ImportError nobody sees until they run the pipeline. These tests
resolve every ``from phonics_benchmark...`` import statically, without executing
the scripts (``inspect_run.py`` does work at import time).
"""

import ast
import importlib
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
SCRIPT_PATHS = sorted(SCRIPTS_DIR.glob("*.py"))


def package_imports(source: str) -> list[tuple[str, str]]:
    """Return every (module, name) imported from phonics_benchmark in ``source``."""
    tree = ast.parse(source)
    return [
        (node.module, alias.name)
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and node.module
        and node.module.startswith("phonics_benchmark")
        for alias in node.names
    ]


def test_scripts_directory_is_not_empty() -> None:
    """Guards against the glob silently matching nothing."""
    assert SCRIPT_PATHS


@pytest.mark.parametrize("script_path", SCRIPT_PATHS, ids=lambda p: p.name)
def test_script_parses(script_path: Path) -> None:
    """Every script is syntactically valid Python."""
    ast.parse(script_path.read_text(encoding="utf-8"))


@pytest.mark.parametrize("script_path", SCRIPT_PATHS, ids=lambda p: p.name)
def test_script_imports_resolve(script_path: Path) -> None:
    """Every name a script imports from the package actually exists."""
    imports = package_imports(script_path.read_text(encoding="utf-8"))
    for module_name, symbol in imports:
        module = importlib.import_module(module_name)
        assert hasattr(module, symbol), (
            f"{script_path.name} imports '{symbol}' from '{module_name}', "
            "which no longer defines it"
        )
