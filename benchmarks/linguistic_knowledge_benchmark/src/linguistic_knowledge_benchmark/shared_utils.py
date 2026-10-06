"""Shared constants and utilities for linguistic_knowledge_benchmark."""

import os
from pathlib import Path

import yaml

BENCH_DIR = Path(__file__).parent.parent.parent
REPO_ROOT = BENCH_DIR.parent.parent
DATA_DIR = BENCH_DIR / "data"
CACHE_BASE_DIR = BENCH_DIR / "cache"

# Config synced from the fab-open-configs bucket on every CLI run (see cmd.py's
# _download_shared_configs) and gitignored, so these live at the repo root rather
# than in DATA_DIR -- none is committed, and the bucket is the single source of truth.
# Defined here rather than in cmd.py because the analysis scripts need the CSV paths:
# importing cmd.py would pull in typer/dotenv and its callback's network side effects,
# which scripts/validate_results.py deliberately avoids.
CUSTOM_MODELS_YAML = REPO_ROOT / "custom_models.yaml"
MODELS_CSV = REPO_ROOT / "models.csv"
PROVIDERS_CSV = REPO_ROOT / "providers.csv"


def require_synced_config(path: str | Path) -> None:
    """Abort with a clear message when one of the synced config files is missing.

    Those files are pulled from the fab-open-configs bucket rather than committed, so a
    fresh checkout has no copy until a CLI command has run. The analysis scripts used to
    treat a missing models.csv/providers.csv as an empty mapping, which renders a
    complete but WRONG figure -- default-coloured bars, an empty provider legend, every
    model untiered. Failing here keeps that from reaching a report unnoticed.
    """
    if not os.path.exists(path):
        raise SystemExit(
            f"Required config file not found: {path}\n"
            "It is synced from the fab-open-configs bucket rather than committed -- "
            "run any SUBCOMMAND to fetch it, e.g.\n"
            "    uv run linguistic-knowledge-benchmark run-bench --help\n"
            "(a bare '--help' will not: typer skips the app callback that syncs it)."
        )


MODELS_LIST_YAML = BENCH_DIR / "list_models.yaml"
with open(MODELS_LIST_YAML, encoding="utf-8") as _f:
    EVAL_MODELS: dict[str, str] = yaml.safe_load(_f)

RATE_LIMITED_MODELS = {
    "gemma-3n-e4b-it",
    "gemma-3n-e2b-it",
    "gemma-3-1b-it",
}


def experiment_dir(language_name: str, experiment_id: str | None = None) -> Path:
    """Return the experiment directory for a language/experiment.

    Always resolves under experiments/; omitting experiment_id uses the 'default'
    experiment.
    """
    return (
        CACHE_BASE_DIR
        / language_name.lower()
        / "experiments"
        / (experiment_id or "default")
    )


# Some registered language_name display values are the language's own endonym
# (e.g. "Kiswahili") rather than the name English text conventionally uses
# (e.g. "Swahili"). Since every prompt sent to a model is English-medium, map
# to the English form here so prompt text stays consistent regardless of
# which form the CLI/display name uses.
ENGLISH_LANGUAGE_NAMES = {
    "Kiswahili": "Swahili",
}


def english_language_name(language_name: str) -> str:
    """Return the English-prose form of a language name for use in prompts."""
    return ENGLISH_LANGUAGE_NAMES.get(language_name, language_name)
