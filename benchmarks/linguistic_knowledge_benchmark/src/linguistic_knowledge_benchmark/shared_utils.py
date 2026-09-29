"""Shared constants and utilities for linguistic_knowledge_benchmark."""

from pathlib import Path

import yaml

BENCH_DIR = Path(__file__).parent.parent.parent
REPO_ROOT = BENCH_DIR.parent.parent
DATA_DIR = BENCH_DIR / "data"
CACHE_BASE_DIR = BENCH_DIR / "cache"

MODELS_LIST_YAML = BENCH_DIR / "list_models.yaml"
with open(MODELS_LIST_YAML, encoding="utf-8") as _f:
    EVAL_MODELS: dict[str, str] = yaml.safe_load(_f)

RATE_LIMITED_MODELS = {
    "gemma-3n-e4b-it",
    "gemma-3n-e2b-it",
    "gemma-3-1b-it",
}


def run_dir(language_name: str, run_id: str | None = None) -> Path:
    """Return the run directory for a language/run.

    Always resolves under runs/; omitting run_id uses the 'default' run.
    """
    return CACHE_BASE_DIR / language_name.lower() / "runs" / (run_id or "default")


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
