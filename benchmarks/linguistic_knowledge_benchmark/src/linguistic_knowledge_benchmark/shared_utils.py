"""Shared constants and utilities for linguistic_knowledge_benchmark."""

import json
import os
import sys
from pathlib import Path

import yaml

BENCH_DIR = Path(__file__).parent.parent.parent
REPO_ROOT = BENCH_DIR.parent.parent
DATA_DIR = BENCH_DIR / "data"
CACHE_BASE_DIR = BENCH_DIR / "cache"

# Add a new language by adding an entry here and dropping its JSON in data/.
# The medium of evaluation is English for every language: questions are written
# in English (only their subject changes from language to language).
#
# This is the NORMAL way commands pick a dataset: each language points directly
# at one fixed file here, no other configuration needed for everyday use.
# `--dataset` on the commands is an EXCEPTIONAL escape hatch that overrides this
# for a single experiment -- e.g. scoring against the repeated dataset built by
# scripts/build_consistency_dataset.py for the "consistency" validation check.
DATASETS: dict[str, dict] = {
    "kiswahili": {
        "path": DATA_DIR / "linguistic_knowledge_eval_kiswahili.json",
        "language_name": "Kiswahili",
    },
    "english": {
        "path": DATA_DIR / "linguistic_knowledge_eval_english.json",
        "language_name": "English",
    },
}

CALL_STATUS_SUCCESS = "success"
CALL_STATUS_ERROR = "error"

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
            "    uv run linguistic-knowledge-benchmark judge-bench --help\n"
            "(a bare '--help' will not: typer skips the app callback that syncs it)."
        )


MODELS_LIST_YAML = BENCH_DIR / "list_models.yaml"
with open(MODELS_LIST_YAML, encoding="utf-8") as _f:
    EVAL_MODELS: dict[str, str] = yaml.safe_load(_f)


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


def load_dataset(dataset_path: Path) -> list[dict]:
    """Load a Linguistic Knowledge evaluation dataset from a JSON file.

    Args:
        dataset_path: Path to the dataset JSON file.

    Returns:
        List of question dicts with keys: question_id, domain, question,
        answer_type, expected_answers, options (MCQ only), etc.
    """
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")
    with open(dataset_path, encoding="utf-8") as f:
        return json.load(f)


def git_hash() -> str | None:
    """Return the short git hash of HEAD, or None (with a warning) if unavailable."""
    import subprocess

    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(BENCH_DIR),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception as e:
        print(f"WARNING: could not read git hash: {e}", file=sys.stderr)
        return None


def update_log(
    experiment_directory: Path,
    model_name: str,
    fields: dict[str, str | list[str] | None],
) -> None:
    """Merge-write model provenance fields into log.json; skip None values."""
    log_path = experiment_directory / "log.json"
    log_data: dict = {}
    if log_path.exists():
        with open(log_path, encoding="utf-8") as _f:
            log_data = json.load(_f)
    entry = log_data.setdefault(model_name, {})
    entry.update({k: v for k, v in fields.items() if v is not None})
    with open(log_path, "w", encoding="utf-8") as _f:
        json.dump(log_data, _f, indent=2)


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
