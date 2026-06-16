"""Shared constants and utilities for linguistic_knowledge_benchmark."""

from pathlib import Path

import yaml

BENCH_DIR = Path(__file__).parent.parent.parent
REPO_ROOT = BENCH_DIR.parent.parent
DATA_DIR = BENCH_DIR / "data"

MODELS_LIST_YAML = BENCH_DIR / "list_models.yaml"
with open(MODELS_LIST_YAML) as _f:
    EVAL_MODELS: dict[str, str] = yaml.safe_load(_f)

RATE_LIMITED_MODELS = {
    "gemma-3n-e4b-it",
    "gemma-3n-e2b-it",
    "gemma-3-1b-it",
}
