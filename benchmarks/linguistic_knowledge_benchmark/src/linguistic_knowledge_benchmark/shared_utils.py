"""Shared constants and utilities for linguistic_knowledge_benchmark."""

from pathlib import Path

import yaml

ROOT_DIR = Path(__file__).parent.parent.parent
DATA_DIR = ROOT_DIR / "data"

MODELS_LIST_YAML = ROOT_DIR / "list_models.yaml"
with open(MODELS_LIST_YAML) as _f:
    EVAL_MODELS: dict[str, str] = yaml.safe_load(_f)

RATE_LIMITED_MODELS = {
    "gemma-3n-e4b-it",
    "gemma-3n-e2b-it",
    "gemma-3-1b-it",
}
