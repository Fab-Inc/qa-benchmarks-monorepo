from collections.abc import Mapping
from functools import cache
from importlib.resources import files
from typing import Any

import yaml
from decodability import DECODABILITY_REGISTRY
from pydantic import ValidationError

from phonics_benchmark.config import ConfigError


@cache
def _load_prompts() -> dict[str, str]:
    """Load and cache prompt templates from the packaged prompts.yaml."""
    text = (
        files("phonics_benchmark").joinpath("prompts.yaml").read_text(encoding="utf-8")
    )
    return yaml.safe_load(text)


def get_prompt(name: str, **kwargs: object) -> str:
    """Return the named prompt template, formatted with the given fields."""
    prompts = _load_prompts()
    try:
        template = prompts[name]
    except KeyError:
        raise KeyError(
            f"Unknown prompt '{name}'; available: {sorted(prompts)}"
        ) from None
    return template.format(**kwargs)


def build_decodability_prompt(
    item: Mapping[str, Any],
    language: str,
) -> str:
    """Validate language-specific knowledge and render one benchmark prompt."""

    try:
        language_integration = DECODABILITY_REGISTRY[language]
    except KeyError:
        raise ConfigError(
            f"Unknown language registry key '{language}'; available: "
            f"{sorted(DECODABILITY_REGISTRY)}"
        ) from None

    try:
        student_knowledge = language_integration[
            "student_knowledge_model"
        ].model_validate(item["student_knowledge"])
    except ValidationError as e:
        raise ConfigError(
            "Invalid student_knowledge for benchmark item "
            f"'{item['benchmark_item_id']}': {e}"
        ) from e

    return get_prompt(
        "decodability_evaluation",
        language=language_integration["name"],
        text=item["content"],
        student_knowledge=student_knowledge,
    )
