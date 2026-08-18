from collections.abc import Callable, Mapping
from functools import cache
from importlib.resources import files
from typing import Any

import yaml
from decodability import DECODABILITY_REGISTRY
from pydantic import ValidationError

from phonics_benchmark.config import ConfigError
from phonics_benchmark.labeling import get_grapheme_breakdown, get_valid_knowledge_str


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


def _baseline_fields(item: Mapping[str, Any], language: str) -> dict[str, Any]:
    return {"text": item["content"]}


def _grapheme_breakdown_fields(
    item: Mapping[str, Any], language: str
) -> dict[str, Any]:
    return {
        "grapheme_breakdown": get_grapheme_breakdown(item["content"], language),
    }


def _valid_graphemes_fields(item: Mapping[str, Any], language: str) -> dict[str, Any]:
    return {
        "text": item["content"],
        "valid_knowledge": get_valid_knowledge_str(language),
    }


PromptFieldBuilder = Callable[[Mapping[str, Any], str], dict[str, Any]]

PROMPT_FIELD_BUILDERS: dict[str, PromptFieldBuilder] = {
    "decodability_evaluation": _baseline_fields,
    "decodability_evaluation_grapheme_breakdown": _grapheme_breakdown_fields,
    "decodability_evaluation_valid_graphemes": _valid_graphemes_fields,
    "decodability_evaluation_clusters": _baseline_fields,
}


def build_decodability_prompt(
    prompt_name: str,
    item: Mapping[str, Any],
    language: str,
) -> str:
    """Validate language-specific knowledge and render one benchmark prompt."""

    try:
        build_fields = PROMPT_FIELD_BUILDERS[prompt_name]
    except KeyError:
        raise ConfigError(
            f"Unknown prompt '{prompt_name}'; available: "
            f"{sorted(PROMPT_FIELD_BUILDERS)}"
        ) from None

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
        prompt_name,
        language=language_integration["name"],
        student_knowledge=student_knowledge,
        **build_fields(item, language),
    )
