"""Load prompt/scenario templates and render one benchmark prompt per item."""

from collections.abc import Mapping
from functools import cache
from importlib.resources import files
from typing import Any

import yaml
from decodability import DECODABILITY_REGISTRY
from pydantic import ValidationError

from phonics_benchmark.config import ConfigError


def _load_packaged_yaml(filename: str) -> dict[str, str]:
    text = files("phonics_benchmark").joinpath(filename).read_text(encoding="utf-8")
    return yaml.safe_load(text)


@cache
def _load_prompts() -> dict[str, str]:
    """Load and cache prompt templates from the packaged prompts.yaml."""
    return _load_packaged_yaml("prompts.yaml")


@cache
def _load_scenarios() -> dict[str, str]:
    """Load and cache generation scenarios from the packaged scenarios.yaml."""
    return _load_packaged_yaml("scenarios.yaml")


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


def get_scenarios() -> dict[str, str]:
    """Return every generation scenario, keyed by scenario_id.

    A copy is returned so callers cannot mutate the cached mapping.
    """
    return dict(_load_scenarios())


def get_scenario(scenario_id: str) -> str:
    """Return the TASK instruction for one scenario_id."""
    scenarios = _load_scenarios()
    try:
        return scenarios[scenario_id]
    except KeyError:
        raise ConfigError(
            f"Unknown scenario '{scenario_id}'; available: {sorted(scenarios)}"
        ) from None


def build_generation_prompt(
    prompt_name: str,
    item: Mapping[str, Any],
    language: str,
) -> str:
    """Validate language-specific knowledge and render one benchmark prompt.

    The item supplies the already-resolved ``scenario`` instruction text; the
    language registry supplies the display name and the ``student_knowledge``
    model whose ``__str__`` becomes the prompt's STUDENT KNOWLEDGE block.

    Raises ``ConfigError`` for an unknown prompt, an unknown language, or
    student knowledge that fails the language model's validation. The runner
    renders every prompt up front, so these failures surface before any model
    call is made.
    """

    prompts = _load_prompts()
    if prompt_name not in prompts:
        raise ConfigError(
            f"Unknown prompt '{prompt_name}'; available: {sorted(prompts)}"
        )

    try:
        language_integration = DECODABILITY_REGISTRY[language]
    except KeyError:
        raise ConfigError(
            f"Unknown language registry key '{language}'; available: "
            f"{sorted(DECODABILITY_REGISTRY)}"
        ) from None

    scenario = item.get("scenario")
    if not isinstance(scenario, str) or not scenario.strip():
        raise ConfigError(
            "Missing scenario for benchmark item "
            f"'{item.get('benchmark_item_id')}': scenario must be a non-empty string"
        )

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
        scenario=scenario,
    )
