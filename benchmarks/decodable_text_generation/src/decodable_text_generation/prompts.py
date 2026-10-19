"""Prompt rendering for the decodable text generation benchmarks."""

from collections.abc import Mapping
from functools import cache
from importlib.resources import files
from typing import Any

import yaml
from decodability import DECODABILITY_REGISTRY
from phonics_benchmark.config import ConfigError
from pydantic import ValidationError

PACKAGE = "decodable_text_generation"


@cache
def _load_yaml(filename: str) -> dict[str, Any]:
    """Load and cache a packaged YAML resource."""
    text = files(PACKAGE).joinpath(filename).read_text(encoding="utf-8")
    return yaml.safe_load(text)


def get_prompt(name: str, **kwargs: object) -> str:
    """Return the named generation prompt template, formatted with the fields."""
    prompts = _load_yaml("prompts.yaml")
    try:
        template = prompts[name]
    except KeyError:
        raise KeyError(
            f"Unknown prompt '{name}'; available: {sorted(prompts)}"
        ) from None
    try:
        return template.format(**kwargs)
    except KeyError as e:
        raise ConfigError(
            f"Prompt '{name}' needs field {e} that was not supplied"
        ) from e


def load_scenarios() -> dict[str, str]:
    """Return the scenario_id -> TASK instruction mapping from scenarios.yaml."""
    return dict(_load_yaml("scenarios.yaml"))


def get_scenario(scenario_id: str) -> str:
    """Return the TASK instruction for a scenario_id from scenarios.yaml."""
    scenarios = _load_yaml("scenarios.yaml")
    try:
        return scenarios[scenario_id]
    except KeyError:
        raise ConfigError(
            f"Unknown scenario_id '{scenario_id}'; available: {sorted(scenarios)}"
        ) from None


def get_decoding_rules(language: str) -> Mapping[str, str]:
    """Return the language's decoding_rules and word_list_name."""
    decoding_rules = _load_yaml("decoding_rules.yaml")
    try:
        return decoding_rules[language]
    except KeyError:
        raise ConfigError(
            f"No decoding rules for language '{language}'; available: "
            f"{sorted(decoding_rules)}"
        ) from None


_TARGET_FIELD_LABELS = {
    "gpcs": "Target GPCs",
    "sight_words": "Target sight words",
    "affixes": "Target affixes",
    "graphemes": "Target graphemes",
    "clusters": "Target clusters",
    "cluster_patterns": "Target cluster patterns",
    "whole_words": "Target whole words",
    "description": "Practice goal",
}


def render_target(target: Mapping[str, Any]) -> str:
    """Render a profile's target object as the TARGET block of a prompt.

    Pair-valued fields (e.g. ``gpcs``) render as ``grapheme -> /phoneme/`` —
    the same notation ``EnglishStudentKnowledge.__str__`` uses for the
    STUDENT KNOWLEDGE block; plain word/symbol lists render comma-separated;
    scalar fields such as ``description`` render verbatim. Empty fields are
    omitted entirely.
    """
    lines = []
    for field, values in target.items():
        if field == "rules":
            continue  # rule ids are evaluation internals, not prompt content
        label = _TARGET_FIELD_LABELS.get(field, f"Target {field}")
        if isinstance(values, str):
            if values.strip():
                lines.append(f"- {label}: {values.strip()}")
            continue
        items = list(values)
        if not items:
            continue
        if isinstance(items[0], (list, tuple)):
            rendered = "; ".join(f"{g} -> /{p}/" for g, p in items)
        else:
            rendered = ", ".join(str(v) for v in sorted(items))
        lines.append(f"- {label}: {rendered}")
    return "\n".join(lines) or "- Nothing new this week"


def build_generation_prompt(
    prompt_name: str,
    item: Mapping[str, Any],
    language: str,
) -> str:
    """Validate language-specific knowledge and render one generation prompt."""

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

    knowledge_text = str(student_knowledge)
    # affixes exist only on the English model; shared __str__ doesn't render
    # them, so append a line here for generation prompts.
    affixes = getattr(student_knowledge, "affixes", None)
    if affixes:
        knowledge_text += f"\n- Known affixes: {', '.join(sorted(affixes))}"

    fields: dict[str, Any] = {
        "decoding_rules": get_decoding_rules(language)["decoding_rules"],
        "word_list_name": get_decoding_rules(language)["word_list_name"],
        "language": language_integration["name"],
        "student_knowledge": knowledge_text,
        "scenario": get_scenario(item["scenario_id"]),
    }
    target = item.get("target_knowledge")
    if target:
        fields["target"] = render_target(target)

    try:
        return get_prompt(prompt_name, **fields)
    except KeyError as e:
        raise ConfigError(str(e)) from e


def prompt_from_item(
    prompt_name: str,  # noqa: ARG001 - kept for the PromptBuilder signature
    item: Mapping[str, Any],
    language: str,  # noqa: ARG001
) -> str:
    """Return the item's baked prompt.

    Generation prompts are rendered at item-build time (see
    ``scripts/build_generation_items.py``), so a run just reads the ``prompt``
    field. The item loader already enforces a non-empty ``prompt``.
    """
    prompt = item.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ConfigError(
            f"Benchmark item '{item.get('benchmark_item_id')}' has no 'prompt'"
        )
    return prompt
