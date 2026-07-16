import json
from collections.abc import Mapping
from functools import cache
from importlib.resources import files
from typing import Any

import yaml


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
    """Render the decodability prompt for a joined benchmark item."""

    student_knowledge = item["student_knowledge"]
    return get_prompt(
        "decodability_evaluation",
        language=language,
        text=item["text"],
        known_graphemes=_json_list(student_knowledge.get("graphemes", [])),
        known_sight_words=_json_list(student_knowledge.get("whole_words", [])),
    )


def _json_list(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)
