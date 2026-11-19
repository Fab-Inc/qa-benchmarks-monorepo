"""Prompt rendering for the reading comprehension question generation benchmark."""

from collections.abc import Mapping
from functools import cache
from importlib.resources import files
from typing import Any

import yaml
from phonics_benchmark.config import ConfigError

PACKAGE = "reading_comprehension_question_generation"
PROMPTS_FILENAME = "prompts.yaml"
DEFAULT_PROMPT_NAME = "question_generation"

# How many question and answer pairs every prompt asks for, and what extraction
# compares each response against. It is one fixed value, deliberately not a run option
# or a per-text setting: making it configurable is a design choice to revisit later.
NUM_PAIRS = 3


@cache
def _load_prompts() -> dict[str, str]:
    """Load and cache the packaged prompt templates."""
    text = files(PACKAGE).joinpath(PROMPTS_FILENAME).read_text(encoding="utf-8")
    return yaml.safe_load(text)


def build_generation_prompt(prompt_name: str, text: str, language: str) -> str:
    """Render the named prompt from the passage and the language name.

    The signature is deliberately just these two values: nothing else about a text
    (title, source, grade, and so on) can reach the prompt. The pair count comes from
    the constant ``NUM_PAIRS``.
    """
    prompts = _load_prompts()
    try:
        template = prompts[prompt_name]
    except KeyError:
        raise ConfigError(
            f"Unknown prompt '{prompt_name}'; available: {sorted(prompts)}"
        ) from None
    try:
        return template.format(text=text, language=language, num_pairs=NUM_PAIRS)
    except KeyError as exc:
        # A template that asks for a placeholder this renderer does not supply.
        raise ConfigError(
            f"Prompt '{prompt_name}' needs field {exc} that was not supplied"
        ) from exc


def prompt_for_item(
    prompt_name: str,
    item: Mapping[str, Any],
    language: str,  # noqa: ARG001 - the run config's language, not the item's
) -> str:
    """Render the prompt for one item; the harness's ``prompt_builder``.

    The harness passes the run config's language, which is one value for the whole run.
    The prompt needs each item's own language name, so it is read from the item.
    """
    return build_generation_prompt(prompt_name, item["text_content"], item["language"])
