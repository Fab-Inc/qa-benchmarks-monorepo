"""The decodable text generation benchmarks, built on fab-benchmark-runner."""

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from decodability import DECODABILITY_REGISTRY
from fab_benchmark_runner import Benchmark, ConfigError, build_app

from decodable_text_generation.items import load_generation_items
from decodable_text_generation.prompts import (
    KNOWLEDGE_FORMAT,
    prompt_from_item,
    prompt_names,
)

GENERATION_CHILD_TASK = "generation-child"
GENERATION_CURRICULUM_TASK = "generation-curriculum"

# Curriculum items carry a `target` object and are prompted with the
# {target}-bearing template; child items use the plain generation template.
DEFAULT_PROMPT_NAMES = {
    GENERATION_CHILD_TASK: "decodable_text_generation",
    GENERATION_CURRICULUM_TASK: "decodable_text_generation_curriculum",
}

# The monorepo's shared .env and custom_models.yaml, as the phonics benchmark
# uses; src/decodable_text_generation/cmd.py sits four levels below the root.
REPO_ROOT = Path(__file__).resolve().parents[4]


@dataclass(frozen=True)
class GenerationParams:
    """The run config keys owned by the generation benchmarks.

    All three are run identity: they go into the config ID and every result
    row. ``prompt_name`` records which template the items were baked with;
    generation items carry their rendered prompt, so it is provenance only.
    """

    language: str
    task: str
    prompt_name: str | None = None

    def __post_init__(self) -> None:
        if self.task not in DEFAULT_PROMPT_NAMES:
            raise ValueError(
                f"Unknown task '{self.task}'; available: {sorted(DEFAULT_PROMPT_NAMES)}"
            )
        if self.language not in DECODABILITY_REGISTRY:
            raise ValueError(
                f"Unknown language '{self.language}'; available: "
                f"{sorted(DECODABILITY_REGISTRY)}"
            )
        if self.language not in KNOWLEDGE_FORMAT:
            raise ValueError(
                f"No student-knowledge format for language '{self.language}'; "
                "add one to KNOWLEDGE_FORMAT in decodable_text_generation/prompts.py"
            )
        if self.prompt_name is None:
            object.__setattr__(self, "prompt_name", DEFAULT_PROMPT_NAMES[self.task])
        elif self.prompt_name not in prompt_names():
            raise ValueError(
                f"Unknown prompt_name '{self.prompt_name}'; available: {prompt_names()}"
            )


def build_prompt(item: Mapping[str, Any], params: GenerationParams) -> str:
    """Check item/task agreement, then return the item's baked prompt.

    ``target_knowledge`` marks a curriculum item; the loader already rejects
    mixed files, so a disagreement here means the config's task does not
    match its items.
    """
    has_target = bool(item.get("target_knowledge"))
    if has_target and params.task == GENERATION_CHILD_TASK:
        raise ConfigError(
            f"Benchmark item '{item['benchmark_item_id']}' carries "
            f"'target_knowledge' but task '{params.task}' is a child run"
        )
    if not has_target and params.task == GENERATION_CURRICULUM_TASK:
        raise ConfigError(
            f"Benchmark item '{item['benchmark_item_id']}' has no "
            f"'target_knowledge' but task '{params.task}' is a curriculum run"
        )
    return prompt_from_item(item)


BENCHMARK = Benchmark(
    name="dtg",
    params_type=GenerationParams,
    build_prompt=build_prompt,
    load_items=load_generation_items,
    default_env_path=REPO_ROOT / ".env",
    default_custom_models_path=REPO_ROOT / "custom_models.yaml",
)
app = build_app(BENCHMARK)


def main() -> None:
    """Main entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()  # pragma: no cover
