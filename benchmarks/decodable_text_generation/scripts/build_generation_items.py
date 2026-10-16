"""Build benchmark items with fully rendered prompts from profile JSONL data.

Reads ``data/<benchmark_dir>/<language>/student_knowledge_profiles.jsonl``
(``child_adapted`` / ``curriculum_aligned``), cross-joins profiles with the
``story`` scenario (or those given via ``--scenarios``), renders each item's
prompt, and writes ``benchmark_items.jsonl`` next to the input.

- ``--benchmark child``: every profile is used, jagged or not.
- ``--benchmark curriculum``: only profiles with a non-empty ``target_knowledge``
  are used, and the target is carried onto each item.

The baked ``prompt`` on each item is what a benchmark run sends to the model,
so changing prompt templates means rerunning this script.

Run from the repository root with (as an example):

    uv run --package decodable-text-generation python \\
        benchmarks/decodable_text_generation/scripts/build_generation_items.py \\
        --language english \\
        --benchmark curriculum
"""

from enum import StrEnum
from pathlib import Path

import typer
from decodability import DECODABILITY_REGISTRY
from decodable_text_generation.models import (
    GenerationBenchmarkItem,
    GenerationStudentProfile,
)
from decodable_text_generation.profiles import (
    profiles_path,
    read_generation_profiles,
    validate_target,
)
from decodable_text_generation.prompts import (
    build_generation_prompt,
    load_scenarios,
)
from phonics_benchmark.data_io import write_jsonl
from pydantic import BaseModel, ValidationError

DEFAULT_DATA_ROOT = Path(__file__).resolve().parents[1] / "data"

ITEMS_FILENAME = "benchmark_items.jsonl"


class Benchmark(StrEnum):
    CHILD = "child"
    CURRICULUM = "curriculum"


DEFAULT_PROMPT_NAMES = {
    Benchmark.CHILD: "decodable_text_generation",
    Benchmark.CURRICULUM: "decodable_text_generation_curriculum",
}


def select_profiles(
    profiles: list[GenerationStudentProfile], benchmark: Benchmark
) -> list[GenerationStudentProfile]:
    """Return the profiles a benchmark uses, or raise if it ends up empty."""
    if benchmark is Benchmark.CURRICULUM:
        selected = [p for p in profiles if p.target_knowledge]
        if not selected:
            raise ValueError(
                "The curriculum benchmark needs at least one profile with a "
                "non-empty target_knowledge; found none."
            )
        return selected
    if not profiles:
        raise ValueError("No student knowledge profiles found.")
    return profiles


def validate_profile_knowledge(profile: GenerationStudentProfile) -> None:
    """Validate student_knowledge and target_knowledge against the registry."""
    language_integration = DECODABILITY_REGISTRY.get(profile.language_iso_code)
    if language_integration is None:
        raise ValueError(
            f"No student_knowledge validator registered for language "
            f"'{profile.language_iso_code}' "
            f"(student_knowledge_id '{profile.student_knowledge_id}')"
        )
    knowledge_model: type[BaseModel] = language_integration["student_knowledge_model"]
    try:
        knowledge_model.model_validate(profile.student_knowledge)
    except ValidationError as exc:
        raise ValueError(
            f"Invalid student_knowledge for student_knowledge_id "
            f"'{profile.student_knowledge_id}': {exc}"
        ) from exc
    if profile.target_knowledge:
        validate_target(
            profile.target_knowledge,
            profile.language_iso_code,
            profile_id=profile.student_knowledge_id,
        )


def build_items(
    profiles: list[GenerationStudentProfile],
    prompt_name: str,
    scenario_ids: list[str] | None = None,
) -> list[GenerationBenchmarkItem]:
    """Cross-join profiles with scenarios, rendering each item's prompt."""
    known = load_scenarios()
    if scenario_ids is None:
        scenario_ids = ["story"]
    unknown = [s for s in scenario_ids if s not in known]
    if unknown:
        raise ValueError(
            f"Unknown scenario(s): {', '.join(unknown)}. "
            f"Available: {', '.join(sorted(known))}"
        )
    items = []
    for profile in profiles:
        for scenario_id in scenario_ids:
            fields = {
                "benchmark_item_id": f"{profile.student_knowledge_id}_{scenario_id}",
                "student_knowledge_id": profile.student_knowledge_id,
                "scenario_id": scenario_id,
                "language_iso_code": profile.language_iso_code,
                "is_jagged": profile.is_jagged,
                "student_knowledge": profile.student_knowledge,
                "target_knowledge": profile.target_knowledge,
            }
            fields["prompt"] = build_generation_prompt(
                prompt_name, fields, profile.language_iso_code
            )
            items.append(GenerationBenchmarkItem(**fields))
    return items


def build_generation_items(
    language: str = typer.Option(
        ...,
        "--language",
        "-l",
        help="Language folder name, e.g. 'english', under the benchmark's data dir.",
    ),
    benchmark: Benchmark = typer.Option(
        ...,
        "--benchmark",
        "-b",
        help="Which benchmark to build items for: 'child' or 'curriculum'.",
    ),
    data_root: Path = typer.Option(
        DEFAULT_DATA_ROOT,
        "--data-root",
        file_okay=False,
        help="Root data directory. Defaults to the package's data/ folder.",
    ),
    prompt_name: str | None = typer.Option(
        None,
        "--prompt-name",
        help=(
            "Prompt template to bake into the items (a key in prompts.yaml). "
            "Defaults to decodable_text_generation / "
            "decodable_text_generation_curriculum."
        ),
    ),
    scenarios: str | None = typer.Option(
        None,
        "--scenarios",
        "-s",
        help=(
            "Comma-separated scenario ids to build (e.g. 'sentences,story'). "
            "Defaults to 'story'."
        ),
    ),
) -> None:
    """Pair profiles with scenarios, render prompts, write benchmark items."""
    profiles = read_generation_profiles(
        profiles_path(benchmark.value, language, data_root)
    )
    for profile in profiles:
        validate_profile_knowledge(profile)
    selected = select_profiles(profiles, benchmark)
    scenario_ids = (
        [s.strip() for s in scenarios.split(",") if s.strip()] if scenarios else None
    )
    items = build_items(
        selected, prompt_name or DEFAULT_PROMPT_NAMES[benchmark], scenario_ids
    )

    output_path = profiles_path(benchmark.value, language, data_root).parent / (
        ITEMS_FILENAME
    )
    write_jsonl(items, output_path)

    typer.echo(
        f"Paired {len(selected)} profile(s) × scenarios into {len(items)} "
        f"benchmark items with rendered prompts in {output_path}"
    )
    typer.echo(
        "Next: point a run config's items_path at this file and run "
        f"'decodable-text-generation run-generation-{benchmark.value}'."
    )


def main() -> None:
    """Entry point."""
    typer.run(build_generation_items)


if __name__ == "__main__":
    main()
