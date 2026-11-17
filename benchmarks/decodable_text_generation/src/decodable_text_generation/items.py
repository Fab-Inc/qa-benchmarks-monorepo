"""Loading of completed benchmark items for decodable-text-generation runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from phonics_benchmark.config import ConfigError
from phonics_benchmark.items import load_benchmark_items

# Generation items pair a student profile with a scenario and carry their fully
# rendered prompt, so `prompt` and `scenario_id` replace the decodability
# benchmark's `content` requirement.
GENERATION_REQUIRED_FIELDS: dict[str, type] = {
    "benchmark_item_id": str,
    "scenario_id": str,
    "student_knowledge": dict,
    "prompt": str,
}

# Dataset-owned fields carried through for prompt building and later scoring.
GENERATION_EXTRA_FIELDS = (
    "student_knowledge_id",
    "language_iso_code",
    "is_jagged",
    "target_knowledge",
)


def load_generation_items(
    items_path: str | Path,
    *,
    curriculum: bool = False,
) -> tuple[list[dict[str, Any]], str]:
    """Load generation benchmark items; hash the exact JSONL bytes consumed.

    With ``curriculum=True`` every item must also carry a non-empty
    ``target_knowledge`` object (the knowledge the student is currently being
    taught).
    """
    items, items_sha256 = load_benchmark_items(
        items_path,
        required_fields=GENERATION_REQUIRED_FIELDS,
        extra_fields=GENERATION_EXTRA_FIELDS,
    )
    if curriculum:
        missing = [
            str(item["benchmark_item_id"])
            for item in items
            if not item.get("target_knowledge")
        ]
        if missing:
            raise ConfigError(
                "Curriculum benchmark items must include a non-empty "
                f"'target_knowledge'; missing on: {', '.join(missing)}"
            )
    return items, items_sha256
