"""Loading of completed benchmark items for decodable-text-generation runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fab_benchmark_runner import ConfigError, load_jsonl_items

# Generation items pair a student profile with a scenario and carry their fully
# rendered prompt, so `prompt` and `scenario_id` replace the decodability
# benchmark's `content` requirement. `benchmark_item_id` is checked by the
# shared loader.
GENERATION_REQUIRED_FIELDS: dict[str, type] = {
    "scenario_id": str,
    "student_knowledge": dict,
    "prompt": str,
}


def load_generation_items(
    items_path: str | Path,
) -> tuple[list[dict[str, Any]], str]:
    """Load generation benchmark items; hash the exact JSONL bytes consumed.

    Every item must carry the generation fields. ``target_knowledge`` — the
    knowledge the student is currently being taught — must be present on
    either all items (a curriculum run) or none (a child run); a file mixing
    the two is rejected. Item/task agreement is checked by the prompt
    builder.
    """
    items, items_sha256 = load_jsonl_items(items_path)
    path = Path(items_path)
    for line_number, item in enumerate(items, start=1):
        for field_name, field_type in GENERATION_REQUIRED_FIELDS.items():
            value = item.get(field_name)
            if field_type is str:
                if not isinstance(value, str) or not value.strip():
                    raise ConfigError(
                        f"{path}:{line_number}: {field_name} must be a "
                        "non-empty, non-whitespace string"
                    )
            elif not isinstance(value, field_type):
                raise ConfigError(
                    f"{path}:{line_number}: {field_name} must be an object"
                    if field_type is dict
                    else f"{path}:{line_number}: {field_name} has the wrong type"
                )

    has_target = [bool(item.get("target_knowledge")) for item in items]
    if any(has_target) and not all(has_target):
        raise ConfigError(
            f"Benchmark items in {path} mix 'target_knowledge' presence; "
            "expected all items or none to carry it."
        )
    return items, items_sha256
