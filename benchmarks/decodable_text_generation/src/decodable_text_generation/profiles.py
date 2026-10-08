"""Student knowledge profiles for the decodable text generation benchmarks.

Profiles are pre-built data artifacts, one JSONL file per benchmark/language:
``data/<benchmark_dir>/<language>/student_knowledge_profiles.jsonl``.
"""

import json
from pathlib import Path
from typing import Any

from decodability import DECODABILITY_REGISTRY
from decodability.kiswahili.definitions import VALID_GRAPHEMES, ClusterPattern
from pydantic import BaseModel, ValidationError

from decodable_text_generation.models import GenerationStudentProfile

PROFILES_FILENAME = "student_knowledge_profiles.jsonl"

# Directory names under data/<dir>/<language>/ that hold each benchmark's
# profiles.
BENCHMARK_DIRS = {
    "child": "child_adapted",
    "curriculum": "curriculum_aligned",
}


def profiles_path(benchmark: str, language: str, data_root: Path) -> Path:
    """Return the profiles JSONL path for a benchmark/language pair."""
    return data_root / BENCHMARK_DIRS[benchmark] / language / PROFILES_FILENAME


def read_generation_profiles(path: Path) -> list[GenerationStudentProfile]:
    """Parse a student_knowledge_profiles.jsonl into profile records."""
    profiles = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            profiles.append(GenerationStudentProfile.model_validate(json.loads(line)))
        except (json.JSONDecodeError, ValidationError) as exc:
            raise ValueError(f"{path}:{line_number}: invalid profile: {exc}") from exc
    if not profiles:
        raise ValueError(f"No student knowledge profiles found in {path}")
    return profiles


def validate_target(
    target: dict[str, Any],
    language_iso_code: str,
    *,
    profile_id: str,
) -> None:
    """Check a target_knowledge object against the language's knowledge model.

    A target is a partial student_knowledge (same field names, e.g. ``gpcs``
    or ``sight_words`` for English), so the existing per-language model
    validates it; absent fields just take their defaults. The exception is
    Kiswahili: a target is a curriculum *diff* — e.g. newly taught clusters
    whose graphemes were learned earlier — so ``KiswahiliStudentKnowledge``'s
    cross-field checks don't apply and each field is validated on its own.
    """
    if language_iso_code == "sw":
        _validate_kiswahili_target(target, profile_id=profile_id)
        return
    language_integration = DECODABILITY_REGISTRY.get(language_iso_code)
    if language_integration is None:
        raise ValueError(
            f"No student_knowledge validator registered for language "
            f"'{language_iso_code}' (student_knowledge_id '{profile_id}')"
        )
    knowledge_model: type[BaseModel] = language_integration["student_knowledge_model"]
    try:
        knowledge_model.model_validate(target)
    except ValidationError as exc:
        raise ValueError(
            f"Invalid target_knowledge for student_knowledge_id '{profile_id}': {exc}"
        ) from exc


def _validate_kiswahili_target(
    target: dict[str, Any],
    *,
    profile_id: str,
) -> None:
    """Field-level validation for a Kiswahili target_knowledge diff."""
    error = f"Invalid target_knowledge for student_knowledge_id '{profile_id}'"
    graphemes = [str(g) for g in target.get("graphemes") or []]
    if bad := [g for g in graphemes if g not in VALID_GRAPHEMES]:
        raise ValueError(f"{error}: graphemes not in VALID_GRAPHEMES: {bad}")
    for field in ("graphemes", "clusters", "whole_words"):
        if bad := [v for v in (target.get(field) or []) if str(v) != str(v).lower()]:
            raise ValueError(f"{error}: non-lowercase {field}: {bad}")
    for pattern in target.get("cluster_patterns") or []:
        try:
            ClusterPattern(pattern)
        except ValueError:
            raise ValueError(f"{error}: unknown cluster pattern '{pattern}'") from None
