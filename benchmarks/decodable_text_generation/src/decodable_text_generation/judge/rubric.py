"""Rubric definitions for LLM-as-a-judge evaluation of generated text.

A rubric is a YAML file mapping section name -> section definition. Each
section holds a list of binary criteria; individual criteria may be *gated*
by tags attached to the story being judged.

Gating semantics:

- ``gating: null`` (or absent) — the criterion always applies.
- ``gating: "S2+"`` — the criterion applies when the story's level tag bands
  at or above S2 in :data:`STORY_LEVEL_BANDS`.
- ``gating: "S2"`` — exact level match: applies iff the story's level tag is
  "S2".
- anything else — membership: applies iff the tag is in the story's tag set
  (feature tags such as ``"-ed taught"`` or ``"non-decodable words"``).

Story tags come from ``derive_tags`` plus any caller-supplied extras.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable, Mapping
from enum import StrEnum
from importlib.resources import files
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, model_validator

# Curriculum-aligned profiles are named ufli-en-lNNN (UFLI lesson NNN) or
# tusome-ke-g{grade}-wk{week} (Tusome grade/week). Jagged variants
# (ufli-en-jNN) and anything else carry no level tag.
_ALIGNED_EN_PROFILE = re.compile(r"^ufli-en-l(\d+)$")
_ALIGNED_KE_PROFILE = re.compile(r"^tusome-ke-g(\d+)-wk(\d+)$")

RUBRICS_PACKAGE = "decodable_text_generation.judge.rubrics"

# ``stages`` in scoring.yaml is keyed by language name; benchmark language
# codes map to those keys here.
_LANGUAGE_STAGE_KEYS = {"en": "English", "sw": "Kiswahili"}

_KISWAHILI_LESSON_MIN = re.compile(r"^G\d+W(\d+)$")


def packaged_scoring_path(name: str = "scoring") -> Path:
    """Path to the packaged deterministic-scoring config (``scoring.yaml``)."""
    return Path(str(files(RUBRICS_PACKAGE).joinpath(f"{name}.yaml")))


def load_scoring_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load the deterministic-scoring config (bands, floors, ceilings)."""
    path = Path(path) if path else packaged_scoring_path()
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def stage_config(
    language: str,
    story_level: str | None,
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Stage config (length bounds, floors, caps) for a language + level."""
    config = load_scoring_config() if config is None else config
    lang = _LANGUAGE_STAGE_KEYS.get(language, language)
    stages = (config.get("stages") or {}).get(lang) or {}
    return stages.get(story_level or "") or {}


def _lesson_min_value(value: Any) -> int:
    """``lesson_min`` as an int — "G1W18" (Kiswahili) means week 18."""
    match = _KISWAHILI_LESSON_MIN.fullmatch(str(value))
    return int(match.group(1)) if match else int(value)


def story_level_bands(
    config: Mapping[str, Any] | None = None,
    language: str = "en",
) -> tuple[tuple[str, int], ...]:
    """Story-level bands (``(name, lesson_min)``) from the scoring config."""
    config = load_scoring_config() if config is None else config
    lang = _LANGUAGE_STAGE_KEYS.get(language, language)
    stages = (config.get("stages") or {}).get(lang) or {}
    return tuple(
        (name, _lesson_min_value(stage["lesson_min"]))
        for name, stage in stages.items()
        if isinstance(stage, Mapping) and "lesson_min" in stage
    )


# Story-level bands keyed by UFLI lesson number, loaded from the packaged
# scoring config (rubrics/scoring.yaml). S1 = lessons 1-19, S2 = 20-83,
# S3 = 84-128, S4 = 129+ (no content today; reserved).
STORY_LEVEL_BANDS: tuple[tuple[str, int], ...] = story_level_bands()

# Kiswahili Tusome profiles band on the grade-1 week axis (lesson_min
# "G1W{n}" in the config): wk01-06 -> S1, wk07-17 -> S2, wk18+ -> S3.
# Grade >=2 profiles follow the grade-1 sequence and are always S3.
_TUSOME_WEEK_BANDS: tuple[tuple[str, int], ...] = story_level_bands(
    language="sw"
)


def language_for_profile(student_knowledge_id: str) -> str:
    """Benchmark language code for a curriculum-aligned profile id."""
    return "sw" if _ALIGNED_KE_PROFILE.fullmatch(student_knowledge_id) else "en"


def language_name_for_profile(student_knowledge_id: str) -> str:
    """Display name of the profile's language ("English"/"Kiswahili")."""
    return _LANGUAGE_STAGE_KEYS.get(
        language_for_profile(student_knowledge_id), "English"
    )


class CriterionType(StrEnum):
    """Type of criterion for scoring logic."""

    POSITIVE = "positive"  # Yes = 1 point
    FAIL_CONDITION = "fail_condition"  # Yes = section score becomes 0
    EVIDENCE_LIST = "evidence_list"  # informational only; returns a list, no Yes/No


class Criterion(BaseModel):
    """A single binary criterion, optionally gated by story tags."""

    id: str
    question: str
    criterion_type: CriterionType
    allows_na: bool = False
    gating: str | None = None
    evidence: bool = False  # judge must also return evidence (see below)
    # Structured evidence: when set, ``evidence`` is a JSON array of objects
    # with these properties, in declared order. Values map property name to
    # "str" (free text) or a list of allowed values (emitted as Literal[...]).
    # Required for ``evidence_list`` criteria, optional for ``evidence: true``.
    evidence_items: dict[str, str | list[str]] | None = None
    # Optional named scalar fields on an ``evidence_list`` response, generated
    # before the item list (e.g. "about": a one-sentence summary the list is
    # judged against). Stored under ``fields`` on the criterion result.
    evidence_fields: dict[str, str] | None = None
    # Derived scoring for ``evidence_list`` criteria: each rule dict is worth
    # one point. Supported rule forms: {"severity_in": [level, ...]} — earned
    # iff every item's ``severity`` is in the allowed set (empty lists pass) —
    # {"max_items": N} — earned iff the list has at most N items — and
    # {"max_count_where": {"field": F, "in"|"not": V|..., "count": N}} — one
    # applicable point, earned iff at most N items match the condition.
    scoring_rules: list[dict[str, Any]] | None = None

    @model_validator(mode="after")
    def check_evidence_items(self) -> Criterion:
        """``evidence_list`` criteria must declare their item properties."""
        if (
            self.criterion_type is CriterionType.EVIDENCE_LIST
            and not self.evidence_items
            and not self.evidence_fields
        ):
            raise ValueError(
                f"{self.id}: evidence_list criteria require evidence_items "
                "or evidence_fields"
            )
        return self


class Section(BaseModel):
    """A rubric section: a named batch of criteria evaluated in one call.

    ``weight`` scales the section's contribution to the final score;
    ``weight: 0`` means the section is judged and reported but excluded
    from the final score entirely.
    """

    name: str
    criteria: list[Criterion]
    allows_na: bool = False
    gating_question: str | None = None
    section_preface: str | None = None
    requires_justification: bool = False
    weight: float = 1.0


def load_rubric(path: str | Path) -> dict[str, Section]:
    """Load and validate a rubric definition from a YAML file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Rubric file not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Rubric file must map section names to sections: {path}")
    return {name: Section(**section_data) for name, section_data in data.items()}


def packaged_rubric_path(name: str = "decodable_text") -> Path:
    """Path to a rubric shipped inside the package's ``rubrics/`` data dir."""
    return Path(str(files(RUBRICS_PACKAGE).joinpath(f"{name}.yaml")))


def rubric_sha256(path: str | Path) -> str:
    """Content hash of a rubric file, stored on judgments for provenance."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _band_for(value: int, bands: tuple[tuple[str, int], ...]) -> str:
    band = bands[0][0]
    for name, minimum in bands:
        if value >= minimum:
            band = name
        else:
            break
    return band


def level_for_lesson(lesson: int) -> str:
    """Map a UFLI lesson number to its story-level band (S1-S4)."""
    return _band_for(lesson, STORY_LEVEL_BANDS)


def lesson_for_profile(student_knowledge_id: str) -> int | None:
    """UFLI lesson number of a curriculum-aligned English profile id.

    Returns e.g. ``50`` for ``ufli-en-l050``; ``None`` for jagged,
    Kiswahili, or otherwise out-of-scope profile ids.
    """
    match = _ALIGNED_EN_PROFILE.fullmatch(student_knowledge_id)
    return int(match.group(1)) if match else None


def level_tag_for_profile(student_knowledge_id: str) -> str | None:
    """Story-level tag for a curriculum-aligned profile id.

    Returns e.g. "S2" for ``ufli-en-l050`` or ``tusome-ke-g1-wk10``.
    Returns ``None`` for jagged profiles (``ufli-en-jNN``) and anything
    else outside the UFLI/Tusome naming schemes.
    """
    lesson = lesson_for_profile(student_knowledge_id)
    if lesson is not None:
        return level_for_lesson(lesson)
    match = _ALIGNED_KE_PROFILE.fullmatch(student_knowledge_id)
    if match is None:
        return None
    if int(match.group(1)) != 1:
        return "S3"
    return _band_for(int(match.group(2)), _TUSOME_WEEK_BANDS)


def _story_level(tags: set[str]) -> str | None:
    """The story's level-band tag: the highest band present in ``tags``."""
    order = [name for name, _ in STORY_LEVEL_BANDS]
    present = [t for t in tags if t in order]
    return max(present, key=order.index) if present else None


def criterion_applies(gating: str | None, tags: set[str]) -> bool:
    """Whether a criterion's gating expression matches a story's tags."""
    if gating is None:
        return True
    order = [name for name, _ in STORY_LEVEL_BANDS]
    if gating.endswith("+") and gating[:-1] in order:
        story_level = _story_level(tags)
        return story_level is not None and order.index(story_level) >= order.index(
            gating[:-1]
        )
    return gating in tags


def filter_section_criteria(section: Section, tags: set[str]) -> list[Criterion]:
    """The criteria of ``section`` that apply to a story with ``tags``."""
    return [c for c in section.criteria if criterion_applies(c.gating, tags)]


def derive_tags(
    student_knowledge_id: str,
    student_knowledge: Mapping[str, Any] | None = None,
    *,
    n_nondecodable_words: int = 0,
    target_knowledge: Mapping[str, Any] | None = None,
    n_target_words: int = 0,
    max_frame_repeats: int = 0,
    extra_tags: Iterable[str] = (),
) -> set[str]:
    """Assemble a story's tag set for criterion gating.

    The level tag comes from the profile's UFLI lesson number or Tusome
    grade/week;
    ``"-ed taught"`` reflects the profile's taught affixes; and
    ``"non-decodable words"`` is set when the deterministic decodability
    check found any non-decodable words (the caller passes the count).
    ``"target knowledge"`` is set when the item/profile carries target
    units (curriculum-aligned runs; child items carry none); ``"target
    words"`` is set when the deterministic check found at least one
    target-bearing word in the story (the caller passes the count).
    ``extra_tags`` (row field or ``--tags-file``) merge on top.
    """
    tags = set(extra_tags)
    level = level_tag_for_profile(student_knowledge_id)
    if level:
        tags.add(level)
    if student_knowledge:
        affixes = student_knowledge.get("affixes") or ()
        if "-ed" in affixes:
            tags.add("-ed taught")
    if n_nondecodable_words > 0:
        tags.add("non-decodable words")
    # ``any`` on values: a dict of empty fields ({gpcs: [], ...}) is not
    # a populated target.
    if target_knowledge and any(target_knowledge.values()):
        tags.add("target knowledge")
    if n_target_words > 0:
        tags.add("target words")
    if max_frame_repeats >= 3:
        tags.add("repeats-frame-3")
    return tags
