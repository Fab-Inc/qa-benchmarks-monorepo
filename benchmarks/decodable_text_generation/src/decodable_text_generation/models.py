"""Data models for decodable_text_generation."""

from typing import Annotated, Any

from decodability.english.definitions import normalize_gpc_grapheme
from pydantic import BaseModel, ConfigDict, Field, field_validator

NonEmptyStr = Annotated[str, Field(min_length=1)]


def _normalize_knowledge(knowledge: Any) -> Any:
    """Normalize a knowledge dict's ``gpcs``/``affixes`` to inventory notation.

    Marker-bearing graphemes in ``gpcs`` split two ways per
    ``EnglishStudentKnowledge``: the stripped pair (``ed`` → ``d``) stays in
    ``gpcs`` as letter knowledge while the marked grapheme (``-ed``, ``un-``,
    ``_ce`` → ``-ce``) is collected into ``affixes``. A medial ``_`` (``a_e``)
    is a split digraph, not an affix marker, and just folds to the inventory's
    hyphen form. Exact duplicate pairs collapse, keeping first-seen order.
    """
    if not isinstance(knowledge, dict):
        return knowledge
    gpcs = knowledge.get("gpcs")
    if not isinstance(gpcs, list):
        return knowledge

    affixes = {str(a) for a in knowledge.get("affixes") or []}
    normalized = []
    for grapheme, phoneme in gpcs:
        raw = str(grapheme).strip()
        phoneme = str(phoneme).strip().strip("/")
        if len(raw) > 1 and (raw.startswith(("-", "_")) or raw.endswith(("-", "_"))):
            affixes.add(raw.replace("_", "-"))
            raw = normalize_gpc_grapheme(raw)
        else:
            raw = raw.replace("_", "-")
        normalized.append((raw, phoneme))
    knowledge["gpcs"] = list(dict.fromkeys(normalized))
    if affixes:
        knowledge["affixes"] = sorted(affixes)
    return knowledge


class GenerationStudentProfile(BaseModel):
    """A student knowledge profile for a generation benchmark.

    Profiles arrive pre-built as JSONL under ``data/<benchmark>/<language>/``:
    child_adapted files carry ``is_jagged``; curriculum_aligned files carry
    ``target_knowledge`` — the knowledge the student is currently being taught
    (e.g. this week's GPCs or sight words), which curriculum prompts ask the
    model to practise.
    """

    model_config = ConfigDict(extra="forbid")
    student_knowledge_id: NonEmptyStr
    language_iso_code: NonEmptyStr
    is_jagged: bool = False
    student_knowledge: dict[str, Any]
    target_knowledge: dict[str, Any] | None = None

    _normalize = field_validator(
        "student_knowledge", "target_knowledge", mode="before"
    )(_normalize_knowledge)


class GenerationBenchmarkItem(BaseModel):
    """A student profile paired with a generation scenario — one benchmark item.

    The ``prompt`` is fully rendered at build time so the item is
    self-contained and the prompt text can be reviewed as an artifact.
    """

    model_config = ConfigDict(extra="forbid")

    benchmark_item_id: NonEmptyStr
    student_knowledge_id: NonEmptyStr
    scenario_id: NonEmptyStr
    language_iso_code: NonEmptyStr

    student_knowledge: dict[str, Any]
    prompt: NonEmptyStr
    is_jagged: bool = False
    target_knowledge: dict[str, Any] | None = None
