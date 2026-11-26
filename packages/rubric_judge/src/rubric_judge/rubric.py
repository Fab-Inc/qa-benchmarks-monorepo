"""Rubric definitions for LLM-as-a-judge evaluation.

A rubric is a YAML file mapping section name -> section definition. Each
section holds a list of binary criteria; individual criteria may be *gated*
by tags attached to the text being judged.

Gating semantics — a ``gating`` string is tested against the caller-supplied
tag set:

- ``gating: null`` (or absent) — the criterion always applies.
- ``gating: "tag"`` — membership: applies iff ``"tag"`` is in the text's tag
  set. Covers binary flags (``"has-dialogue"``) and category axes
  (``"genre:poem"``, ``"lang:sw"``) by caller convention.
- ``gating: "!tag"`` — negated membership: applies iff the tag is absent.
- ``gating: "X+"`` / ``"X-"`` — ordered gate: applies when the text's level
  (the highest member of ``ordered_tags`` present in its tags) is at-or-above
  / at-or-below ``X``. Texts with no ordered tag match neither.

Tags are supplied by the caller; ``ordered_tags`` is the ascending list of
tag names that form the ordered scale for ``"+"``/``"-"`` gating (e.g.
``("S1", "S2", "S3")``). Without it, ordered gatings never match.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from enum import StrEnum
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, model_validator


class CriterionType(StrEnum):
    """Type of criterion for scoring logic."""

    POSITIVE = "positive"  # Yes = 1 point
    FAIL_CONDITION = "fail_condition"  # Yes = section score becomes 0
    EVIDENCE_LIST = "evidence_list"  # informational only; returns a list, no Yes/No


class Criterion(BaseModel):
    """A single binary criterion, optionally gated by tags."""

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
    # Points an ``evidence_list`` criterion earns: each rule dict is one
    # point, earned iff the count of matching items satisfies the rule's
    # ``less_than``/``greater_than`` bound (both together = an exclusive
    # range; {less_than: 3, greater_than: 1} means exactly 2 items).
    # Matchers: none (all items), {"severity_in": [...]}, or
    # {"where": {"field": F, "in"|"not": V|list}}. Legacy forms
    # ({"max_items": N}, bare {"severity_in": [...]}, {"max_count_where":
    # {..., "count": N}}) are still understood when rescoring judgments
    # stored before ``less_than`` existed.
    scoring_rules: list[dict] | None = None
    # Points the criterion is worth (default 1.0): scales each positive
    # verdict and each earned scoring-rule point in ``score_section``.
    # Stored on the judgment row so rescoring sees it (overridable via
    # ``criterion_weights``). Scoring-only: never invalidates call
    # fingerprints. Meaningless on ``fail_condition`` criteria.
    weight: float = 1.0

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
    """A rubric section: a named batch of criteria.

    ``call_per`` controls prompt granularity: ``"section"`` (default)
    evaluates all applicable criteria in one call; ``"question"`` makes one
    call per criterion (run concurrently). ``weight`` scales the section's
    contribution to the final score; ``weight: 0`` means the section is
    judged and reported but excluded from the final score entirely.

    ``prompt_template`` replaces the standard prompt entirely: the rendered
    template is the whole prompt. Available slots are ``{text}``,
    ``{context}``, ``{subject}``, ``{preface}``, ``{criteria}`` (the
    numbered criterion list), ``{instructions}`` (the response-shape text
    matching the built schema) and the item's placeholders — so a section
    can frame the call however it wants while keeping criterion/schema
    parsing.
    """

    name: str
    criteria: list[Criterion]
    allows_na: bool = False
    section_preface: str | None = None
    prompt_template: str | None = None
    requires_justification: bool = False
    weight: float = 1.0
    call_per: Literal["section", "question"] = "section"


def load_rubric(path: str | Path) -> dict[str, Section]:
    """Load and validate a rubric definition from a YAML file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Rubric file not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Rubric file must map section names to sections: {path}")
    return {name: Section(**section_data) for name, section_data in data.items()}


def rubric_sha256(path: str | Path) -> str:
    """Content hash of a rubric file, stored on judgments for provenance."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _story_level(tags: set[str], ordered_tags: Sequence[str]) -> str | None:
    """The text's ordered tag: the highest member of ``ordered_tags`` in ``tags``."""
    order = list(ordered_tags)
    present = [t for t in tags if t in order]
    return max(present, key=order.index) if present else None


def criterion_applies(
    gating: str | None, tags: set[str], ordered_tags: Sequence[str] = ()
) -> bool:
    """Whether a criterion's gating expression matches a text's tags."""
    if gating is None:
        return True
    if gating.startswith("!"):
        return gating[1:] not in tags
    order = list(ordered_tags)
    if gating.endswith(("+", "-")) and gating[:-1] in order:
        level = _story_level(tags, order)
        if level is None:
            return False
        i, bound = order.index(level), order.index(gating[:-1])
        return i >= bound if gating.endswith("+") else i <= bound
    return gating in tags


def filter_section_criteria(
    section: Section, tags: set[str], ordered_tags: Sequence[str] = ()
) -> list[Criterion]:
    """The criteria of ``section`` that apply to a text with ``tags``."""
    return [
        c for c in section.criteria if criterion_applies(c.gating, tags, ordered_tags)
    ]
