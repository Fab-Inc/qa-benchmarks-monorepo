"""Planning: turn a JudgeItem + rubric into the calls the judge model sees.

Phase 1 of three. ``plan_judgment`` renders every prompt and builds every
response schema up front, so the prompts themselves are the identity of a
judgment: editing a question, gating, placeholders, context or subject
changes them; editing scoring-only fields (``scoring_rules``, ``weight``)
does not.

``JudgeCall.inputs_row`` emits the ``{benchmark_item_id, prompt,
system_prompt}`` shape ``fab_benchmark_runner`` consumes, so phase 2 can be
exported to it (``schema_json`` is ready for when inputs can carry a
response schema); phase 3 takes its raw outputs back unchanged.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, create_model

from rubric_judge.rubric import (
    Criterion,
    CriterionType,
    Section,
    filter_section_criteria,
)

JUDGE_MAX_TOKENS = 16384


@dataclass(frozen=True)
class JudgeItem:
    """One text to judge, with everything its prompts need filled in."""

    text: str
    tags: frozenset[str] = frozenset()
    placeholders: Mapping[str, str] | None = None
    context: str | None = None


@dataclass(frozen=True)
class JudgeCall:
    """One model call produced by a rubric for one item.

    ``prompt`` plus ``schema_json`` is exactly what the model is asked, and
    ``fingerprint()`` hashes precisely that — so scoring-only rubric fields
    never invalidate a stored judgment, while template fills always do.
    Calls are unconditional: the whole plan can run in one batch.
    """

    key: str
    prompt: str
    max_tokens: int
    section: str
    kind: str  # "section" | "question"
    criterion_ids: tuple[str, ...]
    label: str
    response_schema: type[BaseModel] = field(compare=False, repr=False)
    system_prompt: str | None = None

    @property
    def schema_json(self) -> dict[str, Any]:
        return self.response_schema.model_json_schema()

    def fingerprint(self) -> dict[str, Any]:
        """The payload a call-level fingerprint hashes — what the model sees."""
        return {
            "criterion_ids": list(self.criterion_ids),
            "kind": self.kind,
            "prompt": self.prompt,
            "schema": self.schema_json,
            "system_prompt": self.system_prompt,
        }

    def inputs_row(self, item_id: str) -> dict[str, Any]:
        """A ``fab_benchmark_runner`` inputs row for this call."""
        return {
            "benchmark_item_id": f"{item_id}::{self.key}",
            "prompt": self.prompt,
            "system_prompt": self.system_prompt,
        }


@dataclass
class SectionPlan:
    """One rubric section's calls and how to assemble its rows."""

    name: str
    weight: float
    criteria: list[Criterion]  # rubric order, raw (unrendered) questions
    rendered: dict[str, Criterion]  # criterion_id -> question-rendered copy
    applicable_ids: frozenset[str]
    gated_out: frozenset[str]
    calls: list[JudgeCall]
    requires_justification: bool


@dataclass
class JudgePlan:
    """Every call needed to judge one item, plus assembly bookkeeping."""

    item: JudgeItem
    rubric: dict[str, Section]
    sections: list[SectionPlan]
    subject: str = "a text"

    @property
    def calls(self) -> list[JudgeCall]:
        return [call for section in self.sections for call in section.calls]

    def call_fingerprints(self) -> dict[str, str]:
        """Call key -> sha256 of what that call asks the model.

        Resume granularity: an unchanged call's stored result can be
        reused, so in a ``call_per: question`` section editing one question
        re-runs only that question's call.
        """
        return {call.key: _sha256_json(call.fingerprint()) for call in self.calls}

    def inputs_rows(self, item_id: str) -> list[dict[str, Any]]:
        """Every call as a ``fab_benchmark_runner`` inputs row."""
        return [call.inputs_row(item_id) for call in self.calls]


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True).encode()
    ).hexdigest()[:16]


def plan_judgment(
    item: JudgeItem,
    *,
    rubric: dict[str, Section],
    ordered_tags: Sequence[str] = (),
    subject: str = "a text",
    max_tokens: int = JUDGE_MAX_TOKENS,
    system_prompt: str | None = None,
) -> JudgePlan:
    """Render every prompt and schema for one item against a rubric.

    No model calls happen here — the returned plan is pure data describing
    what to ask, which also drives resume fingerprints and runner export.
    ``system_prompt`` is the judge persona sent with every call; it is
    rendered with the item's placeholders too.
    """
    if system_prompt is not None and item.placeholders is not None:
        system_prompt = _render_template(system_prompt, item.placeholders)
    sections = [
        _plan_section(
            item, name, section, ordered_tags, subject, max_tokens, system_prompt
        )
        for name, section in rubric.items()
    ]
    return JudgePlan(item=item, rubric=rubric, sections=sections, subject=subject)


def _plan_section(
    item: JudgeItem,
    name: str,
    section: Section,
    ordered_tags: Sequence[str],
    subject: str,
    max_tokens: int,
    system_prompt: str | None,
) -> SectionPlan:
    applicable = filter_section_criteria(section, set(item.tags), ordered_tags)
    applicable_ids = frozenset(c.id for c in applicable)
    rendered: dict[str, Criterion] = {}
    if item.placeholders is not None:
        rendered = {
            c.id: c.model_copy(
                update={
                    "question": _render_template(
                        c.question, item.placeholders, f"criterion '{c.id}'"
                    )
                }
            )
            for c in applicable
        }
    applicable = [rendered.get(c.id, c) for c in applicable]

    calls: list[JudgeCall] = []
    if applicable_ids:
        section_preface = section.section_preface
        if item.placeholders is not None and section_preface:
            section_preface = _render_template(
                section_preface,
                item.placeholders,
                f"section '{name}' section_preface",
            )
        batches = (
            [applicable] if section.call_per == "section" else [[c] for c in applicable]
        )
        for batch in batches:
            key = (
                name
                if len(batch) > 1 or section.call_per == "section"
                else f"{name}::{batch[0].id}"
            )
            prompt = (
                _render_template(
                    section.prompt_template,
                    {
                        **(item.placeholders or {}),
                        "context": item.context or "",
                        "criteria": _criteria_text(batch),
                        "instructions": _response_instruction(
                            batch, section.requires_justification
                        ),
                        "preface": section_preface or "",
                        "subject": subject,
                        "text": item.text,
                    },
                    f"section '{name}' prompt_template",
                )
                if section.prompt_template is not None
                else build_section_prompt(
                    name,
                    batch,
                    text=item.text,
                    section_preface=section_preface,
                    requires_justification=section.requires_justification,
                    context=item.context,
                    subject=subject,
                )
            )
            calls.append(
                JudgeCall(
                    key=key,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    section=name,
                    kind="question" if len(batch) == 1 else "section",
                    criterion_ids=tuple(c.id for c in batch),
                    label=(
                        f"section '{name}'/{batch[0].id}"
                        if len(batch) == 1
                        else f"section '{name}'"
                    ),
                    response_schema=(
                        create_section_response_model_with_justification(name, batch)
                        if section.requires_justification
                        else create_section_response_model(name, batch)
                    ),
                    system_prompt=system_prompt,
                )
            )

    return SectionPlan(
        name=name,
        weight=section.weight,
        criteria=list(section.criteria),
        rendered=rendered,
        applicable_ids=applicable_ids,
        gated_out=frozenset(c.id for c in section.criteria) - applicable_ids,
        calls=calls,
        requires_justification=section.requires_justification,
    )


class CriterionResponseWithJustification(BaseModel):
    """Per-criterion value for sections requiring justification."""

    justification: str
    response: Literal["Yes", "No"]
    severity: Literal["Minor", "Major"] | None = None


def _evidence_item_model(criterion) -> type[BaseModel]:
    """Item model for one criterion's ``evidence_items`` declaration."""
    item_fields = {
        prop: (str if spec == "str" else Literal[tuple(spec)], ...)
        for prop, spec in criterion.evidence_items.items()
    }
    return create_model(
        f"EvidenceItem{re.sub(r'[^a-zA-Z0-9]', '', criterion.id).title()}",
        **item_fields,
    )


def _criterion_evidence_model(criterion) -> type[BaseModel]:
    """Response model for one evidence-flagged criterion."""
    response_type = (
        Literal["Yes", "No", "N/A"] if criterion.allows_na else Literal["Yes", "No"]
    )
    evidence_type: Any = (
        list[_evidence_item_model(criterion)] if criterion.evidence_items else str
    )
    return create_model(
        f"Evidence{re.sub(r'[^a-zA-Z0-9]', '', criterion.id).title()}",
        evidence=(evidence_type, ...),
        response=(response_type, ...),
    )


def _model_name(section_name: str) -> str:
    return re.sub(r"\W+", "", section_name).replace("and", "And") + "Response"


def create_section_response_model(section_name: str, criteria: list) -> type[BaseModel]:
    """Build the response model for a call: one field per criterion id.

    Criteria flagged ``evidence: true`` get a ``{response, evidence}``
    object field; all others stay plain Yes/No(/N/A).
    """
    fields: dict[str, Any] = {}
    for criterion in criteria:
        if criterion.criterion_type == CriterionType.EVIDENCE_LIST:
            if criterion.evidence_fields:
                obj_fields = {name: (str, ...) for name in criterion.evidence_fields}
                if criterion.evidence_items:
                    obj_fields["items"] = (
                        list[_evidence_item_model(criterion)],
                        ...,
                    )
                model_name = (
                    f"EvidenceList{re.sub(r'[^a-zA-Z0-9]', '', criterion.id).title()}"
                )
                fields[criterion.id] = (
                    create_model(
                        model_name,
                        **obj_fields,
                    ),
                    ...,
                )
            else:
                fields[criterion.id] = (
                    list[_evidence_item_model(criterion)],
                    ...,
                )
        elif criterion.evidence:
            fields[criterion.id] = (_criterion_evidence_model(criterion), ...)
        else:
            field_type = (
                Literal["Yes", "No", "N/A"]
                if criterion.allows_na
                else Literal["Yes", "No"]
            )
            fields[criterion.id] = (field_type, ...)
    return create_model(_model_name(section_name), **fields)


def create_section_response_model_with_justification(
    section_name: str, criteria: list
) -> type[BaseModel]:
    """Response model for sections where each criterion needs a rationale."""
    fields: dict[str, Any] = {
        criterion.id: (CriterionResponseWithJustification, ...)
        for criterion in criteria
    }
    return create_model(_model_name(section_name), **fields)


def build_section_prompt(
    section_name: str,
    criteria: list,
    *,
    text: str,
    section_preface: str | None = None,
    requires_justification: bool = False,
    context: str | None = None,
    subject: str = "a text",
) -> str:
    """Build the prompt for one rubric section over a text."""
    preface_section = ""
    if section_preface:
        preface_section = f"## Scoring Guide\n\n{section_preface}\n\n---\n\n"

    context_section = f"{context}\n\n" if context else ""

    return f"""{context_section}You are evaluating {subject}.

{preface_section}TEXT TO EVALUATE:

{text}

Answer ALL of the following questions for the section "{section_name}".

{_criteria_text(criteria)}

{_response_instruction(criteria, requires_justification)}"""


def _criteria_text(criteria: list) -> str:
    """The numbered ``[id] question`` lines the criteria schema answers."""
    return "\n".join(
        f"{i + 1}. [{c.id}] {c.question}"
        + (" *(include evidence)*" if c.evidence else "")
        for i, c in enumerate(criteria)
    )


def _response_instruction(criteria: list, requires_justification: bool) -> str:
    """The response-shape instructions matching the built response schema."""
    if requires_justification:
        return (
            "For each criterion:\n"
            '- Respond with "Yes" or "No"\n'
            "- Provide a brief justification for your selection\n"
            '- If "No", specify whether this is a "Minor" or "Major" issue\n\n'
            "Respond with a JSON object where each key is the criterion ID and "
            'the value is an object containing "justification", "response", and '
            '"severity" (null if response is "Yes").'
        )
    any_allows_na = any(c.allows_na for c in criteria)
    any_evidence = any(c.evidence for c in criteria)
    example_id = criteria[0].id if criteria else "criterion_1"
    na_note = ' or "N/A" (N/A only where indicated as allowed)' if any_allows_na else ""
    response_instruction = (
        f'For each criterion, respond with exactly "Yes" or "No"{na_note}.\n\n'
        "Respond with a JSON object where each key is the criterion ID "
        f'(e.g., "{example_id}") and the value is your response.'
    )
    if any(c.criterion_type == CriterionType.EVIDENCE_LIST for c in criteria):
        response_instruction += (
            " For criteria that ask for a list, the value is instead a "
            "JSON array of objects with the properties named in the "
            "question (empty array if none found)."
        )
        if any(c.evidence_fields for c in criteria):
            response_instruction += (
                " Where the criterion also asks for other named output, "
                "the value is an object {field_name: <value>, ..., "
                '"items": <the array>} instead.'
            )
    if any_evidence:
        response_instruction += (
            "\n\nFor criteria marked *(include evidence)*, the value is an "
            'object {"evidence": <the words or '
            "sentences from the text that determine your answer>, "
            '"response": <your answer>} instead.'
        )
        if any(c.evidence_items for c in criteria):
            response_instruction += (
                " Where the criterion asks for a list, "
                '"evidence" is a JSON array of objects with the properties '
                "named in the question (empty array if none found)."
            )
    return response_instruction


def _render_template(
    template: str, placeholders: Mapping[str, str], what: str = "rubric text"
) -> str:
    """Fill ``{name}`` slots in rubric text.

    A slot with no caller-supplied value raises ``ValueError`` — an unfilled
    placeholder is a rubric/config bug, not literal text. ``{{name}}``
    renders as a literal ``{name}`` for questions that need real braces.
    """
    protected = re.sub(r"\{\{(\w+)\}\}", "\x00\\1\x01", template)
    names = set(re.findall(r"\{(\w+)\}", protected))
    missing = sorted(names - set(placeholders))
    if missing:
        raise ValueError(f"{what}: unfilled placeholder(s) {missing}")
    filled = re.sub(r"\{(\w+)\}", lambda m: str(placeholders[m.group(1)]), protected)
    return filled.replace("\x00", "{").replace("\x01", "}")
