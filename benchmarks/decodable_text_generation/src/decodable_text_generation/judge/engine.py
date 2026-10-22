"""Section-batched LLM judging of generated text against a rubric.

Adapted from evaluate-material-cli's ``base_scorer``, with the material
changed from an attached PDF to text embedded in the prompt. Each rubric
section is evaluated in a single LLM call whose response schema is a
dynamically built pydantic model — one field per criterion id. Criteria
whose gating does not match the story's tags are excluded from the prompt
and recorded as ``gated_out``.
"""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Literal

from phonics_benchmark.llm import call_llm
from pydantic import BaseModel, create_model

from decodable_text_generation.judge.rubric import (
    CriterionType,
    Section,
    filter_section_criteria,
)

logger = logging.getLogger(__name__)

JUDGE_MAX_TOKENS = 16384
GATE_MAX_TOKENS = 2048
MAX_RETRIES = 2


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
    """Dynamically create a pydantic model: one field per criterion id.

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
) -> str:
    """Build the prompt for one rubric section over a generated text."""
    criteria_text = "\n".join(
        f"{i + 1}. [{c.id}] {c.question}"
        + (" *(include evidence)*" if c.evidence else "")
        for i, c in enumerate(criteria)
    )

    preface_section = ""
    if section_preface:
        preface_section = f"## Scoring Guide\n\n{section_preface}\n\n---\n\n"

    if requires_justification:
        response_instruction = (
            "For each criterion:\n"
            '- Respond with "Yes" or "No"\n'
            "- Provide a brief justification for your selection\n"
            '- If "No", specify whether this is a "Minor" or "Major" issue\n\n'
            "Respond with a JSON object where each key is the criterion ID and "
            'the value is an object containing "justification", "response", and '
            '"severity" (null if response is "Yes").'
        )
    else:
        any_allows_na = any(c.allows_na for c in criteria)
        any_evidence = any(c.evidence for c in criteria)
        example_id = criteria[0].id if criteria else "criterion_1"
        na_note = (
            ' or "N/A" (N/A only where indicated as allowed)' if any_allows_na else ""
        )
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

    context_section = f"{context}\n\n" if context else ""

    return f"""{context_section}You are evaluating a decodable text for an early-grade
reader.

{preface_section}TEXT TO EVALUATE:

{text}

Answer ALL of the following questions for the section "{section_name}".

{criteria_text}

{response_instruction}"""


def _parse_response(response_text: str, schema: type[BaseModel]) -> dict[str, Any]:
    """Validate the judge's JSON reply, tolerating surrounding prose."""
    try:
        return schema.model_validate_json(response_text).model_dump()
    except Exception:
        match = re.search(r"\{.*\}", response_text, re.DOTALL)
        if not match:
            raise
        return schema.model_validate_json(match.group(0)).model_dump()


def _call_and_parse(
    prompt: str,
    judge_model: str,
    response_schema: type[BaseModel],
    max_tokens: int,
    context_label: str,
    max_retries: int = MAX_RETRIES,
    usage: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Call the judge model and validate the response, with retries.

    When ``usage`` is given, each completed LLM call appends its raw token
    counts so the caller can report per-call input/output usage.
    """
    last_exc: Exception | None = None
    for attempt in range(1 + max_retries):
        try:
            record = call_llm(
                prompt,
                model=judge_model,
                max_tokens=max_tokens,
                should_log=False,
                response_schema=response_schema,
            )
            if usage is not None:
                usage.append(
                    {
                        "call": context_label,
                        "tokens_used": record["tokens_used"],
                        "tokens_used_completion": record["tokens_used_completion"],
                        "tokens_used_reasoning": record["tokens_used_reasoning"],
                        "latency_seconds": record["latency_seconds"],
                        "model_version": record.get("model_version"),
                    }
                )
            return _parse_response(record["response_text"], response_schema)
        except Exception as exc:
            last_exc = exc
            logger.warning(
                "%s: judge call failed on attempt %d/%d: %s: %s",
                context_label,
                attempt + 1,
                1 + max_retries,
                type(exc).__name__,
                exc,
            )
        if attempt < max_retries:
            time.sleep(2 ** (attempt + 1))
    raise last_exc  # type: ignore[misc]


class SectionGateResult(BaseModel):
    """Answer to a section-level gating question about the text."""

    applies: Literal["Yes", "No"]


def evaluate_section_gate(
    text: str,
    gating_question: str,
    *,
    judge_model: str,
    usage: list[dict[str, Any]] | None = None,
) -> SectionGateResult:
    """Ask the judge a yes/no gating question about the text."""
    prompt = f"""You are evaluating a decodable text written for an early-grade reader.

TEXT TO EVALUATE:

{text}

Question: {gating_question}

Respond in JSON format:
{{"applies": "Yes" or "No"}}"""
    data = _call_and_parse(
        prompt,
        judge_model,
        response_schema=SectionGateResult,
        max_tokens=GATE_MAX_TOKENS,
        context_label=f"gating '{gating_question[:50]}'",
        usage=usage,
    )
    return SectionGateResult(**data)


def evaluate_section(
    text: str,
    section_name: str,
    criteria: list,
    *,
    judge_model: str,
    section_preface: str | None = None,
    requires_justification: bool = False,
    context: str | None = None,
    max_tokens: int = JUDGE_MAX_TOKENS,
    usage: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Evaluate all applicable criteria of one section in a single call."""
    prompt = build_section_prompt(
        section_name,
        criteria,
        text=text,
        section_preface=section_preface,
        requires_justification=requires_justification,
        context=context,
    )
    response_model = (
        create_section_response_model_with_justification(section_name, criteria)
        if requires_justification
        else create_section_response_model(section_name, criteria)
    )
    result_data = _call_and_parse(
        prompt,
        judge_model,
        response_schema=response_model,
        max_tokens=max_tokens,
        context_label=f"section '{section_name}'",
        usage=usage,
    )

    results = []
    for criterion in criteria:
        criterion_data = result_data.get(criterion.id)
        if requires_justification:
            entry = {
                "response": (
                    criterion_data.get("response", "No")
                    if isinstance(criterion_data, dict)
                    else "No"
                ),
                "justification": (
                    criterion_data.get("justification", "")
                    if isinstance(criterion_data, dict)
                    else ""
                ),
                "severity": (
                    criterion_data.get("severity")
                    if isinstance(criterion_data, dict)
                    else None
                ),
            }
        elif criterion.criterion_type == CriterionType.EVIDENCE_LIST:
            entry = {"response": None, "evidence": []}
            if isinstance(criterion_data, dict):
                # evidence_fields criteria return {**scalar fields, "items": [...]}
                entry["fields"] = {
                    k: criterion_data.get(k) for k in criterion.evidence_fields or {}
                }
                items = criterion_data.get("items")
                entry["evidence"] = items if isinstance(items, list) else []
            elif isinstance(criterion_data, list):
                entry["evidence"] = criterion_data
        elif criterion.evidence:
            entry = {
                "response": (
                    criterion_data.get("response", "No")
                    if isinstance(criterion_data, dict)
                    else "No"
                ),
                "evidence": (
                    criterion_data.get("evidence", "")
                    if isinstance(criterion_data, dict)
                    else ""
                ),
            }
        else:
            entry = {
                "response": criterion_data if criterion_data is not None else "No",
            }
        results.append(_criterion_entry(criterion, **entry))
    return results


def _criterion_entry(criterion, **extra) -> dict[str, Any]:
    """A stored criterion result, self-describing for offline rescoring."""
    entry = {
        "criterion_id": criterion.id,
        "question": criterion.question,
        "criterion_type": criterion.criterion_type.value,
        **extra,
    }
    if criterion.scoring_rules is not None:
        entry["scoring_rules"] = criterion.scoring_rules
    return entry


def _na_result(criterion) -> dict[str, Any]:
    return _criterion_entry(criterion, response="N/A")


def _gated_result(criterion) -> dict[str, Any]:
    return _criterion_entry(criterion, response="N/A", gated_out=True)


def _render_question(question: str, placeholders: Mapping[str, str]) -> str:
    """Fill ``{name}`` slots in a criterion question; unknown slots stay literal."""
    return re.sub(
        r"\{(\w+)\}",
        lambda m: placeholders.get(m.group(1), m.group(0)),
        question,
    )


def _evaluate_one_section(
    text: str,
    section_name: str,
    section: Section,
    *,
    tags: set[str],
    judge_model: str,
    context: str | None,
    max_tokens: int,
    placeholders: Mapping[str, str] | None = None,
) -> tuple[str, list[dict], bool, list[dict]]:
    """Evaluate one section: gating filter, optional section gate, LLM call.

    Returns (section_name, ordered_results, is_failed, token_usage). Results
    are ordered to match the rubric's criterion list so gated-out criteria
    keep their original position. ``token_usage`` carries one entry per
    completed judge call (section call plus any gate call).
    """
    applicable_ids = {c.id for c in filter_section_criteria(section, tags)}
    judged: dict[str, dict] = {}
    failed = False
    usage: list[dict] = []

    if applicable_ids:
        if section.gating_question:
            try:
                gate = evaluate_section_gate(
                    text,
                    section.gating_question,
                    judge_model=judge_model,
                    usage=usage,
                )
            except Exception:
                logger.exception("Section '%s': gating evaluation failed", section_name)
                return (
                    section_name,
                    [_na_result(c) for c in section.criteria],
                    True,
                    usage,
                )
            if gate.applies == "No":
                judged = {c.id: _na_result(c) for c in section.criteria}
        if not judged:
            applicable = [c for c in section.criteria if c.id in applicable_ids]
            if placeholders:
                applicable = [
                    c.model_copy(
                        update={"question": _render_question(c.question, placeholders)}
                    )
                    for c in applicable
                ]
            try:
                section_results = evaluate_section(
                    text,
                    section_name,
                    applicable,
                    judge_model=judge_model,
                    section_preface=section.section_preface,
                    requires_justification=section.requires_justification,
                    context=context,
                    max_tokens=max_tokens,
                    usage=usage,
                )
                judged = {r["criterion_id"]: r for r in section_results}
            except Exception:
                logger.exception(
                    "Section '%s': evaluation failed after retries", section_name
                )
                judged = {
                    c.id: _na_result(c)
                    for c in section.criteria
                    if c.id in applicable_ids
                }
                failed = True

    results = [
        judged[c.id]
        if c.id in judged
        else (_na_result(c) if c.id in applicable_ids else _gated_result(c))
        for c in section.criteria
    ]
    for call in usage:
        call["section"] = section_name
    return (section_name, results, failed, usage)


def evaluate_text(
    text: str,
    *,
    rubric: dict[str, Section],
    tags: set[str],
    judge_model: str,
    context: str | None = None,
    max_tokens: int = JUDGE_MAX_TOKENS,
    verbose: bool = True,
    on_section_complete: Callable[[str, int, int], None] | None = None,
    placeholders: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Judge one text against every section of a rubric.

    Sections run concurrently. Returns the raw judgment record: per-section
    criterion responses (gated-out criteria included, flagged), the tags
    used for gating, section weights for later rescoring, and any failed
    sections.
    """
    results: dict[str, Any] = {
        "tags": sorted(tags),
        "section_weights": {name: s.weight for name, s in rubric.items()},
        "sections": {},
        "failed_sections": [],
        "token_usage": [],
    }
    total = len(rubric)
    completed = 0

    with ThreadPoolExecutor() as executor:
        futures = {
            executor.submit(
                _evaluate_one_section,
                text,
                name,
                section,
                tags=tags,
                judge_model=judge_model,
                context=context,
                max_tokens=max_tokens,
                placeholders=placeholders,
            ): name
            for name, section in rubric.items()
        }
        for future in as_completed(futures):
            section_name, section_results, is_failed, usage = future.result()
            results["sections"][section_name] = section_results
            results["token_usage"].extend(usage)
            if is_failed:
                results["failed_sections"].append(section_name)
            completed += 1
            if verbose:
                n_yes = sum(1 for r in section_results if r["response"] == "Yes")
                n_gated = sum(1 for r in section_results if r.get("gated_out"))
                status = "FAILED" if is_failed else f"{n_yes} Yes"
                if n_gated:
                    status += f", {n_gated} gated"
                print(
                    f"  [{completed}/{total}] {section_name}: {status}",
                    flush=True,
                )
            if on_section_complete:
                on_section_complete(section_name, completed, total)

    return results
