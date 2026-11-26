"""Phases 2 and 3: execute a plan's calls, assemble the answers.

``execute_plan`` runs every call in the plan concurrently (plans are
unconditional). The LLM backend is injected: pass any callable
``llm(prompt, model=..., max_tokens=..., response_schema=...)`` that
returns a mapping with ``"response_text"``. Optional ``tokens_used``,
``tokens_used_completion``, ``tokens_used_reasoning``,
``latency_seconds`` and ``model_version`` keys are recorded verbatim in
``token_usage``. A ``fab_benchmark_runner`` run can replace the callable:
feed ``plan.inputs_rows(item_id)`` to the runner and wrap each row's
output in ``CallAnswer(response_text=...)``.

``assemble_judgment`` is pure phase 3: parsed (or raw-text) answers plus
the plan's rubric produce the judgment record ``evaluate_text`` writes,
which ``scoring.score_judgment`` then scores offline.
"""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Protocol

from pydantic import BaseModel

from rubric_judge.plan import (
    JUDGE_MAX_TOKENS,
    JudgeCall,
    JudgeItem,
    JudgePlan,
    build_section_prompt,
    create_section_response_model,
    create_section_response_model_with_justification,
    plan_judgment,
)
from rubric_judge.rubric import CriterionType, Section

logger = logging.getLogger(__name__)

MAX_RETRIES = 2


class JudgeLLM(Protocol):
    """The LLM backend ``execute_plan`` calls.

    Called as ``llm(prompt, model=..., max_tokens=..., response_schema=...)``;
    must return a mapping with ``"response_text"``. Optional ``tokens_used``,
    ``tokens_used_completion``, ``tokens_used_reasoning``,
    ``latency_seconds`` and ``model_version`` keys are reported in
    ``token_usage``. When a ``system_prompt`` is set it arrives as a
    ``system_prompt=...`` kwarg — backends that can't send one can fold it
    into the prompt themselves.
    """

    def __call__(
        self,
        prompt: str,
        *,
        model: str,
        max_tokens: int,
        response_schema: type[BaseModel],
        system_prompt: str | None = None,
    ) -> Mapping[str, Any]: ...


@dataclass
class CallAnswer:
    """One call's outcome: parsed data, or raw text to parse at assembly."""

    key: str
    data: dict[str, Any] | None = None
    response_text: str | None = None
    usage: dict[str, Any] | None = None
    error: str | None = None
    judge: str | None = None  # which ensemble member produced this answer


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
    *,
    llm: JudgeLLM,
    max_retries: int = MAX_RETRIES,
    system_prompt: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Call the judge model, validate the response and return (data, usage).

    The usage record holds the call's label plus whatever token/latency
    fields the ``llm`` record supplied. ``system_prompt`` is only passed to
    the backend when set, so plain single-message backends keep working.
    """
    last_exc: Exception | None = None
    for attempt in range(1 + max_retries):
        try:
            record = llm(
                prompt,
                model=judge_model,
                max_tokens=max_tokens,
                response_schema=response_schema,
                **({"system_prompt": system_prompt} if system_prompt else {}),
            )
            usage = {
                "call": context_label,
                "tokens_used": record.get("tokens_used"),
                "tokens_used_completion": record.get("tokens_used_completion"),
                "tokens_used_reasoning": record.get("tokens_used_reasoning"),
                "latency_seconds": record.get("latency_seconds"),
                "model_version": record.get("model_version"),
            }
            return _parse_response(record["response_text"], response_schema), usage
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


def evaluate_section(
    text: str,
    section_name: str,
    criteria: list,
    *,
    judge_model: str,
    llm: JudgeLLM,
    section_preface: str | None = None,
    requires_justification: bool = False,
    context: str | None = None,
    subject: str = "a text",
    max_tokens: int = JUDGE_MAX_TOKENS,
    usage: list[dict[str, Any]] | None = None,
    call_label: str | None = None,
    system_prompt: str | None = None,
) -> list[dict[str, Any]]:
    """Evaluate all applicable criteria of one section in a single call."""
    prompt = build_section_prompt(
        section_name,
        criteria,
        text=text,
        section_preface=section_preface,
        requires_justification=requires_justification,
        context=context,
        subject=subject,
    )
    response_model = (
        create_section_response_model_with_justification(section_name, criteria)
        if requires_justification
        else create_section_response_model(section_name, criteria)
    )
    result_data, call_usage = _call_and_parse(
        prompt,
        judge_model,
        response_model,
        max_tokens,
        call_label or f"section '{section_name}'",
        llm=llm,
        system_prompt=system_prompt,
    )
    if usage is not None:
        usage.append(call_usage)
    return _criteria_entries(criteria, result_data, requires_justification)


def _criteria_entries(
    criteria: list, result_data: dict[str, Any], requires_justification: bool
) -> list[dict[str, Any]]:
    """Map one call's parsed fields to stored criterion result rows."""
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
    if criterion.weight != 1.0:
        entry["weight"] = criterion.weight
    return entry


def _na_result(criterion) -> dict[str, Any]:
    return _criterion_entry(criterion, response="N/A")


def _gated_result(criterion) -> dict[str, Any]:
    return _criterion_entry(criterion, response="N/A", gated_out=True)


def execute_plan(
    plan: JudgePlan,
    *,
    llm: JudgeLLM,
    judge_model: str,
    max_retries: int = MAX_RETRIES,
    on_call_done: Callable[[JudgeCall, CallAnswer], None] | None = None,
    max_workers: int | None = None,
) -> dict[str, CallAnswer]:
    """Run every call in ``plan``, returning answers keyed by ``call.key``.

    Every call in the plan runs concurrently — plans are unconditional, so
    the whole thing batches in one wave. A call that fails after retries
    gets ``error`` set instead of ``data``.
    """
    return _execute(
        plan,
        {judge_model: llm},
        max_retries=max_retries,
        on_call_done=on_call_done,
        max_workers=max_workers,
    )[judge_model]


def execute_ensemble(
    plan: JudgePlan,
    *,
    judges: Mapping[str, JudgeLLM],
    max_retries: int = MAX_RETRIES,
    on_call_done: Callable[[JudgeCall, CallAnswer], None] | None = None,
    max_workers: int | None = None,
) -> dict[str, dict[str, CallAnswer]]:
    """Run every call with every judge; answers are ``judge -> key -> CallAnswer``.

    Same semantics as ``execute_plan``, one execution per ensemble member;
    all call×judge pairs run in one concurrent wave.
    """
    return _execute(
        plan,
        judges,
        max_retries=max_retries,
        on_call_done=on_call_done,
        max_workers=max_workers,
    )


def _execute(
    plan: JudgePlan,
    judges: Mapping[str, JudgeLLM],
    *,
    max_retries: int,
    on_call_done: Callable[[JudgeCall, CallAnswer], None] | None,
    max_workers: int | None,
) -> dict[str, dict[str, CallAnswer]]:
    """Run every plan call under every judge concurrently."""
    answers: dict[str, dict[str, CallAnswer]] = {judge: {} for judge in judges}
    by_key = {c.key: c for c in plan.calls}

    def _run(call: JudgeCall, judge: str) -> CallAnswer:
        try:
            data, usage = _call_and_parse(
                call.prompt,
                judge,
                call.response_schema,
                call.max_tokens,
                call.label,
                llm=judges[judge],
                max_retries=max_retries,
                system_prompt=call.system_prompt,
            )
            return CallAnswer(key=call.key, judge=judge, data=data, usage=usage)
        except Exception as exc:
            logger.exception("%s [%s]: call failed after retries", call.label, judge)
            return CallAnswer(key=call.key, judge=judge, error=str(exc))

    def _run_job(job: tuple[JudgeCall, str]) -> tuple[JudgeCall, str, CallAnswer]:
        call, judge = job
        return call, judge, _run(call, judge)

    jobs = [(call, judge) for call in plan.calls for judge in judges]
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for call, judge, answer in executor.map(_run_job, jobs):
            answers[judge][call.key] = answer
            if on_call_done:
                on_call_done(by_key[call.key], answer)
    return answers


def _answer_data(call: JudgeCall, answer: CallAnswer | None) -> dict[str, Any] | None:
    """The parsed fields for a call, parsing raw text answers lazily."""
    if answer is None or answer.error is not None:
        return None
    if answer.data is not None:
        return answer.data
    if answer.response_text is not None:
        return _parse_response(answer.response_text, call.response_schema)
    return None


def assemble_judgment(
    plan: JudgePlan,
    answers: Mapping[str, CallAnswer],
    *,
    verbose: bool = False,
    on_section_complete: Callable[[str, int, int], None] | None = None,
) -> dict[str, Any]:
    """Assemble call answers into the judgment record for one item.

    The same shape ``evaluate_text`` writes: per-section criterion rows
    (gated-out criteria included, flagged), ``failed_sections``,
    ``token_usage`` with per-call records, ``tags`` and ``section_weights``.
    """
    results: dict[str, Any] = {
        "tags": sorted(plan.item.tags),
        "section_weights": {name: s.weight for name, s in plan.rubric.items()},
        "sections": {},
        "failed_sections": [],
        "token_usage": [],
    }
    total = len(plan.sections)
    completed = 0
    for section in plan.sections:
        rows, failed, usage = _assemble_section(section, answers)
        results["sections"][section.name] = rows
        results["token_usage"].extend(usage)
        if failed:
            results["failed_sections"].append(section.name)
        completed += 1
        if verbose:
            n_yes = sum(1 for r in rows if r["response"] == "Yes")
            n_gated = sum(1 for r in rows if r.get("gated_out"))
            status = "FAILED" if failed else f"{n_yes} Yes"
            if n_gated:
                status += f", {n_gated} gated"
            print(f"  [{completed}/{total}] {section.name}: {status}", flush=True)
        if on_section_complete:
            on_section_complete(section.name, completed, total)
    return results


def assemble_ensemble(
    plan: JudgePlan,
    answers_by_judge: Mapping[str, Mapping[str, CallAnswer]],
    *,
    primary: str | None = None,
    verbose: bool = False,
    on_section_complete: Callable[[str, int, int], None] | None = None,
) -> dict[str, Any]:
    """Assemble each judge's answers, then majority-vote them into one record.

    ``primary`` (default: the first judge) names the member whose other
    fields (evidence, justification, severity) the merged rows keep when
    its own vote wins; otherwise they come from the first judge voting
    with the majority.
    """
    judgments = {
        judge: assemble_judgment(plan, answers)
        for judge, answers in answers_by_judge.items()
    }
    merged = merge_ensemble(judgments, primary=primary)
    total = len(merged["sections"])
    for completed, (name, rows) in enumerate(merged["sections"].items(), start=1):
        if verbose:
            failed = name in merged["failed_sections"]
            n_yes = sum(1 for r in rows if r["response"] == "Yes")
            status = "FAILED" if failed else f"{n_yes} Yes"
            print(f"  [{completed}/{total}] {name}: {status}", flush=True)
        if on_section_complete:
            on_section_complete(name, completed, total)
    return merged


def merge_ensemble(
    judgments: Mapping[str, dict[str, Any]], *, primary: str | None = None
) -> dict[str, Any]:
    """Merge same-plan judgments from several judges by majority vote.

    Per criterion the merged ``response`` is the majority over judges whose
    row answered Yes/No; N/A rows (errors, gated-out, allowed N/A) don't
    vote. Ties and all-N/A rows keep ``"N/A"``, with ``"tie"`` set when the
    tie was among responders. The merged row's other fields come from the
    ``primary`` judge when it agrees with the verdict, else from the first
    judge that does. ``votes`` and ``agreement`` record the breakdown.
    Rows without a Yes/No verdict (evidence lists) pass through unchanged
    from the primary judge.
    """
    judges = list(judgments)
    primary = primary or judges[0]
    merged = dict(judgments[primary])
    merged["judges"] = judges
    merged["failed_sections"] = [
        name
        for name in judgments[primary]["failed_sections"]
        if all(name in j["failed_sections"] for j in judgments.values())
    ]
    merged["sections"] = {
        section_name: [
            _merge_row(
                {
                    judge: judgments[judge]["sections"][section_name][i]
                    for judge in judges
                },
                judges,
                primary,
            )
            for i in range(len(judgments[primary]["sections"][section_name]))
        ]
        for section_name in judgments[primary]["sections"]
    }
    merged["token_usage"] = [
        {**record, "judge": judge}
        for judge in judges
        for record in judgments[judge]["token_usage"]
    ]
    return merged


def _merge_row(
    rows: Mapping[str, dict[str, Any]], judges: list[str], primary: str
) -> dict[str, Any]:
    """One criterion's merged row from each judge's version of it."""
    votes = {judge: rows[judge].get("response") for judge in judges}
    responders = [v for v in votes.values() if v in ("Yes", "No")]
    if len(responders) < 2:
        return dict(rows[primary])
    n_yes = responders.count("Yes")
    verdict = (
        "Yes"
        if n_yes * 2 > len(responders)
        else "No"
        if n_yes * 2 < len(responders)
        else None
    )
    donor = primary
    if verdict is not None and votes[primary] != verdict:
        donor = next(judge for judge in judges if votes[judge] == verdict)
    row = dict(rows[donor])
    row["response"] = verdict if verdict is not None else "N/A"
    row["votes"] = votes
    row["agreement"] = max(n_yes, len(responders) - n_yes) / len(responders)
    if verdict is None:
        row["tie"] = True
    return row


def evaluate_ensemble(
    item: JudgeItem,
    *,
    rubric: dict[str, Section],
    judges: Mapping[str, JudgeLLM],
    ordered_tags: Sequence[str] = (),
    subject: str = "a text",
    max_tokens: int = JUDGE_MAX_TOKENS,
    system_prompt: str | None = None,
    primary: str | None = None,
    verbose: bool = True,
    on_section_complete: Callable[[str, int, int], None] | None = None,
) -> dict[str, Any]:
    """Judge one item with several judge models, merged by majority vote.

    One plan is built once and run per judge; the merged record carries
    ``votes``/``agreement`` per criterion and a ``judges`` list. ``primary``
    picks which judge's non-verdict fields the merged rows inherit.
    """
    plan = plan_judgment(
        item,
        rubric=rubric,
        ordered_tags=ordered_tags,
        subject=subject,
        max_tokens=max_tokens,
        system_prompt=system_prompt,
    )
    answers = execute_ensemble(plan, judges=judges)
    return assemble_ensemble(
        plan,
        answers,
        primary=primary,
        verbose=verbose,
        on_section_complete=on_section_complete,
    )


def _assemble_section(
    section, answers: Mapping[str, CallAnswer]
) -> tuple[list[dict[str, Any]], bool, list[dict[str, Any]]]:
    """One section's rows, failed flag and usage records from call answers."""
    judged: dict[str, dict] = {}
    failed = False
    usage: list[dict[str, Any]] = []
    for call in section.calls:
        answer = answers.get(call.key)
        if answer is not None and answer.usage is not None:
            usage.append({**answer.usage, "section": section.name})

    for call in section.calls:
        data = _answer_data(call, answers.get(call.key))
        if data is None:
            failed = True
            batch_ids = set(call.criterion_ids)
            judged.update(
                {c.id: _na_result(c) for c in section.criteria if c.id in batch_ids}
            )
            continue
        by_id = {c.id: c for c in section.criteria}
        batch = [section.rendered.get(cid, by_id[cid]) for cid in call.criterion_ids]
        judged.update(
            {
                r["criterion_id"]: r
                for r in _criteria_entries(batch, data, section.requires_justification)
            }
        )

    rows = [
        judged[c.id]
        if c.id in judged
        else (_na_result(c) if c.id in section.applicable_ids else _gated_result(c))
        for c in section.criteria
    ]
    return (rows, failed, usage)


def evaluate_item(
    item: JudgeItem,
    *,
    rubric: dict[str, Section],
    judge_model: str,
    llm: JudgeLLM,
    ordered_tags: Sequence[str] = (),
    subject: str = "a text",
    max_tokens: int = JUDGE_MAX_TOKENS,
    system_prompt: str | None = None,
    verbose: bool = True,
    on_section_complete: Callable[[str, int, int], None] | None = None,
) -> dict[str, Any]:
    """Judge one item: plan the calls, run them, assemble the judgment."""
    plan = plan_judgment(
        item,
        rubric=rubric,
        ordered_tags=ordered_tags,
        subject=subject,
        max_tokens=max_tokens,
        system_prompt=system_prompt,
    )
    answers = execute_plan(plan, llm=llm, judge_model=judge_model)
    return assemble_judgment(
        plan, answers, verbose=verbose, on_section_complete=on_section_complete
    )


def evaluate_text(
    text: str,
    *,
    rubric: dict[str, Section],
    tags: set[str],
    judge_model: str,
    llm: JudgeLLM,
    ordered_tags: Sequence[str] = (),
    subject: str = "a text",
    context: str | None = None,
    max_tokens: int = JUDGE_MAX_TOKENS,
    system_prompt: str | None = None,
    verbose: bool = True,
    on_section_complete: Callable[[str, int, int], None] | None = None,
    placeholders: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Judge one text against every section of a rubric.

    Thin wrapper over the plan -> execute -> assemble pipeline; builds the
    ``JudgeItem`` from the separate arguments for callers that don't plan.
    """
    return evaluate_item(
        JudgeItem(
            text=text,
            tags=frozenset(tags),
            placeholders=placeholders,
            context=context,
        ),
        rubric=rubric,
        judge_model=judge_model,
        llm=llm,
        ordered_tags=ordered_tags,
        subject=subject,
        max_tokens=max_tokens,
        system_prompt=system_prompt,
        verbose=verbose,
        on_section_complete=on_section_complete,
    )
