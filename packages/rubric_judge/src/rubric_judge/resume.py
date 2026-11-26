"""Prompt-granular resume for judge runs.

Judgment rows carry ``call_shas`` — one fingerprint per judge call, from
:meth:`rubric_judge.plan.JudgePlan.call_fingerprints` — so a rerun re-runs
only the calls whose prompts changed. Because fingerprints hash exactly
what the model is asked (rendered prompt, response schema, system prompt,
criterion ids), editing question text, gating, placeholders, context,
subject or the system prompt invalidates the affected calls, while
scoring-only edits (``scoring_rules``, ``weight``) never do.

The flow: :func:`partition_calls` splits a plan's calls into (reusable,
stale) against a cached row; :func:`filter_calls` restricts the plan to
the stale calls for re-judging; :func:`merge_judgment` combines the cached
criterion rows with the fresh judgment.

Legacy rows (no ``call_shas``) get a conservative fallback: a call is
reusable iff the stored rows for its criteria match the current question
texts and types exactly, and none of its criteria need evidence (old rows
cannot contain evidence answers).
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from rubric_judge.plan import JudgePlan


def partition_calls(
    row: dict[str, Any],
    call_shas: dict[str, str],
    plan: JudgePlan,
) -> tuple[set[str], set[str]]:
    """Split the plan's calls into (reusable, stale) for a cached judgment.

    ``call_shas`` is the current item's per-call fingerprint map
    (``plan.call_fingerprints()``); a call is reusable iff the row's stored
    ``call_shas`` entry matches it.
    """
    stored_shas = row.get("call_shas")
    reusable, stale = set(), set()
    for section in plan.sections:
        if stored_shas is not None:
            for call in section.calls:
                (
                    reusable
                    if stored_shas.get(call.key) == call_shas.get(call.key)
                    else stale
                ).add(call.key)
            continue
        stored_rows = {
            r["criterion_id"]: r for r in row.get("sections", {}).get(section.name, [])
        }
        for call in section.calls:
            (
                reusable if _call_reusable_legacy(call, section, stored_rows) else stale
            ).add(call.key)
    return reusable, stale


def _call_reusable_legacy(call, section, stored_rows: dict[str, dict]) -> bool:
    """Conservative reuse for rows stored before ``call_shas`` existed."""
    by_id = {c.id: c for c in section.criteria}
    for cid in call.criterion_ids:
        criterion = by_id[cid]
        if criterion.evidence:
            return False
        r = stored_rows.get(cid)
        if (
            r is None
            or r.get("question") != criterion.question
            or r.get("criterion_type") != criterion.criterion_type.value
        ):
            return False
    return True


def filter_calls(plan: JudgePlan, keys: set[str]) -> JudgePlan:
    """A copy of ``plan`` keeping only the named calls in each section."""
    return replace(
        plan,
        sections=[
            replace(s, calls=[c for c in s.calls if c.key in keys])
            for s in plan.sections
        ],
    )


def merge_judgment(
    cached_row: dict[str, Any],
    fresh: dict[str, Any],
    plan: JudgePlan,
    reusable: set[str],
) -> dict[str, Any]:
    """Combine cached criterion rows with freshly judged ones.

    ``fresh`` is the assemble_judgment result over the stale-call subset
    produced by :func:`filter_calls`; ``reusable`` is the set of call keys
    whose criterion rows come from ``cached_row`` instead.
    """
    reusable_cids = {
        cid
        for section in plan.sections
        for call in section.calls
        if call.key in reusable
        for cid in call.criterion_ids
    }
    merged = dict(fresh)
    sections = {}
    for section in plan.sections:
        cached_rows = {
            r["criterion_id"]: r
            for r in cached_row.get("sections", {}).get(section.name, [])
        }
        fresh_rows = {
            r["criterion_id"]: r
            for r in fresh.get("sections", {}).get(section.name, [])
        }
        rows = [
            cached_rows[c.id]
            if c.id in reusable_cids and c.id in cached_rows
            else fresh_rows.get(c.id, cached_rows.get(c.id))
            for c in section.criteria
        ]
        sections[section.name] = [r for r in rows if r is not None]
    merged["sections"] = sections
    merged["section_weights"] = {name: s.weight for name, s in plan.rubric.items()}
    merged["reused_calls"] = sorted(reusable)
    return merged
