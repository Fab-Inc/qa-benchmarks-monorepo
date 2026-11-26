"""Compute section and final scores from stored rubric judgments.

Judgment rows written by the judge carry every criterion's raw response,
its ``criterion_type``, a ``gated_out`` flag, and a ``section_weights``
snapshot, so scores can be recomputed offline with different weights —
without any further LLM calls.

Scoring rules (mirroring evaluate-material-cli's ``score_calculator``):

- ``positive`` criteria score ``yes_count / applicable_count * 100``;
  ``N/A`` and ``gated_out`` responses are excluded from the denominator.
- A ``fail_condition`` answered ``"Yes"`` forces the section score to 0.
- An ``evidence_list`` criterion carries no verdict; instead each of its
  stored ``scoring_rules`` is one applicable point, earned when the count
  of items matching the rule's matcher satisfies ``less_than`` or
  ``greater_than``.
- A section's effective weight is its ``weight`` (overridable per rescore);
  weight 0 means judged and reported but excluded from the final score.
- The final score is the weighted mean of non-excluded, non-N/A sections,
  plus any deterministic extra section returned by the caller-supplied
  ``extra_section_scorer``.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import yaml

# Fallback for evidence_list rows judged before scoring_rules were stored:
# one point if no medium/high items, one point if none is high.
DEFAULT_EVIDENCE_RULES = [
    {"severity_in": ["medium", "high"], "less_than": 1},
    {"severity_in": ["high"], "less_than": 1},
]

# Optional deterministic extra section for score_judgment: called with the
# judgment row, returns ``(section_name, score, payload)`` or ``None``.
# ``payload`` is merged into the section_scores entry; a ``"weight"`` key in
# it sets the default weight (overridable via ``weights`` like any section).
ExtraSectionScorer = Callable[
    [dict[str, Any]], tuple[str, float | None, dict[str, Any]] | None
]


def _item_matches(item: Any, cond: dict[str, Any]) -> bool:
    """Whether an evidence item satisfies a ``where``/``max_count_where``
    condition: ``{"field": F, "in": V|list}`` or ``{"field": F, "not": V|list}``."""
    if not isinstance(item, dict):
        return False
    val = item.get(cond["field"])
    if "in" in cond:
        allowed = cond["in"]
        if not isinstance(allowed, list):
            allowed = [allowed]
        return val in allowed
    excluded = cond.get("not", ())
    if not isinstance(excluded, list):
        excluded = [excluded]
    return val not in excluded


def _match_count(rule: dict[str, Any], items: list) -> int:
    """How many evidence items match the rule's matcher (all when none)."""
    if "severity_in" in rule:
        allowed = set(rule["severity_in"])
        return sum(isinstance(i, dict) and i.get("severity") in allowed for i in items)
    if "where" in rule:
        return sum(_item_matches(i, rule["where"]) for i in items)
    return len(items)


def _rule_earned(rule: dict[str, Any], items: list, severities: set) -> bool:
    """Whether one evidence scoring rule is earned.

    Canonical form — ``{<optional matcher>, "less_than": N}`` /
    ``"greater_than": N`` — earned iff fewer/more than N items match.
    Both bounds may appear together, giving an exclusive range
    (``{less_than: 3, greater_than: 1}`` = exactly 2 matching items).
    Matchers: none (all items), ``severity_in`` (items whose ``severity``
    is in the set), ``where`` (a field condition).
    Legacy forms kept for rows stored before ``less_than`` existed:
    ``max_items``, ``severity_in`` alone (every item in the allowed set),
    ``max_count_where``.
    """
    if "less_than" in rule or "greater_than" in rule:
        count = _match_count(rule, items)
        if "less_than" in rule and count >= rule["less_than"]:
            return False
        if "greater_than" in rule and count <= rule["greater_than"]:
            return False
        return True
    if "max_count_where" in rule:
        cond = rule["max_count_where"]
        return sum(_item_matches(i, cond) for i in items) <= cond["count"]
    if "max_items" in rule:
        return len(items) <= rule["max_items"]
    if "severity_in" in rule:
        return (
            not items or all(isinstance(i, dict) and "severity" in i for i in items)
        ) and severities <= set(rule["severity_in"])
    return False


def score_section(
    results: list[dict[str, Any]],
    criterion_weights: dict[str, float] | None = None,
    evidence_rules: dict[str, list[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    """Score one section's criterion results.

    ``criterion_weights`` maps criterion_id -> weight, overriding the
    ``weight`` stored on the row (default 1.0); each applicable criterion
    or evidence scoring-rule point is worth its criterion's weight, so the
    section score is a weighted ``yes / applicable`` rate.
    ``evidence_rules`` maps criterion_id -> a
    replacement ``scoring_rules`` list, overriding the rules stored on
    the row (``[]`` removes the criterion's points; the stored legacy
    fallback is bypassed whenever the criterion is named).
    """
    yes_count = applicable_count = na_count = gated_out_count = 0
    triggered_fail: list[str] = []
    criterion_weights = criterion_weights or {}
    evidence_rules = evidence_rules or {}

    for result in results:
        response = result.get("response")
        weight = float(
            criterion_weights.get(result.get("criterion_id"), result.get("weight", 1.0))
        )
        if result.get("gated_out"):
            gated_out_count += 1
            continue
        if response == "N/A":
            na_count += 1
            continue
        if result.get("criterion_type") == "evidence_list":
            items = result.get("evidence") or []
            severities = {i.get("severity") for i in items if isinstance(i, dict)}
            cid = result.get("criterion_id")
            if cid in evidence_rules:
                rules = evidence_rules[cid]
            else:
                # scoring_rules absent on the row -> legacy fallback; explicit
                # empty list -> evidence-only, contributes no points.
                rules = result.get("scoring_rules")
                if rules is None:
                    rules = DEFAULT_EVIDENCE_RULES
            for rule in rules:
                applicable_count += weight
                if _rule_earned(rule, items, severities):
                    yes_count += weight
            continue
        if result.get("criterion_type") == "fail_condition":
            if response == "Yes":
                triggered_fail.append(result["criterion_id"])
            continue
        applicable_count += weight
        if response == "Yes":
            yes_count += weight

    if triggered_fail:
        score = 0.0
    elif applicable_count == 0:
        score = None
    else:
        score = 100 * yes_count / applicable_count

    return {
        "score": score,
        "yes_count": yes_count,
        "applicable_count": applicable_count,
        "na_count": na_count,
        "gated_out_count": gated_out_count,
        "triggered_fail_conditions": triggered_fail,
    }


def score_judgment(
    judgment: dict[str, Any],
    weights: dict[str, float] | None = None,
    criterion_weights: dict[str, float] | None = None,
    evidence_rules: dict[str, list[dict[str, Any]]] | None = None,
    extra_section_scorer: ExtraSectionScorer | None = None,
) -> dict[str, Any]:
    """Score one stored judgment row, optionally with weight overrides.

    ``weights`` maps section name -> weight and takes precedence over the
    ``section_weights`` snapshot stored on the row.
    ``criterion_weights`` maps criterion_id -> weight within its section;
    ``evidence_rules`` maps criterion_id -> replacement scoring_rules;
    ``extra_section_scorer`` may add one deterministic section (e.g.
    code-computed text measures) — see :data:`ExtraSectionScorer`.
    """
    overrides = weights or {}
    stored_weights = judgment.get("section_weights") or {}

    section_scores: dict[str, dict[str, Any]] = {}
    included: list[str] = []
    excluded: list[str] = []
    weighted_sum = 0.0
    weight_total = 0.0

    for name, results in (judgment.get("sections") or {}).items():
        section_score = score_section(results, criterion_weights, evidence_rules)
        weight = float(overrides.get(name, stored_weights.get(name, 1.0)))
        section_score["weight"] = weight
        section_scores[name] = section_score
        if weight <= 0 or section_score["score"] is None:
            excluded.append(name)
        else:
            included.append(name)
            weighted_sum += section_score["score"] * weight
            weight_total += weight

    if extra_section_scorer is not None:
        extra = extra_section_scorer(judgment)
        if extra is not None:
            extra_name, extra_score, extra_payload = extra
            payload = dict(extra_payload)
            weight = float(overrides.get(extra_name, payload.pop("weight", 1.0)))
            if extra_score is not None:
                section_scores[extra_name] = {
                    "score": extra_score,
                    **payload,
                    "weight": weight,
                }
                if weight <= 0:
                    excluded.append(extra_name)
                else:
                    included.append(extra_name)
                    weighted_sum += extra_score * weight
                    weight_total += weight

    final_score = weighted_sum / weight_total if weight_total else None
    return {
        "benchmark_item_id": judgment.get("benchmark_item_id"),
        "model": judgment.get("model"),
        "model_call_id": judgment.get("model_call_id"),
        "judge_model": judgment.get("judge_model"),
        "metrics": judgment.get("metrics") or {},
        "section_scores": section_scores,
        "final_score": final_score,
        "sections_included_in_final": included,
        "sections_excluded": excluded,
        "failed_sections": judgment.get("failed_sections") or [],
        "judge_error": judgment.get("judge_error") or "",
    }


def load_weights(path: str | Path) -> dict[str, float]:
    """Load a ``{SectionName: weight}`` YAML mapping for rescoring."""
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Weights file must map section names to numbers: {path}")
    return {str(name): float(weight) for name, weight in data.items()}


def iter_judgments(path: str | Path):
    """Yield judgment rows from a JSONL file."""
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            yield json.loads(line)


DEFAULT_SEVERITY_COSTS = {"low": 2.0, "medium": 6.0, "high": 15.0}


def deduction_score(
    judgment: dict[str, Any],
    severity_costs: dict[str, float] | None = None,
    per_section_costs: dict[str, dict[str, float]] | None = None,
    criterion_no_cost: float = 0.0,
    fail_condition_cost: float = 0.0,
    max_score: float = 100.0,
    min_score: float = 0.0,
) -> dict[str, Any]:
    """Alternate scoring family: start at ``max_score`` and deduct points.

    - every evidence item on an ``evidence_list`` criterion costs
      ``severity_costs[item['severity']]`` (items without a severity use
      the ``"none"`` bucket; ``per_section_costs`` overrides the global
      map for named sections)
    - every ``positive`` criterion answered ``"No"`` costs
      ``criterion_no_cost``
    - every triggered ``fail_condition`` costs ``fail_condition_cost``

    ``N/A`` and ``gated_out`` results contribute nothing. Returns the
    clamped score plus the deduction breakdown for inspection.
    """
    global_costs = dict(
        DEFAULT_SEVERITY_COSTS if severity_costs is None else severity_costs
    )
    per_section_costs = per_section_costs or {}

    total = 0.0
    breakdown: dict[str, dict[str, Any]] = {}
    for name, results in (judgment.get("sections") or {}).items():
        costs = {**global_costs, **per_section_costs.get(name, {})}
        sec_items: dict[str, int] = {}
        sec_deduction = 0.0
        for result in results:
            if result.get("gated_out") or result.get("response") == "N/A":
                continue
            ctype = result.get("criterion_type")
            if ctype == "evidence_list":
                for item in result.get("evidence") or []:
                    sev = item.get("severity") if isinstance(item, dict) else None
                    sev = sev if sev in costs else "none"
                    sec_items[sev] = sec_items.get(sev, 0) + 1
                    sec_deduction += costs.get(sev, 0.0)
            elif ctype == "fail_condition":
                if result.get("response") == "Yes":
                    sec_items["fail_condition"] = sec_items.get("fail_condition", 0) + 1
                    sec_deduction += fail_condition_cost
            elif result.get("response") == "No":
                sec_items["criterion_no"] = sec_items.get("criterion_no", 0) + 1
                sec_deduction += criterion_no_cost
        if sec_items or sec_deduction:
            breakdown[name] = {
                "items": sec_items,
                "deduction": round(sec_deduction, 2),
            }
        total += sec_deduction

    score = max(min_score, min(max_score, max_score - total))
    return {
        "benchmark_item_id": judgment.get("benchmark_item_id"),
        "model": judgment.get("model"),
        "score": round(score, 2),
        "deducted": round(total, 2),
        "breakdown": breakdown,
    }


def rescore_file(
    judgments_path: str | Path,
    weights_path: str | Path | None = None,
    **score_kwargs: Any,
) -> list[dict[str, Any]]:
    """Recompute scores for every judgment in a JSONL file.

    Purely local: reads stored responses, never calls an LLM. Extra
    keyword arguments are forwarded to :func:`score_judgment`.
    """
    weights = load_weights(weights_path) if weights_path else None
    return [
        score_judgment(row, weights, **score_kwargs)
        for row in iter_judgments(judgments_path)
    ]
