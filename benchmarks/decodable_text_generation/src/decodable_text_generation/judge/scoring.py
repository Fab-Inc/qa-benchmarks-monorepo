"""Compute section and final scores from stored rubric judgments.

Judgment rows written by the judge script carry every criterion's raw
response, its ``criterion_type``, a ``gated_out`` flag, and a
``section_weights`` snapshot, so scores can be recomputed offline with
different weights — without any further LLM calls.

Scoring rules (mirroring evaluate-material-cli's ``score_calculator``):

- ``positive`` criteria score ``yes_count / applicable_count * 100``;
  ``N/A`` and ``gated_out`` responses are excluded from the denominator.
- A ``fail_condition`` answered ``"Yes"`` forces the section score to 0.
- An ``evidence_list`` criterion carries no verdict; instead each of its
  stored ``scoring_rules`` is one applicable point, earned when every item's
  ``severity`` is in the rule's allowed set (empty lists pass all rules).
- A section's effective weight is its ``weight`` (overridable per rescore);
  weight 0 means judged and reported but excluded from the final score.
- ``Text measures`` is a deterministic section bundling the code-computed
  scores, averaged in with weight 2 (overridable like any section weight):

  - ``text_decodable``: ``pct_decodable_words`` rescaled so the stage floor
    maps to 0 and the ceiling to 100.
  - ``text_target``: ``n_target_words`` — 100 at or above the band's
    minimum count of distinct target-bearing words, a straight line below
    it (0 → cap).

  Gates that zero the whole section: zero target-bearing words (a story
  that practises no targets has no meaningful text-measure score), under
  ``DECODABLE_HARD_FLOOR``% decodable words, or below half the length of
  the band's shortest reference story.
  - ``text_length``: word count vs the band's reference range — 100 inside
    the range, falling linearly to 0 at half the minimum and at double
    the maximum.

  All parameters (band cutoffs, length ranges, target-word minimums,
  decodability floor/ceiling) live in ``rubrics/scoring.yaml``; run
  ``report_reference_stats.py`` to see what the human stories suggest.
- The final score is the weighted mean of non-excluded, non-N/A sections.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from decodable_text_generation.judge.rubric import (
    STORY_LEVEL_BANDS,
    language_for_profile,
    load_scoring_config,
    stage_config,
)

TEXT_MEASURES_SECTION = "Text measures"
TEXT_MEASURES_WEIGHT = 2.0


def _length_score(n_words: int | None, bounds: tuple[float, float]) -> float | None:
    """Length sub-score in [0, 1]: 1 inside ``bounds``, linear falloff to 0
    at half the minimum and double the maximum."""
    if n_words is None:
        return None
    lo, hi = bounds
    if n_words < lo:
        return max(0.0, (n_words - lo / 2) / (lo / 2))
    if n_words <= hi:
        return 1.0
    return max(0.0, 1.0 - (n_words - hi) / hi)


def text_measure_scores(
    metrics: dict[str, Any],
    story_level: str | None,
    language: str = "en",
    stage_overrides: dict[str, Any] | None = None,
) -> dict[str, float | None]:
    """Deterministic Text measures sub-scores from stored metrics.

    Floors, ceilings, and per-band caps come from the language's section
    of ``rubrics/scoring.yaml``, optionally overridden per call by
    ``stage_overrides`` (flat keys merged over the stage's config, e.g.
    ``{"decodable_floor": 80.0}`` or ``{"length": [30, 100]}``).
    """
    metrics = metrics or {}
    stage = dict(stage_config(language, story_level, load_scoring_config()))
    if stage_overrides:
        stage.update(stage_overrides)
    pct = metrics.get("pct_decodable_words")
    floor = stage.get("decodable_floor", 70.0)
    ceiling = stage.get("decodable_ceiling", 90.0)
    decodable = (
        round(
            min(100.0, max(0.0, (pct - floor) * 100 / (ceiling - floor))),
            2,
        )
        if pct is not None
        else None
    )
    n_target = metrics.get("n_target_words")
    cap = stage.get("target_words_min", 2)
    target = (
        round(min(100.0, 100.0 * n_target / cap), 2) if n_target is not None else None
    )
    bounds = stage.get("length")
    if bounds and metrics.get("n_words") is not None:
        # Recompute from word count + bounds so ``length`` overrides apply;
        # identical to the stored ``length_score`` when bounds are unchanged.
        score = _length_score(metrics["n_words"], tuple(bounds))
    else:
        score = metrics.get("length_score")
    length = round(score * 100, 2) if score is not None else None
    return {
        "text_decodable": decodable,
        "text_target": target,
        "text_length": length,
    }


DECODABLE_HARD_FLOOR = 90.0

# Fallback for evidence_list rows judged before scoring_rules were stored:
# one point if no medium/high items, one point if none is high.
DEFAULT_EVIDENCE_RULES = [
    {"severity_in": ["low"]},
    {"severity_in": ["low", "medium"]},
]


def _text_measures_gated(
    metrics: dict[str, Any], hard_floor: float = DECODABLE_HARD_FLOOR
) -> bool:
    """Hard-fail conditions that zero the Text measures section."""
    if metrics.get("n_target_words") == 0:
        return True
    pct = metrics.get("pct_decodable_words")
    if pct is not None and pct < hard_floor:
        return True
    ref_min = metrics.get("reference_words_min")
    return ref_min is not None and (metrics.get("n_words") or 0) < ref_min / 2


def text_measures_section_score(
    metrics: dict[str, Any] | None,
    story_level: str | None,
    language: str = "en",
    stage_overrides: dict[str, Any] | None = None,
) -> float | None:
    """The Text measures section score, including hard-fail gates.

    ``stage_overrides`` is merged over the stage config like in
    ``text_measure_scores``; the extra key ``hard_floor`` overrides
    ``DECODABLE_HARD_FLOOR`` for the gating check.
    """
    metrics = metrics or {}
    overrides = dict(stage_overrides or {})
    hard_floor = float(overrides.pop("hard_floor", DECODABLE_HARD_FLOOR))
    if overrides.get("length"):
        # The half-length gate reads the stored reference minimum; an
        # overridden range must update it too.
        metrics = {**metrics, "reference_words_min": overrides["length"][0]}
    measures = text_measure_scores(metrics, story_level, language, overrides)
    present = [v for v in measures.values() if v is not None]
    if not present:
        return None
    if _text_measures_gated(metrics, hard_floor):
        return 0.0
    return round(sum(present) / len(present), 2)


def score_section(
    results: list[dict[str, Any]],
    criterion_weights: dict[str, float] | None = None,
    evidence_rules: dict[str, list[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    """Score one section's criterion results.

    ``criterion_weights`` maps criterion_id -> weight (default 1.0); each
    applicable criterion or evidence scoring-rule point is worth its
    criterion's weight, so the section score is a weighted ``yes /
    applicable`` rate. ``evidence_rules`` maps criterion_id -> a
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
        weight = float(criterion_weights.get(result.get("criterion_id"), 1.0))
        if result.get("gated_out"):
            gated_out_count += 1
            continue
        if response == "N/A":
            na_count += 1
            continue
        if result.get("criterion_type") == "evidence_list":
            items = result.get("evidence") or []
            severities = {i.get("severity") for i in items if isinstance(i, dict)}

            def _matches(item: Any, cond: dict[str, Any]) -> bool:
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
                if "max_count_where" in rule:
                    cond = rule["max_count_where"]
                    applicable_count += weight
                    if sum(_matches(i, cond) for i in items) <= cond["count"]:
                        yes_count += weight
                    continue
                applicable_count += weight
                if "max_items" in rule:
                    earned = len(items) <= rule["max_items"]
                elif "severity_in" in rule:
                    earned = (
                        not items
                        or all(isinstance(i, dict) and "severity" in i for i in items)
                    ) and severities <= set(rule["severity_in"])
                else:
                    earned = False
                if earned:
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
    scoring_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Score one stored judgment row, optionally with weight overrides.

    ``weights`` maps section name -> weight and takes precedence over the
    ``section_weights`` snapshot stored on the row.
    ``criterion_weights`` maps criterion_id -> weight within its section;
    ``evidence_rules`` maps criterion_id -> replacement scoring_rules;
    ``scoring_config`` carries flat Text-measure stage overrides (e.g.
    ``decodable_floor``, ``target_words_min``, ``length``, ``hard_floor``).
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

    tags = judgment.get("tags") or []
    level_names = [name for name, _ in STORY_LEVEL_BANDS]
    story_level = next((t for t in tags if t in level_names), None)
    language = language_for_profile(judgment.get("profile_id") or "")

    measures = text_measure_scores(
        judgment.get("metrics"), story_level, language, scoring_config
    )
    measures_score = text_measures_section_score(
        judgment.get("metrics"), story_level, language, scoring_config
    )
    if measures_score is not None:
        weight = float(overrides.get(TEXT_MEASURES_SECTION, TEXT_MEASURES_WEIGHT))
        section_scores[TEXT_MEASURES_SECTION] = {
            "score": measures_score,
            "measures": measures,
            "deterministic": True,
            "weight": weight,
        }
        if weight <= 0:
            excluded.append(TEXT_MEASURES_SECTION)
        else:
            included.append(TEXT_MEASURES_SECTION)
            weighted_sum += measures_score * weight
            weight_total += weight

    final_score = weighted_sum / weight_total if weight_total else None
    return {
        "benchmark_item_id": judgment.get("benchmark_item_id"),
        "model": judgment.get("model"),
        "model_call_id": judgment.get("model_call_id"),
        "judge_model": judgment.get("judge_model"),
        "story_level": story_level,
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
    global_costs = dict(DEFAULT_SEVERITY_COSTS if severity_costs is None else severity_costs)
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
) -> list[dict[str, Any]]:
    """Recompute scores for every judgment in a JSONL file.

    Purely local: reads stored responses, never calls an LLM.
    """
    weights = load_weights(weights_path) if weights_path else None
    return [score_judgment(row, weights) for row in iter_judgments(judgments_path)]
