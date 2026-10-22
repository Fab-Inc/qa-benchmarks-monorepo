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
    load_scoring_config,
)

TEXT_MEASURES_SECTION = "Text measures"
TEXT_MEASURES_WEIGHT = 2.0


def text_measure_scores(
    metrics: dict[str, Any], story_level: str | None
) -> dict[str, float | None]:
    """Deterministic Text measures sub-scores from stored metrics.

    Floors, ceilings, and per-band caps come from ``rubrics/scoring.yaml``
    (see ``report_reference_stats.py`` for suggested values).
    """
    metrics = metrics or {}
    cfg = load_scoring_config()
    stage = (cfg.get("stages") or {}).get(story_level or "") or {}
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
    length_score = metrics.get("length_score")
    length = round(length_score * 100, 2) if length_score is not None else None
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


def _text_measures_gated(metrics: dict[str, Any]) -> bool:
    """Hard-fail conditions that zero the Text measures section."""
    if metrics.get("n_target_words") == 0:
        return True
    pct = metrics.get("pct_decodable_words")
    if pct is not None and pct < DECODABLE_HARD_FLOOR:
        return True
    ref_min = metrics.get("reference_words_min")
    return ref_min is not None and (metrics.get("n_words") or 0) < ref_min / 2


def text_measures_section_score(
    metrics: dict[str, Any] | None, story_level: str | None
) -> float | None:
    """The Text measures section score, including hard-fail gates."""
    metrics = metrics or {}
    measures = text_measure_scores(metrics, story_level)
    present = [v for v in measures.values() if v is not None]
    if not present:
        return None
    if _text_measures_gated(metrics):
        return 0.0
    return round(sum(present) / len(present), 2)


def score_section(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Score one section's criterion results."""
    yes_count = applicable_count = na_count = gated_out_count = 0
    triggered_fail: list[str] = []

    for result in results:
        response = result.get("response")
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

            # scoring_rules absent on the row -> legacy fallback; explicit
            # empty list -> evidence-only, contributes no points.
            rules = result.get("scoring_rules")
            if rules is None:
                rules = DEFAULT_EVIDENCE_RULES
            for rule in rules:
                if "max_count_where" in rule:
                    cond = rule["max_count_where"]
                    applicable_count += 1
                    if sum(_matches(i, cond) for i in items) <= cond["count"]:
                        yes_count += 1
                    continue
                applicable_count += 1
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
                    yes_count += 1
            continue
        if result.get("criterion_type") == "fail_condition":
            if response == "Yes":
                triggered_fail.append(result["criterion_id"])
            continue
        applicable_count += 1
        if response == "Yes":
            yes_count += 1

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
) -> dict[str, Any]:
    """Score one stored judgment row, optionally with weight overrides.

    ``weights`` maps section name -> weight and takes precedence over the
    ``section_weights`` snapshot stored on the row.
    """
    overrides = weights or {}
    stored_weights = judgment.get("section_weights") or {}

    section_scores: dict[str, dict[str, Any]] = {}
    included: list[str] = []
    excluded: list[str] = []
    weighted_sum = 0.0
    weight_total = 0.0

    for name, results in (judgment.get("sections") or {}).items():
        section_score = score_section(results)
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

    measures = text_measure_scores(judgment.get("metrics"), story_level)
    measures_score = text_measures_section_score(judgment.get("metrics"), story_level)
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


def rescore_file(
    judgments_path: str | Path,
    weights_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Recompute scores for every judgment in a JSONL file.

    Purely local: reads stored responses, never calls an LLM.
    """
    weights = load_weights(weights_path) if weights_path else None
    return [score_judgment(row, weights) for row in iter_judgments(judgments_path)]
