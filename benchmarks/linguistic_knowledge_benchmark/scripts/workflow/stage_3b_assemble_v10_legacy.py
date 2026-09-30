#!/usr/bin/env python3
"""
DEPRECATED / RETIRED — no longer part of the pipeline. This feature-map / hard-coded-quota
assembler has been replaced as Stage 3b by IRT-based selection; see estimate_irt.py. It is
kept only for reference and to reproduce older runs.

Stage 3b of the Linguistic Knowledge Benchmark pipeline: assemble the final
dataset deterministically from the Stage-3a annotated candidate pool.

Where it sits:
    1a  digest    -> resource_digest        (per-source survey; verbatim examples)
    1b  merge     -> feature_map            (JSON->JSON, no PDFs)
    2   generate  -> [candidate, ...]       (sharded; oversampled; per-shard PDFs)
    3a  pool & QC -> [annotated candidate]  (qc_verdict, dup_group_id, quality_rank)
    3b  assemble  -> THIS SCRIPT            (deterministic selection to a quota)

Given the 3a pool, this script:
    1. keeps only QC-passed candidates;
    2. collapses each dup_group_id to its best member (quality_rank == 1);
    3. fills a per-domain x grade quota, nudging the overall multiple_choice / short_form
       mix toward ANSWER_TYPE_TARGET and preferring better-attested candidates. The quota
       is set by the ALLOCATION constant (top of file): "feature_map" derives it
       proportionally from the Stage-1b feature map (scaled to TOTAL_ITEMS, with
       EXCLUDE_GRADES dropped), "hard-coded" uses the DEFAULT_QUOTA table;
    4. rebalances the correct MCQ letter roughly evenly across A-D (reordering options);
    5. assigns sequential question_id and emits the item schema.

It is deterministic: same pool + same constants -> byte-identical output. No LLM call.

Configuration: there are NO command-line flags. Edit the RUN CONFIGURATION block at the
top of this file (POOL_PATH, FEATURE_MAP_PATH, OUTDIR, PREFIX, ALLOCATION, TOTAL_ITEMS,
EXCLUDE_GRADES, ANSWER_TYPE_TARGET), then run the script with no arguments.

Outputs (under OUTDIR, with PREFIX):
    <prefix>_dataset.json      final benchmark, a JSON array in the item schema
    <prefix>_dataset_full.json  EVERY QC-passed, de-duplicated item in the same item schema
                               (ignores the quota; the un-capped superset of _dataset.json)
    <prefix>_backups.json      QC-passed candidates not selected, ranked (for the human gate)
    <prefix>_rejected.json     QC-failed candidates, with their verdicts (audit trail)
    <prefix>_gap_report.json   per-cell quota vs filled, plus answer-type & letter tallies
    <prefix>_quota.json        the derived per-D x G quota table (ONLY when ALLOCATION="feature_map")

Run:
    python benchmarks/linguistic_knowledge_benchmark/scripts/workflow/stage_3b_assemble_v10_legacy.py
"""

from __future__ import annotations

import json
import math
import os
from collections import Counter

# =========================================================================== #
# RUN CONFIGURATION  —  edit these constants, then run:  python stage_3b_assemble_v10.py
# (there are no CLI flags; everything is configured here in the file).
# =========================================================================== #
# Paths are anchored to this script's location, so the script runs from any working
# directory. Point them at a different language workflow by editing _PROCESS_DIR.
_PROCESS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "..",
    "data",
    "english_workflow",
    "process",
)
POOL_PATH = os.path.join(
    _PROCESS_DIR, "stage_3a.json"
)  # Stage-3a annotated pool (input)
FEATURE_MAP_PATH = os.path.join(
    _PROCESS_DIR, "stage_1b_merge_all_files.json"
)  # Stage-1b feature_map
OUTDIR = _PROCESS_DIR  # output directory
PREFIX = "stage_3b"  # output filename prefix

# How the per-domain x grade quota is decided:
#   "feature_map" -> derive it proportionally from FEATURE_MAP_PATH (scaled to TOTAL_ITEMS,
#                    with EXCLUDE_GRADES dropped from the count and the denominator);
#   "hard-coded"  -> use the DEFAULT_QUOTA table defined below, unchanged.
ALLOCATION = "feature_map"

TOTAL_ITEMS = 150  # target dataset size when ALLOCATION == "feature_map"
EXCLUDE_GRADES = (
    "G0",
    "beyond_primary",
)  # grades dropped when deriving (G0/beyond_primary are out-of-band)
ANSWER_TYPE_TARGET = 0.5  # target fraction of the dataset that is multiple_choice

# --------------------------------------------------------------------------- #
# Canonical axes (same fixed spine as eda_pipeline.py) so absent cells stay visible.
# --------------------------------------------------------------------------- #
DOMAINS = [f"D{i}" for i in range(1, 10)]
GRADES = ["G0", "G1", "G2", "G3", "G4", "G5", "beyond_primary"]

# Tier authority, best first (curriculum > teacher > student > other).
TIER_RANK = {"curriculum": 0, "teacher_material": 1, "student_material": 2, "other": 3}

# Fields that belong ONLY to the pipeline plumbing and must be stripped from the
# final dataset (which follows the item schema).
_PLUMBING_FIELDS = (
    "shard_id",
    "candidate_id",
    "qc_verdict",
    "dup_group_id",
    "quality_rank",
)

# Order of keys in each final item (matches linguistic_bench_swahili.json).
_ITEM_ORDER = (
    "question_id",
    "domain",
    "subdomain",
    "grade",
    "answer_type",
    "question",
    "options",
    "expected_answers",
    "linguistic_feature",
    "svr_strand",
    "geeap_component",
    "sources",
    "grade_placement_evidence",
)

# --------------------------------------------------------------------------- #
# HARD-CODED ALLOCATION TABLE. Used only when ALLOCATION == "hard-coded" (above);
# ignored when ALLOCATION == "feature_map". Edit this table to reshape the dataset.
# Cells not listed (and grades not listed within a domain) default to a quota of 0.
# Totals: 150 items overall (D1 18, D2 14, D3 20, D4 20, D5 16, D6 16, D7 12, D8 16,
# D9 18). A cell is a CEILING: it fills only if the pool supplies enough candidates,
# so under-supplied cells under-fill and surface in the gap_report.
# --------------------------------------------------------------------------- #
DEFAULT_QUOTA: dict[str, dict[str, int]] = {
    "D1": {"G0": 1, "G1": 5, "G2": 4, "G3": 3, "G4": 2, "G5": 2, "beyond_primary": 1},
    "D2": {"G0": 1, "G1": 4, "G2": 2, "G3": 2, "G4": 2, "G5": 2, "beyond_primary": 1},
    "D3": {"G1": 5, "G2": 5, "G3": 4, "G4": 2, "G5": 3, "beyond_primary": 1},
    "D4": {"G1": 4, "G2": 5, "G3": 4, "G4": 3, "G5": 3, "beyond_primary": 1},
    "D5": {"G1": 3, "G2": 4, "G3": 3, "G4": 3, "G5": 3},
    "D6": {"G0": 1, "G1": 4, "G2": 3, "G3": 3, "G4": 2, "G5": 3},
    "D7": {"G0": 1, "G1": 3, "G2": 4, "G3": 4},
    "D8": {"G0": 1, "G1": 4, "G2": 3, "G3": 3, "G4": 1, "G5": 2, "beyond_primary": 2},
    "D9": {"G1": 3, "G2": 3, "G3": 4, "G4": 3, "G5": 2, "beyond_primary": 3},
}


# --------------------------------------------------------------------------- #
# Quota derivation from the Stage-1b feature map
# --------------------------------------------------------------------------- #
def _round_half_up(x: float) -> int:
    """Round a NON-NEGATIVE value with halves going up (0.5 -> 1), matching Excel ROUND.

    Uses floor(x + 0.5) rather than Python's round() (which banker's-rounds 0.5 -> 0).
    Callers only pass x >= 0 (a proportion times a positive target), where this equals
    Excel ROUND; for x < 0 it would round toward +inf, not away from zero.
    """
    return math.floor(x + 0.5)


def _count_included_features(feature_map: dict, exclude_grades: tuple[str, ...]) -> int:
    """Total feature count over in-spine (domain, grade) cells, minus excluded grades.

    This is the denominator quota_from_feature_map divides by; exposed so the caller can
    report it in the gap_report.
    """
    excluded = set(exclude_grades)
    by_domain = (feature_map or {}).get("feature_map", {}).get("by_domain") or []
    total = 0
    for block in by_domain:
        if block.get("domain") not in DOMAINS:
            continue
        for feat in block.get("features", []):
            g = feat.get("grade")
            if g in GRADES and g not in excluded:
                total += 1
    return total


def quota_from_feature_map(
    feature_map: dict,
    total_items: int,
    exclude_grades: tuple[str, ...] = EXCLUDE_GRADES,
) -> dict[str, dict[str, int]]:
    """Proportional per-(domain, grade) quota derived from a Stage-1b feature_map.

        cell_quota = round_half_up(cell_feature_count / total_feature_count * total_items)

    Grades in `exclude_grades` are dropped from BOTH the per-cell count and the total
    (denominator) and forced to a quota of 0, so `total_items` spreads over the remaining
    grades only. Mirrors eda_pipeline.eda_merge's traversal so per-cell counts match the
    EDA. Rounding is naive per cell, so the realized sum may drift from total_items (the
    drift is reported in the gap_report).
    """
    excluded = set(exclude_grades)
    by_domain = (feature_map or {}).get("feature_map", {}).get("by_domain")
    if not isinstance(by_domain, list):
        raise SystemExit(
            f"FEATURE_MAP_PATH must be a Stage-1b JSON with feature_map.by_domain (a list): {FEATURE_MAP_PATH}"
        )

    # Count features per (domain, grade), skipping excluded grades entirely.
    counts: dict[str, dict[str, int]] = {d: {g: 0 for g in GRADES} for d in DOMAINS}
    total = 0
    for block in by_domain:
        d = block.get("domain")
        if d not in DOMAINS:
            continue
        for feat in block.get("features", []):
            g = feat.get("grade")
            if g in GRADES and g not in excluded:
                counts[d][g] += 1
                total += 1

    quota: dict[str, dict[str, int]] = {d: {g: 0 for g in GRADES} for d in DOMAINS}
    if total <= 0:
        return quota  # nothing to allocate (avoids ZeroDivisionError)
    for d in DOMAINS:
        for g in GRADES:
            if g not in excluded and counts[d][g] > 0:
                quota[d][g] = _round_half_up(counts[d][g] / total * total_items)
    return quota


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _best_tier_rank(item: dict) -> int:
    """Lowest (best) tier rank among an item's sources; unknown tiers sort last."""
    ranks = [TIER_RANK.get(s.get("tier"), 99) for s in item.get("sources", [])]
    return min(ranks) if ranks else 99


def _quality_key(item: dict) -> tuple:
    """Deterministic 'better candidate first' sort key.

    Prefer: better-attested tier, then more corroborating sources, then a stable
    candidate_id tie-break.
    """
    return (
        _best_tier_rank(item),
        -len(item.get("sources", [])),
        str(item.get("candidate_id", "")),
    )


def _passed(item: dict) -> bool:
    verdict = item.get("qc_verdict") or {}
    return bool(verdict.get("pass", False))


def _quota_for(quota: dict, domain: str, grade: str) -> int:
    return int(quota.get(domain, {}).get(grade, 0))


def _pick_balanced(candidates: list[dict], q: int, running: Counter) -> list[dict]:
    """Pick up to q candidates from a cell, nudging the global MCQ/short_form mix
    toward ANSWER_TYPE_TARGET. `running` holds global counts and is updated in place."""
    by_type = {"multiple_choice": [], "short_form": []}
    for c in candidates:
        by_type.setdefault(c.get("answer_type"), []).append(c)
    for lst in by_type.values():
        lst.sort(key=_quality_key)

    picked: list[dict] = []
    while len(picked) < q:
        avail = {t: lst for t, lst in by_type.items() if lst}
        if not avail:
            break
        total = running["multiple_choice"] + running["short_form"]
        # Preferred type = the one that moves the MCQ fraction toward target.
        if "multiple_choice" in avail and "short_form" in avail:
            mcq_frac_if_mcq = (running["multiple_choice"] + 1) / (total + 1)
            mcq_frac_if_sf = running["multiple_choice"] / (total + 1)
            pref = (
                "multiple_choice"
                if abs(mcq_frac_if_mcq - ANSWER_TYPE_TARGET)
                <= abs(mcq_frac_if_sf - ANSWER_TYPE_TARGET)
                else "short_form"
            )
        else:
            pref = next(iter(avail))
        chosen = by_type[pref].pop(0)
        picked.append(chosen)
        running[pref] += 1
    return picked


def _rebalance_letters(mcq_items: list[dict]) -> None:
    """Reorder each MCQ's options so the correct answer lands on a target letter,
    cycling A->B->C->D across items (in question_id order) for an even spread."""
    letters = ["A", "B", "C", "D"]
    for i, item in enumerate(mcq_items):
        options = item.get("options") or {}
        expected = item.get("expected_answers") or []
        if len(options) != 4 or len(expected) != 1 or expected[0] not in options:
            continue  # malformed MCQ; leave as-is (3a should have flagged it)
        target = letters[i % 4]
        correct_text = options[expected[0]]
        others = [options[l] for l in letters if l != expected[0]]  # keep A..D order
        new_options: dict[str, str] = {}
        oi = 0
        for l in letters:
            if l == target:
                new_options[l] = correct_text
            else:
                new_options[l] = others[oi]
                oi += 1
        item["options"] = new_options
        item["expected_answers"] = [target]


def _to_final_item(item: dict, question_id: int) -> dict:
    """Strip plumbing fields, set question_id, and emit keys in the item order."""
    clean = {k: v for k, v in item.items() if k not in _PLUMBING_FIELDS}
    clean["question_id"] = question_id
    if clean.get("answer_type") == "short_form":
        clean.pop("options", None)
    ordered = {k: clean[k] for k in _ITEM_ORDER if k in clean}
    # Preserve any extra fields deterministically after the known ones.
    for k in sorted(clean):
        if k not in ordered:
            ordered[k] = clean[k]
    return ordered


# --------------------------------------------------------------------------- #
# Core
# --------------------------------------------------------------------------- #
def _cell_sort_key(cell: tuple[str, str]) -> tuple:
    """Order (domain, grade) cells along the fixed spine; unknown values sort last."""
    domain, grade = cell
    di = DOMAINS.index(domain) if domain in DOMAINS else len(DOMAINS)
    gi = GRADES.index(grade) if grade in GRADES else len(GRADES)
    return (di, gi, str(domain), str(grade))


def assemble(pool: list[dict], quota: dict) -> dict:
    """Return {'dataset', 'dataset_full', 'backups', 'rejected', 'gap_report'} from a 3a pool."""
    rejected = [c for c in pool if not _passed(c)]
    passed = [c for c in pool if _passed(c)]

    # Collapse each dup_group to its best (quality_rank == 1, tie-broken deterministically).
    groups: dict[str, list[dict]] = {}
    for c in passed:
        groups.setdefault(str(c.get("dup_group_id", c.get("candidate_id"))), []).append(
            c
        )

    reps: list[dict] = []
    non_reps: list[dict] = []
    for gid in sorted(groups):
        members = sorted(
            groups[gid], key=lambda c: (int(c.get("quality_rank", 1)), _quality_key(c))
        )
        reps.append(members[0])
        non_reps.extend(members[1:])

    # Bucket representatives by (domain, grade).
    by_cell: dict[tuple[str, str], list[dict]] = {}
    for r in reps:
        by_cell.setdefault((r.get("domain"), r.get("grade")), []).append(r)

    selected: list[dict] = []
    selected_ids: set[str] = set()
    running = Counter({"multiple_choice": 0, "short_form": 0})
    gap_cells = []

    # Deterministic cell order: domain spine, then grade order.
    for domain in DOMAINS:
        for grade in GRADES:
            q = _quota_for(quota, domain, grade)
            avail = sorted(by_cell.get((domain, grade), []), key=_quality_key)
            picked = _pick_balanced(avail, q, running) if q > 0 else []
            selected.extend(picked)
            for p in picked:
                selected_ids.add(str(p.get("candidate_id")))
            if q > 0 or picked:
                deficit = q - len(picked)
                if deficit != 0:
                    gap_cells.append(
                        {
                            "domain": domain,
                            "grade": grade,
                            "quota": q,
                            "filled": len(picked),
                            "deficit": deficit,  # >0 under-filled, <0 quota below supply
                            "available": len(avail),
                        }
                    )

    # Assign question_ids in selection (cell-major) order, then rebalance letters.
    dataset: list[dict] = []
    for idx, item in enumerate(selected, start=1):
        dataset.append(_to_final_item(item, idx))
    _rebalance_letters(
        [d for d in dataset if d.get("answer_type") == "multiple_choice"]
    )

    # Full dataset: EVERY QC-passed, de-duplicated representative (quota ignored), in the
    # same item schema and the same cell-major order, with its own sequential question_ids.
    full_ordered: list[dict] = []
    for cell in sorted(by_cell, key=_cell_sort_key):
        full_ordered.extend(sorted(by_cell[cell], key=_quality_key))
    dataset_full: list[dict] = [
        _to_final_item(item, idx) for idx, item in enumerate(full_ordered, start=1)
    ]
    _rebalance_letters(
        [d for d in dataset_full if d.get("answer_type") == "multiple_choice"]
    )

    # Backups: passing reps not selected + all non-rep group members, ranked.
    backups = [
        r for r in reps if str(r.get("candidate_id")) not in selected_ids
    ] + non_reps
    backups.sort(key=lambda c: (int(c.get("quality_rank", 1)), _quality_key(c)))

    letter_dist = Counter(
        d["expected_answers"][0]
        for d in dataset
        if d.get("answer_type") == "multiple_choice" and d.get("expected_answers")
    )
    type_dist = Counter(d.get("answer_type") for d in dataset)

    gap_report = {
        "total_selected": len(dataset),
        "total_full": len(dataset_full),
        "total_quota": sum(sum(g.values()) for g in quota.values()),
        "answer_type_distribution": dict(type_dist),
        "letter_distribution": {l: letter_dist.get(l, 0) for l in ["A", "B", "C", "D"]},
        "cells": gap_cells,
        "counts": {
            "pool": len(pool),
            "qc_passed": len(passed),
            "qc_rejected": len(rejected),
            "dup_groups": len(groups),
            "full_dataset": len(dataset_full),
            "backups": len(backups),
        },
    }
    return {
        "dataset": dataset,
        "dataset_full": dataset_full,
        "backups": backups,
        "rejected": rejected,
        "gap_report": gap_report,
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _load_json(path: str):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _dump_json(obj, path: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=2)
        fh.write("\n")


def main() -> None:
    pool = _load_json(POOL_PATH)
    if not isinstance(pool, list):
        raise SystemExit(
            f"POOL_PATH must be a JSON array of annotated candidate items: {POOL_PATH}"
        )

    # Quota comes from ALLOCATION (configured at the top of this file):
    #   "feature_map" -> derive proportionally from FEATURE_MAP_PATH;
    #   "hard-coded"  -> use the DEFAULT_QUOTA table.
    if ALLOCATION == "feature_map":
        fmap = _load_json(FEATURE_MAP_PATH)
        quota = quota_from_feature_map(fmap, TOTAL_ITEMS, exclude_grades=EXCLUDE_GRADES)
        quota_meta = {
            "quota_source": "feature_map",
            "quota_target_total": TOTAL_ITEMS,
            "feature_total": _count_included_features(fmap, EXCLUDE_GRADES),
            "excluded_grades": list(EXCLUDE_GRADES),
        }
    elif ALLOCATION == "hard-coded":
        quota = DEFAULT_QUOTA
        quota_meta = {"quota_source": "hard-coded"}
    else:
        raise SystemExit(
            f"ALLOCATION must be 'feature_map' or 'hard-coded', got {ALLOCATION!r}."
        )

    result = assemble(pool, quota)
    result["gap_report"].update(quota_meta)

    outdir = OUTDIR
    os.makedirs(outdir, exist_ok=True)
    _dump_json(result["dataset"], os.path.join(outdir, f"{PREFIX}_dataset.json"))
    _dump_json(
        result["dataset_full"], os.path.join(outdir, f"{PREFIX}_dataset_full.json")
    )
    _dump_json(result["backups"], os.path.join(outdir, f"{PREFIX}_backups.json"))
    _dump_json(result["rejected"], os.path.join(outdir, f"{PREFIX}_rejected.json"))
    _dump_json(result["gap_report"], os.path.join(outdir, f"{PREFIX}_gap_report.json"))
    n_files = 5
    if quota_meta["quota_source"] == "feature_map":
        _dump_json(quota, os.path.join(outdir, f"{PREFIX}_quota.json"))
        n_files = 6

    gr = result["gap_report"]
    print(
        f"pool={gr['counts']['pool']}  passed={gr['counts']['qc_passed']}  "
        f"rejected={gr['counts']['qc_rejected']}  dup_groups={gr['counts']['dup_groups']}"
    )
    print(
        f"quota_source={gr['quota_source']}  selected={gr['total_selected']} / quota={gr['total_quota']}  "
        f"full={gr['total_full']}  "
        f"types={gr['answer_type_distribution']}  letters={gr['letter_distribution']}"
    )
    under = [c for c in gr["cells"] if c["deficit"] > 0]
    if under:
        print(f"under-filled cells: {len(under)} (see {PREFIX}_gap_report.json)")
    print(f"wrote {n_files} files to {outdir}/ with prefix '{PREFIX}'")


if __name__ == "__main__":
    main()
