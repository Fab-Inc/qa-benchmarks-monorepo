#!/usr/bin/env python3
"""
Build the final CLEAN benchmark dataset from the human domain-review spreadsheet.

Takes the domain-reviewed workbook (Kiswahili_LLK_FINAL_DATASET_dom_reviewed.xlsx),
keeps only the rows whose `Final Verdict` is "true" (i.e. retained = Kept or Fixed;
Dropped rows are removed), and emits a JSON array in the exact item schema of
`data/linguistic_knowledge_eval_kiswahili.json`.

Field provenance
    The spreadsheet is the source of truth for everything the reviewer could change:
        Domain, Grade, Type, Question, Options, Answer, SVR, GEEAP
    (re-tagging fixes live here — e.g. an item re-filed D1 -> D3). The descriptive
    fields the sheet does not carry (subdomain, linguistic_feature, sources,
    grade_placement_evidence) are joined back from --source-json by matching the
    sheet's `Q#` to that file's `question_id`.

Rebalancing
    After selection, the correct-answer letter of the MCQs is spread evenly across
    A-D by _rebalance_letters (options are reordered, the correct letter cycles
    A->B->C->D). This is applied whenever the current spread exceeds
    --max-letter-diff (default 5% of the MCQ count); it drives the max pairwise
    letter difference to <=1 item, well under the 5% target.

question_id is reassigned 1..N over the retained items, in sheet (Q#) order.

Run:
    python benchmarks/linguistic_knowledge_benchmark/scripts/workflow/create_clean_dataset.py
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter

import pandas as pd

# Spreadsheet Type -> schema answer_type.
TYPE_MAP = {"MCQ": "multiple_choice", "short_form": "short_form"}

# Descriptive fields the sheet lacks; pulled from --source-json by question_id.
_JOINED_FIELDS = (
    "subdomain",
    "linguistic_feature",
    "sources",
    "grade_placement_evidence",
)

# Final key order in each emitted item (matches linguistic_knowledge_eval_kiswahili.json).
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


def parse_options(raw: str) -> dict[str, str]:
    """'A) foo | B) bar | C) baz | D) qux' -> {'A': 'foo', ...} (splits only on the
    ' | ' that precedes a 'X)' label, so option text may itself contain '|')."""
    out: dict[str, str] = {}
    for part in re.split(r"\s*\|\s*(?=[A-D]\))", str(raw).strip()):
        m = re.match(r"([A-D])\)\s*(.*)", part.strip(), flags=re.DOTALL)
        if m:
            out[m.group(1)] = m.group(2).strip()
    return out


def parse_short_answers(raw: str) -> list[str]:
    """Short-form Answer cell -> accepted answers; '5 | five | tano' -> 3 answers."""
    return [a.strip() for a in str(raw).split("|") if a.strip()]


def build_item(row: pd.Series, joined: dict) -> dict:
    """Assemble one schema item from a retained sheet row + its joined descriptive fields."""
    answer_type = TYPE_MAP.get(str(row["Type"]).strip(), str(row["Type"]).strip())
    item: dict = {
        "domain": str(row["Domain"]).strip(),
        "subdomain": joined.get("subdomain"),
        "grade": str(row["Grade"]).strip(),
        "answer_type": answer_type,
        "question": str(row["Question"]).strip(),
    }
    if answer_type == "multiple_choice":
        item["options"] = parse_options(row["Options"])
        item["expected_answers"] = [str(row["Answer"]).strip()]
    else:
        item["expected_answers"] = parse_short_answers(row["Answer"])
    item["linguistic_feature"] = joined.get("linguistic_feature")
    item["svr_strand"] = str(row["SVR"]).strip()
    item["geeap_component"] = str(row["GEEAP"]).strip()
    item["sources"] = joined.get("sources")
    item["grade_placement_evidence"] = joined.get("grade_placement_evidence")
    return item


def order_item(item: dict, question_id: int) -> dict:
    """Set question_id and emit keys in _ITEM_ORDER (extras kept, sorted, at the end)."""
    item["question_id"] = question_id
    ordered = {k: item[k] for k in _ITEM_ORDER if k in item}
    for k in sorted(item):
        if k not in ordered:
            ordered[k] = item[k]
    return ordered


def _rebalance_letters(mcq_items: list[dict]) -> None:
    """Reorder each MCQ's options so the correct answer lands on a target letter,
    cycling A->B->C->D across items (in order) for an even spread. The correct
    answer TEXT is unchanged; the distractors keep their A..D relative order and
    slide into the remaining slots. (Mirrors stage_3b_assemble_v10._rebalance_letters.)"""
    letters = ["A", "B", "C", "D"]
    for i, item in enumerate(mcq_items):
        options = item.get("options") or {}
        expected = item.get("expected_answers") or []
        if len(options) != 4 or len(expected) != 1 or expected[0] not in options:
            continue  # malformed MCQ; leave as-is
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


def letter_stats(items: list[dict]) -> tuple[Counter, int, float]:
    """(letter counts, max pairwise count diff, that diff as % of MCQ count)."""
    dist = Counter(
        it["expected_answers"][0]
        for it in items
        if it.get("answer_type") == "multiple_choice" and it.get("expected_answers")
    )
    n = sum(dist.values())
    counts = [dist.get(x, 0) for x in ["A", "B", "C", "D"]]
    spread = (max(counts) - min(counts)) if n else 0
    return dist, spread, (spread / n * 100 if n else 0.0)


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(os.path.dirname(here))
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--xlsx",
        default=os.path.join(
            root,
            "data",
            "kiswahili_workflow",
            "process",
            "Kiswahili_LLK_FINAL_DATASET_dom_reviewed.xlsx",
        ),
        help="domain-reviewed workbook (input)",
    )
    ap.add_argument(
        "--source-json",
        dest="source_json",
        default=os.path.join(root, "data", "linguistic_knowledge_eval_kiswahili.json"),
        help="dataset the sheet was reviewed from; supplies the descriptive fields "
        "(joined by question_id == sheet Q#)",
    )
    ap.add_argument(
        "--out",
        default=os.path.join(
            root, "data", "linguistic_knowledge_eval_kiswahili_clean.json"
        ),
        help="output clean dataset JSON",
    )
    ap.add_argument(
        "--verdict-col",
        dest="verdict_col",
        default="Final Verdict",
        help="column holding the keep/drop decision",
    )
    ap.add_argument(
        "--drop-values",
        dest="drop_values",
        nargs="+",
        default=["Dropped"],
        help="verdict values treated as FALSE (removed); everything else is kept",
    )
    ap.add_argument(
        "--max-letter-diff",
        dest="max_letter_diff",
        type=float,
        default=5.0,
        help="max allowed pairwise MCQ correct-letter gap, as %% of MCQ count; "
        "rebalance is applied when the raw spread exceeds this",
    )
    args = ap.parse_args()

    df = pd.read_excel(args.xlsx)
    if args.verdict_col not in df.columns:
        raise SystemExit(
            f"--verdict-col {args.verdict_col!r} not in workbook columns: {list(df.columns)}"
        )
    verdict = df[args.verdict_col].astype(str).str.strip()
    drop = {v.strip() for v in args.drop_values}
    retained = df[~verdict.isin(drop)].copy()

    source = json.load(open(args.source_json, encoding="utf-8"))
    jmap = {int(it["question_id"]): it for it in source}

    missing = [int(q) for q in retained["Q#"] if int(q) not in jmap]
    if missing:
        raise SystemExit(
            f"{len(missing)} retained Q# have no match in --source-json (question_id): "
            f"{missing[:10]}{'...' if len(missing) > 10 else ''}"
        )

    # Build items in sheet (Q#) order.
    retained = retained.sort_values("Q#", kind="stable")
    items: list[dict] = []
    for _, row in retained.iterrows():
        src = jmap[int(row["Q#"])]
        joined = {k: src.get(k) for k in _JOINED_FIELDS}
        items.append(build_item(row, joined))

    # Reassign question_id 1..N, then rebalance MCQ letters if the spread is too wide.
    dataset = [order_item(it, i) for i, it in enumerate(items, start=1)]
    _, spread0, pct0 = letter_stats(dataset)
    dist_before, _, _ = letter_stats(dataset)
    rebalanced = pct0 > args.max_letter_diff
    if rebalanced:
        _rebalance_letters(
            [d for d in dataset if d.get("answer_type") == "multiple_choice"]
        )
    dist_after, spread1, pct1 = letter_stats(dataset)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(dataset, fh, ensure_ascii=False, indent=2)
        fh.write("\n")

    n_mcq = sum(1 for d in dataset if d["answer_type"] == "multiple_choice")
    n_sf = sum(1 for d in dataset if d["answer_type"] == "short_form")
    print(
        f"retained {len(dataset)} items ({n_mcq} MCQ, {n_sf} short-form) "
        f"from {len(df)} reviewed  (dropped {len(df) - len(dataset)})"
    )
    print(
        f"MCQ letters before: {dict(sorted(dist_before.items()))}  "
        f"(spread {spread0} = {pct0:.1f}%)"
    )
    if rebalanced:
        print(
            f"MCQ letters after : {dict(sorted(dist_after.items()))}  "
            f"(spread {spread1} = {pct1:.1f}%)  [rebalanced]"
        )
    else:
        print(f"no rebalance needed (spread {pct0:.1f}% <= {args.max_letter_diff}%)")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
