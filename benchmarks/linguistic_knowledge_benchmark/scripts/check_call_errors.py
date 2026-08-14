#!/usr/bin/env python3
"""
Table of per-model error counts (total/mcq/short-form), for responses vs
responses_judged.

Scans cache/<language>/runs/<run-id>/responses/ and .../responses_judged/ -- every
per-model CSV written by run-bench / judge-bench -- and prints one row per model that
has at least one error in either file, split by answer_type so you can tell whether
--judge-fallback (which only re-judges null MCQ rows) is likely to help:

  * responses        a row is an error when call_status == "error" (the model call
                      itself failed in run-bench).
  * responses_judged  a row is an error when is_correct is null. judge_bench.py
                      carries call_status=="error" rows straight through as
                      is_correct=None, so this is normally >= the responses count --
                      it's higher when the LLM judge(s) themselves failed to score an
                      otherwise-successful response (judge call error, or an
                      exact-tie majority vote) or a regex couldn't parse an MCQ
                      answer, and lower if --judge-fallback already recovered some of
                      them (fallback columns aren't checked here).

If responses_judged's mcq count is high, most of the unscored rows are MCQs that
--judge-fallback can re-evaluate; if it's mostly short-form, fallback won't help
(it only targets multiple_choice).

Run:
    python benchmarks/linguistic_knowledge_benchmark/scripts/check_call_errors.py --language kiswahili --run-id final-135-items
    python .../check_call_errors.py --language english --run-id default
"""

from __future__ import annotations

import argparse
import os

import pandas as pd

LANGUAGES = ["english", "kiswahili"]
CALL_STATUS_ERROR = "error"
RESPONSES_PREFIX = "responses_"
JUDGED_PREFIX = "responses_judged_"


def _breakdown(df: pd.DataFrame, mask: pd.Series) -> tuple[int, int, int]:
    """(total, mcq, short_form) counts of rows where mask is True."""
    total = int(mask.sum())
    if "answer_type" not in df.columns:
        return total, -1, -1
    mcq = int((mask & (df["answer_type"] == "multiple_choice")).sum())
    sf = int((mask & (df["answer_type"] == "short_form")).sum())
    return total, mcq, sf


def scan_responses_dir(dir_path: str) -> dict[str, tuple[int, int, int]]:
    """model -> (total, mcq, short_form) call_status=="error" rows, for every
    responses_<model>.csv present. Empty dict if the dir doesn't exist."""
    counts: dict[str, tuple[int, int, int]] = {}
    if not os.path.isdir(dir_path):
        return counts
    for name in sorted(os.listdir(dir_path)):
        if not (name.startswith(RESPONSES_PREFIX) and name.endswith(".csv")):
            continue
        model = name[len(RESPONSES_PREFIX) : -len(".csv")]
        df = pd.read_csv(os.path.join(dir_path, name))
        if "call_status" not in df.columns:
            continue
        counts[model] = _breakdown(df, df["call_status"] == CALL_STATUS_ERROR)
    return counts


def scan_judged_dir(dir_path: str) -> dict[str, tuple[int, int, int]]:
    """model -> (total, mcq, short_form) is_correct==None rows, for every
    responses_judged_<model>.csv present. Empty dict if the dir doesn't exist."""
    counts: dict[str, tuple[int, int, int]] = {}
    if not os.path.isdir(dir_path):
        return counts
    for name in sorted(os.listdir(dir_path)):
        if not (name.startswith(JUDGED_PREFIX) and name.endswith(".csv")):
            continue
        model = name[len(JUDGED_PREFIX) : -len(".csv")]
        df = pd.read_csv(os.path.join(dir_path, name))
        if "is_correct" not in df.columns:
            continue
        counts[model] = _breakdown(df, df["is_correct"].isna())
    return counts


def _fmt(counts: dict[str, tuple[int, int, int]], model: str) -> str:
    if model not in counts:
        return "-"
    total, mcq, sf = counts[model]
    if total == 0:
        return "0/0/0"
    if mcq == -1:
        return f"{total}/?/?"
    return f"{total}/{mcq}/{sf}"


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--language", required=True, choices=LANGUAGES)
    ap.add_argument("--run-id", required=True)
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    run_path = os.path.join(root, "cache", args.language, "runs", args.run_id)

    responses_counts = scan_responses_dir(os.path.join(run_path, "responses"))
    judged_counts = scan_judged_dir(os.path.join(run_path, "responses_judged"))

    models = sorted(
        m
        for m in set(responses_counts) | set(judged_counts)
        if responses_counts.get(m, (0, 0, 0))[0] > 0
        or judged_counts.get(m, (0, 0, 0))[0] > 0
    )

    if not models:
        print(f"No errors found under {run_path}")
        return

    col1, col2 = "responses (total/mcq/sf)", "responses_judged (total/mcq/sf)"
    name_w = max(len("model"), *(len(m) for m in models))
    col1_w = max(len(col1), *(len(_fmt(responses_counts, m)) for m in models))
    col2_w = max(len(col2), *(len(_fmt(judged_counts, m)) for m in models))

    header = f"{'model':<{name_w}}  {col1:<{col1_w}}  {col2:<{col2_w}}"
    print(header)
    print("-" * len(header))
    for m in models:
        print(
            f"{m:<{name_w}}  {_fmt(responses_counts, m):<{col1_w}}  "
            f"{_fmt(judged_counts, m):<{col2_w}}"
        )


if __name__ == "__main__":
    main()
