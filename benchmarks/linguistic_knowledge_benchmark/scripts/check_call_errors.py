#!/usr/bin/env python3
"""
Table of per-model error counts (total/mcq/short-form), for responses vs
responses_judged.

Scans cache/<language>/experiments/<experiment-id>/responses/ and .../responses_judged/ -- every
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

  * after fallback   shown ONLY when responses_judged_fallback/ exists for this
                      experiment-id (i.e. --judge-fallback has been run). A row counts
                      when it is STILL unscored with is_correct_fallback taking
                      precedence over is_correct -- the same best-available-verdict rule
                      compute_summary() applies -- so this is what the fallback pass
                      could not rescue, and the drop from the previous column is what it
                      did. What remains should be roughly the rows with no output at all
                      (call errors) plus short-form rows, which fallback never targets.

If responses_judged's mcq count is high, most of the unscored rows are MCQs that
--judge-fallback can re-evaluate; if it's mostly short-form, fallback won't help
(it only targets multiple_choice).

The last column applies judge_bench.MAX_CALL_ERROR_RATE to the responses counts, i.e.
it says which of these models judge-bench will now refuse outright: above that share of
failed calls a model is not judged and is left out of the summary/detailed CSVs
entirely, because an accuracy computed from only the items that happened to succeed
would sit in the summary next to models that answered everything. SKIP rows need
'run-bench --retry-errors' before judging; "ok" rows have errors but few enough to be
judged (their failed rows stay unscored). The limit is imported, not restated here, so
this table can't drift from the gate that enforces it.

Run:
    python benchmarks/linguistic_knowledge_benchmark/scripts/check_call_errors.py --language kiswahili --experiment-id final-135-items
    python .../check_call_errors.py --language english --experiment-id default
"""

from __future__ import annotations

import argparse
import os

import pandas as pd
from linguistic_knowledge_benchmark.judge_bench import MAX_CALL_ERROR_RATE

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


def scan_responses_dir(
    dir_path: str,
) -> tuple[dict[str, tuple[int, int, int]], dict[str, int]]:
    """(error counts, row counts) per model, for every responses_<model>.csv present.

    Error counts are (total, mcq, short_form) call_status=="error" rows; row counts are
    the file's total length, needed to turn those into the error RATE that judge-bench's
    MAX_CALL_ERROR_RATE gate actually tests. Both empty if the dir doesn't exist.
    """
    counts: dict[str, tuple[int, int, int]] = {}
    row_counts: dict[str, int] = {}
    if not os.path.isdir(dir_path):
        return counts, row_counts
    for name in sorted(os.listdir(dir_path)):
        if not (name.startswith(RESPONSES_PREFIX) and name.endswith(".csv")):
            continue
        model = name[len(RESPONSES_PREFIX) : -len(".csv")]
        df = pd.read_csv(os.path.join(dir_path, name))
        if "call_status" not in df.columns:
            continue
        counts[model] = _breakdown(df, df["call_status"] == CALL_STATUS_ERROR)
        row_counts[model] = len(df)
    return counts, row_counts


def scan_judged_dir(
    dir_path: str, *, fallback: bool = False
) -> dict[str, tuple[int, int, int]]:
    """model -> (total, mcq, short_form) still-unscored rows, for every
    responses_judged_<model>.csv present. Empty dict if the dir doesn't exist.

    A row is unscored when is_correct is null. With fallback=True (i.e. scanning
    responses_judged_fallback/, whose files carry the same name) is_correct_fallback
    takes precedence where it's populated -- the same "best available verdict" rule
    compute_summary() uses -- so the count is what the fallback pass could NOT rescue,
    not what it started from.
    """
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
        effective = df["is_correct"]
        if fallback and "is_correct_fallback" in df.columns:
            effective = df["is_correct_fallback"].fillna(effective)
        counts[model] = _breakdown(df, effective.isna())
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
    ap.add_argument("--experiment-id", required=True)
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    experiment_path = os.path.join(
        root, "cache", args.language, "experiments", args.experiment_id
    )

    responses_counts, row_counts = scan_responses_dir(
        os.path.join(experiment_path, "responses")
    )
    judged_counts = scan_judged_dir(os.path.join(experiment_path, "responses_judged"))
    # Only present once --judge-fallback has been run for this experiment-id; the
    # column is left out entirely rather than shown as a wall of "-" when it hasn't.
    fallback_counts = scan_judged_dir(
        os.path.join(experiment_path, "responses_judged_fallback"), fallback=True
    )

    models = sorted(
        m
        for m in set(responses_counts) | set(judged_counts) | set(fallback_counts)
        if responses_counts.get(m, (0, 0, 0))[0] > 0
        or judged_counts.get(m, (0, 0, 0))[0] > 0
        or fallback_counts.get(m, (0, 0, 0))[0] > 0
    )

    if not models:
        print(f"No errors found under {experiment_path}")
        return

    # The gate judge-bench applies, per model: call errors / rows, against
    # MAX_CALL_ERROR_RATE. "-" where the responses CSV is absent (a judged-only model),
    # since without it there is no rate to compare.
    gate = {}
    for m in models:
        n_rows = row_counts.get(m, 0)
        if not n_rows:
            gate[m] = "-"
            continue
        n_errors = responses_counts.get(m, (0, 0, 0))[0]
        rate = n_errors / n_rows
        verdict = "SKIP" if rate > MAX_CALL_ERROR_RATE else "ok"
        gate[m] = f"{verdict:<4}  {n_errors}/{n_rows} = {rate:5.1%}"

    # (header, per-model cell) for each column after the model name. The fallback
    # column only joins the table when that pass has actually been run.
    columns = [
        ("responses (total/mcq/sf)", lambda m: _fmt(responses_counts, m)),
        ("responses_judged (total/mcq/sf)", lambda m: _fmt(judged_counts, m)),
    ]
    if fallback_counts:
        columns.append(
            ("after fallback (total/mcq/sf)", lambda m: _fmt(fallback_counts, m))
        )
    columns.append(
        (f"judge-bench (limit {MAX_CALL_ERROR_RATE:.0%})", lambda m: gate[m])
    )

    name_w = max(len("model"), *(len(m) for m in models))
    widths = [
        max(len(title), *(len(str(cell(m))) for m in models)) for title, cell in columns
    ]

    header = f"{'model':<{name_w}}"
    for (title, _), w in zip(columns, widths):
        header += f"  {title:<{w}}"
    print(header)
    print("-" * len(header))
    for m in models:
        row = f"{m:<{name_w}}"
        for (_, cell), w in zip(columns, widths):
            row += f"  {str(cell(m)):<{w}}"
        print(row)

    n_skip = sum(1 for m in models if gate[m].startswith("SKIP"))
    if n_skip:
        print(
            f"\n{n_skip} model(s) marked SKIP: judge-bench will not judge them and "
            "they will be absent from\nthe summary and detailed CSVs. Re-run them with "
            "'run-bench --retry-errors' first."
        )


if __name__ == "__main__":
    main()
