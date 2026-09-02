#!/usr/bin/env python3
"""
Estimate the $ cost of one "model run" -- run-bench + judge-bench for a single model on
all items of one language -- from data already sitting in
cache/<language>/experiments/<experiment-id>/. No new API calls are made.

Two cost components, tracked with different precision:

  * run-bench (the model under test): EXACT. Every row of responses_<model>.csv already
    carries total_tokens/completion_tokens from the real API call (fdllm's
    TokensUsed/TokensUsedCompletion); input_tokens = total_tokens - completion_tokens.
    Cost = input_tokens/1e6 * input_cost + completion_tokens/1e6 * output_cost, priced
    from data/models.csv (USD per 1M tokens) for the model under test.
    completion_tokens include the reasoning tokens.

  * judge-bench (the main LLM-judge pass): ESTIMATED. judge_bench.py never persists
    judge-call token usage (score_with_llm_judge() doesn't read response.TokensUsed at
    all), so there's nothing exact to sum here. Instead this script reconstructs the same
    prompt text judge_bench.py actually built for each call (system prompt + "Question:
    ... Model's answer: ... Acceptable expected answers: ... Answer type: ..." --
    mirrored in build_judge_user_prompt() below, kept in sync with
    judge_bench.score_with_llm_judge()) and the real response text that came back (the
    judged CSV's justification_<judge> column -- a genuine LLM response, not a guess),
    then approximates tokens as len(text) / CHARS_PER_TOKEN (~4 chars/token, the standard
    rule of thumb for English text -- judge prompts/responses are always English,
    regardless of --language). Priced per-judge from models.csv. This is a rough but
    data-grounded estimate, not a flat per-call guess.

  Regex-scored MCQ rows (most rows, most models) cost nothing at judge time -- only
  short_form rows ever reach an LLM judge. Which judge(s) were actually used for a given
  model/experiment is read from that experiment's log.json (llm_judges), not assumed,
  since the judge ensemble is a per-experiment choice (see judge_bench.judge_results()'s judge_models arg), not
  a fixed constant.

  Only run-bench + the main judge-bench pass are covered here -- a --judge-fallback pass,
  if this run has one, is not read or priced. Its cost is ignored for simplicity: it's a
  small LLM-judge re-pass over just the previously-unscored MCQ rows, so it's cheap
  relative to the two components above, and skipping it keeps this script's logic (and
  the console/CSV output) to exactly the two-component estimate described above.

Note on --experiment-id: unlike scripts/analyze_results.py / validate_results.py, this script
reads cache/<language>/experiments/<experiment-id>/ directly (responses/, responses_judged/) rather
than results/ CSVs, so it takes the bare cache experiment-id with no "_fallback" suffix (e.g.
"final-137-items", not "final-137-items_fallback").

Output: printed to the terminal only, nothing written to disk. A compact per-model table
    (run-bench cost exact, judge cost estimated, total), sorted priciest first, plus
    min/mean/max total cost across this run's priced models -- directly answers "how
    much does a single model run cost", since every model here already IS one run-bench
    + judge-bench pass over one language's full item set. A model missing pricing in
    data/models.csv (open-weight/self-hosted models often are) shows "-" and is excluded
    from the min/mean/max, not silently treated as free.

Run:
    uv run python benchmarks/linguistic_knowledge_benchmark/scripts/estimate_experiment_cost.py --language english --experiment-id final-137-items
    uv run python .../estimate_experiment_cost.py --language kiswahili --experiment-id final-135-items
"""

from __future__ import annotations

import argparse
import json
import os

import pandas as pd
from linguistic_knowledge_benchmark.judge_bench import (
    DEFAULT_JUDGE_MODELS,
    _build_judge_system_prompt,
    strip_think_block,
)
from linguistic_knowledge_benchmark.run_bench import load_dataset

LANGUAGES = ["english", "kiswahili"]
LANGUAGE_DISPLAY_NAMES = {"english": "English", "kiswahili": "Kiswahili"}

# len(text) / CHARS_PER_TOKEN is the estimator for every judge-call token count below
# (run-bench's own tokens are exact -- see the module docstring). ~4 characters/token is
# the standard rule-of-thumb for English text.
CHARS_PER_TOKEN = 4.0
# Small fixed allowance for the structured-output JSON wrapper around the judge's
# justification text (`{"is_correct": true, "justification": "..."}`), which isn't
# reflected in the justification string itself.
JSON_WRAPPER_TOKENS = 10


def est_tokens(text) -> int:
    """len(text) / CHARS_PER_TOKEN, 0 for missing/non-string text."""
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return 0
    return int(len(str(text)) / CHARS_PER_TOKEN)


def load_model_prices(models_csv: str) -> dict[str, tuple[float | None, float | None]]:
    """model_id -> (input_cost, output_cost), USD per 1M tokens, from data/models.csv.
    Either value is None where models.csv leaves that model's pricing blank."""
    df = pd.read_csv(models_csv)
    prices: dict[str, tuple[float | None, float | None]] = {}
    for _, row in df.iterrows():
        ic, oc = row["input_cost"], row["output_cost"]
        prices[row["model_id"]] = (
            float(ic) if pd.notna(ic) else None,
            float(oc) if pd.notna(oc) else None,
        )
    return prices


def run_bench_cost(
    responses_df: pd.DataFrame, input_cost: float | None, output_cost: float | None
) -> tuple[float | None, int, int]:
    """(cost, prompt_tokens, completion_tokens) for one model's run-bench pass -- exact,
    from the real total_tokens/completion_tokens columns fdllm recorded per call. cost is
    None if this model has no pricing in models.csv."""
    completion = responses_df["completion_tokens"].fillna(0)
    total = responses_df["total_tokens"].fillna(0)
    input_tokens = (total - completion).clip(lower=0)
    prompt_tokens, completion_tokens = int(input_tokens.sum()), int(completion.sum())
    if input_cost is None or output_cost is None:
        return None, prompt_tokens, completion_tokens
    cost = prompt_tokens / 1e6 * input_cost + completion_tokens / 1e6 * output_cost
    return cost, prompt_tokens, completion_tokens


def build_judge_user_prompt(
    question: str, output: str, expected_answers: list[str], answer_type: str
) -> str:
    """Mirrors judge_bench.score_with_llm_judge()'s user_prompt f-string exactly --
    keep in sync if that changes. Needed to reconstruct judge-call input length, since
    judge_bench.py doesn't persist the prompt text itself."""
    return (
        f"Question: {question}\n\n"
        f"Model's answer: {output}\n\n"
        f"Acceptable expected answers: {', '.join(expected_answers)}\n\n"
        f"Answer type: {answer_type}\n\n"
        "Is the model's answer correct?"
    )


def judge_call_cost(
    judged_df: pd.DataFrame,
    question_map: dict,
    judge_models: list[str],
    system_prompt: str,
    *,
    prices: dict[str, tuple[float | None, float | None]],
) -> tuple[float | None, int, int, int]:
    """Estimated (cost, n_calls, prompt_tokens, completion_tokens) across every judge in
    `judge_models`, for whichever rows the main judge-bench pass actually sent to a judge
    (short_form rows). A row counts as "sent to this judge" when its
    is_correct_<judge> column is non-null (judge_bench.py always populates it, even for
    a False verdict; only never-attempted rows are null). cost is 0.0 when there's
    nothing to price (n_calls == 0); None if there WERE calls but at least one judge
    involved has no pricing in models.csv (a true "can't price this")."""
    total_cost = 0.0
    unpriced_calls = 0
    n_calls = prompt_tokens = completion_tokens = 0
    for jm in judge_models:
        col, just_col = f"is_correct_{jm}", f"justification_{jm}"
        if col not in judged_df.columns:
            continue
        scored = judged_df[judged_df[col].notna()]
        if scored.empty:
            continue
        n_calls += len(scored)
        input_cost, output_cost = prices.get(jm, (None, None))
        priced = input_cost is not None and output_cost is not None
        for _, row in scored.iterrows():
            question = question_map.get(row["question_id"], "")
            expected = str(row["expected_answers"]).split("|")
            output = strip_think_block(row["output"])
            user_prompt = build_judge_user_prompt(
                question, output, expected, row["answer_type"]
            )
            p_tok = est_tokens(system_prompt) + est_tokens(user_prompt)
            c_tok = est_tokens(row.get(just_col)) + JSON_WRAPPER_TOKENS
            prompt_tokens += p_tok
            completion_tokens += c_tok
            if priced:
                total_cost += p_tok / 1e6 * input_cost + c_tok / 1e6 * output_cost
            else:
                unpriced_calls += 1
    cost = None if unpriced_calls > 0 else total_cost
    return cost, n_calls, prompt_tokens, completion_tokens


def discover_models(responses_dir: str) -> list[str]:
    prefix, suffix = "responses_", ".csv"
    return sorted(
        f[len(prefix) : -len(suffix)]
        for f in os.listdir(responses_dir)
        if f.startswith(prefix) and f.endswith(suffix)
    )


def _fmt_usd(x) -> str:
    """ "$0.12" / "-" for None or NaN. 2dp is plenty for a per-model summary line."""
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "-"
    return f"${x:.2f}"


def print_cost_table(result_df: pd.DataFrame) -> None:
    """Compact console table: $-formatted costs only (no token/call counts), sorted by
    total cost descending so the priciest model(s) are immediately visible; unpriced
    models sort last."""
    cols = ["model", "run_bench_cost_usd", "judge_cost_est_usd", "total_cost_est_usd"]
    headers = ["model", "run-bench", "judge", "total"]
    ordered = result_df.sort_values(
        "total_cost_est_usd", ascending=False, na_position="last"
    )
    fmted = [
        [str(row["model"])] + [_fmt_usd(row[c]) for c in cols[1:]]
        for _, row in ordered.iterrows()
    ]
    widths = [
        max(len(headers[i]), *(len(r[i]) for r in fmted)) for i in range(len(headers))
    ]
    align = [str.ljust] + [str.rjust] * (len(headers) - 1)
    header_line = "  ".join(pad(h, w) for h, w, pad in zip(headers, widths, align))
    print(header_line)
    print("-" * len(header_line))
    for r in fmted:
        print("  ".join(pad(v, w) for v, w, pad in zip(r, widths, align)))


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--language", required=True, choices=LANGUAGES)
    ap.add_argument(
        "--experiment-id",
        required=True,
        help="Bare cache experiment-id, e.g. 'final-137-items' (no '_fallback' suffix -- see "
        "the module docstring's note on --experiment-id).",
    )
    ap.add_argument(
        "--dataset",
        default=None,
        help="Filename only (resolved as data/<dataset>), matching run-bench/judge-bench's "
        "own --dataset override. Defaults to this language's normal dataset; only needed "
        "if this experiment-id was produced against a different one (e.g. a consistency-check "
        "repeated dataset).",
    )
    ap.add_argument("--models-csv", default=None)
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    models_csv = args.models_csv or os.path.join(root, "data", "models.csv")

    dataset_filename = args.dataset or f"linguistic_knowledge_eval_{args.language}.json"
    questions = load_dataset(os.path.join(root, "data", dataset_filename))
    question_map = {q["question_id"]: q["question"] for q in questions}
    judge_system_prompt = _build_judge_system_prompt(
        LANGUAGE_DISPLAY_NAMES[args.language]
    )

    experiment_path = os.path.join(
        root, "cache", args.language, "experiments", args.experiment_id
    )
    responses_dir = os.path.join(experiment_path, "responses")
    judged_dir = os.path.join(experiment_path, "responses_judged")
    log_path = os.path.join(experiment_path, "log.json")

    if not os.path.isdir(responses_dir):
        raise SystemExit(
            f"No experiment found: {responses_dir} does not exist.\n"
            f"Check --language/--experiment-id -- --experiment-id is the bare cache experiment-id "
            "(no '_fallback' suffix)."
        )

    log_data: dict = {}
    if os.path.exists(log_path):
        with open(log_path, encoding="utf-8") as f:
            log_data = json.load(f)

    prices = load_model_prices(models_csv)
    models = discover_models(responses_dir)

    rows = []
    for model in models:
        responses_df = pd.read_csv(
            os.path.join(responses_dir, f"responses_{model}.csv")
        )
        input_cost, output_cost = prices.get(model, (None, None))
        rb_cost, rb_prompt_tok, rb_completion_tok = run_bench_cost(
            responses_df, input_cost, output_cost
        )

        judge_cost, judge_calls, judge_prompt_tok, judge_completion_tok = (
            None,
            0,
            0,
            0,
        )
        judged_path = os.path.join(judged_dir, f"responses_judged_{model}.csv")
        has_judged = os.path.exists(judged_path)
        if has_judged:
            judged_df = pd.read_csv(judged_path)
            judge_models = log_data.get(model, {}).get("llm_judges") or list(
                DEFAULT_JUDGE_MODELS[:1]
            )
            judge_cost, judge_calls, judge_prompt_tok, judge_completion_tok = (
                judge_call_cost(
                    judged_df,
                    question_map,
                    judge_models,
                    judge_system_prompt,
                    prices=prices,
                )
            )

        missing_pricing = rb_cost is None or (has_judged and judge_cost is None)
        # total_cost_est_usd is None (not a partial sum) whenever any component is
        # unpriced -- a partial total would understate cost while looking complete.
        total_cost = None if missing_pricing else (rb_cost or 0.0) + (judge_cost or 0.0)

        rows.append(
            {
                "model": model,
                "run_bench_cost_usd": (
                    round(rb_cost, 4) if rb_cost is not None else None
                ),
                "run_bench_prompt_tokens": rb_prompt_tok,
                "run_bench_completion_tokens": rb_completion_tok,
                "judge_cost_est_usd": (
                    round(judge_cost, 4) if judge_cost is not None else None
                ),
                "judge_calls": judge_calls,
                "total_cost_est_usd": (
                    round(total_cost, 4) if total_cost is not None else None
                ),
                "missing_pricing": missing_pricing,
            }
        )

    result_df = pd.DataFrame(rows)
    print_cost_table(result_df)

    priced = result_df[result_df["total_cost_est_usd"].notna()]
    if len(priced):
        print(
            f"\nPer-model total cost (run-bench + judge-bench), "
            f"n={len(priced)}/{len(result_df)} priced model(s):"
        )
        print(f"  min:  ${priced['total_cost_est_usd'].min():.2f}")
        print(f"  mean: ${priced['total_cost_est_usd'].mean():.2f}")
        print(f"  max:  ${priced['total_cost_est_usd'].max():.2f}")

    unpriced = result_df.loc[result_df["missing_pricing"], "model"].tolist()
    if unpriced:
        print(
            f"\n{len(unpriced)} model(s) missing pricing in {models_csv} "
            f"(excluded from the stats above): {', '.join(unpriced)}"
        )


if __name__ == "__main__":
    main()
