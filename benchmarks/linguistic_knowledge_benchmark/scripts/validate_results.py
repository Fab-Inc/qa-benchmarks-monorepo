#!/usr/bin/env python3
"""
Robustness/validation checks for a Linguistic-Knowledge benchmark experiment.
Each check is a subcommand; --language and --experiment-id are mandatory on
every one of them and are what locate the three CSVs `judge-bench` already produced for
that experiment:

    results/summary_linguistic-knowledge_<language>_<experiment-id>.csv                 SUMMARY
    results/summary_linguistic-knowledge_<language>_<experiment-id>_detailed.csv        DETAILED
    results/summary_linguistic-knowledge_<language>_<experiment-id>_judge_agreement.csv JUDGE AGREEMENT

Three checks (judge-agreement's kappa, self-preference-bias, fallback-rate) additionally
read the raw per-judge CSVs under cache/<language>/experiments/<experiment-id>/responses_judged[_fallback]/
-- the bare cache experiment-id is derived by stripping a trailing '_fallback' from --experiment-id,
since that suffix is only ever appended to the results/ filenames, never to the cache
directory (see cache_paths_for() below).

Subcommands and their outputs (under --outdir/<language>/, --outdir defaults to
results/validation/, filenames tagged with --label, default "<language>_<experiment-id>"):

  judge-agreement        Raw pairwise judge agreement (re-reads judge-bench's own
                          judge_agreement CSV) plus Cohen's kappa per judge pair (chance-
                          corrected -- raw agreement is inflated when most answers are
                          correct) and a full-consensus / split breakdown, as counts and
                          as percentages of the rows the judges actually scored (rows
                          where the answering model returned no output are reported
                          separately, not folded into the consensus rates).
                          -> judge_agreement_kappa_<label>.csv

  self-preference-bias   For each judge, the rate at which its verdict is more lenient
                          (says correct) than the majority of the other two judges,
                          split by whether the answering model shares the judge's
                          provider -- a materially higher same-provider rate is the bias
                          signal.
                          -> self_preference_bias_<label>.csv

  fallback-rate           Per model, the fraction of MCQ rows resolved by the
                          --judge-fallback LLM pass vs still unscored after it, joined
                          against overall accuracy -- flags whether a low score is a
                          knowledge gap or a format-compliance/parsing issue. Requires a
                          experiment-id with a fallback pass (cache/.../responses_judged_fallback/).
                          -> fallback_rate_<label>.csv

  bootstrap-ci            Per-model 95% CI on accuracy via a paired/cluster bootstrap over
                          question_id (same resampled item set applied to every model each
                          replicate), from the detailed CSV. The plot matches
                          analyze_results.py's plot_accuracy() (provider-colored bars,
                          value labels, provider legend), with the CI as error bars.
                          -> bootstrap_ci_<label>.csv, bootstrap_ci_<label>.png

  family-grid              A grid of small bar charts (one per model family, from
                          data/model_families.csv), each family's models sorted by
                          accuracy (highest first). The CSV also records each model's
                          generation/size order_index (release_date, falling back to
                          parameter size, both from models.csv), for a from-scratch
                          old->new / small->big consistency check even though the plot
                          itself is sorted by accuracy.
                          -> family_grid_<label>.png, family_grid_<label>.csv

  consistency             Replicate-to-replicate accuracy variance, from an experiment
                          scored against a
                          REPEATED dataset (see build_consistency_dataset.py: N copies of
                          every question, question_id "<original_id>.<replicate>") --
                          decodes the replicate back out of question_id, groups the
                          detailed CSV's final judged (post-majority-vote) `<model>_correct`
                          by replicate, and reports/plots each model's accuracy spread
                          across the N replicates. Aborts with next-step instructions if
                          the experiment's 3 result CSVs aren't present yet, or if the detailed
                          CSV looks unscored (see build_consistency_dataset.py's docstring
                          for why that happens and how to fix it).
                          -> consistency_<label>.csv, consistency_<label>_raw.csv,
                             consistency_<label>.png

  all                     Runs every check above, in order, for this --language/--experiment-id.
                          A check is SKIPPED (not overwritten) if its output file(s)
                          already exist under <outdir>/<language>/ -- pass a different
                          --label, or delete those files, to force a rerun. A check that
                          errors (e.g. fallback-rate with no fallback pass for this
                          experiment, or consistency with no repeated-dataset experiment
                          yet) is reported and
                          skipped, not fatal to the rest.

Run (--language and --experiment-id are mandatory on every subcommand):
    uv run python benchmarks/linguistic_knowledge_benchmark/scripts/validate_results.py judge-agreement --language english --experiment-id final-137-items_fallback
    uv run python .../validate_results.py self-preference-bias --language kiswahili --experiment-id final-135-items_fallback
    uv run python .../validate_results.py fallback-rate --language english --experiment-id final-137-items_fallback
    uv run python .../validate_results.py bootstrap-ci --language english --experiment-id final-137-items_fallback --n-bootstrap 2000
    uv run python .../validate_results.py family-grid --language english --experiment-id final-137-items_fallback
    uv run python .../validate_results.py consistency --language english --experiment-id validation-20x
    uv run python .../validate_results.py all --language english --experiment-id final-137-items_fallback
    uv run python .../validate_results.py all --language kiswahili --experiment-id final-135-items_fallback

Producing an experiment for the "consistency" check (external to this script):
    uv run python scripts/build_consistency_dataset.py --language english
    # run-bench/judge-bench both take --dataset directly -- pass the SAME one to both,
    # no dataset-file swapping and no separate detailed-CSV rebuild step needed.
    # --dataset is FILENAME ONLY (always resolved as DATA_DIR/<dataset>):
    uv run linguistic-knowledge-benchmark run-bench --language english --experiment-id validation-20x \\
        --dataset linguistic_knowledge_eval_english_20x.json
    uv run linguistic-knowledge-benchmark judge-bench --language english --experiment-id validation-20x \\
        --dataset linguistic_knowledge_eval_english_20x.json --num-judges 3
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import sys
from itertools import combinations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from analyze_results import (
    LANGUAGES,
    _bar_colors,
    _labels,
    _provider_legend_handles,
    load_detailed,
    load_display_names,
    load_judged_model,
    load_provider_colors,
    load_summary,
)
from linguistic_knowledge_benchmark.judge_bench import (
    DEFAULT_JUDGE_MODELS,
    compute_judge_agreement,
)

# scripts/ isn't part of the installed package, but the package itself is -- this import
# is safe (unlike importing linguistic_knowledge_benchmark.cmd) because it never touches
# cmd.py's app.callback(), the only place that pulls in typer/dotenv/network side effects.
from linguistic_knowledge_benchmark.shared_utils import MODELS_CSV, PROVIDERS_CSV

CHECKS = [
    "judge-agreement",
    "self-preference-bias",
    "fallback-rate",
    "bootstrap-ci",
    "family-grid",
    "consistency",
]

DEFAULT_SEED = 0  # bootstrap-ci's --seed default, for reproducible CIs across reruns

# benchmark_domain key to read out of a benchmark_default_models.json (written by
# scripts/results_for_website.py, which ids its pairs "<benchmark>-<ISO 639-1 code>").
DEFAULT_MODELS_DOMAIN = {"english": "LK-en", "kiswahili": "LK-sw"}
# Appended to the auto-generated --label when --models-json filters the model set, so a
# filtered run cannot silently overwrite the all-models outputs. An explicit --label wins.
DEFAULT_MODELS_LABEL_SUFFIX = "_default-models"
DEFAULT_N_REPLICATES = 20  # consistency's --n-replicates default; must match whatever
# build_consistency_dataset.py --n-replicates was used to build the experiment's dataset

# --------------------------------------------------------------------------- #
# Console summary formatting (same pattern as filter_dataset.py / estimate_irt.py)
# --------------------------------------------------------------------------- #
_RULE = "=" * 70


def _section(title: str) -> None:
    print(f"\n{_RULE}\n{title}\n{_RULE}")


def _kv(label: str, value) -> None:
    print(f"  {label:<28}: {value}")


def _print_pair_range(matrix: pd.DataFrame, what: str) -> None:
    """Print the min/max of a judge x judge matrix over its OFF-DIAGONAL entries.

    The diagonal is 1.0 by construction (a judge always agrees with itself), so a
    range including it would always max out at 1.000 and say nothing. The off-diagonal
    is one value per judge PAIR -- the range actually quoted for a judge panel. NaN
    pairs (two judges that never scored a common row) are skipped.
    """
    pairs = {
        (a, b): matrix.loc[a, b]
        for a, b in combinations(matrix.index, 2)
        if pd.notna(matrix.loc[a, b])
    }
    if not pairs:
        print(f"\n  (no judge pair shares a scored row -- no {what} range)")
        return
    lo = min(pairs, key=pairs.get)
    hi = max(pairs, key=pairs.get)
    print(f"\n  min {what}  {pairs[lo]:.4f}   {lo[0]} vs {lo[1]}")
    print(f"  max {what}  {pairs[hi]:.4f}   {hi[0]} vs {hi[1]}")


# --------------------------------------------------------------------------- #
# Path resolution
# --------------------------------------------------------------------------- #
def default_paths_for(language: str, results_dir: str, experiment_id: str) -> dict:
    """summary/detailed/judge_agreement CSV paths -- --experiment-id used verbatim, exactly like
    analyze_results.py's own default_paths_for (include the literal '_fallback' suffix in
    --experiment-id if that's how the experiment was produced)."""
    base = f"summary_linguistic-knowledge_{language}_{experiment_id}"
    return {
        "summary": os.path.join(results_dir, f"{base}.csv"),
        "detailed": os.path.join(results_dir, f"{base}_detailed.csv"),
        "judge_agreement": os.path.join(results_dir, f"{base}_judge_agreement.csv"),
    }


def cache_paths_for(language: str, root: str, experiment_id: str) -> tuple[str, bool]:
    """(cache experiment-dir under cache/<language>/experiments/, is_fallback).

    The cache directory is never suffixed with '_fallback' -- that suffix is only ever
    appended to the results/ filenames (fallback data lives in a responses_judged_fallback/
    subdirectory of the SAME experiment). Since --experiment-id here may already have '_fallback' baked
    into it (per the results/ naming convention), strip that one known suffix to recover
    the bare cache experiment-id, rather than adding a separate CLI flag.
    """
    is_fallback = experiment_id.endswith("_fallback")
    bare_experiment_id = (
        experiment_id[: -len("_fallback")] if is_fallback else experiment_id
    )
    return os.path.join(
        root, "cache", language, "experiments", bare_experiment_id
    ), is_fallback


def discover_models_in_experiment(cache_experiment_dir: str) -> list[str]:
    """Model ids with a primary judged CSV under cache_experiment_dir/responses_judged/."""
    judged_dir = os.path.join(cache_experiment_dir, "responses_judged")
    if not os.path.isdir(judged_dir):
        return []
    prefix, suffix = "responses_judged_", ".csv"
    return sorted(
        f[len(prefix) : -len(suffix)]
        for f in os.listdir(judged_dir)
        if f.startswith(prefix) and f.endswith(suffix)
    )


def load_full_judged(
    cache_experiment_dir: str, model: str, is_fallback: bool
) -> pd.DataFrame | None:
    """The fully-resolved judged CSV for `model`: the fallback pass's file (with its
    final, post-fallback score_method) if is_fallback, else the primary pass's file.
    None if the expected file doesn't exist."""
    subdir = "responses_judged_fallback" if is_fallback else "responses_judged"
    path = os.path.join(cache_experiment_dir, subdir, f"responses_judged_{model}.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def judge_models_from_summary(
    summary_df: pd.DataFrame,
    column: str = "short_form_llm_judges",
    default: list[str] | None = None,
) -> list[str]:
    """The judge ensemble recorded in a summary CSV column (a stringified python list).

    `column` selects which pass: 'short_form_llm_judges' is the short-form ensemble
    (the default), 'mcq_llm_judges_fallback' the MCQ --judge-fallback pass, which is
    usually a single judge rather than the full ensemble. Returns `default`
    (DEFAULT_JUDGE_MODELS when not given) if the column is missing/unparseable -- pass
    an explicit [] where guessing would be worse than reporting 'unknown'.
    """
    if column in summary_df.columns:
        for val in summary_df[column].dropna():
            try:
                parsed = ast.literal_eval(val) if isinstance(val, str) else val
            except (ValueError, SyntaxError):
                continue
            if isinstance(parsed, list) and parsed:
                return parsed
    return list(DEFAULT_JUDGE_MODELS if default is None else default)


def load_all_judged(
    cache_experiment_dir: str, models: list[str]
) -> dict[str, pd.DataFrame]:
    """model -> load_judged_model() result, skipping models with no judged CSV."""
    out: dict[str, pd.DataFrame] = {}
    for model in models:
        df = load_judged_model(cache_experiment_dir, model)
        if df is not None:
            out[model] = df
    return out


# --------------------------------------------------------------------------- #
# 1. judge-agreement
# --------------------------------------------------------------------------- #
def compute_kappa(judged_df: pd.DataFrame, judge_models: list[str]) -> pd.DataFrame:
    """Cohen's kappa per judge pair, over rows both judges scored (is_correct_<judge>
    columns). Chance-corrected -- unlike raw agreement, kappa isn't inflated just because
    most answers happen to be correct."""
    cols = {jm: f"is_correct_{jm}" for jm in judge_models}
    kappa = pd.DataFrame(index=judge_models, columns=judge_models, dtype=float)
    for a in judge_models:
        for b in judge_models:
            if a == b:
                kappa.loc[a, b] = 1.0
                continue
            both = judged_df[[cols[a], cols[b]]].dropna()
            if both.empty:
                kappa.loc[a, b] = float("nan")
                continue
            po = (both[cols[a]] == both[cols[b]]).mean()
            pa_true = both[cols[a]].mean()
            pb_true = both[cols[b]].mean()
            pe = pa_true * pb_true + (1 - pa_true) * (1 - pb_true)
            kappa.loc[a, b] = float("nan") if pe >= 1.0 else (po - pe) / (1 - pe)
    return kappa


def run_judge_agreement(args: argparse.Namespace, root: str) -> None:
    paths = default_paths_for(
        args.language, os.path.join(root, "results"), args.experiment_id
    )
    cache_experiment_dir, _ = cache_paths_for(args.language, root, args.experiment_id)

    summary = load_summary(paths["summary"])
    judge_models = judge_models_from_summary(summary)

    _section("CONFIG")
    _kv("language", args.language)
    _kv("experiment_id", args.experiment_id)
    _kv("judge_models", judge_models)
    _kv("cache_experiment_dir", cache_experiment_dir)

    if len(judge_models) < 2:
        raise SystemExit(
            f"Only {len(judge_models)} judge(s) were used for this experiment -- nothing "
            "compute agreement/kappa over (need >=2)."
        )

    if os.path.exists(paths["judge_agreement"]):
        precomputed = pd.read_csv(paths["judge_agreement"], index_col=0)
        _section("RAW PAIRWISE AGREEMENT (from judge-bench's own output)")
        print(precomputed.round(4).to_string())
        _print_pair_range(precomputed, "pairwise agreement")
    else:
        print(
            f"\n(no {paths['judge_agreement']} found -- recomputing raw agreement below)"
        )

    models = discover_models_in_experiment(cache_experiment_dir)
    if not models:
        raise SystemExit(
            f"No judged CSVs found under {cache_experiment_dir}/responses_judged/."
        )
    judged_by_model = load_all_judged(cache_experiment_dir, models)
    all_judged = pd.concat(
        [df.assign(model=m) for m, df in judged_by_model.items()], ignore_index=True
    )

    cols = [f"is_correct_{jm}" for jm in judge_models]
    missing = [c for c in cols if c not in all_judged.columns]
    if missing:
        raise SystemExit(
            f"judged CSVs are missing column(s) {missing} -- this experiment may have used a "
            "different judge ensemble than the summary CSV's short_form_llm_judges."
        )

    raw_agreement = compute_judge_agreement(all_judged, judge_models, fallback=False)
    kappa = compute_kappa(all_judged, judge_models)

    _section("RAW AGREEMENT (recomputed across all models' judged rows)")
    print(raw_agreement.round(4).to_string())
    _print_pair_range(raw_agreement, "pairwise agreement")
    _section("COHEN'S KAPPA (chance-corrected)")
    print(kappa.round(4).to_string())
    _print_pair_range(kappa, "kappa")

    # judge_agreement is NaN when the ANSWERING model returned no output (call error or
    # empty completion), so the judges were never called on that row -- those rows are
    # not a judge-agreement outcome and are excluded from the consensus percentages, the
    # same denominator compute_judge_agreement/compute_kappa already use (they correlate
    # over rows both judges scored). Both counts are printed so the two reconcile.
    sf = all_judged[all_judged["answer_type"] == "short_form"]
    unscored = sf["judge_agreement"].isna()
    n_expected = len(sf)
    n_unscored = int(unscored.sum())
    n_judged = n_expected - n_unscored

    def pct(count: int, total: int) -> str:
        return f"{100.0 * count / total:6.2f}%" if total else "   n/a"

    _section("CONSENSUS DISTRIBUTION (short_form rows)")
    print(f"  {'expected to judge':<30} {n_expected:6d}")
    print(
        f"  {'unscored (no model output)':<30} {n_unscored:6d}  "
        f"{pct(n_unscored, n_expected)} of expected"
    )
    print(
        f"  {'judged (effective denominator)':<30} {n_judged:6d}  "
        f"{pct(n_judged, n_expected)} of expected"
    )
    print(f"\n  of those {n_judged} judged rows:")
    dist = (
        sf.loc[~unscored, "judge_agreement"]
        .value_counts()
        .sort_index(ascending=False)  # full consensus (1.0) first
    )
    for val, count in dist.items():
        label = (
            "full consensus (responding judges)"
            if val == 1.0
            else f"split, resolved by majority vote (agreement={val:.3f})"
        )
        print(f"    {label:<52} {count:6d}  {pct(count, n_judged)}")

    os.makedirs(args.outdir, exist_ok=True)
    out_path = os.path.join(args.outdir, f"judge_agreement_kappa_{args.label}.csv")
    combined = (
        raw_agreement.round(4)
        .add_suffix("_raw_agreement")
        .join(kappa.round(4).add_suffix("_kappa"))
    )
    combined.to_csv(out_path)
    print(f"\nWrote: {out_path}")


# --------------------------------------------------------------------------- #
# 2. self-preference-bias
# --------------------------------------------------------------------------- #
def compute_self_preference(
    judged_by_model: dict[str, pd.DataFrame],
    judge_models: list[str],
    model_provider: dict,
) -> pd.DataFrame:
    """Per (judge, same_provider) leniency rate: among rows all `judge_models` scored,
    the fraction where `judge` alone says correct while the OTHER two judges agree with
    each other and say incorrect (rows where the other two disagree with each other have
    no well-defined "majority of the others" and are excluded)."""
    cols = {jm: f"is_correct_{jm}" for jm in judge_models}
    rows: list[dict] = []
    for model, df in judged_by_model.items():
        if not all(cols[jm] in df.columns for jm in judge_models):
            continue
        model_prov = model_provider.get(model)
        sub = df[list(cols.values())].dropna()
        if sub.empty:
            continue
        for jm in judge_models:
            other_cols = [cols[k] for k in judge_models if k != jm]
            other_sum = sub[other_cols].sum(axis=1)
            defined = other_sum.isin([0, len(other_cols)])  # the others agree
            other_majority_true = other_sum[defined] == len(other_cols)
            judge_verdict = sub.loc[defined, cols[jm]]
            lenient = judge_verdict & ~other_majority_true
            same_provider = model_provider.get(jm) == model_prov
            rows.append(
                {
                    "judge": jm,
                    "same_provider": same_provider,
                    "n_rows": int(defined.sum()),
                    "n_lenient": int(lenient.sum()),
                }
            )
    if not rows:
        return pd.DataFrame(
            columns=[
                "judge",
                "same_provider_leniency_rate",
                "other_provider_leniency_rate",
                "delta",
                "n_same",
                "n_other",
            ]
        )
    long = pd.DataFrame(rows).groupby(["judge", "same_provider"], as_index=False).sum()

    out_rows = []
    for jm in judge_models:
        same = long[(long["judge"] == jm) & (long["same_provider"])]
        other = long[(long["judge"] == jm) & (~long["same_provider"])]
        n_same = int(same["n_rows"].sum())
        n_other = int(other["n_rows"].sum())
        same_rate = same["n_lenient"].sum() / n_same if n_same else float("nan")
        other_rate = other["n_lenient"].sum() / n_other if n_other else float("nan")
        out_rows.append(
            {
                "judge": jm,
                "same_provider_leniency_rate": same_rate,
                "other_provider_leniency_rate": other_rate,
                "delta": same_rate - other_rate,
                "n_same": n_same,
                "n_other": n_other,
            }
        )
    return pd.DataFrame(out_rows)


def run_self_preference_bias(args: argparse.Namespace, root: str) -> None:
    paths = default_paths_for(
        args.language, os.path.join(root, "results"), args.experiment_id
    )
    cache_experiment_dir, _ = cache_paths_for(args.language, root, args.experiment_id)

    summary = load_summary(paths["summary"])
    judge_models = judge_models_from_summary(summary)
    model_provider, _ = load_provider_colors(args.models_csv, args.providers_csv)

    _section("CONFIG")
    _kv("language", args.language)
    _kv("experiment_id", args.experiment_id)
    _kv("judge_models", judge_models)

    if len(judge_models) < 3:
        raise SystemExit(
            f"self-preference-bias needs a >=3-judge ensemble (this experiment used "
            f"{len(judge_models)}) -- 'majority of the other judges' isn't well-defined "
            "with fewer than 2 others."
        )

    missing_provider = [jm for jm in judge_models if jm not in model_provider]
    if missing_provider:
        print(
            f"WARNING: no provider found in {args.models_csv} for judge(s) "
            f"{missing_provider} -- their same-provider rows will be undercounted."
        )

    models = discover_models_in_experiment(cache_experiment_dir)
    if not models:
        raise SystemExit(
            f"No judged CSVs found under {cache_experiment_dir}/responses_judged/."
        )
    judged_by_model = load_all_judged(cache_experiment_dir, models)

    report = compute_self_preference(judged_by_model, judge_models, model_provider)

    _section(
        "SELF-PREFERENCE BIAS (leniency = says correct when the other judges agree it's not)"
    )
    for _, r in report.iterrows():
        print(
            f"  {r['judge']:<20} same-provider {r['same_provider_leniency_rate']:.3f} "
            f"(n={r['n_same']})   other-provider {r['other_provider_leniency_rate']:.3f} "
            f"(n={r['n_other']})   delta {r['delta']:+.3f}"
        )

    os.makedirs(args.outdir, exist_ok=True)
    out_path = os.path.join(args.outdir, f"self_preference_bias_{args.label}.csv")
    report.round(4).to_csv(out_path, index=False)
    print(f"\nWrote: {out_path}")


# --------------------------------------------------------------------------- #
# 3. fallback-rate
# --------------------------------------------------------------------------- #
def run_fallback_rate(args: argparse.Namespace, root: str) -> None:
    paths = default_paths_for(
        args.language, os.path.join(root, "results"), args.experiment_id
    )
    cache_experiment_dir, is_fallback = cache_paths_for(
        args.language, root, args.experiment_id
    )

    _section("CONFIG")
    _kv("language", args.language)
    _kv("experiment_id", args.experiment_id)
    _kv("cache_experiment_dir", cache_experiment_dir)

    fallback_dir = os.path.join(cache_experiment_dir, "responses_judged_fallback")
    if not is_fallback or not os.path.isdir(fallback_dir):
        raise SystemExit(
            f"No fallback pass found for this experiment (expected {fallback_dir}) -- "
            "fallback-rate needs --experiment-id to be an experiment judged with "
            "'judge-bench --judge-fallback' (pass the experiment-id with its '_fallback' suffix, "
            "e.g. --experiment-id final-137-items_fallback)."
        )

    summary = load_summary(paths["summary"])
    accuracy_by_model = dict(zip(summary["model"], summary["accuracy"]))

    # Who re-judged the unparseable MCQ rows. Recorded per experiment in the summary's
    # own 'mcq_llm_judges_fallback' column; [] rather than DEFAULT_JUDGE_MODELS as the
    # default, since a wrong judge name in a provenance line is worse than "unknown".
    # It is usually ONE judge (not the short-form ensemble), which is why the fallback
    # verdicts below have no majority vote behind them.
    fallback_judges = judge_models_from_summary(
        summary, column="mcq_llm_judges_fallback", default=[]
    )

    models = discover_models_in_experiment(cache_experiment_dir)
    rows = []
    judges_in_files: set[str] = set()
    for model in models:
        df = load_full_judged(cache_experiment_dir, model, is_fallback=True)
        if df is None:
            continue
        judges_in_files.update(
            c[len("is_correct_fallback_") :]
            for c in df.columns
            if c.startswith("is_correct_fallback_")
        )
        mcq = df[df["answer_type"] == "multiple_choice"]
        mcq_total = len(mcq)
        if mcq_total == 0:
            continue
        mcq_fallback_sent = int((mcq["score_method"] == "llm_fallback").sum())
        mcq_still_unscored = int((mcq["score_method"] == "unscored").sum())

        # Accuracy before vs after the fallback pass, over ALL answer types (that is
        # what the headline accuracy is). "Before" = the rows the primary pass scored
        # (is_correct non-null: regex-parsed MCQ + judged short-form); "after" adds the
        # rows the fallback judge rescued, using is_correct_fallback where present --
        # the same best-available-verdict rule compute_summary() applies. The two have
        # DIFFERENT denominators by design: that is exactly the effect of the fallback
        # pass, which converts unscored rows into scored ones.
        primary = df["is_correct"].map(CORRECT_MAP)
        effective = (
            df["is_correct_fallback"].map(CORRECT_MAP).fillna(primary)
            if "is_correct_fallback" in df.columns
            else primary
        )
        n_before, n_after = int(primary.notna().sum()), int(effective.notna().sum())
        correct_before, correct_after = int(primary.sum()), int(effective.sum())
        fb_rows = effective[mcq.index].where(mcq["score_method"] == "llm_fallback")

        rows.append(
            {
                "model": model,
                "mcq_total": mcq_total,
                "mcq_fallback_sent": mcq_fallback_sent,
                "mcq_fallback_rate": round(mcq_fallback_sent / mcq_total, 4),
                "mcq_fallback_correct": int(fb_rows.sum()),
                "mcq_still_unscored": mcq_still_unscored,
                "n_scored_before": n_before,
                "correct_before": correct_before,
                "accuracy_before": round(100.0 * correct_before / n_before, 2)
                if n_before
                else None,
                "n_scored_after": n_after,
                "correct_after": correct_after,
                "accuracy_after": round(100.0 * correct_after / n_after, 2)
                if n_after
                else None,
                "accuracy": accuracy_by_model.get(model),
            }
        )

    report = pd.DataFrame(rows).sort_values("mcq_fallback_rate", ascending=False)
    used = report[report["mcq_fallback_sent"] > 0]
    n_models, n_used = len(report), len(used)

    def share(count: int, total: int) -> str:
        return f"{100.0 * count / total:5.1f}%" if total else "  n/a"

    _section("FALLBACK USE")
    named = ", ".join(fallback_judges) if fallback_judges else "unknown"
    print(f"  {'fallback judge(s)':<34} {named}")
    # The summary says who was configured; the judged CSVs' is_correct_fallback_<judge>
    # columns say who actually voted. Disagreement means the summary and the cache come
    # from different fallback passes -- the verdicts counted below are the files'.
    if judges_in_files and judges_in_files != set(fallback_judges):
        print(
            f"  {'':<34} WARNING: judged files carry verdicts from "
            f"{', '.join(sorted(judges_in_files))}"
        )
    print(f"  {'models judged':<34} {n_models:4d}")
    print(
        f"  {'with >=1 fallback-judged row':<34} {n_used:4d}   "
        f"{share(n_used, n_models)} of models judged"
    )
    print(
        f"  {'with no fallback at all':<34} {n_models - n_used:4d}   "
        f"{share(n_models - n_used, n_models)} of models judged"
    )

    _section("PER-MODEL RECOVERY (only models with >=1 fallback-judged row)")
    print(
        f"  {'model':<32}{'fallback judged':>19}{'of which correct':>20}"
        f"{'accuracy before':>22}{'accuracy after':>22}"
    )
    for _, r in used.iterrows():
        sent, correct = int(r["mcq_fallback_sent"]), int(r["mcq_fallback_correct"])
        print(
            f"  {r['model']:<32}"
            f"{f'{sent}/{int(r.mcq_total)} MCQ':>11}{f' ({r.mcq_fallback_rate:.1%})':>8}"
            f"{correct:>11}{f'({share(correct, sent).strip()})':>9}"
            f"{f'{r.accuracy_before:.1f}%':>9}{f'({int(r.correct_before)}/{int(r.n_scored_before)})':>13}"
            f"{f'{r.accuracy_after:.1f}%':>9}{f'({int(r.correct_after)}/{int(r.n_scored_after)})':>13}"
        )

    os.makedirs(args.outdir, exist_ok=True)
    out_path = os.path.join(args.outdir, f"fallback_rate_{args.label}.csv")
    report.to_csv(out_path, index=False)
    print(
        f"\n  (the CSV below keeps all {n_models} models, including the "
        f"{n_models - n_used} with no fallback)"
    )
    print(f"\nWrote: {out_path}")


# --------------------------------------------------------------------------- #
# 4. bootstrap-ci
# --------------------------------------------------------------------------- #
CORRECT_MAP = {True: 1, False: 0, "True": 1, "False": 0}


def bootstrap_ci_table(
    detailed_path: str, n_bootstrap: int, ci_level: float, seed: int
) -> pd.DataFrame:
    """Paired/cluster bootstrap over question_id: the SAME resampled item index set is
    applied to every model in a given replicate (not independent per-model resampling),
    so the replicate distributions stay comparable across models (e.g. for a later
    paired-significance check), at no extra cost."""
    wide, _, models = load_detailed(detailed_path)
    corr_cols = [f"{m}_correct" for m in models]
    matrix = (
        wide[corr_cols].map(lambda v: CORRECT_MAP.get(v, np.nan)).to_numpy(dtype=float)
    )
    n_items = matrix.shape[0]

    # IMPORTANT: one shared index array per replicate (with replacement), applied to
    # every model's column below via matrix[idx] -- NOT a fresh independent draw per
    # model. This is deliberate ("paired"/cluster bootstrap): all models answer the same
    # fixed item set, and item difficulty is correlated across models (a badly-worded
    # item is hard for everyone), so a shared resample preserves that correlation instead
    # of pretending each model saw its own random item pool. It doesn't change any single
    # model's own CI below (a model's marginal bootstrap distribution only depends on its
    # own draws, regardless of what other models' columns happened to share that draw) --
    # what it buys is that replicate_acc's ROWS stay comparable across model columns, so
    # a future paired-significance test between two models (accuracy_A[b] - accuracy_B[b]
    # per replicate b) has correctly-small variance, instead of the inflated variance an
    # independent-per-model resample would give that same subtraction.
    rng = np.random.default_rng(seed)
    replicate_acc = np.empty((n_bootstrap, len(models)), dtype=float)
    for b in range(n_bootstrap):
        idx = rng.integers(0, n_items, size=n_items)  # with replacement
        with np.errstate(invalid="ignore"):
            replicate_acc[b] = np.nanmean(matrix[idx], axis=0) * 100.0

    alpha = 1.0 - ci_level
    lo = np.nanpercentile(replicate_acc, 100 * alpha / 2, axis=0)
    hi = np.nanpercentile(replicate_acc, 100 * (1 - alpha / 2), axis=0)
    with np.errstate(invalid="ignore"):
        point = np.nanmean(matrix, axis=0) * 100.0

    return pd.DataFrame(
        {
            "model": models,
            "accuracy": point,
            "ci_low": lo,
            "ci_high": hi,
            "n_bootstrap": n_bootstrap,
        }
    ).sort_values("accuracy", ascending=False)


def plot_bootstrap_ci(
    report: pd.DataFrame,
    out: str,
    title: str,
    display_names: dict | None,
    model_provider: dict | None,
    provider_color: dict | None,
) -> None:
    """Same design as analyze_results.py's plot_accuracy() -- provider-colored bars,
    value labels, provider legend to the right -- with the bootstrap CI overlaid as
    asymmetric error bars."""
    model_provider = model_provider or {}
    provider_color = provider_color or {}
    s = report.sort_values("accuracy", ascending=False)
    models = s["model"].tolist()
    x = np.arange(len(models))
    colors = _bar_colors(models, model_provider, provider_color)
    handles = _provider_legend_handles(models, model_provider, provider_color)

    lower = (s["accuracy"] - s["ci_low"]).clip(lower=0).to_numpy()
    upper = (s["ci_high"] - s["accuracy"]).clip(lower=0).to_numpy()

    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.bar(x, s["accuracy"], color=colors)
    ax.errorbar(
        x,
        s["accuracy"],
        yerr=[lower, upper],
        fmt="none",
        ecolor="black",
        elinewidth=1,
        capsize=3,
        zorder=3,
    )
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 106)
    ax.set_xticks(x)
    ax.set_xticklabels(
        _labels(models, display_names), rotation=35, ha="right", fontsize=7
    )
    ax.set_title(title, fontweight="bold", pad=16, fontsize=15)
    for xi, (a, hi) in enumerate(zip(s["accuracy"], s["ci_high"])):
        ax.text(xi, hi + 1.5, f"{a:.0f}", ha="center", fontsize=7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#eeeeee")
    ax.set_axisbelow(True)

    if handles:
        ax.legend(
            handles=handles,
            title="Provider",
            loc="center left",
            bbox_to_anchor=(1.01, 0.5),
            frameon=False,
            fontsize=8,
            title_fontsize=9,
        )
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def load_default_models(models_json: str, language: str) -> list[str]:
    """The model ids this language's leaderboard pre-selects, from a
    benchmark_default_models.json written by scripts/results_for_website.py.

    That file is keyed by benchmark_domain ("LK-en"/"LK-sw"), one entry per published
    (benchmark, language) pair, so the language picks the entry. Each run folder under
    results/web/<timestamp>/ has its own copy and they differ by curation setting --
    point at a specific one rather than assuming "the latest".
    """
    with open(models_json, encoding="utf-8") as fh:
        by_domain = json.load(fh)
    domain = DEFAULT_MODELS_DOMAIN.get(language)
    if domain not in by_domain:
        raise SystemExit(
            f"ERROR: no '{domain}' entry in {models_json} "
            f"(found: {', '.join(sorted(by_domain)) or 'nothing'})"
        )
    return list(by_domain[domain])


def filter_to_models(
    report: pd.DataFrame, keep: list[str], models_json: str
) -> pd.DataFrame:
    """Restrict a per-model table to `keep`, reporting both directions of mismatch."""
    have = set(report["model"])
    missing = [m for m in keep if m not in have]
    if missing:
        print(
            f"\nWARNING: {len(missing)} model(s) in {models_json} have no result in "
            f"this experiment and are not plotted: {', '.join(sorted(missing))}"
        )
    out = report[report["model"].isin(keep)].copy()
    if out.empty:
        raise SystemExit(
            f"ERROR: none of the {len(keep)} models in {models_json} have a result in "
            "this experiment -- is it the right language/experiment-id?"
        )
    print(f"Filtered to {len(out)}/{len(report)} models from {models_json}")
    return out


def run_bootstrap_ci(args: argparse.Namespace, root: str) -> None:
    paths = default_paths_for(
        args.language, os.path.join(root, "results"), args.experiment_id
    )

    _section("CONFIG")
    _kv("language", args.language)
    _kv("experiment_id", args.experiment_id)
    _kv("n_bootstrap", args.n_bootstrap)
    _kv("ci_level", args.ci_level)
    _kv("seed", args.seed)

    report = bootstrap_ci_table(
        paths["detailed"], args.n_bootstrap, args.ci_level, args.seed
    )
    # Filter AFTER the bootstrap, not before: each replicate resamples items and every
    # model is scored on that same resampled set, so a model's CI does not depend on
    # which other models are in the table. Dropping rows here therefore gives exactly
    # the intervals an all-models run would report, just fewer bars.
    if args.models_json:
        _kv("models_json", args.models_json)
        report = filter_to_models(
            report,
            load_default_models(args.models_json, args.language),
            args.models_json,
        )

    _section(f"BOOTSTRAP {args.ci_level:.0%} CI PER MODEL")
    for _, r in report.iterrows():
        print(
            f"  {r['model']:<32} {r['accuracy']:5.1f}%   "
            f"[{r['ci_low']:5.1f}, {r['ci_high']:5.1f}]"
        )

    os.makedirs(args.outdir, exist_ok=True)
    out_path = os.path.join(args.outdir, f"bootstrap_ci_{args.label}.csv")
    report.round(2).to_csv(out_path, index=False)

    display_names = load_display_names(args.models_csv)
    model_provider, provider_color = load_provider_colors(
        args.models_csv, args.providers_csv
    )
    language_name = {"kiswahili": "Kiswahili"}.get(
        args.language, args.language.capitalize()
    )
    png_path = os.path.join(args.outdir, f"bootstrap_ci_{args.label}.png")
    plot_bootstrap_ci(
        report,
        png_path,
        # The CI level belongs in the figure caption, not the title -- the error bars
        # are self-evidently intervals, and the exact level is printed to the console
        # and recorded in bootstrap_ci_<label>.csv.
        title=f"{language_name} Linguistic Knowledge Benchmark Performance",
        display_names=display_names,
        model_provider=model_provider,
        provider_color=provider_color,
    )

    print(f"\nWrote:\n  {out_path}\n  {png_path}")


# --------------------------------------------------------------------------- #
# 5. family-grid
# --------------------------------------------------------------------------- #
def parse_release_date(value) -> float:
    """'MM/YYYY' -> a sortable float (year*12 + month); NaN if blank/unparseable."""
    if pd.isna(value):
        return float("nan")
    try:
        month, year = str(value).split("/")
        return float(int(year) * 12 + int(month))
    except (ValueError, AttributeError):
        return float("nan")


def load_family_order_keys(models_csv: str) -> tuple[dict, dict]:
    """(model_id -> release_date sort key, model_id -> size) from models.csv."""
    m = pd.read_csv(models_csv)
    release = {
        row.model_id: parse_release_date(row.release_date) for row in m.itertuples()
    }
    size = {
        row.model_id: (float(row.size) if pd.notna(row.size) else float("nan"))
        for row in m.itertuples()
    }
    return release, size


def build_family_table(
    families_csv: str, summary: pd.DataFrame, models_csv: str
) -> pd.DataFrame:
    """family, model, order_index, accuracy -- restricted to models present in `summary`,
    ordered within each family by release_date (fallback: size; models with neither sort
    last, tiebroken by model_id)."""
    families = pd.read_csv(families_csv)
    accuracy_by_model = dict(zip(summary["model"], summary["accuracy"]))
    release, size = load_family_order_keys(models_csv)

    present = families[families["model_id"].isin(accuracy_by_model)].copy()
    missing = families[~families["model_id"].isin(accuracy_by_model)]
    if len(missing):
        print(
            f"\n({len(missing)} model(s) in {families_csv} not in this experiment's "
            f"summary -- skipped: {', '.join(missing['model_id'])})"
        )

    def sort_key(model_id: str):
        r = release.get(model_id, float("nan"))
        if not pd.isna(r):
            return (0, r, model_id)
        s = size.get(model_id, float("nan"))
        return (1, s if not pd.isna(s) else float("inf"), model_id)

    rows = []
    for family, group in present.groupby("family"):
        ordered = sorted(group["model_id"], key=sort_key)
        for i, model_id in enumerate(ordered):
            rows.append(
                {
                    "family": family,
                    "model": model_id,
                    "order_index": i,
                    "accuracy": accuracy_by_model[model_id],
                }
            )
    return pd.DataFrame(rows)


def plot_family_grid(
    table: pd.DataFrame,
    out: str,
    display_names: dict,
    model_provider: dict | None = None,
    provider_color: dict | None = None,
    language: str | None = None,
) -> None:
    """One shared axis, not a subplot grid -- a subplot per family gave every family
    the same on-screen WIDTH regardless of how many models it has, so bars in a
    10-model family were squeezed thin next to bars in a 3-model family stretched
    wide (bar width has no meaning here, so that variation was misleading). Every bar
    here is the same width; families stay visually separated by an extra gap, a light
    divider line, and a centered label above each cluster. Bars colored by provider
    (providers.csv, via analyze_results.py's own _bar_colors -- same palette as
    plot_accuracy()'s per-model bars); `language`, if given, is prefixed onto the
    title."""
    model_provider = model_provider or {}
    provider_color = provider_color or {}
    families = [f for f, g in table.groupby("family") if len(g) >= 2]
    if not families:
        print("\n(no family has >=2 models in this experiment -- skipping grid plot)")
        return

    gap = 1.0  # extra spacing (in bar-width units) between the last bar of one
    # family and the first bar of the next, on top of the normal 1-unit bar spacing
    x_positions: list[float] = []
    accuracies: list[float] = []
    colors: list[str] = []
    tick_labels: list[str] = []
    family_bounds: list[tuple[str, float, float]] = []  # (family, first_x, last_x)
    plotted_models: list[str] = []

    cursor = 0.0
    for family in families:
        g = table[table["family"] == family].sort_values("accuracy", ascending=False)
        models = g["model"].tolist()
        start = cursor
        for _ in models:
            x_positions.append(cursor)
            cursor += 1.0
        accuracies.extend(g["accuracy"].tolist())
        colors.extend(_bar_colors(models, model_provider, provider_color))
        tick_labels.extend(display_names.get(m, m) for m in models)
        family_bounds.append((family, start, cursor - 1.0))
        plotted_models.extend(models)
        cursor += gap

    fig, ax = plt.subplots(figsize=(max(14, cursor * 0.42), 5))
    bars = ax.bar(x_positions, accuracies, width=0.8, color=colors)
    for b, a in zip(bars, accuracies):
        ax.text(
            b.get_x() + b.get_width() / 2, a + 2, f"{a:.0f}", ha="center", fontsize=7
        )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(tick_labels, rotation=35, ha="right", fontsize=7)
    ax.set_ylim(0, 118)
    ax.set_yticks(range(0, 101, 20))
    ax.set_ylabel("Accuracy (%)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#eeeeee")
    ax.set_axisbelow(True)

    for family, start, end in family_bounds:
        ax.text(
            (start + end) / 2, 110, family, ha="center", fontsize=10, fontweight="bold"
        )
    for (_, _, prev_end), (_, next_start, _) in zip(family_bounds, family_bounds[1:]):
        ax.axvline((prev_end + next_start) / 2, color="#dddddd", linewidth=1, zorder=0)

    title = "Model-family consistency — accuracy by model"
    if language:
        title = f"{language.capitalize()} Linguistic Knowledge Benchmark -- {title}"
    ax.set_title(title, fontweight="bold", pad=16, fontsize=13)
    # Built from every plotted model at once, so the providers come out in
    # providers.csv order like every other provider legend -- accumulating them
    # family by family ordered the legend by family iteration instead.
    handles = _provider_legend_handles(plotted_models, model_provider, provider_color)
    if handles:
        ax.legend(
            handles=handles,
            title="Provider",
            loc="center left",
            bbox_to_anchor=(1.005, 0.5),
            frameon=False,
            fontsize=8,
            title_fontsize=9,
        )
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_family_grid(args: argparse.Namespace, root: str) -> None:
    paths = default_paths_for(
        args.language, os.path.join(root, "results"), args.experiment_id
    )

    _section("CONFIG")
    _kv("language", args.language)
    _kv("experiment_id", args.experiment_id)
    _kv("model_families_csv", args.model_families_csv)

    summary = load_summary(paths["summary"])
    table = build_family_table(args.model_families_csv, summary, args.models_csv)
    if table.empty:
        raise SystemExit(
            f"No model in {args.model_families_csv} matched this experiment's summary CSV."
        )

    _section("FAMILY TABLE (old->new / small->big within each family)")
    for family, g in table.groupby("family"):
        line = "  ".join(
            f"{m}={a:.0f}"
            for m, a in zip(g.sort_values("order_index")["model"], g["accuracy"])
        )
        print(f"  {family:<12} {line}")

    display_names = load_display_names(args.models_csv)
    model_provider, provider_color = load_provider_colors(
        args.models_csv, args.providers_csv
    )
    os.makedirs(args.outdir, exist_ok=True)
    png_path = os.path.join(args.outdir, f"family_grid_{args.label}.png")
    plot_family_grid(
        table, png_path, display_names, model_provider, provider_color, args.language
    )
    csv_path = os.path.join(args.outdir, f"family_grid_{args.label}.csv")
    table.to_csv(csv_path, index=False)

    print("\nWrote:")
    if os.path.exists(png_path):
        print(f"  {png_path}")
    print(f"  {csv_path}")


# --------------------------------------------------------------------------- #
# 6. consistency
# --------------------------------------------------------------------------- #
def decode_replicate_width(n_replicates: int) -> int:
    """Digit width the replicate suffix was zero-padded to -- see
    build_consistency_dataset.py's docstring for why the padding matters."""
    return max(len(str(n_replicates)), 2)


def compute_consistency_table(
    detailed_path: str, n_replicates: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(agg, raw) accuracy-per-replicate tables from a detailed CSV whose question_id
    values are "<original_id>.<replicate>"-encoded (build_consistency_dataset.py).
    Groups each model's final judged `<model>_correct` by the decoded replicate, then
    aggregates mean/std/min/max across replicates per model."""
    wide, _, models = load_detailed(detailed_path)
    denom = 10 ** decode_replicate_width(n_replicates)

    qid = pd.to_numeric(wide["question_id"], errors="coerce")
    original_id = np.floor(qid)
    replicate = ((qid - original_id) * denom).round()
    valid = qid.notna() & replicate.between(1, n_replicates)
    n_bad = int((~valid).sum())
    if n_bad:
        print(
            f"WARNING: {n_bad}/{len(wide)} row(s) don't decode to a replicate in "
            f"[1, {n_replicates}] -- excluded (check --n-replicates matches what "
            "build_consistency_dataset.py used to build this experiment's dataset)."
        )

    rows = []
    for model in models:
        col = f"{model}_correct"
        if col not in wide.columns:
            continue
        corr = wide[col].map(CORRECT_MAP)
        for rep in range(1, n_replicates + 1):
            mask = valid & (replicate == rep)
            scored = corr[mask].dropna()
            if len(scored) == 0:
                continue
            rows.append(
                {"rep": rep, "model": model, "accuracy": float(scored.mean()) * 100.0}
            )

    raw = pd.DataFrame(rows, columns=["rep", "model", "accuracy"])
    if raw.empty:
        return pd.DataFrame(
            columns=[
                "model",
                "mean_accuracy",
                "std_accuracy",
                "min_accuracy",
                "max_accuracy",
                "n_reps",
            ]
        ), raw

    agg = (
        raw.groupby("model")["accuracy"]
        .agg(["mean", "std", "min", "max", "count"])
        .rename(
            columns={
                "mean": "mean_accuracy",
                "std": "std_accuracy",
                "min": "min_accuracy",
                "max": "max_accuracy",
                "count": "n_reps",
            }
        )
        .reset_index()
        .sort_values("mean_accuracy", ascending=False)
    )
    return agg, raw


def plot_consistency(
    raw: pd.DataFrame,
    agg: pd.DataFrame,
    out: str,
    display_names: dict | None,
    model_provider: dict | None,
    provider_color: dict | None,
    language: str | None = None,
) -> None:
    """One box (spread across all replicates) per model, ordered by mean accuracy,
    colored by provider -- same palette as the other plots. `language`, if given,
    becomes a figure-level suptitle above the existing (replicate-count) title."""
    model_provider = model_provider or {}
    provider_color = provider_color or {}
    order = agg.sort_values("mean_accuracy", ascending=False)["model"].tolist()
    data = [raw.loc[raw["model"] == m, "accuracy"].to_numpy() for m in order]
    colors = _bar_colors(order, model_provider, provider_color)

    fig, ax = plt.subplots(figsize=(max(12, len(order) * 0.5), 5))
    bp = ax.boxplot(data, patch_artist=True, widths=0.6, showfliers=True)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.85)
    for median in bp["medians"]:
        median.set_color("black")

    ax.set_xticks(range(1, len(order) + 1))
    ax.set_xticklabels(
        _labels(order, display_names), rotation=35, ha="right", fontsize=7
    )
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 106)
    n_reps = raw["rep"].nunique()
    ax.set_title(
        f"Replicate-to-replicate consistency -- accuracy across {n_reps} repeated "
        f"replicates per model",
        fontweight="bold",
        pad=16,
        fontsize=13,
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#eeeeee")
    ax.set_axisbelow(True)

    handles = _provider_legend_handles(order, model_provider, provider_color)
    if handles:
        ax.legend(
            handles=handles,
            title="Provider",
            loc="center left",
            bbox_to_anchor=(1.01, 0.5),
            frameon=False,
            fontsize=8,
            title_fontsize=9,
        )
    if language:
        fig.suptitle(
            f"{language.capitalize()} Linguistic Knowledge Benchmark Performance",
            fontweight="bold",
            fontsize=15,
            y=1.03,
        )
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_consistency(args: argparse.Namespace, root: str) -> None:
    paths = default_paths_for(
        args.language, os.path.join(root, "results"), args.experiment_id
    )

    _section("CONFIG")
    _kv("language", args.language)
    _kv("experiment_id", args.experiment_id)
    _kv("n_replicates", args.n_replicates)

    # --dataset is filename-only: always resolved as DATA_DIR/<dataset> by cmd.py.
    dataset_arg = f"linguistic_knowledge_eval_{args.language}_{args.n_replicates}x.json"
    missing = {name: p for name, p in paths.items() if not os.path.exists(p)}
    if missing:
        listing = "\n".join(f"  {name}: {p}" for name, p in missing.items())
        raise SystemExit(
            f"Missing result file(s) for --language {args.language} --experiment-id "
            f"{args.experiment_id}:\n{listing}\n\n"
            "This check needs the summary/detailed/judge_agreement CSVs from an "
            "experiment scored against the REPEATED (consistency) dataset, not a normal "
            "one. To "
            "produce them:\n"
            f"  1. uv run python scripts/build_consistency_dataset.py --language "
            f"{args.language} --n-replicates {args.n_replicates}   (if not already built)\n"
            "  2. Pass that file as --dataset to BOTH commands (same --experiment-id for both):\n"
            f"       uv run linguistic-knowledge-benchmark run-bench --language "
            f"{args.language} --experiment-id {args.experiment_id} --dataset {dataset_arg}\n"
            f"       uv run linguistic-knowledge-benchmark judge-bench --language "
            f"{args.language} --experiment-id {args.experiment_id} --dataset {dataset_arg} "
            "--num-judges 3"
        )

    wide, _, _models = load_detailed(paths["detailed"])
    corr_cols = [c for c in wide.columns if c.endswith("_correct")]
    if not corr_cols or wide[corr_cols].notna().to_numpy().sum() == 0:
        raise SystemExit(
            f"{paths['detailed']} has no scored rows for any model.\n"
            "This is the failure mode where judge-bench's detailed CSV was merged "
            "against the wrong dataset -- --dataset must be passed consistently to "
            "BOTH run-bench and judge-bench (the same file, same --experiment-id). Re-run "
            "judge-bench with it:\n"
            f"  uv run linguistic-knowledge-benchmark judge-bench --language "
            f"{args.language} --experiment-id {args.experiment_id} --dataset {dataset_arg} "
            "--num-judges 3 --force"
        )

    agg, raw = compute_consistency_table(paths["detailed"], args.n_replicates)
    if raw.empty:
        raise SystemExit(
            "No row decoded to a valid replicate -- is --n-replicates set to what "
            "build_consistency_dataset.py used to build this experiment's dataset?"
        )

    _section(
        f"CONSISTENCY ACROSS {args.n_replicates} REPEATED RUNS (final judged score)"
    )
    for _, r in agg.iterrows():
        print(
            f"  {r['model']:<32} mean {r['mean_accuracy']:5.1f}%   "
            f"std {r['std_accuracy']:4.1f}   "
            f"[{r['min_accuracy']:5.1f}, {r['max_accuracy']:5.1f}]   n={int(r['n_reps'])}"
        )

    display_names = load_display_names(args.models_csv)
    model_provider, provider_color = load_provider_colors(
        args.models_csv, args.providers_csv
    )

    os.makedirs(args.outdir, exist_ok=True)
    agg_path = os.path.join(args.outdir, f"consistency_{args.label}.csv")
    agg.round(2).to_csv(agg_path, index=False)
    raw_path = os.path.join(args.outdir, f"consistency_{args.label}_raw.csv")
    raw.round(2).to_csv(raw_path, index=False)
    png_path = os.path.join(args.outdir, f"consistency_{args.label}.png")
    plot_consistency(
        raw, agg, png_path, display_names, model_provider, provider_color, args.language
    )

    print(f"\nWrote:\n  {agg_path}\n  {raw_path}\n  {png_path}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def build_parser(root: str) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--language",
        required=True,
        choices=LANGUAGES,
        help="which language's result CSVs to validate",
    )
    common.add_argument(
        "--experiment-id",
        required=True,
        help="experiment-id passed to judge-bench; selects "
        "summary_linguistic-knowledge_<language>_<experiment-id>[_detailed|_judge_agreement].csv "
        "-- include the literal '_fallback' suffix if that's how the experiment was produced "
        "(e.g. 'final-137-items_fallback')",
    )
    common.add_argument(
        "--outdir",
        default=os.path.join(root, "results", "validation"),
        help="base output dir; this experiment's files go under <outdir>/<language>/",
    )
    common.add_argument("--label", default=None, help="tag for output filenames")
    common.add_argument(
        "--models-csv",
        dest="models_csv",
        default=str(MODELS_CSV),
    )
    common.add_argument(
        "--providers-csv",
        dest="providers_csv",
        default=str(PROVIDERS_CSV),
    )

    sub = ap.add_subparsers(dest="check", required=True)
    sub.add_parser(
        "judge-agreement",
        parents=[common],
        help="raw + chance-corrected judge agreement",
    )
    sub.add_parser(
        "self-preference-bias",
        parents=[common],
        help="does a judge score same-provider models more leniently?",
    )
    sub.add_parser(
        "fallback-rate",
        parents=[common],
        help="per-model MCQ fallback-judging rate vs accuracy",
    )
    bootstrap = sub.add_parser(
        "bootstrap-ci", parents=[common], help="per-model bootstrap accuracy CI"
    )
    bootstrap.add_argument("--n-bootstrap", type=int, default=1000)
    bootstrap.add_argument("--ci-level", type=float, default=0.95)
    bootstrap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    bootstrap.add_argument(
        "--models-json",
        default=None,
        help="plot only the models this language is pre-selected on, from a "
        "benchmark_default_models.json written by results_for_website.py (e.g. "
        "results/web/<timestamp>/benchmark_default_models.json). Each run folder has "
        "its own and they differ by curation setting, so name the one you mean. "
        f"Auto-appends '{DEFAULT_MODELS_LABEL_SUFFIX}' to --label unless you set it.",
    )
    family = sub.add_parser(
        "family-grid", parents=[common], help="accuracy grid by model family/generation"
    )
    family.add_argument(
        "--model-families-csv",
        default=os.path.join(root, "data", "model_families.csv"),
    )
    consistency = sub.add_parser(
        "consistency",
        parents=[common],
        help="replicate-to-replicate accuracy variance from an experiment scored against a "
        "repeated "
        "dataset (see build_consistency_dataset.py); suggested --experiment-id: validation-20x",
    )
    consistency.add_argument(
        "--n-replicates",
        type=int,
        default=DEFAULT_N_REPLICATES,
        help="must match what build_consistency_dataset.py used for this experiment's dataset "
        f"(default {DEFAULT_N_REPLICATES})",
    )
    all_checks = sub.add_parser(
        "all",
        parents=[common],
        help="run every check for this language/experiment-id; skips (doesn't overwrite) any "
        "check whose output file(s) already exist under <outdir>/<language>/",
    )
    all_checks.add_argument("--n-bootstrap", type=int, default=1000)
    all_checks.add_argument("--ci-level", type=float, default=0.95)
    all_checks.add_argument("--seed", type=int, default=DEFAULT_SEED)
    all_checks.add_argument(
        "--model-families-csv",
        default=os.path.join(root, "data", "model_families.csv"),
    )
    all_checks.add_argument("--n-replicates", type=int, default=DEFAULT_N_REPLICATES)
    return ap


RUNNERS = {
    "judge-agreement": run_judge_agreement,
    "self-preference-bias": run_self_preference_bias,
    "fallback-rate": run_fallback_rate,
    "bootstrap-ci": run_bootstrap_ci,
    "family-grid": run_family_grid,
    "consistency": run_consistency,
}

# Output file(s) each check writes, keyed by --label -- used by run_all() to decide
# whether a check has already been run for this language/experiment-id/label and should be
# skipped rather than overwritten.
CHECK_OUTPUT_FILES = {
    "judge-agreement": lambda label: [f"judge_agreement_kappa_{label}.csv"],
    "self-preference-bias": lambda label: [f"self_preference_bias_{label}.csv"],
    "fallback-rate": lambda label: [f"fallback_rate_{label}.csv"],
    "bootstrap-ci": lambda label: [
        f"bootstrap_ci_{label}.csv",
        f"bootstrap_ci_{label}.png",
    ],
    "family-grid": lambda label: [f"family_grid_{label}.csv"],
    "consistency": lambda label: [
        f"consistency_{label}.csv",
        f"consistency_{label}_raw.csv",
        f"consistency_{label}.png",
    ],
}


def run_all(args: argparse.Namespace, root: str) -> None:
    """Run every check in CHECKS for this --language/--experiment-id, in order. A check is
    SKIPPED (not overwritten) if all of its expected output files already exist under
    <outdir>/<language>/ -- rerun with a different --label, or delete those files, to
    force it. A check that errors (e.g. fallback-rate with no fallback pass for this
    run) is reported and skipped too, rather than aborting the remaining checks."""
    _section("RUNNING ALL CHECKS")
    _kv("language", args.language)
    _kv("experiment_id", args.experiment_id)
    _kv("label", args.label)
    _kv("outdir", args.outdir)

    for check in CHECKS:
        expected = [
            os.path.join(args.outdir, name)
            for name in CHECK_OUTPUT_FILES[check](args.label)
        ]
        if all(os.path.exists(p) for p in expected):
            print(f"\n[{check}] output already exists -- skipping:")
            for p in expected:
                print(f"    {p}")
            continue
        print(f"\n[{check}] running...")
        try:
            RUNNERS[check](args, root)
        except SystemExit as e:
            print(f"[{check}] ABORTED: {e}")
        except Exception as e:  # noqa: BLE001 -- keep running the remaining checks
            print(f"[{check}] FAILED: {e}")


RUNNERS["all"] = run_all


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    if here not in sys.path:
        sys.path.insert(0, here)

    args = build_parser(root).parse_args()
    if args.label is None:
        args.label = f"{args.language}_{args.experiment_id}"
        # A filtered run writes different numbers under the same name otherwise, and
        # would silently overwrite the all-models CSV/PNG.
        if getattr(args, "models_json", None):
            args.label += DEFAULT_MODELS_LABEL_SUFFIX
    args.outdir = os.path.join(args.outdir, args.language)

    RUNNERS[args.check](args, root)


if __name__ == "__main__":
    main()
