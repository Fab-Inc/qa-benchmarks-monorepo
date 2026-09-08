#!/usr/bin/env python3
"""
Analyze a Linguistic-Knowledge "consistency" experiment: print per-model summary
statistics and plot replicate-to-replicate accuracy variance, for an experiment scored
against the N-times-repeated
dataset build_consistency_dataset.py builds (question_id "<original_id>.<replicate>").

Inputs (both produced by `judge-bench` for a --experiment-id scored against that repeated
dataset -- see build_consistency_dataset.py's docstring for how to produce them):
    results/summary_linguistic-knowledge_<language>_<experiment-id>.csv          SUMMARY
    results/summary_linguistic-knowledge_<language>_<experiment-id>_detailed.csv DETAILED

Prints five tables:
  1. SUMMARY -- each model's accuracy/judge_agreement pooled across all replicates at
     once (straight from the summary CSV, one number per model).
  2. REPLICATE-TO-REPLICATE CONSISTENCY -- each model's accuracy computed SEPARATELY per replicate,
     then the mean/std/min/max across those N replicate-level accuracies (decoded from
     the detailed CSV's question_id; reuses validate_results.py's
     compute_consistency_table(), the same logic its own "consistency" check uses).
  3. ITEM-LEVEL VARIANCE -- (1) and (2) are both experiment-level accuracy: they can look
     perfectly stable even if a model flips its verdict on individual items across
     replicates, as long as roughly as many items flip right-to-wrong as wrong-to-right
     within each replicate (the flips cancel out in the average). This instead asks,
     per item: across the N replicates, what fraction were scored correct (p_correct)?
     The Bernoulli variance p_correct*(1-p_correct) is 0 for an item that's always right
     or always wrong (perfectly stable) and maximal (0.25) for an item that's a coin
     flip -- averaged per model, and split by answer_type (multiple_choice vs
     short_form), this is the headline signal for whether instability concentrates in
     one format. Comparing this table's numbers across two separate invocations of this script
     (--language english vs --language kiswahili) answers whether one language is more
     unstable at the item level than the other.
  4. VARIANCE COMPONENTS / BETA-BINOMIAL -- (3) measures only the response noise. The
     score's uncertainty has a second, independent source: these 135/137 items are one
     draw from a larger population of possible items, so even a perfectly deterministic
     model's score would move if we'd written different items. Modelling item i's latent
     success probability as p_i ~ Beta(alpha, beta) and each replicate as
     Bernoulli(p_i) (a hierarchical beta-binomial), the variance of the reported score
     over n items x m replicates splits exactly into

         Var(theta_hat) = Var(p_i)/n  +  E[p_i(1-p_i)]/(n*m)
                          ^item draw     ^response stochasticity

     Both components are estimated in closed form (method of moments -- no MCMC, no
     bootstrap) from the same per-item p_correct column table (3) already builds, so
     each model gets an item SE, a response SE and a total SE. More items shrink both
     terms; more replicates shrink only the second, which is what makes the last table
     possible.
  5. RERUN / ITEM ALLOCATION -- with the two components in hand, Var(theta_hat) above is
     a formula in (n, m), so it answers the design questions directly: what the total SE
     would have been at m=1,2,5,10,20,...; the irreducible floor sqrt(Var(p_i)/n) that no
     amount of rerunning can beat; how many replicates are needed before response noise
     stops dominating; and, at a fixed budget of model calls, whether 135 items x 20
     replicates beats 2700 items x 1 replicate (it generally doesn't -- unique items buy
     more precision than repeats, which is worth knowing before the next data round).
     The judge (3 judges + majority vote) is treated as noise-free here: repeated
     judging of a FIXED response would be a third component and is not measured by this
     experiment, so the "response" component is really generation+judge noise combined.
A model that is stable replicate-to-replicate has (1) and (2)'s mean close together with
a small std/range; a big spread in (2) despite a healthy (1) number means individual
replicates are landing all over the place even though they average out fine -- (3) is what
tells you whether that's item-level flipping or something else, and (4)/(5) put a number
on how much of the score's total uncertainty that flipping is actually responsible for.

Also writes:
  * a box plot (one box per model, ordered by mean accuracy, colored by provider) of
    the per-replicate accuracy distribution, plus the underlying aggregate CSV
  * a grouped bar plot of mean item-level variance per model, split MCQ vs short_form
    (the mean hides the per-item distribution -- most items sit at variance=0 exactly)
  * a boxplot of the actual per-item variance distribution per model, split MCQ vs
    short_form, same model order as the bar plot -- shows the zero-inflation and the
    outlier tail the mean bar averages away
  plus the underlying per-(model, item) CSV shared by both plots
  * a two-panel variance-components figure: total SE as a function of replicates per
    item (with each model's item-only floor), and how much of the score variance the
    reruns actually removed (response noise's share at m=1 vs at the m run)
  plus the per-model variance-components CSV and the (model, n_items, m) design-grid CSV
-- all under --outdir/<language>/, tagged with --label.

Run:
    uv run python benchmarks/linguistic_knowledge_benchmark/scripts/analyze_consistency.py --language kiswahili
    uv run python .../analyze_consistency.py --language kiswahili --experiment-id consistency-20x --n-replicates 20
    uv run python .../analyze_consistency.py --language english --experiment-id consistency-20x
"""

from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from analyze_results import (
    LANGUAGES,
    _labels,
    load_detailed,
    load_display_names,
    load_provider_colors,
    load_summary,
)
from matplotlib.patches import Patch
from validate_results import (
    CORRECT_MAP,
    compute_consistency_table,
    decode_replicate_width,
    plot_consistency,
)

DEFAULT_EXPERIMENT_ID = "consistency-20x"
DEFAULT_N_REPLICATES = 20


def _section(title: str) -> None:
    print(f"\n=== {title} ===")


def print_pooled_summary(summary_df: pd.DataFrame) -> None:
    _section("SUMMARY (pooled across all replicates at once)")
    for _, r in summary_df.iterrows():
        # .get() so a summary CSV without the column degrades to "n/a", not a KeyError.
        agreement = r.get("short_form_judge_agreement_mean")
        agreement_str = f"{agreement:.2%}" if pd.notna(agreement) else "n/a"
        print(
            f"  {r['model']:<32} accuracy {r['accuracy']:5.1f}%   "
            f"scored {int(r['scored_count'])}   "
            f"short_form_judge_agreement {agreement_str}"
        )


def print_consistency_table(agg: pd.DataFrame, n_replicates: int) -> None:
    _section(f"RUN-TO-RUN CONSISTENCY ACROSS {n_replicates} REPEATED RUNS")
    for _, r in agg.iterrows():
        print(
            f"  {r['model']:<32} mean {r['mean_accuracy']:5.1f}%   "
            f"std {r['std_accuracy']:4.1f}   "
            f"[{r['min_accuracy']:5.1f}, {r['max_accuracy']:5.1f}]   n={int(r['n_reps'])}"
        )


def compute_item_variance_table(detailed_path: str, n_replicates: int) -> pd.DataFrame:
    """Per (model, item) instability across the N replicates.

    Unlike experiment-level accuracy (compute_consistency_table): that stays flat even if
    DIFFERENT items flip correct<->incorrect each replicate, as long as the flips
    cancel out within each replicate's average. This instead asks whether the SAME
    item gets the same verdict every time. `p_correct` is the fraction of scored
    replicates that item was marked correct; `variance` = p_correct * (1 - p_correct),
    the Bernoulli variance of a single replicate's verdict for that item (0 = always
    the same verdict either way, 0.25 = a coin flip). One row per (model, question_id).
    """
    wide, _, models = load_detailed(detailed_path)
    denom = 10 ** decode_replicate_width(n_replicates)

    qid = pd.to_numeric(wide["question_id"], errors="coerce")
    original_id = np.floor(qid)
    replicate = ((qid - original_id) * denom).round()
    valid = qid.notna() & replicate.between(1, n_replicates)

    rows = []
    for model in models:
        col = f"{model}_correct"
        if col not in wide.columns:
            continue
        item_df = pd.DataFrame(
            {
                "original_id": original_id[valid],
                "answer_type": wide.loc[valid, "answer_type"],
                "correct": wide.loc[valid, col].map(CORRECT_MAP),
            }
        )
        grouped = item_df.dropna(subset=["correct"]).groupby(
            ["original_id", "answer_type"]
        )["correct"]
        for (question_id, answer_type), scored in grouped:
            p = float(scored.mean())
            rows.append(
                {
                    "model": model,
                    "question_id": question_id,
                    "answer_type": answer_type,
                    "n_scored": int(scored.count()),
                    "p_correct": p,
                    "variance": p * (1 - p),
                }
            )
    return pd.DataFrame(rows)


def print_item_variance_summary(item_var: pd.DataFrame) -> None:
    _section("ITEM-LEVEL VARIANCE (does the SAME item flip verdict across replicates?)")
    unstable = (item_var["p_correct"] > 0) & (item_var["p_correct"] < 1)
    item_var = item_var.assign(unstable=unstable)

    overall = item_var.groupby("model")["variance"].mean()
    unstable_rate = item_var.groupby("model")["unstable"].mean() * 100.0
    by_type = (
        item_var.groupby(["model", "answer_type"])["variance"]
        .mean()
        .unstack("answer_type")
    )
    for model in overall.sort_values().index:
        parts = [f"overall {overall[model]:.4f}"]
        for atype in ("multiple_choice", "short_form"):
            if atype in by_type.columns and pd.notna(by_type.loc[model, atype]):
                parts.append(f"{atype} {by_type.loc[model, atype]:.4f}")
        print(
            f"  {model:<32} {'   '.join(parts):<58} unstable {unstable_rate[model]:5.1f}%"
        )

    _section("ITEM-LEVEL VARIANCE BY ANSWER TYPE (pooled across all models)")
    pooled = item_var.groupby("answer_type").agg(
        mean_variance=("variance", "mean"),
        pct_unstable=("unstable", "mean"),
        n_item_model_pairs=("variance", "size"),
    )
    for answer_type, r in pooled.iterrows():
        print(
            f"  {answer_type:<16} mean item variance {r['mean_variance']:.4f}   "
            f"unstable {r['pct_unstable'] * 100:5.1f}%   "
            f"n={int(r['n_item_model_pairs'])}"
        )


def compute_variance_components(
    item_var: pd.DataFrame, min_scored: int = 2
) -> pd.DataFrame:
    """Split each model's score uncertainty into an item component and a response
    component, via a hierarchical beta-binomial / variance-components fit.

    Model: item i has a latent success probability p_i ~ Beta(alpha, beta) (the fraction
    of runs the model would get item i right if we could run it forever); each replicate
    is an independent Bernoulli(p_i) draw. The reported score theta_hat is the mean over
    n items of the per-item mean over m replicates, so

        Var(theta_hat) = Var(p_i)/n  +  E[p_i(1-p_i)]/(n*m)

    where the first term is item-sampling uncertainty (we drew THESE items out of the
    population of possible items) and the second is response stochasticity (the model
    answers the same item differently on different replicates). Both are estimated in closed
    form -- method of moments, so no MCMC or bootstrap:

      * response component, E[p_i(1-p_i)]: mean over items of the unbiased within-item
        variance s_i^2 = m_i/(m_i - 1) * p_i_hat * (1 - p_i_hat).
      * item component, Var(p_i): the observed spread of the per-item means is
        Var_i(p_i_hat) = Var(p_i) + E[p_i(1-p_i)/m_i], i.e. real item heterogeneity PLUS
        leftover finite-replicate noise, so the noise term is subtracted off (clamped at
        0). Taking Var_i(p_i_hat) at face value -- the mistake this correction exists to
        avoid -- overstates item heterogeneity, here by W/m.
        Replicate counts differ per item (a replicate whose call errored is unscored), so
        the correction is the per-item mean of s_i^2/m_i rather than W/m at a single m.

    Given those two, the Beta hyperparameters follow from Var(p_i) = mu(1-mu)/(kappa+1):

        mu = E[p_i],  kappa = mu(1-mu)/Var(p_i) - 1,  alpha = mu*kappa,
        beta = (1-mu)*kappa

    kappa is the concentration of the item population: large kappa means all items are
    about equally hard for this model and most of the observed movement is response
    noise; small kappa means the items are spread out (many near-certain right, many
    near-certain wrong) and most of the uncertainty is about WHICH items are in the set.
    `beta_response_var` is the response component the fitted Beta implies
    (mu(1-mu) - Var(p_i), by the law of total variance for a binary outcome) -- it is
    estimated by a different route than `response_var`, so the two agreeing is a check
    that the Beta shape is not badly wrong. They can disagree because a single Beta
    cannot be both U-shaped and heavy in the middle; see check_beta_fit().

    One row per model, with the two components, the SEs they imply at the replicate count
    actually run, and the floor sqrt(Var(p_i)/n) that no amount of rerunning can beat.
    Items with fewer than `min_scored` scored replicates carry no within-item variance
    estimate and are dropped.
    """
    rows = []
    usable = item_var[item_var["n_scored"] >= min_scored]
    for model, df in usable.groupby("model", sort=False):
        p = df["p_correct"].to_numpy(dtype=float)
        m = df["n_scored"].to_numpy(dtype=float)
        n_items = len(p)
        if n_items < 2:
            continue

        # E[p_i(1-p_i)] -- response stochasticity, averaged over the item population.
        within = m / (m - 1.0) * p * (1.0 - p)
        response_var = float(within.mean())
        # Var_i(p_i_hat) still contains E[p_i(1-p_i)/m_i]; subtract it off for Var(p_i).
        observed_spread = float(p.var(ddof=1))
        finite_replicate_noise = float((within / m).mean())
        item_var_est = max(observed_spread - finite_replicate_noise, 0.0)

        mu = float(p.mean())
        spread_max = mu * (1.0 - mu)  # Var(p_i) + E[p_i(1-p_i)], for binary outcomes
        kappa = spread_max / item_var_est - 1.0 if item_var_est > 0 else float("inf")
        beta_response_var = max(spread_max - item_var_est, 0.0)

        # SEs of the score actually reported for this experiment: n items, m_i replicates each.
        se_item = float(np.sqrt(item_var_est / n_items))
        se_response = float(np.sqrt(finite_replicate_noise / n_items))
        se_total = float(np.sqrt(se_item**2 + se_response**2))
        rows.append(
            {
                "model": model,
                "score": mu,
                "n_items": n_items,
                "mean_replicates": float(m.mean()),
                "item_var": item_var_est,
                "response_var": response_var,
                "beta_response_var": beta_response_var,
                "kappa": kappa,
                "alpha": mu * kappa,
                "beta": (1.0 - mu) * kappa,
                "se_item": se_item,
                "se_response": se_response,
                "se_total": se_total,
                "pct_var_item": (
                    100.0 * se_item**2 / se_total**2 if se_total > 0 else float("nan")
                ),
                # Replicates at which the response term would equal the item term, and
                # at which the total SE lands within 10% of the sqrt(item_var/n) floor.
                "m_balanced": (
                    response_var / item_var_est if item_var_est > 0 else float("inf")
                ),
                "m_near_floor": (
                    np.ceil(response_var / item_var_est / (1.1**2 - 1))
                    if item_var_est > 0
                    else float("inf")
                ),
            }
        )
    return pd.DataFrame(rows)


def score_se(item_var_est: float, response_var: float, n_items: int, m: float) -> float:
    """SE of a benchmark score built from `n_items` items x `m` replicates each, under
    the fitted components. m=inf is the item-sampling floor (response noise averaged
    away entirely); n_items may exceed the items actually run, which assumes new items
    would be exchangeable draws from the same population as these ones."""
    return float(np.sqrt(item_var_est / n_items + response_var / (n_items * m)))


def compute_design_table(
    components: pd.DataFrame,
    m_grid: tuple[int, ...] = (1, 2, 3, 5, 10, 20, 50, 100),
) -> pd.DataFrame:
    """Long-form (model, n_items, m, se) grid for two design questions: how the total SE
    falls as replicates per item increase (n fixed at what was run), and how the same
    fixed budget of model calls performs when spent on items instead of on repeats."""
    rows = []
    for _, r in components.iterrows():
        n_actual = int(r["n_items"])
        budget = n_actual * int(round(r["mean_replicates"]))
        for m in m_grid:
            rows.append(
                {
                    "model": r["model"],
                    "arm": "fixed_items",
                    "n_items": n_actual,
                    "m": m,
                    "calls": n_actual * m,
                    "se": score_se(r["item_var"], r["response_var"], n_actual, m),
                }
            )
        rows.append(
            {
                "model": r["model"],
                "arm": "fixed_items",
                "n_items": n_actual,
                "m": float("inf"),
                "calls": float("inf"),
                "se": r["se_item"],
            }
        )
        for m in m_grid:
            n_alt = budget // m
            if n_alt < 2:
                continue
            rows.append(
                {
                    "model": r["model"],
                    "arm": "fixed_budget",
                    "n_items": n_alt,
                    "m": m,
                    "calls": n_alt * m,
                    "se": score_se(r["item_var"], r["response_var"], n_alt, m),
                }
            )
    return pd.DataFrame(rows)


def check_beta_fit(
    item_var: pd.DataFrame,
    components: pd.DataFrame,
    min_scored: int = 2,
    seed: int = 0,
) -> pd.DataFrame:
    """Posterior-predictive check on the Beta(alpha, beta) assumption for the item
    population, per model: draw p_i from the fitted Beta and y_i ~ Binomial(m_i, p_i)
    with the SAME per-item replicate counts, then compare the simulated per-item
    p_correct distribution with the observed one (its spread, and the share of items that
    come out always-correct / always-wrong / unstable).

    This matters because the item population here is strongly U-shaped -- most items are
    stably right or stably wrong -- and a single Beta can be U-shaped (kappa < 1) but
    cannot also carry a heavy middle. Large gaps mean the Beta's mu and Var(p_i) are
    still fine (they are moment estimates, not shape-dependent) but its per-item
    predictions, and any simulation drawing new items from it, should be distrusted; a
    logistic-normal or mixture item distribution would be the fix.
    """
    rng = np.random.default_rng(seed)
    usable = item_var[item_var["n_scored"] >= min_scored]
    fitted = components.set_index("model")
    rows = []
    for model, df in usable.groupby("model", sort=False):
        if model not in fitted.index:
            continue
        r = fitted.loc[model]
        m = df["n_scored"].to_numpy(dtype=float)
        p_obs = df["p_correct"].to_numpy(dtype=float)
        # kappa = inf (no item heterogeneity) or kappa <= 0 (item spread already at its
        # binary maximum, every p_i a hard 0/1): a degenerate Beta, nothing to draw from.
        if (
            not (np.isfinite(r["alpha"]) and np.isfinite(r["beta"]))
            or min(r["alpha"], r["beta"]) <= 0
        ):
            continue
        p_sim = rng.beta(r["alpha"], r["beta"], size=len(m))
        p_rep = rng.binomial(m.astype(int), p_sim) / m
        rows.append(
            {
                "model": model,
                "sd_p_observed": float(p_obs.std(ddof=1)),
                "sd_p_beta": float(p_rep.std(ddof=1)),
                "pct_always_correct_observed": float((p_obs == 1).mean() * 100),
                "pct_always_correct_beta": float((p_rep == 1).mean() * 100),
                "pct_always_wrong_observed": float((p_obs == 0).mean() * 100),
                "pct_always_wrong_beta": float((p_rep == 0).mean() * 100),
                "pct_unstable_observed": float(
                    ((p_obs > 0) & (p_obs < 1)).mean() * 100
                ),
                "pct_unstable_beta": float(((p_rep > 0) & (p_rep < 1)).mean() * 100),
            }
        )
    return pd.DataFrame(rows)


def _fmt_m(value: float) -> str:
    """Replicate counts, which can legitimately land below 1 (m_balanced when item
    heterogeneity already dwarfs response noise at a single replicate)."""
    if not np.isfinite(value):
        return "inf"
    return f"{value:.2f}" if value < 10 else f"{value:.0f}"


def print_variance_components(components: pd.DataFrame, design: pd.DataFrame) -> None:
    _section("VARIANCE COMPONENTS (beta-binomial: item draw vs response stochasticity)")
    print(
        "  score = mean over items of per-item accuracy; SE splits as "
        "Var(p_i)/n + E[p_i(1-p_i)]/(n*m)"
    )
    print(
        f"  {'model':<32} {'score':>6} {'SE':>6} {'SE_item':>8} {'SE_resp':>8} "
        f"{'%item':>6} {'Var(p)':>8} {'E[p(1-p)]':>10} {'kappa':>7}"
    )
    for _, r in components.sort_values("score", ascending=False).iterrows():
        kappa = "inf" if not np.isfinite(r["kappa"]) else f"{r['kappa']:7.2f}"
        print(
            f"  {r['model']:<32} {r['score'] * 100:5.1f}% {r['se_total'] * 100:5.2f}% "
            f"{r['se_item'] * 100:7.2f}% {r['se_response'] * 100:7.2f}% "
            f"{r['pct_var_item']:5.0f}% {r['item_var']:8.4f} {r['response_var']:10.4f} "
            f"{kappa}"
        )
    print(
        "  (SEs are for n items x the replicates actually run; %item is the item "
        "component's share of the score variance)"
    )

    _section("REPLICATES NEEDED (how far the response term can still be pushed down)")
    print(
        f"  {'model':<32} {'m=1':>7} {'m=5':>7} {'m=10':>7} {'m=20':>7} "
        f"{'m=inf':>7}  {'m_bal':>6} {'m_10%':>6}"
    )
    fixed = design[design["arm"] == "fixed_items"]
    for _, r in components.sort_values("score", ascending=False).iterrows():
        curve = fixed[fixed["model"] == r["model"]].set_index("m")["se"]
        cells = " ".join(
            f"{curve.get(m, float('nan')) * 100:6.2f}%"
            for m in (1, 5, 10, 20, float("inf"))
        )
        print(
            f"  {r['model']:<32} {cells}  {_fmt_m(r['m_balanced']):>6} "
            f"{_fmt_m(r['m_near_floor']):>6}"
        )
    print(
        "  m=inf is the item-sampling floor; m_bal = replicates at which response noise "
        "equals item noise,\n  m_10% = replicates to land within 10% of the floor "
        "(both independent of the number of items)"
    )

    _section("SAME BUDGET, SPENT DIFFERENTLY (total model calls held fixed per model)")
    budget = design[design["arm"] == "fixed_budget"]
    for model in components.sort_values("score", ascending=False)["model"]:
        arms = budget[budget["model"] == model].sort_values("m")
        if arms.empty:
            continue
        best = arms.loc[arms["se"].idxmin()]
        cells = "   ".join(
            f"{int(a['n_items'])}x{int(a['m'])}: {a['se'] * 100:.2f}%"
            for _, a in arms.iterrows()
            if a["m"] in (1, 2, 5, 10, 20)
        )
        print(f"  {model:<32} {cells}")
        print(
            f"  {'':<32} -> best: {int(best['n_items'])} items x {int(best['m'])} "
            f"replicates, SE {best['se'] * 100:.2f}%"
        )


def print_beta_fit_check(fit: pd.DataFrame) -> None:
    _section("BETA FIT CHECK (observed item distribution vs the fitted Beta's)")
    print(
        f"  {'model':<32} {'sd(p) obs/beta':>18} {'always-right %':>18} "
        f"{'always-wrong %':>18} {'unstable %':>18}"
    )
    for _, r in fit.iterrows():
        print(
            f"  {r['model']:<32} "
            f"{r['sd_p_observed']:8.3f}/{r['sd_p_beta']:<9.3f} "
            f"{r['pct_always_correct_observed']:8.1f}/{r['pct_always_correct_beta']:<9.1f} "
            f"{r['pct_always_wrong_observed']:8.1f}/{r['pct_always_wrong_beta']:<9.1f} "
            f"{r['pct_unstable_observed']:8.1f}/{r['pct_unstable_beta']:<9.1f}"
        )
    print(
        "  close pairs => the Beta item population is a fair description; big gaps => "
        "trust Var(p_i)/E[p(1-p)]\n  (moment estimates, shape-free) but not the Beta's "
        "own per-item or new-item predictions"
    )


def plot_variance_components(
    components: pd.DataFrame,
    design: pd.DataFrame,
    out: str,
    display_names: dict | None,
    accuracy_order: list[str],
    language: str | None = None,
) -> None:
    """Two panels. Left: total score SE against replicates per item, one curve per model,
    with each model's item-sampling floor as a dashed line -- where a curve meets its
    floor is where more replicates stop buying precision. Right: the response
    component's share of the score variance, at one replicate per item vs at the
    replicate count actually run -- i.e. how much of the score's instability the reruns
    actually removed, and how little is left to remove. (Not the item/response split as a
    100% stack: with item sampling at ~99% every bar would be full and the interesting
    number invisible.) Curves are colored per model rather than per provider -- several
    providers share a color, which is unreadable with one line each."""
    order = [m for m in accuracy_order if m in set(components["model"])]
    comp = components.set_index("model").reindex(order)
    fixed = design[(design["arm"] == "fixed_items") & np.isfinite(design["m"])]
    colors = plt.get_cmap("tab10")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    for i, model in enumerate(order):
        curve = fixed[fixed["model"] == model].sort_values("m")
        label = (display_names or {}).get(model, model)
        (line,) = ax1.plot(
            curve["m"],
            curve["se"] * 100,
            marker="o",
            markersize=3,
            linewidth=1.4,
            color=colors(i % 10),
            label=label,
        )
        ax1.axhline(
            comp.loc[model, "se_item"] * 100,
            color=line.get_color(),
            linewidth=0.7,
            linestyle="--",
            alpha=0.5,
        )
    ax1.set_xscale("log")
    ax1.set_xlabel("replicates per item (m)")
    ax1.set_ylabel("total SE of the benchmark score (pp)")
    ax1.set_title(
        f"Score SE vs replicates ({int(comp['n_items'].max())} items fixed)",
        fontsize=11,
        fontweight="bold",
    )
    ax1.text(
        0.98,
        0.96,
        "dashed = item-sampling floor\n(m -> inf)",
        transform=ax1.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        color="#666666",
    )
    ax1.legend(frameon=False, fontsize=7, ncol=2)
    ax1.spines[["top", "right"]].set_visible(False)
    ax1.grid(color="#eeeeee")
    ax1.set_axisbelow(True)

    x = np.arange(len(order))
    w = 0.4
    m_actual = comp["mean_replicates"].to_numpy()
    # Response share of the score variance: E[p(1-p)]/m / (Var(p) + E[p(1-p)]/m).
    at_1 = comp["response_var"] / (comp["item_var"] + comp["response_var"])
    at_actual = (comp["response_var"] / m_actual) / (
        comp["item_var"] + comp["response_var"] / m_actual
    )
    ax2.bar(x - w / 2, at_1 * 100, w, label="1 replicate per item", color="#e15759")
    ax2.bar(
        x + w / 2,
        at_actual * 100,
        w,
        label=f"{m_actual.mean():.0f} replicates per item (this experiment)",
        color="#76b7b2",
    )
    ax2.set_xticks(x)
    ax2.set_xticklabels(
        _labels(order, display_names), rotation=35, ha="right", fontsize=7
    )
    ax2.set_ylabel("response noise's share of score variance (%)")
    ax2.set_title(
        "What the reruns removed (rest is item sampling)",
        fontsize=11,
        fontweight="bold",
    )
    ax2.legend(frameon=False, fontsize=8)
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.grid(axis="y", color="#eeeeee")
    ax2.set_axisbelow(True)

    title = "Score uncertainty decomposition -- item draw vs response stochasticity"
    if language:
        title = f"{language.capitalize()} Linguistic Knowledge Benchmark -- {title}"
    fig.suptitle(title, fontweight="bold", fontsize=13)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_item_variance(
    item_var: pd.DataFrame,
    out: str,
    display_names: dict | None,
    accuracy_order: list[str],
    language: str | None = None,
) -> None:
    """Grouped bars per model: mean item-level (Bernoulli) variance, split MCQ vs
    short_form, ordered by each model's overall accuracy across the 20 replicates
    (highest first, per `accuracy_order` -- same order as the REPLICATE-TO-REPLICATE CONSISTENCY
    table). A tall short_form bar next to a low MCQ bar means short-form items are
    where verdicts actually flip replicate-to-replicate, even if overall accuracy looks
    stable."""
    order = [m for m in accuracy_order if m in set(item_var["model"])]
    by_type = (
        item_var.groupby(["model", "answer_type"])["variance"]
        .mean()
        .unstack("answer_type")
    )
    by_type = by_type.reindex(order)
    mcq = by_type["multiple_choice"] if "multiple_choice" in by_type.columns else 0.0
    sf = by_type["short_form"] if "short_form" in by_type.columns else 0.0

    x = np.arange(len(order))
    w = 0.4
    fig, ax = plt.subplots(figsize=(max(12, len(order) * 0.6), 4.5))
    ax.bar(
        x - w / 2,
        pd.Series(mcq, index=order).fillna(0.0).to_numpy(),
        w,
        label="multiple_choice",
        color="#4e79a7",
    )
    ax.bar(
        x + w / 2,
        pd.Series(sf, index=order).fillna(0.0).to_numpy(),
        w,
        label="short_form",
        color="#f28e2b",
    )
    ax.axhline(0.25, color="#999999", linewidth=1, linestyle="--", zorder=1)
    ax.text(
        len(order) - 0.5,
        0.25,
        "coin flip (p=0.5)",
        color="#666666",
        fontsize=8,
        ha="right",
        va="bottom",
    )
    ax.set_ylabel("Mean item-level variance  (p·(1−p))")
    ax.set_ylim(0, 0.26)
    ax.set_xticks(x)
    ax.set_xticklabels(
        _labels(order, display_names), rotation=35, ha="right", fontsize=7
    )
    title = "Item-level instability across replicates -- MCQ vs short-form"
    if language:
        title = f"{language.capitalize()} Linguistic Knowledge Benchmark -- {title}"
    ax.set_title(title, fontweight="bold", pad=28, fontsize=13)
    ax.text(
        0.5,
        1.06,
        "models ordered left to right by accuracy (highest first)",
        transform=ax.transAxes,
        ha="center",
        fontsize=9,
        color="#666666",
    )
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#eeeeee")
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_item_variance_boxplot(
    item_var: pd.DataFrame,
    out: str,
    display_names: dict | None,
    accuracy_order: list[str],
    language: str | None = None,
) -> None:
    """Boxplot of each item's own variance (one pair of boxes per model, MCQ vs
    short_form), same model order as plot_item_variance's bars. Most items sit at
    variance=0 exactly (stably right or stably wrong every replicate), so the box
    itself is usually squashed near zero -- the outlier points above it are where
    genuine item-level instability actually is, which the mean bars average away."""
    order = [m for m in accuracy_order if m in set(item_var["model"])]
    positions = np.arange(len(order)) * 3.0
    width = 1.0

    def _series_for(answer_type: str) -> list[np.ndarray]:
        return [
            item_var.loc[
                (item_var["model"] == m) & (item_var["answer_type"] == answer_type),
                "variance",
            ].to_numpy()
            for m in order
        ]

    fig, ax = plt.subplots(figsize=(max(12, len(order) * 0.7), 5))

    def _draw(data: list[np.ndarray], offset: float, color: str) -> None:
        pos = positions + offset
        # boxplot errors on an empty array -- skip slots with no items of this type.
        kept = [(p, d) for p, d in zip(pos, data) if len(d)]
        if not kept:
            return
        kept_pos, kept_data = zip(*kept)
        ax.boxplot(
            kept_data,
            positions=list(kept_pos),
            widths=width,
            patch_artist=True,
            showfliers=True,
            flierprops={
                "marker": "o",
                "markersize": 2.5,
                "alpha": 0.5,
                "markerfacecolor": color,
                "markeredgecolor": "none",
            },
            boxprops={"facecolor": color, "alpha": 0.5},
            medianprops={"color": "#222222", "linewidth": 1.2},
            whiskerprops={"color": color},
            capprops={"color": color},
        )

    _draw(_series_for("multiple_choice"), -width * 0.6, "#4e79a7")
    _draw(_series_for("short_form"), width * 0.6, "#f28e2b")

    ax.axhline(0.25, color="#999999", linewidth=1, linestyle="--", zorder=0)
    ax.text(
        positions[-1] + width,
        0.25,
        "coin flip (p=0.5)",
        color="#666666",
        fontsize=8,
        ha="right",
        va="bottom",
    )
    ax.set_xticks(positions)
    ax.set_xticklabels(
        _labels(order, display_names), rotation=35, ha="right", fontsize=7
    )
    ax.set_ylabel("Item-level variance  (p·(1−p))")
    ax.set_ylim(-0.01, 0.27)
    title = "Per-item variance distribution across replicates -- MCQ vs short-form"
    if language:
        title = f"{language.capitalize()} Linguistic Knowledge Benchmark -- {title}"
    ax.set_title(title, fontweight="bold", pad=28, fontsize=13)
    ax.text(
        0.5,
        1.06,
        "models ordered left to right by accuracy (highest first)",
        transform=ax.transAxes,
        ha="center",
        fontsize=9,
        color="#666666",
    )
    handles = [
        Patch(facecolor="#4e79a7", label="multiple_choice", alpha=0.5),
        Patch(facecolor="#f28e2b", label="short_form", alpha=0.5),
    ]
    ax.legend(handles=handles, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#eeeeee")
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--language", required=True, choices=LANGUAGES)
    ap.add_argument(
        "--experiment-id",
        default=DEFAULT_EXPERIMENT_ID,
        help=f"selects summary_linguistic-knowledge_<language>_<experiment-id>[.csv|"
        f"_detailed.csv] (default: {DEFAULT_EXPERIMENT_ID})",
    )
    ap.add_argument("--n-replicates", type=int, default=DEFAULT_N_REPLICATES)
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--label", default=None)
    ap.add_argument("--models-csv", default=None)
    ap.add_argument("--providers-csv", default=None)
    ap.add_argument(
        "--seed",
        type=int,
        default=0,
        help="seed for the beta-binomial posterior-predictive fit check (default: 0)",
    )
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    outdir = os.path.join(
        args.outdir or os.path.join(root, "results", "consistency"), args.language
    )
    label = args.label or f"{args.language}_{args.experiment_id}"
    models_csv = args.models_csv or os.path.join(root, "data", "models.csv")
    providers_csv = args.providers_csv or os.path.join(root, "data", "providers.csv")

    results_dir = os.path.join(root, "results")
    base = f"summary_linguistic-knowledge_{args.language}_{args.experiment_id}"
    summary_path = os.path.join(results_dir, f"{base}.csv")
    detailed_path = os.path.join(results_dir, f"{base}_detailed.csv")

    missing = [p for p in (summary_path, detailed_path) if not os.path.exists(p)]
    if missing:
        listing = "\n".join(f"  {p}" for p in missing)
        raise SystemExit(
            f"Missing result file(s) for --language {args.language} --experiment-id "
            f"{args.experiment_id}:\n{listing}\n\n"
            "These are produced by run-bench/judge-bench scored against the repeated "
            "(consistency) dataset -- see build_consistency_dataset.py's docstring."
        )

    summary_df = load_summary(summary_path)
    print_pooled_summary(summary_df)

    agg, raw = compute_consistency_table(detailed_path, args.n_replicates)
    if raw.empty:
        raise SystemExit(
            "No row decoded to a valid replicate -- is --n-replicates set to what "
            "build_consistency_dataset.py used to build this run's dataset?"
        )
    print_consistency_table(agg, args.n_replicates)

    display_names = load_display_names(models_csv)
    model_provider, provider_color = load_provider_colors(models_csv, providers_csv)

    os.makedirs(outdir, exist_ok=True)
    agg_path = os.path.join(outdir, f"consistency_{label}.csv")
    agg.round(2).to_csv(agg_path, index=False)
    png_path = os.path.join(outdir, f"consistency_{label}.png")
    plot_consistency(
        raw, agg, png_path, display_names, model_provider, provider_color, args.language
    )

    item_var = compute_item_variance_table(detailed_path, args.n_replicates)
    print_item_variance_summary(item_var)
    item_var_path = os.path.join(outdir, f"item_variance_{label}.csv")
    item_var.round(4).to_csv(item_var_path, index=False)
    item_var_png = os.path.join(outdir, f"item_variance_{label}.png")
    accuracy_order = agg["model"].tolist()
    plot_item_variance(
        item_var, item_var_png, display_names, accuracy_order, args.language
    )
    item_var_box_png = os.path.join(outdir, f"item_variance_box_{label}.png")
    plot_item_variance_boxplot(
        item_var, item_var_box_png, display_names, accuracy_order, args.language
    )

    components = compute_variance_components(item_var)
    design = compute_design_table(components)
    print_variance_components(components, design)
    print_beta_fit_check(check_beta_fit(item_var, components, seed=args.seed))
    components_path = os.path.join(outdir, f"variance_components_{label}.csv")
    components.round(6).to_csv(components_path, index=False)
    design_path = os.path.join(outdir, f"design_grid_{label}.csv")
    design.round(6).to_csv(design_path, index=False)
    components_png = os.path.join(outdir, f"variance_components_{label}.png")
    plot_variance_components(
        components, design, components_png, display_names, accuracy_order, args.language
    )

    print(
        f"\nWrote:\n  {agg_path}\n  {png_path}\n  {item_var_path}\n  {item_var_png}\n"
        f"  {item_var_box_png}\n  {components_path}\n  {design_path}\n  {components_png}"
    )


if __name__ == "__main__":
    main()
