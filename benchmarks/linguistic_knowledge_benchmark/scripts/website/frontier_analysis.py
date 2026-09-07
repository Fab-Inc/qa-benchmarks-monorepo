"""
Frontier analysis and default-model selection for the leaderboard website.

Vendored from the CDPK benchmark-website pipeline
(``benchmark_website_datasets.frontier_analysis``) and trimmed to what this
cluster's publisher needs: Pareto/concave frontier computation, successive layer
peeling, and the default-model selection that combines them with a
provider/family dedup. Used by ``scripts/results_for_website.py``.

Dropped from the original, deliberately:
  * ``build_benchmarks_models_df`` - parses the CDPK published-JSON shape; this
    repo reads the judge-bench summary CSVs directly.
  * ``create_benchmark_frontier_models_dict_per_cat`` - hardcodes the
    CDPK/SEND/MATHV/VISUAL_REASONING benchmark names (and raises on anything
    else), and needs subject/level category columns this cluster does not have.
  * ``check_df`` / ``enforce_price_ratio`` - results_for_website.py already
    validates its inputs and blends prices identically.

NOTHING HERE NAMES A BENCHMARK. Every function is driven by whatever
benchmark_domain keys the caller passes in, so LK/DA/DTG x any language work
with no change here.
"""

import re
from collections import defaultdict

import numpy as np
import pandas as pd

# Columns every frame in `benchmark_dfs` must carry for selection to work.
REQUIRED_COLUMNS = ("model", "accuracy", "open")

# Per-provider family filters for model deduplication in the default list.
# Keys are provider names matching models.csv. Values are ordered lists of
# regex patterns (re.search), from most specific to least specific.
# A model is assigned to the first pattern that matches its model_id.
# Order matters: more specific patterns must come before broader ones, e.g.
# "gpt-5.*nano" before "gpt-5", "flash-lite" before "flash".
# Rules applied per (provider, family):
#   - Always ensure the most recent model (by release_date) is included.
#   - If >= 2 models of the same family are present, remove all but the most
#     recent one, UNLESS the older model appears on a frontier line.
PROVIDER_FAMILY_FILTERS = {
    "Anthropic": [
        r"claude.*-sonnet",
        r"claude.*-opus",
        r"claude.*-haiku",
    ],
    "Google": [
        r"gemini.*-flash-lite",
        r"gemini.*-flash",
        r"gemini.*-pro",
    ],
    "OpenAI": [
        # nano tier: "luna" is OpenAI's post-rename name for nano
        r"gpt-5.*-(nano|luna)(-|$)",  # gpt-5-nano-*, gpt-5.4-nano-*, gpt-5.6-luna-*
        # mini tier: "terra" is OpenAI's post-rename name for mini
        r"gpt-5.*-(mini|terra)(-|$)",  # gpt-5-mini-*, gpt-5.6-terra-* (not "minimal")
        # base/normal tier: catches gpt-5-*, gpt-5.1-*, ... and the "sol" suffix
        r"gpt-5",
        r"^o.*mini",  # o-series mini: o1-mini, o3-mini, o4-mini (not gpt-4o-mini)
        r"^o",  # o-series base: o1, o3, o4
    ],
    "xAI": [
        r"grok.*fast",  # grok-4-fast, grok-4.1-fast
        r"grok",  # grok-2, grok-3, grok-4, grok-3-mini, etc.
    ],
    "AI21labs": [
        r"jamba.*large",  # jamba-1-5-large, jamba-1-6-large, jamba-large-1.7
        r"jamba.*mini",  # jamba-1-5-mini, jamba-1-6-mini, jamba-mini-1.7
    ],
    "Z.ai": [
        r"glm.*v",  # glm-4.1v-9b-thinking, glm-4.6v (vision models)
        r"glm",  # fw-glm-4.5, fw-glm-4.6, fw-glm-4.7, glm-5, glm-5.1
    ],
    "Alibaba": [
        r"qwen.*flash",  # API flash: qwen3.5-flash, qwen3.6-flash
        r"qwen.*plus",  # API plus: qwen3.5-plus, qwen3.6-plus
        r"qwen.*max",  # API max: qwen3.6-max-preview
        r"qwen.*-a3b",  # small MoE (~3B active): 30b-a3b, 35b-a3b
        r"qwen.*-a\d{2,}b",  # large MoE (>=10B active): 122b-a10b, 235b-a22b
        r"qwen",  # dense models (all sizes): qwen2.5-*, qwen-3-*, qwen3.5-*
    ],
    "DeepSeek-AI": [
        r"deepseek.*-r",  # R-series reasoning: r1, r1-0528
        r"deepseek",  # V-series: v2.5, v3, v3.1, v3.2, v4-flash, v4-pro
    ],
    "Mistral": [
        r"magistral",  # Magistral reasoning: magistral-small, magistral-medium
        r"ministral",  # Ministral small models: 3B, 8B, 14B
        r"mistral.*small",  # Small: mistral-small-3/3.1/3.2/4
        r"mistral.*medium",  # Medium: mistral-medium-3/3.1/3.5
        r"mistral.*large",  # Large: mistral-large-2411, mistral-large-3
    ],
    "Meta": [
        r"llama-4|llama.*-\d{3}b",  # large (100B+): Llama 4 maverick/scout, 405B
        r"llama.*-\d{2}b",  # medium (10-99B): 11B, 70B, 90B
        r"llama",  # small (1-9B): 1B, 3B, 8B
    ],
    "Xiaomi": [
        r"mimo.*pro",  # MiMo Pro: mimo-v2-pro, mimo-v2.5-pro
        r"mimo",  # MiMo base/flash: mimo-v2-flash, mimo-v2.5
    ],
}


def compute_frontier(data, acc_col="accuracy", sorting_col="input_cost", smooth=None):
    """
    Computes the value frontier from a set of models.

    1. Sorts models by cost.
    2. Finds the Pareto frontier (models where accuracy is the highest seen so far).
    3. If smooth=True, filters this frontier to find the concave hull, ensuring
       the slope (Accuracy vs. log(sorting_col)) is always decreasing.

    Args:
        data: DataFrame containing model data
        acc_col: Column name for accuracy metric (default: 'accuracy')
        sorting_col: Column name for cost/size variable (default: 'input_cost')
        smooth: If True, enforces concavity (decreasing slope) (default: None)

    Returns:
        DataFrame containing frontier models
    """
    data = data.sort_values(sorting_col, ascending=True)
    frontier, best = [], -1e9

    for _, row in data.iterrows():
        if row[acc_col] > best:
            frontier.append(row)
            best = row[acc_col]

    if not frontier:
        return pd.DataFrame(columns=data.columns)

    df_return = pd.DataFrame(frontier)

    # if rows have same input cost, keep the one with highest accuracy
    df_return = df_return.sort_values(sorting_col).drop_duplicates(
        sorting_col, keep="last"
    )

    # If not smoothing or < 3 points, return the simple Pareto frontier
    # (Concavity can't be computed with 2 or fewer points)
    if not smooth or len(df_return) < 3:
        return df_return.reset_index(drop=True)

    # 3. Enforce Concavity
    # We use the log of the variable, matching the plot's x-axis
    # Add a tiny epsilon to cost to avoid log(0) if it exists
    df_return["log_var"] = np.log(df_return[sorting_col].astype(float) + 1e-9)

    # Convert to a list of tuples (index, log_var, accuracy)
    # We store the *original index* to filter the DataFrame later
    points = list(zip(df_return.index, df_return["log_var"], df_return[acc_col]))

    # This list will hold the *indices* of the points on the concave hull
    concave_indices = []

    # Add the first two points to start
    concave_indices.append(points[0][0])  # index of first point
    concave_indices.append(points[1][0])  # index of second point

    for i in range(2, len(points)):
        # Get the new point to consider
        p_new_idx, x_new, y_new = points[i]

        # Keep checking the last segment on the hull
        while len(concave_indices) >= 2:
            # Get the last two points *from the hull we are building*
            p_mid_idx = concave_indices[-1]
            p_prev_idx = concave_indices[-2]

            # Get their coordinates from the DataFrame
            x_mid, y_mid = df_return.loc[p_mid_idx][["log_var", acc_col]]
            x_prev, y_prev = df_return.loc[p_prev_idx][["log_var", acc_col]]

            # Calculate the slopes of the two segments
            # Handle potential vertical lines (though drop_duplicates should
            # prevent this)
            slope1 = (
                (y_mid - y_prev) / (x_mid - x_prev) if (x_mid - x_prev) != 0 else np.inf
            )
            slope2 = (
                (y_new - y_mid) / (x_new - x_mid) if (x_new - x_mid) != 0 else np.inf
            )

            # --- Concavity Check ---
            # If the new slope (slope2) is *greater* than the previous slope (slope1),
            # it means the middle point (p_mid) creates a "convex" bump.
            # We must remove it and check the point before it.
            if slope2 > slope1:
                concave_indices.pop()  # Remove the middle point
            else:
                # The slope is decreasing (or flat), which is concave.
                # Stop checking and keep the middle point.
                break

        # Add the new point to the hull
        concave_indices.append(p_new_idx)

    # Filter the original dataframe to keep only the points in our concave hull
    df_return = df_return.loc[concave_indices]

    # Clean up the helper column
    df_return = df_return.drop(columns=["log_var"])

    return df_return.reset_index(drop=True)


def compute_layers(
    df, frontier_var="input_cost", acc_col="accuracy", smooth=True, max_models=None
):
    """
    Computes successive concave frontier layers ("peeling").

    1. Computes the concave frontier (Layer 0).
    2. Removes those points from the dataset.
    3. Computes the next frontier (Layer 1) from the remaining points.
    4. Repeats until no more concave frontiers (min 3 points) can be found.

    Args:
        df: DataFrame containing model data
        frontier_var: Variable for x-axis (e.g., 'cost' or 'size')
        acc_col: Variable for y-axis (e.g., 'accuracy')
        smooth: Whether to enforce concavity (default: True)
        max_models: Maximum models to collect across layers (default: None)

    Returns:
        List of DataFrames, each representing a frontier layer
    """

    all_frontiers = []
    # We must reset the index of the source df to use .merge for removal
    remaining_df = df.copy().reset_index(drop=True)

    # Safety counter to prevent infinite loops
    max_iterations = len(df) + 1
    iteration_count = 0

    while True:
        iteration_count += 1
        if iteration_count > max_iterations:
            # Safety break to prevent infinite loops
            break
        # 1. Compute frontier on remaining points
        frontier = compute_frontier(
            remaining_df, acc_col=acc_col, sorting_col=frontier_var, smooth=smooth
        )

        # 2. Stop if the frontier is not a meaningful concave line (needs >= 3 pts)
        if len(frontier) < 3:
            break

        all_frontiers.append(frontier)

        # 3. "Peel off" the layer.
        # NOTE: the peel matches on (frontier_var, acc_col), not on model, so two
        # distinct models sharing an identical (cost, accuracy) pair are peeled
        # together and only one of them is ever recorded on a layer.
        key_cols = [frontier_var, acc_col]
        temp_frontier = frontier[key_cols].copy()
        temp_frontier["_is_frontier"] = True

        merged = remaining_df.merge(temp_frontier, on=key_cols, how="left")

        # 4. Keep only the rows that were *not* tagged
        remaining_df = merged[merged["_is_frontier"].isnull()].drop(
            columns=["_is_frontier"]
        )

        # 5. Stop if no points are left
        if remaining_df.empty:
            break

    # if less than 3 models on frontier, rerun compute_frontier to have
    # 2 models-frontier
    if not all_frontiers:
        frontier = compute_frontier(
            remaining_df, acc_col=acc_col, sorting_col=frontier_var, smooth=smooth
        )
        all_frontiers.append(frontier)
        return all_frontiers

    # Compute number of models collected so far
    max_models_collected = sum(len(layer) for layer in all_frontiers)

    # Filter out some layers if too many models collected
    # Safety counter to prevent infinite loops
    reduction_iteration_count = 0
    max_reduction_iterations = len(all_frontiers) + 1

    while max_models is not None and max_models_collected > max_models:
        reduction_iteration_count += 1
        if reduction_iteration_count > max_reduction_iterations:
            # Safety break - can't reduce further
            break

        # If only one layer left, can't reduce by filtering
        if len(all_frontiers) <= 1:
            break

        # keep only even layers: 0, 2, 4, ...
        indices = np.arange(0, len(all_frontiers), 2)
        all_frontiers = [all_frontiers[i] for i in indices]
        # update max_models_collected
        max_models_collected = sum(len(layer) for layer in all_frontiers)

    return all_frontiers


def _positive(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Rows where `column` is present and > 0.

    Every frontier must be computed over the same population the website draws,
    which filters cost > 0 and requires a known size. Without this a model with a
    null cost sorts last, can be admitted to the frontier at "infinite" cost, and
    ends up a default model with no dot on the drawn line.
    """
    if column not in df.columns:
        return df.iloc[0:0]
    return df[df[column].notna() & (df[column] > 0)]


def create_default_models_list(
    benchmark_dfs,
    *,
    models_csv_path,
    cost_var="cost",
    max_models=10,
    add_missing_models=False,
    filter_by_provider_family=False,
    manual_default_models=(),
):
    """
    Creates a default list of models based on frontier analysis.

    The selection process has four stages:
    1. Frontier layer collection: For each benchmark, computes successive
       concave frontier layers (accuracy vs. cost, and accuracy vs. size for
       open-source models). All models on any layer are selected. The total
       number of models per frontier type (cost or size) per benchmark is
       capped at max_models by progressively dropping every other layer when
       the total exceeds the cap (see compute_layers).
    2. Missing frontier models: If add_missing_models=True, adds models that
       appear on a non-smooth (raw Pareto) frontier but were not already
       selected by the smoothed layer peeling. These frontiers are computed
       inline per benchmark and open/closed status (smooth=False). This is what
       guarantees every vertex of the drawn frontier line is a default model.
    3. Provider/family deduplication: If filter_by_provider_family=True,
       applies PROVIDER_FAMILY_FILTERS to group models by provider and family.
       For each family, the most recent model is always kept. If >= 2 models
       of the same family are present, older ones are removed unless they
       appear on a non-smooth (raw Pareto) frontier computed in stage 2.
       Release dates are read from models_csv_path; a release_date that is not
       MM/YYYY raises, naming every offending row, since the ranking this stage
       depends on would otherwise be silently wrong.
    4. Manual additions: manual_default_models are force-included last, so the
       dedup above cannot strip them.

    Args:
        benchmark_dfs: Dictionary of benchmark_domain -> DataFrame. Each frame
            must carry REQUIRED_COLUMNS plus `cost_var` and `size`.
        models_csv_path: models.csv to read release dates from for stage 3. Pass
            the same file the caller drew its model metadata from, or the dedup
            silently ranks against a different catalogue.
        cost_var: Column name for cost variable (default: 'cost')
        max_models: Soft cap on the number of models collected per frontier
            type (cost or size) per benchmark (default: 10). Passed to
            compute_layers, which progressively discards every other layer
            until the total across layers is within the cap.
        add_missing_models: If True, adds models on the non-smooth (raw
            Pareto) frontiers that were missed by the smoothed layer peeling
            (default: False)
        filter_by_provider_family: If True, applies PROVIDER_FAMILY_FILTERS
            to deduplicate models per provider/family using release dates
            from models_csv_path (default: False)
        manual_default_models: Model ids always forced into the default list.
            Each must have benchmark data or it cannot appear in any benchmark's
            list; a warning is printed for those that do not. Keep it short.

    Returns:
        Tuple of (selected_models_list, benchmark_default_models_dict)
    """
    selected_models_set = set()
    # Non-smooth (raw Pareto) frontier models, used by add_missing_models
    non_smooth_frontier_set = set()
    # model_id -> {"DOMAIN/frontier_var", ...}, i.e. which raw Pareto
    # frontiers each model sits on. Only used to explain why the provider/family
    # dedup below spared a model.
    non_smooth_frontier_sources = defaultdict(set)

    # Loop through each benchmark
    for benchmark_name, df in benchmark_dfs.items():
        missing_columns = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing_columns:
            # The original skipped silently here, which produced a default list
            # derived from whatever OTHER benchmarks happened to select.
            raise ValueError(
                f"benchmark_dfs['{benchmark_name}'] is missing required column(s): "
                f"{', '.join(missing_columns)}"
            )

        # Process open and closed source separately
        for open_status in [True, False]:
            subset_df = df[df["open"] == open_status].copy()
            if subset_df.empty:
                continue

            # Compute cost frontier layers (smooth=True for layer peeling)
            cost_df = _positive(subset_df, cost_var)
            if not cost_df.empty:
                cost_layers = compute_layers(
                    cost_df,
                    frontier_var=cost_var,
                    acc_col="accuracy",
                    smooth=True,
                    max_models=max_models,
                )
                for layer_df in cost_layers:
                    selected_models_set.update(layer_df["model"].tolist())

                # Non-smooth cost frontier for add_missing_models
                cost_frontier = compute_frontier(
                    cost_df, acc_col="accuracy", sorting_col=cost_var, smooth=False
                )
                non_smooth_frontier_set.update(cost_frontier["model"].tolist())
                for model in cost_frontier["model"].tolist():
                    non_smooth_frontier_sources[model].add(
                        f"{benchmark_name}/{cost_var}"
                    )

            # Compute size frontier layers (open source only)
            size_df = (
                _positive(subset_df, "size") if open_status else subset_df.iloc[0:0]
            )
            if not size_df.empty:
                size_layers = compute_layers(
                    size_df,
                    frontier_var="size",
                    acc_col="accuracy",
                    smooth=True,
                    max_models=max_models,
                )
                for layer_df in size_layers:
                    selected_models_set.update(layer_df["model"].tolist())

                # Non-smooth size frontier for add_missing_models
                size_frontier = compute_frontier(
                    size_df, acc_col="accuracy", sorting_col="size", smooth=False
                )
                non_smooth_frontier_set.update(size_frontier["model"].tolist())
                for model in size_frontier["model"].tolist():
                    non_smooth_frontier_sources[model].add(f"{benchmark_name}/size")

    selected_models = sorted(selected_models_set)
    print(f"Completed. Found {len(selected_models)} unique models on all frontiers.")

    # Models that have a result in at least one benchmark. Anything outside this
    # set can never reach a per-benchmark default list.
    models_with_data = set()
    for df in benchmark_dfs.values():
        models_with_data.update(df["model"].values)

    # Add missing frontier models if requested (uses non-smooth Pareto frontiers
    # computed inline)
    if add_missing_models:
        missing_models = [
            model for model in non_smooth_frontier_set if model not in selected_models
        ]
        if len(missing_models) > 0:
            print(
                f"Adding {len(missing_models)} missing frontier models to the "
                f"default list."
            )
            selected_models.extend(missing_models)
            selected_models = sorted(set(selected_models))
            print(f"Updated total unique models: {len(selected_models)}")

    # Apply provider/family filters using PROVIDER_FAMILY_FILTERS
    filter_triggered = False
    # (provider, family, model, most_recent) for older models that dedup would
    # have dropped but kept because they sit on >= 1 raw Pareto frontier.
    frontier_protected = []
    if filter_by_provider_family:
        models_df = pd.read_csv(models_csv_path)
        # errors="coerce" only to collect EVERY bad row for the message below, not to
        # tolerate them: an unparsed date becomes NaT, pandas sorts NaT last, and that
        # model would then be taken as the newest in its family and drop its genuinely
        # newer siblings. Silently picking the wrong "most recent" model is worse than
        # stopping, and the fix is a one-character edit in models.csv, so abort here.
        models_df["release_date_parsed"] = pd.to_datetime(
            models_df["release_date"], format="%m/%Y", errors="coerce"
        )
        unparsed = models_df.loc[models_df["release_date_parsed"].isna()]
        if not unparsed.empty:
            rows = "\n".join(
                f"  row {i + 2}: {r['model_id']} -> release_date={r['release_date']!r}"
                for i, r in unparsed.iterrows()
            )
            raise ValueError(
                f"{len(unparsed)} model(s) in {models_csv_path} have an unparseable "
                f"release_date; expected MM/YYYY (e.g. 07/2026):\n{rows}\n"
                "Provider/family deduplication ranks models by release date, so it "
                "cannot run until these are fixed."
            )
        selected_models_set = set(selected_models)

        for provider, families in PROVIDER_FAMILY_FILTERS.items():
            provider_df = models_df[models_df["provider"] == provider]

            def _matches(model_id, pattern):
                """Return True if model_id matches the regex pattern."""
                return bool(re.search(pattern, model_id))

            def _first_match(model_id, families=families):
                return next((fk for fk in families if _matches(model_id, fk)), None)

            for family_keyword in families:
                # Assign each model to its most-specific matching family
                family_df = provider_df[
                    provider_df["model_id"].apply(
                        lambda m, fk=family_keyword: _first_match(m) == fk
                    )
                ].sort_values("release_date_parsed")

                if family_df.empty:
                    continue

                # Only the models that actually have benchmark data can end up in
                # a per-benchmark default list, so a family with none of them is
                # not our problem.
                family_with_data = family_df[
                    family_df["model_id"].isin(models_with_data)
                ]
                if family_with_data.empty:
                    continue
                most_recent = family_with_data.iloc[-1]["model_id"]
                family_model_ids = family_df["model_id"].tolist()
                family_label = family_keyword

                # Ensure the most recent model is always included
                if most_recent not in selected_models_set:
                    if not filter_triggered:
                        # Only print once, even if multiple families trigger
                        print("=" * 70)
                        filter_triggered = True
                    print(
                        f"Adding most recent {provider}/{family_label} model: "
                        f"{most_recent}"
                    )
                    selected_models_set.add(most_recent)

                # If >= 2 models present, remove older ones unless they are
                # on a non-smooth (raw) Pareto frontier
                present = [m for m in family_model_ids if m in selected_models_set]
                if len(present) >= 2:
                    to_remove = [
                        m
                        for m in present
                        if m != most_recent and m not in non_smooth_frontier_set
                    ]
                    # Same condition, inverted: older models spared only because
                    # they are on a raw Pareto frontier.
                    frontier_protected.extend(
                        (provider, family_label, m, most_recent)
                        for m in present
                        if m != most_recent and m in non_smooth_frontier_set
                    )
                    if to_remove:
                        if not filter_triggered:
                            # Only print once, even if multiple families trigger
                            print("=" * 70)
                            filter_triggered = True
                        print(
                            f"Filtering {provider}/{family_label}: removing "
                            f"{to_remove}, keeping {most_recent}"
                        )
                        selected_models_set -= set(to_remove)

        selected_models = sorted(selected_models_set)
        print(
            f"Final list has {len(selected_models)} models after provider/family "
            f"filtering."
        )

        if frontier_protected:
            if not filter_triggered:
                print("=" * 70)
                filter_triggered = True
            print(
                f"\n{len(frontier_protected)} older model(s) kept despite provider/"
                f"family dedup (on >= 1 raw Pareto frontier):"
            )
            for provider, family_label, model, most_recent in frontier_protected:
                frontiers = ", ".join(sorted(non_smooth_frontier_sources[model]))
                print(f"  {provider}/{family_label}: kept {model}")
                print(f"      newer model in family: {most_recent}")
                print(f"      on frontier(s): {frontiers}")

        if filter_triggered:
            print("=" * 70)

    # Force-include manually specified models (after all filtering so they can't
    # be stripped by the provider/family dedup logic).
    if manual_default_models:
        # Warn (don't fail) for manual models that have no benchmark data: they
        # would be silently dropped from every benchmark's default list below.
        for model in manual_default_models:
            if model not in models_with_data:
                print(
                    f"Warning: manual default model '{model}' has no benchmark "
                    f"data; it will not appear in any benchmark's default list."
                )
        for model in manual_default_models:
            if model not in selected_models:
                print(f"Adding manual default model: {model}")
                selected_models.append(model)
        selected_models = sorted(set(selected_models))

    # Build benchmark-specific default models dict
    benchmark_default_models_dict = {}
    for benchmark_name, df in benchmark_dfs.items():
        models_in_benchmark = [
            model for model in selected_models if model in df["model"].values
        ]
        benchmark_default_models_dict[benchmark_name] = models_in_benchmark

    return selected_models, benchmark_default_models_dict
