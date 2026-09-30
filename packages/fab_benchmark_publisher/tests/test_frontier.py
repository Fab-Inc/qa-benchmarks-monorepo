"""Tests for the vendored frontier maths and default-model selection.

These functions decide which models the leaderboard pre-selects and where the drawn
frontier lines bend, so a regression here is visible on the website but silent in
every other check.
"""

import pandas as pd
import pytest
from fab_benchmark_publisher.frontier import (
    _positive,
    compute_frontier,
    compute_layers,
    create_default_models_list,
)


def frame(rows: list[tuple], columns: tuple[str, ...] = ("model", "cost", "accuracy")):
    return pd.DataFrame(rows, columns=list(columns))


def models_csv(tmp_path, rows: list[tuple[str, str, str]]):
    """A minimal models.csv with the columns the family dedup reads."""
    path = tmp_path / "models.csv"
    lines = ["model_id,provider,release_date"]
    lines += [f"{mid},{provider},{date}" for mid, provider, date in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #
# compute_frontier
# --------------------------------------------------------------------------- #
def test_compute_frontier_keeps_only_improving_models() -> None:
    """Pareto: sorted by cost, a model is on the frontier only if it beats every
    cheaper one."""
    df = frame([("a", 1.0, 50.0), ("b", 2.0, 40.0), ("c", 3.0, 60.0)])

    assert list(compute_frontier(df, sorting_col="cost")["model"]) == ["a", "c"]


def test_compute_frontier_empty_input_returns_empty_frame_with_columns() -> None:
    result = compute_frontier(frame([]), sorting_col="cost")

    assert result.empty
    assert list(result.columns) == ["model", "cost", "accuracy"]


def test_compute_frontier_ties_on_cost_keep_the_more_accurate_model() -> None:
    df = frame([("cheap-bad", 1.0, 50.0), ("cheap-good", 1.0, 70.0)])

    assert list(compute_frontier(df, sorting_col="cost")["model"]) == ["cheap-good"]


def test_compute_frontier_smooth_drops_a_convex_bump() -> None:
    """The middle point sits below the line joining its neighbours, so enforcing a
    decreasing slope removes it -- that is the whole point of smooth=True."""
    df = frame([("a", 1.0, 10.0), ("b", 10.0, 20.0), ("c", 100.0, 90.0)])

    assert list(compute_frontier(df, sorting_col="cost", smooth=False)["model"]) == [
        "a",
        "b",
        "c",
    ]
    assert list(compute_frontier(df, sorting_col="cost", smooth=True)["model"]) == [
        "a",
        "c",
    ]


def test_compute_frontier_smooth_keeps_an_already_concave_curve() -> None:
    df = frame([("a", 1.0, 10.0), ("b", 10.0, 80.0), ("c", 100.0, 90.0)])

    assert list(compute_frontier(df, sorting_col="cost", smooth=True)["model"]) == [
        "a",
        "b",
        "c",
    ]


def test_compute_frontier_smooth_is_skipped_below_three_points() -> None:
    """Concavity is undefined for two points, so smooth=True must not drop either."""
    df = frame([("a", 1.0, 10.0), ("b", 10.0, 20.0)])

    assert list(compute_frontier(df, sorting_col="cost", smooth=True)["model"]) == [
        "a",
        "b",
    ]


def test_compute_frontier_sorts_by_the_requested_column() -> None:
    df = frame(
        [("big", 70.0, 90.0), ("small", 7.0, 60.0)],
        columns=("model", "size", "accuracy"),
    )

    assert list(compute_frontier(df, sorting_col="size")["model"]) == ["small", "big"]


# --------------------------------------------------------------------------- #
# compute_layers
# --------------------------------------------------------------------------- #
def test_compute_layers_peels_successive_frontiers() -> None:
    """Layer 0 is the frontier; layer 1 is the frontier of what is left."""
    df = frame(
        [
            ("a", 1.0, 30.0),
            ("b", 10.0, 60.0),
            ("c", 100.0, 70.0),
            ("d", 2.0, 20.0),
            ("e", 20.0, 40.0),
            ("f", 200.0, 50.0),
        ]
    )

    layers = compute_layers(df, frontier_var="cost", smooth=False)

    assert [sorted(layer["model"]) for layer in layers] == [
        ["a", "b", "c"],
        ["d", "e", "f"],
    ]


def test_compute_layers_falls_back_to_a_single_short_frontier() -> None:
    """Fewer than 3 points can form no concave layer, but the caller still needs the
    models that are on the frontier."""
    df = frame([("a", 1.0, 30.0), ("b", 10.0, 60.0)])

    layers = compute_layers(df, frontier_var="cost")

    assert len(layers) == 1
    assert sorted(layers[0]["model"]) == ["a", "b"]


def test_compute_layers_drops_alternate_layers_when_over_max_models() -> None:
    """The cap is enforced by discarding every other layer, keeping layer 0."""
    rows = []
    for layer in range(4):
        for i in range(3):
            rows.append((f"m{layer}{i}", 10.0**i + layer, 90.0 - 10 * layer + i))
    df = frame(rows)

    uncapped = compute_layers(df, frontier_var="cost", smooth=False)
    capped = compute_layers(df, frontier_var="cost", smooth=False, max_models=6)

    assert len(uncapped) == 4
    assert len(capped) == 2
    assert sum(len(layer) for layer in capped) <= 6


def test_compute_layers_cannot_reduce_a_single_layer_below_the_cap() -> None:
    """One layer of 3 against max_models=1: the loop must exit, not spin."""
    df = frame([("a", 1.0, 10.0), ("b", 10.0, 20.0), ("c", 100.0, 30.0)])

    layers = compute_layers(df, frontier_var="cost", smooth=False, max_models=1)

    assert len(layers) == 1


# --------------------------------------------------------------------------- #
# _positive
# --------------------------------------------------------------------------- #
def test_positive_drops_null_and_non_positive_values() -> None:
    df = frame([("a", 1.0, 10.0), ("b", 0.0, 20.0), ("c", None, 30.0)])

    assert list(_positive(df, "cost")["model"]) == ["a"]


def test_positive_on_a_missing_column_returns_no_rows() -> None:
    """A frame with no `size` column must yield nothing, not raise: closed-weight
    subsets legitimately have no size."""
    assert _positive(frame([("a", 1.0, 10.0)]), "size").empty


# --------------------------------------------------------------------------- #
# create_default_models_list
# --------------------------------------------------------------------------- #
def selection_frame(rows: list[tuple]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["model", "accuracy", "cost", "size", "open"])


def test_create_default_models_list_requires_its_columns(tmp_path) -> None:
    """Skipping a malformed frame silently would produce a default list derived from
    whatever OTHER benchmarks happened to select."""
    df = pd.DataFrame([{"model": "a", "cost": 1.0}])

    with pytest.raises(ValueError, match="missing required column"):
        create_default_models_list(
            {"LK-en": df}, models_csv_path=models_csv(tmp_path, [])
        )


def test_create_default_models_list_collects_frontier_models(tmp_path) -> None:
    dfs = {
        "LK-en": selection_frame(
            [
                ("cheap", 30.0, 1.0, 1.0, True),
                ("mid", 60.0, 10.0, 10.0, True),
                ("dear", 90.0, 100.0, 100.0, True),
                ("bad", 10.0, 50.0, 50.0, True),
            ]
        )
    }

    selected, by_domain = create_default_models_list(
        dfs, models_csv_path=models_csv(tmp_path, [])
    )

    assert "dear" in selected
    assert set(by_domain["LK-en"]) <= set(selected)


def test_create_default_models_list_add_missing_adds_raw_pareto_vertices(
    tmp_path,
) -> None:
    """A vertex the concave peeling smoothed away still has a dot on the drawn line,
    so it has to be pre-selected."""
    dfs = {
        "LK-en": selection_frame(
            [
                ("a", 10.0, 1.0, 1.0, True),
                ("bump", 20.0, 10.0, 10.0, True),
                ("c", 90.0, 100.0, 100.0, True),
            ]
        )
    }
    csv_path = models_csv(tmp_path, [])

    without = create_default_models_list(
        dfs, models_csv_path=csv_path, add_missing_models=False
    )[0]
    with_missing = create_default_models_list(
        dfs, models_csv_path=csv_path, add_missing_models=True
    )[0]

    assert "bump" not in without
    assert "bump" in with_missing


def test_create_default_models_list_rejects_unparseable_release_dates(
    tmp_path,
) -> None:
    """An unparsed date sorts last as NaT, so that model would be taken as its
    family's newest and drop genuinely newer siblings -- abort instead."""
    dfs = {"LK-en": selection_frame([("claude-4-sonnet", 90.0, 1.0, None, False)])}
    csv_path = models_csv(tmp_path, [("claude-4-sonnet", "Anthropic", "2026-01")])

    with pytest.raises(ValueError, match="unparseable release_date"):
        create_default_models_list(
            dfs, models_csv_path=csv_path, filter_by_provider_family=True
        )


def test_family_filter_keeps_only_the_most_recent_of_a_family(tmp_path, capsys) -> None:
    """The older model has to be SELECTED before dedup can drop it, so it sits on a
    peeled second layer (selected) while staying off the raw Pareto line (droppable).
    Two same-cost models would leave it unselected and pass for the wrong reason.
    """
    dfs = {
        "LK-en": selection_frame(
            [
                # layer 0, concave
                ("cheap", 30.0, 1.0, None, False),
                ("claude-4-sonnet", 80.0, 10.0, None, False),
                ("dear", 90.0, 100.0, None, False),
                # layer 1, concave, entirely under layer 0
                ("claude-3-sonnet", 20.0, 2.0, None, False),
                ("mid", 40.0, 20.0, None, False),
                ("costly", 50.0, 200.0, None, False),
            ]
        )
    }
    csv_path = models_csv(
        tmp_path,
        [
            ("claude-3-sonnet", "Anthropic", "01/2025"),
            ("claude-4-sonnet", "Anthropic", "01/2026"),
        ],
    )

    before, _ = create_default_models_list(dfs, models_csv_path=csv_path)
    after, _ = create_default_models_list(
        dfs, models_csv_path=csv_path, filter_by_provider_family=True
    )

    assert "claude-3-sonnet" in before  # selected by the layer peeling ...
    assert "claude-3-sonnet" not in after  # ... and removed by the dedup
    assert "claude-4-sonnet" in after
    assert "Filtering Anthropic" in capsys.readouterr().out


def test_family_filter_spares_an_older_model_that_is_on_a_frontier(tmp_path) -> None:
    """The dedup must not delete a model the drawn line bends at."""
    dfs = {
        "LK-en": selection_frame(
            [
                ("claude-3-sonnet", 50.0, 0.5, None, False),
                ("claude-4-sonnet", 90.0, 50.0, None, False),
                ("other", 95.0, 100.0, None, False),
            ]
        )
    }
    csv_path = models_csv(
        tmp_path,
        [
            ("claude-3-sonnet", "Anthropic", "01/2025"),
            ("claude-4-sonnet", "Anthropic", "01/2026"),
            ("other", "OtherCo", "01/2026"),
        ],
    )

    selected, _ = create_default_models_list(
        dfs,
        models_csv_path=csv_path,
        add_missing_models=True,
        filter_by_provider_family=True,
    )

    assert "claude-3-sonnet" in selected


def test_family_filter_adds_a_familys_most_recent_model_even_off_frontier(
    tmp_path,
) -> None:
    dfs = {
        "LK-en": selection_frame(
            [
                ("claude-3-sonnet", 90.0, 1.0, None, False),
                ("claude-4-sonnet", 20.0, 90.0, None, False),
            ]
        )
    }
    csv_path = models_csv(
        tmp_path,
        [
            ("claude-3-sonnet", "Anthropic", "01/2025"),
            ("claude-4-sonnet", "Anthropic", "01/2026"),
        ],
    )

    selected, _ = create_default_models_list(
        dfs, models_csv_path=csv_path, filter_by_provider_family=True
    )

    assert "claude-4-sonnet" in selected


def test_family_filter_ignores_families_with_no_benchmark_data(tmp_path) -> None:
    """A family nobody scored cannot reach a default list, so it must not be added."""
    dfs = {"LK-en": selection_frame([("other", 90.0, 1.0, None, False)])}
    csv_path = models_csv(
        tmp_path,
        [
            ("claude-4-sonnet", "Anthropic", "01/2026"),
            ("other", "OtherCo", "01/2026"),
        ],
    )

    selected, _ = create_default_models_list(
        dfs, models_csv_path=csv_path, filter_by_provider_family=True
    )

    assert selected == ["other"]


def test_manual_models_survive_the_family_filter(tmp_path) -> None:
    """Manual additions land last precisely so the dedup cannot strip them."""
    dfs = {
        "LK-en": selection_frame(
            [
                ("claude-3-sonnet", 50.0, 5.0, None, False),
                ("claude-4-sonnet", 90.0, 5.0, None, False),
            ]
        )
    }
    csv_path = models_csv(
        tmp_path,
        [
            ("claude-3-sonnet", "Anthropic", "01/2025"),
            ("claude-4-sonnet", "Anthropic", "01/2026"),
        ],
    )

    selected, by_domain = create_default_models_list(
        dfs,
        models_csv_path=csv_path,
        filter_by_provider_family=True,
        manual_default_models=["claude-3-sonnet"],
    )

    assert "claude-3-sonnet" in selected
    assert "claude-3-sonnet" in by_domain["LK-en"]


def test_manual_model_without_data_is_warned_about_not_published(
    tmp_path, capsys
) -> None:
    dfs = {"LK-en": selection_frame([("a", 90.0, 1.0, None, False)])}

    selected, by_domain = create_default_models_list(
        dfs,
        models_csv_path=models_csv(tmp_path, []),
        manual_default_models=["ghost"],
    )

    assert "ghost" in selected
    assert "ghost" not in by_domain["LK-en"]
    assert "has no benchmark data" in capsys.readouterr().out


def test_size_frontier_is_computed_for_open_models_only(tmp_path) -> None:
    """Closed-weight models have no deployable size, so they must never reach the
    size frontier even when models.csv lists one."""
    dfs = {
        "LK-en": selection_frame(
            [
                ("open-small", 40.0, 1.0, 1.0, True),
                ("open-big", 70.0, 10.0, 100.0, True),
                ("closed", 99.0, 50.0, 5.0, False),
            ]
        )
    }

    _, by_domain = create_default_models_list(
        dfs, models_csv_path=models_csv(tmp_path, []), add_missing_models=True
    )

    assert "open-small" in by_domain["LK-en"]


def test_per_benchmark_lists_only_hold_models_scored_in_that_benchmark(
    tmp_path,
) -> None:
    dfs = {
        "LK-en": selection_frame([("a", 90.0, 1.0, None, False)]),
        "LK-sw": selection_frame([("b", 80.0, 2.0, None, False)]),
    }

    _, by_domain = create_default_models_list(
        dfs, models_csv_path=models_csv(tmp_path, [])
    )

    assert by_domain["LK-en"] == ["a"]
    assert by_domain["LK-sw"] == ["b"]
