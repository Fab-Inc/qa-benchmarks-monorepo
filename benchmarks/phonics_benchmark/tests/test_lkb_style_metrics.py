"""Metadata alignment, Pareto selection and preservation of saved bootstrap bounds."""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from matplotlib.container import ErrorbarContainer


@pytest.fixture
def plotting(monkeypatch):
    scripts = Path(__file__).parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location(
        "plot_lkb_style_metrics", scripts / "plot_lkb_style_metrics.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def inputs(tmp_path, plotting):
    # Shuffled metadata must join by ID, not inherit summary row positions.
    pd.DataFrame(
        {
            "model_id": ["large", "small", "closed"],
            "display_name": ["Large model", "Small model", "Closed model"],
            "provider": ["Second", "First", "Second"],
            "size": [35e9, 4e9, None],
            "input_cost": [4, 1, 2],
            "output_cost": [8, 5, 6],
        }
    ).to_csv(tmp_path / "models.csv", index=False)
    pd.DataFrame({"provider": ["Second", "First"], "color": ["red", "blue"]}).to_csv(
        tmp_path / "providers.csv", index=False
    )
    pd.DataFrame(
        {
            "model_id": ["small", "large", "closed"],
            "family": ["Family", "Family", "Singleton"],
        }
    ).to_csv(tmp_path / "families.csv", index=False)
    rows = pd.DataFrame(
        {
            "model": ["small", "large", "closed", "partial"],
            "finished": [True, True, True, False],
            "run_id": ["run"] * 4,
            "n_scored": [99, 100, 100, 10],
            "n_call_failed": [1, 0, 0, 0],
            "n_parse_failed": [0] * 4,
        }
    )
    for i, metric in enumerate(plotting.METRICS):
        rows[metric] = np.array([0.4, 0.8, 0.9, 1]) - i * 0.05
        rows[f"{metric}_ci_low"] = rows[metric] - 0.1
        rows[f"{metric}_ci_high"] = (rows[metric] + 0.05).clip(upper=1)
    rows.to_csv(tmp_path / "summary_by_model.csv", index=False)
    (tmp_path / "analysis.json").write_text(
        json.dumps({"bootstrap": {"ci_level": 0.95, "n_bootstrap": 10000, "seed": 0}})
    )
    return tmp_path


def test_metadata_join_retains_errors_and_uses_total_size(inputs, plotting):
    metadata, colors = plotting.load_metadata(
        inputs / "models.csv", inputs / "providers.csv", inputs / "families.csv"
    )
    table = plotting.metric_table(
        inputs / "summary_by_model.csv", "pooled_f2", metadata, colors
    )
    rows = table.set_index("model")
    assert set(rows.index) == {"small", "large", "closed"}
    assert rows.loc["small", "n_call_failed"] == 1
    assert rows.loc["small", "display_name"] == "Small model"
    assert rows.loc["small", "blended_price"] == 2
    assert rows.loc["small", "hardware_tier"] == "Tablet (4–12B)"
    assert rows.loc["large", "hardware_tier"] == "Cloud (≥35B)"
    assert rows.loc["closed", "hardware_tier"] == "Unknown size"
    assert list(colors) == ["Second", "First"]
    fig, _ = plotting.plot_bars(table, "pooled_f2", colors)
    try:
        assert [t.get_text() for t in fig.axes[0].get_legend().get_texts()] == [
            "Second",
            "First",
        ]
    finally:
        plotting.plt.close(fig)


def test_frontier_drops_dominated_models_and_handles_equal_cost(plotting):
    table = pd.DataFrame(
        {
            "model": ["cheap", "worse", "better", "tied", "best"],
            "cost": [1, 2, 2, 3, 4],
            "score": [0.5, 0.4, 0.7, 0.7, 0.9],
        }
    )
    frontier = plotting.compute_frontier(table, "cost")
    assert frontier["model"].tolist() == ["cheap", "better", "best"]


@pytest.mark.parametrize("horizontal", [True, False])
def test_ci_endpoints_are_not_clipped_to_estimate(plotting, horizontal):
    table = pd.DataFrame(
        {"score": [0.8, 0.5], "ci_low": [0.85, np.nan], "ci_high": [0.9, np.nan]}
    )
    fig, ax = plotting.plt.subplots()
    try:
        plotting.draw_intervals(ax, table, [2, 3], horizontal=horizontal)
        bars = next(c for c in ax.containers if isinstance(c, ErrorbarContainer))
        segments = bars.lines[2][0].get_segments()
        assert len(segments) == 1
        expected = [[0.85, 2], [0.9, 2]] if horizontal else [[2, 0.85], [2, 0.9]]
        np.testing.assert_allclose(segments[0], expected)
    finally:
        plotting.plt.close(fig)


def test_tier_bootstrap_resamples_model_scores_not_items(plotting):
    table = pd.DataFrame(
        {
            "hardware_tier": ["one", "one", "two"],
            "score": [0, 1, 0.7],
            "metric": ["pooled_f1"] * 3,
            "n_scored": [1, 1000, 5],
        }
    )
    means = plotting.hardware_means(table).set_index("hardware_tier")
    assert means.loc["one", "score"] == 0.5
    np.testing.assert_allclose(
        means.loc["one", ["ci_low", "ci_high"]].astype(float), [0, 1]
    )
    np.testing.assert_allclose(
        means.loc["two", ["ci_low", "ci_high"]].astype(float), [0.7, 0.7]
    )


def test_full_export_preserves_metrics_and_never_overwrites(
    inputs, plotting, monkeypatch
):
    out = inputs / "figures"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "plot_lkb_style_metrics.py",
            "--summary-csv",
            str(inputs / "summary_by_model.csv"),
            "--models-csv",
            str(inputs / "models.csv"),
            "--providers-csv",
            str(inputs / "providers.csv"),
            "--families-csv",
            str(inputs / "families.csv"),
            "--output-dir",
            str(out),
            "--formats",
            "png",
            "svg",
            "--dpi",
            "30",
        ],
    )
    plotting.main()
    original = pd.read_csv(inputs / "summary_by_model.csv").set_index("model")
    for metric in plotting.METRICS:
        csv = pd.read_csv(out / f"{metric}_leaderboard.csv").set_index("model")
        assert set(csv.index) == {"small", "large", "closed"}
        np.testing.assert_allclose(csv["score"], original.loc[csv.index, metric])
        np.testing.assert_allclose(
            csv["ci_low"], original.loc[csv.index, f"{metric}_ci_low"]
        )
        np.testing.assert_allclose(
            csv["ci_high"], original.loc[csv.index, f"{metric}_ci_high"]
        )
        size = pd.read_csv(out / f"{metric}_vs_size.csv")
        assert set(size["model"]) == {"small", "large"}
        assert set(pd.read_csv(out / f"{metric}_families.csv")["model"]) == {
            "small",
            "large",
            "closed",
        }
    assert len(list(out.glob("*.png"))) == 21
    assert len(list(out.glob("*.svg"))) == 21
    assert "item_averaged_f1" not in " ".join(p.name for p in out.iterdir())
    assert (
        "95% bootstrap intervals (10,000 resamples)" in (out / "README.md").read_text()
    )
    before = {p.name: p.read_bytes() for p in out.iterdir() if p.is_file()}
    with pytest.raises(FileExistsError):
        plotting.main()
    assert before == {p.name: p.read_bytes() for p in out.iterdir() if p.is_file()}
