"""Check metric selection and that plotted whiskers preserve bootstrap bounds."""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from matplotlib.container import ErrorbarContainer


@pytest.fixture
def plotting():
    path = Path(__file__).parents[1] / "scripts" / "plot_model_metrics.py"
    spec = importlib.util.spec_from_file_location("plot_model_metrics", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_leaderboard_selects_only_requested_metric_and_orders_bars(tmp_path, plotting):
    path = tmp_path / "summary.csv"
    pd.DataFrame(
        {
            "model": ["lower", "higher", "unfinished"],
            "finished": ["true", "true", "false"],
            "perfect_match_accuracy": [0.3, 0.8, 1.0],
            "pooled_f1": [float("nan"), 0.9, 1.0],
        }
    ).to_csv(path, index=False)
    summary = plotting.load_summary(path, "perfect_match_accuracy")
    assert summary["model"].tolist() == ["higher", "lower"]
    fig = plotting.plot_metrics(
        summary,
        metric="perfect_match_accuracy",
        theme="light",
        annotate=True,
        title="Leaderboard",
        subtitle="No intervals",
    )
    try:
        ax = fig.axes[0]
        assert [bar.get_width() for bar in ax.patches] == [0.3, 0.8]
        assert [text.get_text() for text in ax.get_yticklabels()] == ["lower", "higher"]
        assert ax.get_xlim() == (0, 1)
        assert ax.get_xlabel() == "Perfect-match accuracy"
        assert not any(isinstance(c, ErrorbarContainer) for c in ax.containers)
    finally:
        plotting.plt.close(fig)


@pytest.mark.parametrize("theme", ["light", "dark"])
@pytest.mark.parametrize("metric", ["pooled_f1", "item_averaged_f1", "pooled_f2"])
def test_whiskers_show_exact_bounds_even_when_interval_excludes_estimate(
    plotting, theme, metric
):
    summary = pd.DataFrame(
        {
            "model": ["model"],
            metric: [0.8],
            f"{metric}_ci_low": [0.85],
            f"{metric}_ci_high": [0.9],
        }
    )
    fig = plotting.plot_metrics(
        summary,
        metric=metric,
        theme=theme,
        annotate=True,
        title="Leaderboard",
        subtitle="95% bootstrap intervals",
    )
    try:
        ax = fig.axes[0]
        whiskers = next(c for c in ax.containers if isinstance(c, ErrorbarContainer))
        segment = whiskers.lines[2][0].get_segments()[0]
        np.testing.assert_allclose(segment, [[0.85, 0], [0.9, 0]])
        assert ax.patches[0].get_width() == 0.8
        fig.canvas.draw()
    finally:
        plotting.plt.close(fig)


@pytest.mark.parametrize("low,high", [(0.9, 0.8), (-0.1, 0.9), (0.1, 1.1), (None, 0.9)])
def test_rejects_invalid_interval_bounds(tmp_path, plotting, low, high):
    path = tmp_path / "summary.csv"
    pd.DataFrame(
        {
            "model": ["a"],
            "finished": [True],
            "pooled_f1": [0.8],
            "pooled_f1_ci_low": [low],
            "pooled_f1_ci_high": [high],
        }
    ).to_csv(path, index=False)
    with pytest.raises(ValueError, match="Bootstrap intervals"):
        plotting.load_summary(path, "pooled_f1")


def test_older_summary_without_intervals_and_confidence_metadata(tmp_path, plotting):
    path = tmp_path / "summary.csv"
    pd.DataFrame({"model": ["a"], "finished": [True], "pooled_f1": [0.8]}).to_csv(
        path, index=False
    )
    summary = plotting.load_summary(path, "pooled_f1")
    assert (
        plotting.interval_note(summary, "pooled_f1", path)
        == "Bootstrap intervals unavailable"
    )
    summary["pooled_f1_ci_low"] = 0.7
    summary["pooled_f1_ci_high"] = 0.9
    assert (
        plotting.interval_note(summary, "pooled_f1", path)
        == "Whiskers: Bootstrap intervals"
    )
    (tmp_path / "analysis.json").write_text(
        json.dumps(
            {
                "bootstrap": {"ci_level": 0.9, "n_bootstrap": 500},
            }
        )
    )
    assert (
        plotting.interval_note(summary, "pooled_f1", path)
        == "Whiskers: 90% bootstrap intervals (500 resamples)"
    )


def test_cli_preserves_existing_image(tmp_path, plotting, monkeypatch):
    path = tmp_path / "summary.csv"
    pd.DataFrame({"model": ["a"], "finished": [True], "pooled_f1": [0.8]}).to_csv(
        path, index=False
    )
    output = tmp_path / "existing.png"
    output.write_bytes(b"keep this image")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "plot_model_metrics.py",
            "--summary-csv",
            str(path),
            "--output",
            str(output),
        ],
    )
    with pytest.raises(SystemExit) as error:
        plotting.main()
    assert error.value.code == 2
    assert output.read_bytes() == b"keep this image"


def test_repeated_models_are_not_silently_combined(tmp_path, plotting):
    path = tmp_path / "summary.csv"
    pd.DataFrame(
        {
            "model": ["a", "a"],
            "run_id": ["r1", "r2"],
            "finished": [True, True],
            "pooled_f1": [0.8, 0.9],
        }
    ).to_csv(path, index=False)
    with pytest.raises(ValueError, match="one result per model"):
        plotting.load_summary(path, "pooled_f1")


def test_item_panels_preserve_coordinates_and_count_undefined_items(tmp_path, plotting):
    path = tmp_path / "summary.csv"
    pd.DataFrame(
        {
            "run_id": ["r", "r"],
            "model": ["undefined", "scored"],
            "finished": [True, True],
            "item_averaged_f1": [float("nan"), 0.4],
            "n_scored": [1, 3],
        }
    ).to_csv(path, index=False)
    summary = plotting.load_summary(path, "item_averaged_f1", include_undefined=True)
    assert summary["model"].tolist() == ["scored", "undefined"]
    # Leaderboards omit undefined model metrics, but panels retain those models.
    assert plotting.load_summary(path, "item_averaged_f1")["model"].tolist() == [
        "scored"
    ]
    scored_path = tmp_path / "scored.csv"
    pd.DataFrame(
        {
            "run_id": ["r"] * 4,
            "model": ["scored"] * 3 + ["undefined"],
            "benchmark_item_id": ["i1", "i2", "i3", "i3"],
            "item_f1": [0.8, 0, float("nan"), float("nan")],
            "nondecodable_ratio": [0.2, 0, 0, 0],
        }
    ).to_csv(scored_path, index=False)
    scored = plotting.load_item_scores(scored_path, summary)
    fig = plotting.plot_item_f1_panels(scored, summary, ncols=2)
    try:
        ax, empty = fig.axes
        np.testing.assert_allclose(
            ax.collections[0].get_offsets(), [[0.2, 0.8], [0, 0]]
        )
        assert "n=2 · undefined F1=1" in ax.get_title(loc="left")
        assert empty.collections[0].get_offsets().shape == (0, 2)
        assert "n=0 · undefined F1=1" in empty.get_title(loc="left")
        assert ax.get_xlim() == empty.get_xlim()
        assert ax.get_ylim() == empty.get_ylim()
        assert ax.get_xlim()[0] < 0 < ax.get_xlim()[1]
        assert ax.get_ylim()[0] < 0 < 1 < ax.get_ylim()[1]
    finally:
        plotting.plt.close(fig)
    with pytest.raises(ValueError, match="coverage differs"):
        plotting.load_item_scores(scored_path, summary.assign(n_scored=99))


@pytest.mark.parametrize(
    "plot_args",
    [
        ["--metric", "item_averaged_f1"],
        ["--metric", "pooled_f2"],
        ["--plot", "item-panels"],
    ],
)
def test_new_plots_cli_writes_png_once(tmp_path, plotting, monkeypatch, plot_args):
    path = tmp_path / "summary_by_model.csv"
    pd.DataFrame(
        {
            "run_id": ["r"],
            "model": ["a"],
            "finished": [True],
            "item_averaged_f1": [0.8],
            "n_scored": [1],
            "item_averaged_f1_ci_low": [0.7],
            "item_averaged_f1_ci_high": [0.9],
            "pooled_f2": [0.85],
            "pooled_f2_ci_low": [0.8],
            "pooled_f2_ci_high": [0.9],
        }
    ).to_csv(path, index=False)
    pd.DataFrame(
        {
            "run_id": ["r"],
            "model": ["a"],
            "benchmark_item_id": ["i1"],
            "item_f1": [0.8],
            "nondecodable_ratio": [0.2],
        }
    ).to_csv(tmp_path / "results_with_scoring.csv", index=False)
    output = tmp_path / "plot.png"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "plot_model_metrics.py",
            "--summary-csv",
            str(path),
            "--output",
            str(output),
            *plot_args,
        ],
    )
    plotting.main()
    original = output.read_bytes()
    assert original.startswith(b"\x89PNG")
    with pytest.raises(SystemExit):
        plotting.main()
    assert output.read_bytes() == original
