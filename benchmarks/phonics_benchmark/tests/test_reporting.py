"""Exercise normal-run reporting and saved-table plotting through the package API."""

import json
from hashlib import sha256

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from phonics_benchmark import bootstrap, reporting
from phonics_benchmark.hardware import minimum_scorable_items, report_hardware
from phonics_benchmark.plots import METRIC_SPECS, eligible_models, plot_leaderboard


@pytest.fixture
def ordinary_run(tmp_path):
    run = tmp_path / "example"
    (run / "responses").mkdir(parents=True)
    items = [
        {
            "benchmark_item_id": key,
            "content": "cat dog",
            "student_knowledge": "cat",
            "nondecodable_words": ["dog"],
        }
        for key in ("a", "b", "c")
    ]
    item_path = run / "items.jsonl"
    item_path.write_text("".join(json.dumps(row) + "\n" for row in items))
    config = {
        "run_id": "example",
        "language": "en",
        "models": ["complete", "partial", "failed", "absent"],
        "items_path": "items.jsonl",
        "items_sha256": sha256(item_path.read_bytes()).hexdigest(),
    }
    (run / "config.json").write_text(json.dumps({"resolved_config": config}))
    rows = [
        ("complete", "a", "success", '["dog"]'),
        ("complete", "b", "success", '["cat"]'),
        ("partial", "a", "success", '["dog"]'),
        ("failed", "a", "error", ""),
        ("failed", "b", "success", "I cannot answer this question."),
    ]
    responses = [
        dict(
            result_id=str(i),
            model=model,
            benchmark_item_id=key,
            call_status=status,
            output=output,
        )
        for i, (model, key, status, output) in enumerate(rows)
    ]
    (run / "responses/responses.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in responses)
    )
    return run


def test_reporting_includes_missing_items_and_all_configured_models(
    ordinary_run, tmp_path
):
    destination = tmp_path / "report"
    reporting.report_on_run(
        ordinary_run.name,
        ordinary_run.parent,
        output_dir=destination,
        n_bootstrap=40,
        min_coverage=0.5,
        inspect_sample_size=1,
        make_plots=False,
    )
    coverage = reporting.read_csv(destination / "coverage_by_model.csv").set_index(
        "model"
    )
    assert coverage.loc[
        "complete", ["scorable_items", "missing_responses"]
    ].tolist() == [2, 1]
    assert coverage.loc[
        "partial", ["scorable_items", "missing_responses"]
    ].tolist() == [1, 2]
    assert coverage.loc[
        "failed", ["call_failed", "parse_failed", "missing_responses"]
    ].tolist() == [1, 1, 1]
    assert coverage.loc[
        "absent", ["response_rows", "scorable_items", "missing_responses"]
    ].tolist() == [0, 0, 3]
    summary = reporting.read_csv(
        destination / "model_summary_all_scored.csv"
    ).set_index("model")
    assert summary.loc["complete", "pooled_f2"] == 0.5
    assert summary.loc["complete", "balanced_accuracy"] == 0.5
    assert summary.loc["complete", "adjusted_balanced_accuracy"] == 0
    assert (
        reporting.read_csv(destination / "bootstrap_ci.csv")
        .n_items_resampled.eq(3)
        .all()
    )
    assert reporting.read_csv(
        destination / "model_summary_eligible.csv"
    ).model.tolist() == ["complete"]
    assert (destination / "sampled_items_for_inspection.csv").is_file()
    assert not (ordinary_run / "artifact_manifest.json").exists()


def test_replot_reads_only_saved_tables(ordinary_run, tmp_path, monkeypatch):
    report = reporting.report_on_run(
        ordinary_run.name, ordinary_run.parent, n_bootstrap=20, make_plots=False
    )
    before = {p.name: p.read_bytes() for p in report.iterdir()}

    def forbidden(*args, **kwargs):
        pytest.fail("Replotting must not rescore responses or bootstrap")

    for name in (
        "load_run_responses",
        "add_per_item_confusion_matrix",
        "bootstrap_metric_cis",
        "summarize_balanced_accuracy",
    ):
        monkeypatch.setattr(reporting, name, forbidden)
    paths = reporting.plot_report(report, output_dir=tmp_path / "plots")
    assert len(paths) == 8
    assert all(path.stat().st_size for path in paths)
    assert before == {p.name: p.read_bytes() for p in report.iterdir()}


def test_every_metric_uses_same_full_target_item_draws(ordinary_run, monkeypatch):
    recorded = []
    original = bootstrap.item_bootstrap_draws

    def record(n_items, n_bootstrap, seed):
        draws = list(original(n_items, n_bootstrap, seed))
        recorded.append(np.array(draws))
        yield from draws

    monkeypatch.setattr(bootstrap, "item_bootstrap_draws", record)
    reporting.report_on_run(
        ordinary_run.name, ordinary_run.parent, n_bootstrap=25, seed=7, make_plots=False
    )
    assert len(recorded) == 2
    assert recorded[0].shape == (25, 3)
    np.testing.assert_array_equal(recorded[0], recorded[1])
    assert 2 in recorded[0]  # Target item c has no response from any model.


def test_no_bootstrap_removes_stale_interval_file(ordinary_run):
    report = reporting.report_on_run(
        ordinary_run.name, ordinary_run.parent, n_bootstrap=10, make_plots=False
    )
    reporting.report_on_run(
        ordinary_run.name, ordinary_run.parent, n_bootstrap=0, make_plots=False
    )
    assert not (report / "bootstrap_ci.csv").exists()
    summary = reporting.read_csv(report / "model_summary_all_scored.csv")
    assert summary.filter(regex="_ci_(low|high)$").isna().all().all()
    # Point-only plots must also work with explicitly absent intervals.
    assert len(reporting.plot_report(report)) == 8


@pytest.mark.parametrize(
    "total,counts,expected",
    [(991, [891, 892], [False, True]), (1000, [900, 901], [False, True])],
)
def test_eligibility_uses_strict_target_coverage(total, counts, expected):
    frame = pd.DataFrame(
        {
            "scorable_items": counts,
            "score": [0.9, 0.8],
            "ci_low": [0.7, 0.6],
            "ci_high": [0.95, 0.9],
        }
    )
    required = minimum_scorable_items(total, 700, 0.9)
    assert frame.scorable_items.ge(required).tolist() == expected
    pd.testing.assert_frame_equal(
        eligible_models(frame, required), frame.iloc[[1]].reset_index(drop=True)
    )


@pytest.mark.parametrize("metric", list(METRIC_SPECS))
def test_metric_labels_and_units_come_from_renderer_spec(metric):
    adjusted = metric == "adjusted_balanced_accuracy"
    frame = pd.DataFrame(
        {
            "display_name": ["A"],
            "score": [-40 if adjusted else 0.4],
            "ci_low": [-30 if adjusted else 0.3],
            "ci_high": [-20 if adjusted else 0.6],
            "provider": ["test"],
            "color": ["#376b94"],
            "n_items": [10],
            "expected_items": [10],
            "ci_level": [0.95],
            "n_bootstrap": [1000],
        }
    )
    fig = plot_leaderboard(frame, {"test": "#376b94"}, "english", metric=metric)
    try:
        ax = fig.axes[0]
        assert METRIC_SPECS[metric].label in ax.get_title(loc="left")
        assert ax.get_xlabel() == METRIC_SPECS[metric].axis_label
        assert ax.patches[0].get_width() == (-0.4 if adjusted else 0.4)
        if adjusted:
            assert ax.get_xlim()[0] < -0.4
    finally:
        plt.close(fig)


def test_hardware_report_supports_one_language_and_no_eligible_models(
    ordinary_run, tmp_path
):
    report = reporting.report_on_run(
        ordinary_run.name,
        ordinary_run.parent,
        n_bootstrap=10,
        min_items=100,
        make_plots=False,
    )
    metadata = tmp_path / "models.csv"
    pd.DataFrame(
        {
            "model_id": ["complete"],
            "display_name": ["Complete"],
            "provider": ["test"],
            "open": [False],
            "size": [1e9],
        }
    ).to_csv(metadata, index=False)
    destination = tmp_path / "hardware"
    report_hardware([report], metadata, destination)
    audit = reporting.read_csv(
        destination / "pooled_f2/hardware_tiers_english_audit.csv"
    )
    assert len(audit) == 4
    assert not audit.included.any()


def test_display_metadata_preserves_known_aliases_and_defaults_for_new_models():
    from pathlib import Path

    from phonics_benchmark.hardware import assign_hardware, load_metadata

    metadata = Path(__file__).parents[1] / "scripts/f2_plot_metadata"
    specs = load_metadata(metadata / "models.csv")
    assert specs["qwen3-8b-alibaba"]["size"] == 8e9
    assert (
        assign_hardware(specs["qwen3-8b-alibaba"])["hardware_tier"] == "Mobile + Tablet"
    )
    display, colors = reporting.display_metadata(
        ["qwen3-8b-alibaba", "new-model"], metadata
    )
    display = display.set_index("model")
    assert (
        display.loc["qwen3-8b-alibaba", "display_name"]
        == specs["qwen3-8b-alibaba"]["display_name"]
    )
    assert display.loc["new-model", "display_name"] == "new-model"
    assert display.loc["new-model", "color"] == colors["Unspecified"]
