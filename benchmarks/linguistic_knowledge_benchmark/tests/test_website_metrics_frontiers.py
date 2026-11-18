"""Tests for scripts/website_metrics_frontiers.py, loaded by path since scripts/ is
not a package. Delete together with that script."""

import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest
from fab_benchmark_publisher.metrics import USAGE_COLUMNS

SCRIPT = (
    Path(__file__).resolve().parents[1] / "scripts" / "website_metrics_frontiers.py"
)
spec = importlib.util.spec_from_file_location("website_metrics_frontiers", SCRIPT)
script = importlib.util.module_from_spec(spec)
spec.loader.exec_module(script)

STEM = "LK-en_accuracy_test-4-items"

# 4 questions; model-b has one badly formatted answer (empty _correct)
DETAILED = {
    "question_id": ["1", "2", "3", "4"],
    "domain": ["D1", "D1", "D2", "D2"],
    "model-a_output": ["A", "B", "C", "D"],
    "model-a_correct": ["True", "True", "True", "False"],
    "model-b_output": ["A", "???", "C", "D"],
    "model-b_correct": ["True", "", "False", "False"],
}
SUMMARY = "model,accuracy\nmodel-a,75.0\nmodel-b,33.33\n"


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    results = tmp_path / "results"
    results.mkdir()
    pair = {
        "benchmark_slug": "LK-en",
        "display_name": "LK English",
        "language": "english",
        "experiment": "test-4-items",
        "fallback": False,
    }
    monkeypatch.setattr(script, "PAIRS", [pair])
    detailed, summary = script.source_paths(pair, results)
    pd.DataFrame(DETAILED).to_csv(detailed, index=False)
    summary.write_text(SUMMARY)

    models = tmp_path / "models.csv"
    models.write_text(
        "model_id,provider,open,size,input_cost,output_cost,release_date\n"
        "model-a,P,False,,2.0,2.0,01/2026\n"
        "model-b,P,True,7,1.0,1.0,01/2026\n"
    )
    return tmp_path, results, models, summary


def run(tmp_path, results, models):
    script.main(
        [
            "--results-dir",
            str(results),
            "--models-csv",
            str(models),
            "--out-root",
            str(tmp_path / "out"),
            "--n-bootstrap",
            "200",
        ]
    )
    return tmp_path / "out" / "linguistic-knowledge" / "LK-en"


def test_writes_the_three_benchmark_files(inputs) -> None:
    outdir = run(*inputs[:3])

    assert sorted(p.name for p in outdir.iterdir()) == [
        f"{STEM}.csv",
        f"{STEM}_frontiers.json",
        "benchmark_metadata.json",
    ]
    meta = json.loads((outdir / "benchmark_metadata.json").read_text())
    assert meta["benchmark_slug"] == "LK-en"
    assert [m["metric_slug"] for m in meta["metrics"]] == ["accuracy"]
    assert meta["metrics"][0]["unit"] == "%"
    assert meta["metrics"][0]["is_bad_format_scored"] is False


def test_metrics_follow_the_schema(inputs) -> None:
    metrics = pd.read_csv(
        run(*inputs[:3]) / f"{STEM}.csv", keep_default_na=False, na_values=[""]
    )
    by_model = metrics.set_index("model_slug")

    assert set(metrics["benchmark_slug"]) == {"LK-en"}
    assert set(metrics["metric_slug"]) == {"accuracy"}
    assert set(metrics["experiment_slug"]) == {"test-4-items"}
    assert metrics["benchmark_subset"].isna().all()  # overall only
    assert by_model.loc["model-a", "metric_value"] == 75.0
    assert by_model.loc["model-b", "metric_value"] == pytest.approx(33.3333)
    assert by_model.loc["model-b", "n_items_scored"] == 3
    assert by_model.loc["model-b", "n_items_bad_format"] == 1
    assert set(metrics["n_repeats"]) == {1}
    # not filled yet
    assert metrics[list(USAGE_COLUMNS)].isna().all().all()


def test_frontiers_price_and_size_only(inputs) -> None:
    frontiers = json.loads((run(*inputs[:3]) / f"{STEM}_frontiers.json").read_text())

    by_axis = {
        f["axis"]: [p["model_slug"] for p in f["points"]]
        for f in frontiers["frontiers"]
    }
    assert by_axis == {"price": ["model-b", "model-a"], "size": ["model-b"]}


def test_disagreement_with_summary_accuracy_stops_the_run(inputs) -> None:
    tmp_path, results, models, summary = inputs
    summary.write_text(SUMMARY.replace("75.0", "80.0"))

    with pytest.raises(SystemExit, match="differs from the summary CSV"):
        run(tmp_path, results, models)
    assert not (tmp_path / "out").exists()


def test_missing_source_file_is_reported(inputs) -> None:
    tmp_path, results, models, summary = inputs
    summary.unlink()

    with pytest.raises(SystemExit, match="not found"):
        run(tmp_path, results, models)
