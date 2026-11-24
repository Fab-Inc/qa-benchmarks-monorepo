"""Tests for scripts/write_result_files.py, loaded by path since scripts/ is
not a package. Delete together with that script."""

import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest
from fab_benchmark_publisher.metrics import PLACEHOLDER_COLUMNS
from linguistic_knowledge_benchmark import shared_utils

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "write_result_files.py"
spec = importlib.util.spec_from_file_location("write_result_files", SCRIPT)
script = importlib.util.module_from_spec(spec)
spec.loader.exec_module(script)

BENCHMARK = {
    "benchmark_slug": "LK-en",
    "language": "english",
    "experiment": "test-4-items",
    "judged_pass": script.JUDGED_FALLBACK_DIR_NAME,
}


def judged_file_name(model: str) -> str:
    """The judged CSV name the script expects for `model`."""
    return f"{script.JUDGED_FILE_PREFIX}{model}.csv"


def default_judged_dir() -> Path:
    """BENCHMARK's judged folder, under the cache the fixture points shared_utils at."""
    return (
        shared_utils.experiment_dir(BENCHMARK["language"], BENCHMARK["experiment"])
        / BENCHMARK["judged_pass"]
    )


# (is_correct, is_correct_fallback) per model, 4 questions each. None = no verdict.
#   model-a: the fallback judge flips question 3 to correct -> 3/4 = 75.0
#   model-b: question 2 unscored -> 1/3 = 33.33
#   model-old: not in the model list (EVAL_MODELS) -> left out
VERDICTS_BY_MODEL = {
    "model-a": ([True, True, False, False], [None, None, True, None]),
    "model-b": ([True, None, False, False], [None, None, None, None]),
    "model-old": ([True, True, True, True], [None, None, None, None]),
}
# model-new is listed but was never judged in this experiment.
LISTED_MODELS = {"model-a": "A", "model-b": "B", "model-new": "New"}

ACCURACY = {
    "metric_slug": "accuracy",
    "display_name": "Accuracy",
    "display_description": "Percentage of questions answered correctly.",
    "unit": "%",
    "is_higher_better": True,
    "is_bad_format_scored": False,
}


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    """A temporary cache with VERDICTS_BY_MODEL judged, plus a models CSV and a
    leaderboard config; the script is pointed at them. Returns (tmp_path, models
    CSV, config)."""
    monkeypatch.setattr(shared_utils, "CACHE_BASE_DIR", tmp_path / "cache")
    monkeypatch.setattr(script, "EVAL_MODELS", LISTED_MODELS)
    monkeypatch.setattr(script, "BENCHMARKS", [BENCHMARK])
    judged_dir = default_judged_dir()
    judged_dir.mkdir(parents=True)
    for model, (is_correct, fallback) in VERDICTS_BY_MODEL.items():
        pd.DataFrame(
            {
                "question_id": [1, 2, 3, 4],
                "answer_type": ["multiple_choice"] * 2 + ["short_form"] * 2,
                "is_correct": is_correct,
                "is_correct_fallback": fallback,
            }
        ).to_csv(judged_dir / judged_file_name(model), index=False)

    models = tmp_path / "models.csv"
    models.write_text(
        "model_id,provider,open,size,input_price_usd_per_mtok,"
        "output_price_usd_per_mtok,release_date\n"
        "model-a,P,False,,2.0,2.0,01/2026\n"
        "model-b,P,True,7,1.0,1.0,01/2026\n"
    )
    config = tmp_path / "leaderboard_config.json"
    config.write_text(
        json.dumps(
            {
                "cluster_name": "foundational-literacy",
                "benchmarks": [
                    {
                        "benchmark_domain": "LK-en",
                        "display_name": "LK English",
                        "metrics": [ACCURACY],
                    }
                ],
            }
        )
    )
    return tmp_path, models, config


def run(tmp_path, models, config, *extra):
    """Run the script on the fixture inputs (plus `extra` arguments) and return the
    LK-en output folder."""
    script.main(
        [
            "--models-csv",
            str(models),
            "--config",
            str(config),
            "--out-root",
            str(tmp_path / "out"),
            *extra,
        ]
    )
    return tmp_path / "out" / "linguistic-knowledge" / "LK-en"


def read_metrics(outdir: Path) -> pd.DataFrame:
    """The LK-en metrics CSV written to `outdir`, empty cells read as NaN."""
    return pd.read_csv(
        outdir / "LK-en_accuracy.csv", keep_default_na=False, na_values=[""]
    )


def test_writes_the_three_files(inputs) -> None:
    """Metrics, metadata and frontier files are written; the metadata is the
    config's metric definition plus the benchmark slug."""
    outdir = run(*inputs)

    assert sorted(p.name for p in outdir.iterdir()) == [
        "LK-en_accuracy.csv",
        "LK-en_accuracy_frontiers.json",
        "LK-en_accuracy_metadata.json",
    ]
    meta = json.loads((outdir / "LK-en_accuracy_metadata.json").read_text())
    assert meta == {"benchmark_slug": "LK-en", **ACCURACY}


def test_metrics_come_from_the_judged_files(inputs, capsys) -> None:
    """Accuracy is computed from the judged files for listed models only, and the
    mismatches with the model list are reported."""
    metrics = read_metrics(run(*inputs))

    assert set(metrics["benchmark_slug"]) == {"LK-en"}
    assert set(metrics["metric_slug"]) == {"accuracy"}
    assert set(metrics["experiment_slug"]) == {"test-4-items"}
    assert metrics["benchmark_subset"].isna().all()  # overall only
    assert dict(zip(metrics["model_slug"], metrics["metric_value"])) == {
        "model-a": 75.0,
        "model-b": 33.33,
    }
    assert metrics[list(PLACEHOLDER_COLUMNS)].isna().all().all()

    out = capsys.readouterr().out
    assert "not judged here: model-new" in out
    assert f"not in {script.MODELS_LIST_YAML.name} (left out): model-old" in out


def test_frontier_is_built_from_the_metrics_file(inputs) -> None:
    """The frontiers file has the expected price and size frontiers and carries the
    metrics file's experiment_slug."""
    outdir = run(*inputs)
    frontier = json.loads((outdir / "LK-en_accuracy_frontiers.json").read_text())

    by_axis = {
        f["axis"]: [p["model_slug"] for p in f["points"]] for f in frontier["frontiers"]
    }
    assert by_axis == {"price": ["model-b", "model-a"], "size": ["model-b"]}
    assert frontier["experiment_slug"] == "test-4-items"


def test_judged_dir_override_sets_the_experiment(inputs) -> None:
    """--judged-dir reads another folder, and experiment_slug follows it."""
    tmp_path, models, config = inputs
    other = (
        shared_utils.experiment_dir("english", "rerun-2")
        / script.JUDGED_FALLBACK_DIR_NAME
    )
    other.mkdir(parents=True)
    src = default_judged_dir() / judged_file_name("model-a")
    (other / src.name).write_bytes(src.read_bytes())

    metrics = read_metrics(
        run(tmp_path, models, config, "--judged-dir", "LK-en", str(other))
    )

    assert set(metrics["experiment_slug"]) == {"rerun-2"}
    assert list(metrics["model_slug"]) == ["model-a"]


@pytest.mark.parametrize(
    "override", [["XX-zz", "some/folder"], ["LK-en"]], ids=["unknown", "no folder"]
)
def test_bad_judged_dir_override_is_rejected(inputs, override, capsys) -> None:
    """An unknown benchmark or a missing folder in --judged-dir is a usage error."""
    with pytest.raises(SystemExit):
        run(*inputs, "--judged-dir", *override)
    assert "--judged-dir" in capsys.readouterr().err


def test_missing_judged_folder_stops_the_run(inputs) -> None:
    """A judged folder that does not exist stops the run with 'no scored model'."""
    tmp_path, models, config = inputs
    missing = (
        shared_utils.experiment_dir("english", "nope") / script.JUDGED_FALLBACK_DIR_NAME
    )

    with pytest.raises(SystemExit, match="no scored model"):
        run(tmp_path, models, config, "--judged-dir", "LK-en", str(missing))


# --------------------------------------------------------------------------- #
# is_bad_format_scored
# --------------------------------------------------------------------------- #
# 6 rows: 2 right, 1 wrong, 1 rescued by the fallback judge (right),
# 1 bad format (answered, no verdict), 1 call error (never answered).
MIXED = pd.DataFrame(
    {
        "answer_type": ["multiple_choice"] * 6,
        "output": ["A", "B", "C", "zzz", "", None],
        "call_status": ["success"] * 5 + ["error"],
        "is_correct": [True, True, False, None, None, None],
        "is_correct_fallback": [None, None, None, True, None, None],
    }
)


def test_bad_format_left_out_when_not_scored() -> None:
    """is_bad_format_scored False: 3 right of 4 scored; bad format and call error
    are both left out."""
    assert script.accuracy_from_judged(MIXED, is_bad_format_scored=False) == 75.0


def test_bad_format_counts_as_wrong_when_scored() -> None:
    """is_bad_format_scored True: 3 right of 5; bad format counts as wrong, the call
    error is still left out."""
    assert script.accuracy_from_judged(MIXED, is_bad_format_scored=True) == 60.0


def test_config_flag_drives_the_published_accuracy(inputs) -> None:
    """Setting is_bad_format_scored in the config changes the published accuracy."""
    tmp_path, models, config = inputs
    raw = json.loads(config.read_text())
    raw["benchmarks"][0]["metrics"][0]["is_bad_format_scored"] = True
    config.write_text(json.dumps(raw))

    metrics = read_metrics(run(tmp_path, models, config))

    # model-b's unscored question now counts as wrong: 1/4 instead of 1/3
    assert dict(zip(metrics["model_slug"], metrics["metric_value"])) == {
        "model-a": 75.0,
        "model-b": 25.0,
    }


def test_metric_missing_from_the_config_is_reported(inputs) -> None:
    """A benchmark whose config entry lacks the metric fails with a clear error."""
    tmp_path, models, config = inputs
    raw = json.loads(config.read_text())
    raw["benchmarks"][0]["metrics"] = []
    config.write_text(json.dumps(raw))

    with pytest.raises(ValueError, match="defines no metric 'accuracy'"):
        run(tmp_path, models, config)
