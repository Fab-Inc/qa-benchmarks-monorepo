"""Tests for scripts/upload_result_files.py with a fake bucket (no network), loaded
by path since scripts/ is not a package. Delete together with that script."""

import datetime as dt
import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "upload_result_files.py"
spec = importlib.util.spec_from_file_location("upload_result_files", SCRIPT)
urf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(urf)

SUFFIXES = (".csv", "_metadata.json", "_frontiers.json")


class FakeBlob:
    """Stands in for a storage Blob: stores uploads in its FakeBucket."""

    def __init__(self, bucket, name):
        self.bucket, self.name = bucket, name

    def exists(self):
        return self.name in self.bucket.objects

    def upload_from_filename(self, path, content_type):
        self.bucket.objects[self.name] = (Path(path).read_text(), content_type)


class FakeBucket:
    """Stands in for a storage Bucket: {object name: (content, content type)}."""

    name = "fab-benchmarks-hub"

    def __init__(self):
        self.objects = {}

    def blob(self, name):
        return FakeBlob(self, name)


@pytest.fixture
def results(tmp_path):
    """A config with LK-en and LK-sw accuracy, an LK entry with no metric and another
    benchmark's metric, plus the LK result files. Returns the folder holding both."""
    config = {
        "cluster_name": "c",
        "benchmarks": [
            {"benchmark_domain": b, "metrics": [{"metric_slug": "accuracy"}]}
            for b in ("LK-en", "LK-sw", "DA-en")
        ]
        + [{"benchmark_domain": "LK-fr", "metrics": []}],
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    for b in ("LK-en", "LK-sw"):
        (tmp_path / b).mkdir()
        for s in SUFFIXES:
            (tmp_path / b / f"{b}_accuracy{s}").write_text(f"{b}{s}")
    return tmp_path


def plan(tmp, date="2026-11-20"):
    """The planned uploads for the fixture's config and files on `date`."""
    return urf.planned_uploads(urf.benchmark_metrics(tmp / "config.json"), tmp, date)


def test_only_this_benchmarks_metrics_are_uploaded(results):
    """Other benchmarks' entries in the shared config are ignored."""
    assert urf.benchmark_metrics(results / "config.json") == [
        ("LK-en", "accuracy"),
        ("LK-sw", "accuracy"),
    ]


def test_objects_go_to_a_dated_folder_per_benchmark_metric(results):
    """Each (benchmark, metric) gets metrics/<benchmark>_<metric>/dt=<date>/."""
    names = [name for _, name, _ in plan(results)]
    assert names[:3] == [
        "metrics/LK-en_accuracy/dt=2026-11-20/LK-en_accuracy.csv",
        "metrics/LK-en_accuracy/dt=2026-11-20/LK-en_accuracy_metadata.json",
        "metrics/LK-en_accuracy/dt=2026-11-20/LK-en_accuracy_frontiers.json",
    ]
    assert len(names) == 6


def test_bucket_folder_sets_the_path_inside_the_bucket(results):
    """bucket_folder is stripped of '/' and may be nested or empty."""
    pairs = urf.benchmark_metrics(results / "config.json")
    names = [
        n for _, n, _ in urf.planned_uploads(pairs, results, "2026-11-20", "/m/v2/")
    ]
    assert names[0] == "m/v2/LK-en_accuracy/dt=2026-11-20/LK-en_accuracy.csv"
    names = [n for _, n, _ in urf.planned_uploads(pairs, results, "2026-11-20", "")]
    assert names[0] == "LK-en_accuracy/dt=2026-11-20/LK-en_accuracy.csv"


def test_missing_file_stops_before_any_upload(results):
    """A missing file is reported before anything is uploaded."""
    (results / "LK-sw" / "LK-sw_accuracy_frontiers.json").unlink()
    with pytest.raises(FileNotFoundError, match="LK-sw_accuracy_frontiers.json"):
        plan(results)


def test_second_upload_same_day_overwrites_with_a_warning(results, capsys):
    """Re-uploading the same day overwrites with a warning; another day is a new
    folder."""
    bucket = FakeBucket()
    assert urf.upload(plan(results), bucket) == []
    assert "WARNING" not in capsys.readouterr().out
    assert bucket.objects[
        "metrics/LK-sw_accuracy/dt=2026-11-20/LK-sw_accuracy.csv"
    ] == ("LK-sw.csv", "text/csv")

    assert len(urf.upload(plan(results), bucket)) == 6
    assert "WARNING: overwriting gs://fab-benchmarks-hub/metrics/LK-en_accuracy" in (
        capsys.readouterr().out
    )
    assert urf.upload(plan(results, "2026-11-21"), bucket) == []
    assert len(bucket.objects) == 12


def run_dry(results, *extra):
    """main() with --dry-run on the fixture's config and files."""
    urf.main(
        [
            "--dry-run",
            "--config",
            str(results / "config.json"),
            "--results-dir",
            str(results),
            *extra,
        ]
    )


def test_dry_run_uploads_nothing(results, capsys):
    """--dry-run prints the plan for the selected benchmark only."""
    run_dry(results, "--benchmark", "LK-sw", "--date", "2026-11-20")
    out = capsys.readouterr().out
    assert "gs://fab-benchmarks-hub/metrics/LK-sw_accuracy/dt=2026-11-20/" in out
    assert "LK-en" not in out
    assert "dry run: 3 files" in out


def test_unknown_benchmark_is_rejected(results, capsys):
    """--benchmark with no metric of this benchmark in the config is a usage error."""
    with pytest.raises(SystemExit):
        run_dry(results, "--benchmark", "DA-en")
    assert "--benchmark" in capsys.readouterr().err


def test_default_date_is_today_in_utc(results, capsys):
    """Without --date the folder is today's date in UTC."""
    run_dry(results)
    today_utc = dt.datetime.now(dt.UTC).date().isoformat()
    assert f"dt={today_utc}" in capsys.readouterr().out
