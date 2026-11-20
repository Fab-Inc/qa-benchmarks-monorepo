"""Versions: each run records what it ran with; dry runs warn when they differ.

Decisions this file describes:

- The runner doesn't pin FDLLM, and a project that imports it may choose FDLLM's
  version, so local and cloud runs of one experiment can drift apart. That is
  accepted, and made visible instead of prevented.
- Each model's record lists every run that wrote its results, with the runner's,
  FDLLM's and Pillow's versions (FDLLM resizes images with Pillow), Python's, the
  container image, the token limit and the run config.
- A dry run, and a run, warn when the experiment already has results from other
  runner, FDLLM or Pillow versions. Python and the image are recorded only.
"""

from __future__ import annotations

from typing import Any

import pytest

from fab_benchmark_runner import runner

from .conftest import Bench


def test_each_run_is_recorded_with_the_versions_it_used(bench: Bench) -> None:
    bench.create(max_tokens=1000)

    bench.run("model-a")
    bench.run("model-a", force=True)

    record = bench.record("model-a")
    assert len(record["runs"]) == 2
    run = record["runs"][-1]
    assert set(run["environment"]) == {"fdllm", "image", "pillow", "python", "runner"}
    assert run["environment"]["fdllm"]["version"]
    assert (run["max_tokens"], run["executed_calls"], run["force"]) == (1000, 3, True)
    assert run["run_config"]["models"] == ["model-a"]
    assert record["experiment_id"] == bench.rows()[0]["experiment_id"]


def test_a_dry_run_warns_when_earlier_results_used_other_versions(
    bench: Bench, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Here model-a ran with an older FDLLM commit; adding model-b now warns."""

    bench.create()
    current = runner.run_environment()
    older = {**current, "fdllm": {"commit": "0123456789", "version": "0.29.0"}}
    monkeypatch.setattr(runner, "run_environment", lambda: older)
    bench.run("model-a")
    monkeypatch.setattr(runner, "run_environment", lambda: current)

    plan = bench.plan("model-b")

    assert len(plan.version_warnings) == 1
    assert plan.version_warnings[0].startswith("fdllm: 0.29.0 (0123456)")


def test_python_and_image_differences_are_recorded_but_not_warned_about(
    bench: Bench, monkeypatch: pytest.MonkeyPatch
) -> None:
    bench.create()
    current: dict[str, Any] = runner.run_environment()
    cloud = {
        **current,
        "image": "europe-docker.pkg.dev/p/r/fab-bench:1",
        "python": "3.12.0",
    }
    monkeypatch.setattr(runner, "run_environment", lambda: cloud)
    bench.run("model-a")
    monkeypatch.setattr(runner, "run_environment", lambda: current)

    plan = bench.plan("model-b")

    assert plan.version_warnings == ()
    assert bench.record("model-a")["runs"][0]["environment"]["python"] == "3.12.0"
