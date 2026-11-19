"""Jobs: mirroring an experiment on GCS, upload failures, fetching and uploading.

What jobs are meant to do is described in ``tests/behaviour/test_jobs_on_gcs.py``;
these tests pin the mechanics underneath.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from fab_benchmark_runner import (
    ConfigError,
    create_experiment,
    fetch_results,
    parse_run_config,
    plan_job,
    run_job,
    upload_experiment,
)
from fab_benchmark_runner.jobs import experiment_mirror
from fab_benchmark_runner.run_sync import RunSync, RunSyncConflictError

EXPERIMENT = "gs://bucket/exp"
ITEMS = [{"benchmark_item_id": f"item-{n}", "prompt": f"Prompt {n}?"} for n in (1, 2)]


def _echo(model: str, model_input: Any, _max_tokens: Any) -> dict[str, str]:
    return {"output": f"{model}:{model_input.benchmark_item_id}"}


@pytest.fixture
def setup(tmp_path: Path, storage: Any, write_registry) -> dict[str, Any]:
    inputs = tmp_path / "inputs.jsonl"
    inputs.write_text("".join(json.dumps(row) + "\n" for row in ITEMS))
    create_experiment(
        inputs, dest=EXPERIMENT, benchmark="t", max_tokens=100, storage_client=storage
    )
    registry = write_registry(("model-a", "model-b"))
    return {"tmp": tmp_path, "storage": storage, "registry": registry}


def _config(setup: dict[str, Any], *models: str, experiment: str = EXPERIMENT, **raw):
    return parse_run_config(
        {
            "experiment": experiment,
            "models": list(models or ("model-a",)),
            "custom_models_path": str(setup["registry"]),
            "model_workers": 1,
            **raw,
        },
        base_dir=setup["tmp"],
    )


def _run(setup: dict[str, Any], *models: str, job: str = "job", **kwargs: Any):
    return run_job(
        _config(setup, *models),
        workdir=setup["tmp"] / job,
        download_public_registry=False,
        call_model=kwargs.pop("call_model", _echo),
        storage_client=setup["storage"],
        **kwargs,
    )


def test_the_mirror_lives_in_the_workdir(tmp_path: Path) -> None:
    assert experiment_mirror(tmp_path, "gs://bucket/a/b/") == (
        tmp_path.resolve() / "experiments" / "bucket" / "a" / "b"
    )


def test_a_transient_upload_failure_warns_and_is_retried_at_the_end(
    setup: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    storage = setup["storage"]
    storage.fail_next_upload.add("exp/responses/responses_model-a.jsonl")

    _run(setup)

    assert "WARNING: could not upload" in capsys.readouterr().err
    remote = storage.objects["bucket"]["exp/responses/responses_model-a.jsonl"][1]
    assert len(remote.decode().splitlines()) == 2


def test_a_failed_final_retry_does_not_hide_the_runs_own_failure(
    setup: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    def crash(*_args: Any) -> dict:
        raise KeyError("run failed")

    with (
        patch.object(RunSync, "push_pending", side_effect=RuntimeError("retry failed")),
        pytest.raises(KeyError, match="run failed"),
    ):
        _run(setup, call_model=crash)

    assert "retrying uploads failed" in capsys.readouterr().err


def test_a_failed_final_retry_after_a_successful_run_propagates(
    setup: dict[str, Any],
) -> None:
    with (
        patch.object(RunSync, "push_pending", side_effect=RuntimeError("retry failed")),
        pytest.raises(RuntimeError, match="retry failed"),
    ):
        _run(setup)


def test_a_generation_conflict_stops_the_run(setup: dict[str, Any]) -> None:
    with (
        patch.object(RunSync, "push_file", side_effect=RunSyncConflictError("changed")),
        pytest.raises(RunSyncConflictError),
    ):
        _run(setup)


def test_a_missing_experiment_is_reported(setup: dict[str, Any]) -> None:
    config = _config(setup, experiment="gs://bucket/missing")

    with pytest.raises(ConfigError, match="holds no experiment"):
        plan_job(
            config,
            workdir=setup["tmp"] / "job",
            download_public_registry=False,
            call_model=_echo,
            storage_client=setup["storage"],
        )


@pytest.mark.parametrize(
    ("inputs", "max_tokens", "message"),
    [
        (
            [
                {
                    "benchmark_item_id": "v",
                    "prompt": "?",
                    "images": [{"uri": "f.png", "detail": "low"}],
                }
            ],
            100,
            "has images",
        ),
        (ITEMS, None, "has no token limit"),
    ],
)
def test_fdllm_runs_refuse_what_they_cant_send_yet_before_any_call(
    tmp_path: Path, write_registry, inputs, max_tokens, message: str
) -> None:
    (tmp_path / "f.png").write_bytes(b"pixels")
    path = tmp_path / "inputs.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in inputs))
    create_experiment(path, dest=tmp_path / "exp", benchmark="t", max_tokens=max_tokens)
    config = parse_run_config(
        {
            "experiment": "exp",
            "models": ["model-a"],
            "custom_models_path": str(write_registry(("model-a",))),
        },
        base_dir=tmp_path,
    )

    with (
        patch("fab_benchmark_runner.jobs.prepare_fdllm", return_value={}),
        patch("fab_benchmark_runner.jobs.call_model_with_fdllm") as call,
        pytest.raises(ConfigError, match=message),
    ):
        run_job(config, workdir=tmp_path / "job", download_public_registry=False)
    call.assert_not_called()


def test_fetch_results_downloads_everything_but_images_or_one_model(
    setup: dict[str, Any],
) -> None:
    _run(setup, "model-a", "model-b")
    dest = setup["tmp"] / "fetched"

    one = fetch_results(
        EXPERIMENT, dest / "one", model="model-b", storage_client=setup["storage"]
    )
    everything = fetch_results(
        EXPERIMENT, dest / "all", storage_client=setup["storage"]
    )

    files = {p.relative_to(dest / "one").as_posix() for p in (dest / "one").rglob("*")}
    assert one == 5 and everything == 6
    assert "responses/responses_model-b.jsonl" in files
    assert "responses/responses_model-a.jsonl" not in files
    with pytest.raises(ConfigError, match="holds no experiment"):
        fetch_results("gs://bucket/none", dest / "x", storage_client=setup["storage"])


def test_upload_sends_the_record_after_the_inputs_and_never_overwrites(
    tmp_path: Path, storage: Any
) -> None:
    (tmp_path / "f.png").write_bytes(b"pixels")
    inputs = tmp_path / "inputs.jsonl"
    inputs.write_text(
        json.dumps(
            {
                "benchmark_item_id": "v",
                "prompt": "?",
                "images": [{"uri": "f.png", "detail": "low"}],
            }
        )
        + "\n"
    )
    create_experiment(
        inputs, dest=tmp_path / "exp", benchmark="t", max_tokens=1, copy_images=True
    )

    sent = upload_experiment(tmp_path / "exp", "gs://bucket/up", storage_client=storage)
    again = upload_experiment(
        tmp_path / "exp", "gs://bucket/up", storage_client=storage
    )

    names = [name for name, _ in storage.upload_preconditions]
    assert sent == again == 3
    assert names[1:3] == ["up/inputs.jsonl", "up/experiment.json"]
    assert names[0].startswith("up/images/")
    storage.objects["bucket"]["up/inputs.jsonl"] = (999, b"other")
    with pytest.raises(RunSyncConflictError):
        upload_experiment(tmp_path / "exp", "gs://bucket/up", storage_client=storage)


def test_upload_refuses_images_at_local_paths(tmp_path: Path, storage: Any) -> None:
    (tmp_path / "f.png").write_bytes(b"pixels")
    inputs = tmp_path / "inputs.jsonl"
    inputs.write_text(
        json.dumps(
            {
                "benchmark_item_id": "v",
                "prompt": "?",
                "images": [{"uri": "f.png", "detail": "low"}],
            }
        )
        + "\n"
    )
    create_experiment(inputs, dest=tmp_path / "exp", benchmark="t", max_tokens=1)

    with pytest.raises(ConfigError, match="--copy-images"):
        upload_experiment(tmp_path / "exp", "gs://bucket/up", storage_client=storage)
    assert storage.upload_preconditions == []
