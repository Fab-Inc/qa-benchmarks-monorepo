"""
Runs go through the real shared harness with an injected fake evaluator and a temporary
fake ``custom_models.yaml``. No provider, network, or GCS call is made.
"""

import json
import sys
from importlib.metadata import entry_points
from pathlib import Path

import phonics_benchmark.cmd as phonics_cmd
import pytest
import typer
from reading_comprehension_question_generation import cmd
from reading_comprehension_question_generation.cmd import app
from reading_comprehension_question_generation.prompts import (
    DEFAULT_PROMPT_NAME,
    build_generation_prompt,
)
from typer.testing import CliRunner

THREE = json.dumps([{"question": f"Q{i}?", "answer": f"A{i}"} for i in (1, 2, 3)])
ITEMS = [
    {"benchmark_item_id": "en-1", "text_content": "A hen sat.", "language": "English"},
    {
        "benchmark_item_id": "sw-1",
        "text_content": "Pili aliona paka.",
        "language": "Kiswahili",
    },
]


@pytest.fixture()
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture()
def items_file(tmp_path: Path) -> Path:
    path = tmp_path / "items.jsonl"
    path.write_text("".join(json.dumps(i) + "\n" for i in ITEMS), encoding="utf-8")
    return path


@pytest.fixture()
def models_file(tmp_path: Path) -> Path:
    path = tmp_path / "custom_models.yaml"
    path.write_text(
        json.dumps({"provider-a": {"models": {"model-a": {}}}}), encoding="utf-8"
    )
    return path


@pytest.fixture()
def evaluator(monkeypatch):
    """Route the command's harness call through a controllable fake evaluator.

    ``state["output"]`` is what the fake model returns; ``state["prompts"]`` records
    every prompt it was called with.
    """
    state = {"output": THREE, "prompts": []}

    def fake(_config, _model, _item, prompt):
        state["prompts"].append(prompt)
        return {"output": state["output"]}

    real_execute_run = phonics_cmd.execute_run
    monkeypatch.setattr(
        phonics_cmd,
        "execute_run",
        lambda config, **kwargs: real_execute_run(config, evaluator=fake, **kwargs),
    )
    return state


def _options_args(
    tmp_path: Path,
    items_file: Path,
    models_file: Path,
    *extra: str,
    with_items_path: bool = True,
):
    args = [
        "run-generation",
        "--language",
        "english",
        "--models",
        "model-a",
        "--run-id",
        "opt-run",
        "--output-dir",
        str(tmp_path / "out"),
        "--custom-models-path",
        str(models_file),
        "--skip-models-download",
        "--model-workers",
        "1",
        *extra,
    ]
    if with_items_path:
        args += ["--items-path", str(items_file)]
    return args


def _write_config(tmp_path, models_file, **overrides) -> Path:
    config = {
        "run_id": "cfg-run",
        "language": "english",
        "items_path": "items.jsonl",
        "output_dir": "out",
        "models": ["model-a"],
        "custom_models_path": str(models_file),
        "model_workers": 1,
        **overrides,
    }
    path = tmp_path / "run.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


def _rows(run_dir: Path) -> list[dict]:
    return [
        json.loads(line)
        for path in sorted((run_dir / "responses").glob("*.jsonl"))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_both_console_scripts_resolve_to_main_and_help_lists_the_commands(
    runner, monkeypatch, capsys
) -> None:
    scripts = {
        ep.name: ep.value
        for ep in entry_points(group="console_scripts")
        if ep.name in {"reading-comprehension-question-generation", "rcqg"}
    }
    target = "reading_comprehension_question_generation.cmd:main"
    assert scripts == {
        "reading-comprehension-question-generation": target,
        "rcqg": target,
    }

    assert "Usage:" in runner.invoke(app).output
    monkeypatch.setattr(sys, "argv", ["rcqg", "--help"])
    with pytest.raises(SystemExit) as exit_info:
        cmd.main()
    assert exit_info.value.code == 0
    help_text = capsys.readouterr().out
    for command in (
        "run-generation",
        "extract-pairs",
        "upload-results",
        "fetch-results",
    ):
        assert command in help_text


def test_run_from_options_writes_the_standard_layout_with_run_time_prompts(
    runner, tmp_path, monkeypatch, evaluator, models_file, items_file
) -> None:
    # The default items path is data/<language>/benchmark_items.jsonl, so point the
    # data root at tmp_path and omit --items-path.
    monkeypatch.setattr(cmd, "DEFAULT_DATA_ROOT", tmp_path / "data")
    (tmp_path / "data" / "english").mkdir(parents=True)
    (tmp_path / "data" / "english" / "benchmark_items.jsonl").write_bytes(
        items_file.read_bytes()
    )
    args = _options_args(tmp_path, items_file, models_file, with_items_path=False)

    result = runner.invoke(app, args)

    assert result.exit_code == 0, result.output
    run_dir = tmp_path / "out" / "opt-run"
    for name in ("config.json", "log.json", "responses/responses_model-a.jsonl"):
        assert (run_dir / name).is_file(), name
    rows = {r["benchmark_item_id"]: r for r in _rows(run_dir)}
    assert set(rows) == {"en-1", "sw-1"}
    for item in ITEMS:
        row = rows[item["benchmark_item_id"]]
        assert row["task"] == "question-generation"
        assert row["language"] == "english"  # the run's data folder, not the item's
        assert row["prompt"] == build_generation_prompt(
            DEFAULT_PROMPT_NAME, item["text_content"], item["language"]
        )
    assert "Kiswahili" in rows["sw-1"]["prompt"]
    assert "prompt" not in items_file.read_text(encoding="utf-8")


def test_run_from_a_config_file_uses_the_default_prompt_name(
    runner, tmp_path, evaluator, models_file, items_file
) -> None:
    config_path = _write_config(tmp_path, models_file)

    result = runner.invoke(
        app, ["run-generation", "--config", str(config_path), "--skip-models-download"]
    )

    assert result.exit_code == 0, result.output
    rows = _rows(tmp_path / "out" / "cfg-run")
    assert len(rows) == 2
    assert {r["prompt_name"] for r in rows} == {"question_generation"}
    assert {r["task"] for r in rows} == {"question-generation"}


def test_an_identical_rerun_uses_the_cache(
    runner, tmp_path, evaluator, models_file, items_file
) -> None:
    args = _options_args(tmp_path, items_file, models_file)
    assert runner.invoke(app, args).exit_code == 0
    assert len(evaluator["prompts"]) == 2

    second = runner.invoke(app, args)

    assert second.exit_code == 0, second.output
    assert len(evaluator["prompts"]) == 2


def test_force_reruns_every_pair(
    runner, tmp_path, evaluator, models_file, items_file
) -> None:
    assert (
        runner.invoke(app, _options_args(tmp_path, items_file, models_file)).exit_code
        == 0
    )

    forced = runner.invoke(
        app, _options_args(tmp_path, items_file, models_file, "--force")
    )

    assert forced.exit_code == 0, forced.output
    assert len(evaluator["prompts"]) == 4


def test_retry_errors_reruns_only_the_failed_rows(
    runner, tmp_path, evaluator, models_file, items_file
) -> None:
    evaluator["output"] = ""
    failed = runner.invoke(app, _options_args(tmp_path, items_file, models_file))
    assert failed.exit_code == 1
    assert {r["call_status"] for r in _rows(tmp_path / "out" / "opt-run")} == {"error"}

    evaluator["output"] = THREE
    retried = runner.invoke(
        app, _options_args(tmp_path, items_file, models_file, "--retry-errors")
    )

    assert retried.exit_code == 0, retried.output
    assert len(evaluator["prompts"]) == 4  # 2 failed calls, then 2 retried calls
    assert {r["call_status"] for r in _rows(tmp_path / "out" / "opt-run")} == {
        "success"
    }


def test_a_task_key_in_the_config_is_rejected(
    runner, tmp_path, evaluator, models_file, items_file
) -> None:
    config_path = _write_config(tmp_path, models_file, task="something-else")

    result = runner.invoke(
        app, ["run-generation", "--config", str(config_path), "--skip-models-download"]
    )

    assert result.exit_code == 1
    assert "'task' is selected by the CLI command" in result.output


def test_a_run_without_models_is_a_readable_error(
    runner, tmp_path, evaluator, items_file
) -> None:
    result = runner.invoke(
        app,
        [
            "run-generation",
            "--language",
            "english",
            "--items-path",
            str(items_file),
            "--skip-models-download",
        ],
    )

    assert result.exit_code == 1
    assert "ERROR:" in result.output
    assert "Traceback" not in result.output


def test_no_option_or_config_key_changes_the_pair_count(
    runner, tmp_path, evaluator, models_file, items_file
) -> None:
    command = typer.main.get_group(app).commands["run-generation"]
    option_names = {name for p in command.params for name in p.opts}
    assert not [n for n in option_names if "pair" in n or "num" in n or "count" in n]

    config_path = _write_config(tmp_path, models_file, num_pairs=7, pair_count=7)
    result = runner.invoke(
        app, ["run-generation", "--config", str(config_path), "--skip-models-download"]
    )

    assert result.exit_code == 0, result.output
    assert len(evaluator["prompts"]) == 2
    for prompt in evaluator["prompts"]:
        assert "exactly 3 question and answer pairs" in prompt


def test_extract_pairs_reports_parse_problems_and_still_exits_zero(
    runner, tmp_path, write_run_dir
) -> None:
    out = tmp_path / "out"
    out.mkdir()
    write_run_dir(
        out,
        {
            "good": [{"benchmark_item_id": "i-1", "output": THREE}],
            "bad": [{"benchmark_item_id": "i-1", "output": "no json"}],
        },
        run_id="r1",
    )

    result = runner.invoke(
        app, ["extract-pairs", "--run-id", "r1", "--output-dir", str(out)]
    )

    assert result.exit_code == 0, result.output
    assert "good: 1 response(s), 3 pair(s); ok=1" in result.output
    assert "bad: 1 response(s), 0 pair(s); parse_failed=1" in result.output
    for name in ("pairs.jsonl", "response_status.jsonl", "extraction_summary.json"):
        assert (out / "r1" / name).is_file()


def test_extract_pairs_needs_a_run_to_read(runner) -> None:
    result = runner.invoke(app, ["extract-pairs"])

    assert result.exit_code == 1
    assert "give --run-dir, or --run-id with --output-dir" in result.output


def test_extract_pairs_for_a_missing_run_is_a_readable_error(runner, tmp_path) -> None:
    result = runner.invoke(
        app, ["extract-pairs", "--run-id", "nope", "--output-dir", str(tmp_path)]
    )

    assert result.exit_code == 1
    assert "No response files found" in result.output
    assert "Traceback" not in result.output


def test_results_dest_is_forwarded_to_the_run_config(
    runner, tmp_path, monkeypatch, models_file, items_file
) -> None:
    seen = {}

    def fake_sync(run_config, **kwargs):
        seen["config"] = run_config
        return {"error_count": 0}

    monkeypatch.setattr(cmd, "_execute_with_optional_sync", fake_sync)
    monkeypatch.setattr(cmd, "_echo_run_output", lambda output: None)
    dest = "gs://example-bucket/rcqg/runs/opt-run/"

    result = runner.invoke(
        app, _options_args(tmp_path, items_file, models_file, "--results-dest", dest)
    )

    assert result.exit_code == 0, result.output
    assert seen["config"].results_dest == dest
