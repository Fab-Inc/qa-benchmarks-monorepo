"""Tests for run_bench: parsing, prompt building, dataset loading, run loop."""

import json

import pandas as pd
import pytest
from conftest import VALID_MODEL, FakeCaller, make_response, write_dataset
from linguistic_knowledge_benchmark import run_bench, shared_utils


# --------------------------------------------------------------------------- #
# _build_system_prompts
# --------------------------------------------------------------------------- #
def test_build_system_prompts_has_all_answer_types():
    prompts = run_bench._build_system_prompts("Luganda")
    assert set(prompts) == {"multiple_choice", "short_form"}


def test_build_system_prompts_prefix_gives_language_context_for_all_types():
    prompts = run_bench._build_system_prompts("Luganda")
    # Every answer type is prefixed with English-medium context naming the
    # language the question is about.
    for answer_type in ("multiple_choice", "short_form"):
        assert prompts[answer_type].startswith("The following question")
        assert "Luganda" in prompts[answer_type]


def test_build_system_prompts_normalizes_kiswahili_to_swahili():
    # Kiswahili is the CLI/display name (endonym); prompts stay English-medium
    # and should use the English form "Swahili" instead.
    prompts = run_bench._build_system_prompts("Kiswahili")
    for answer_type in ("multiple_choice", "short_form"):
        assert "Swahili" in prompts[answer_type]
        assert "Kiswahili" not in prompts[answer_type]


# --------------------------------------------------------------------------- #
# load_dataset
# --------------------------------------------------------------------------- #
def test_load_dataset_reads_json(tmp_path):
    path = write_dataset(tmp_path / "ds.json", [{"question_id": 1}])
    assert run_bench.load_dataset(path) == [{"question_id": 1}]


def test_load_dataset_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        run_bench.load_dataset(tmp_path / "nope.json")


# --------------------------------------------------------------------------- #
# git_hash
# --------------------------------------------------------------------------- #
def test_git_hash_returns_none_on_failure(monkeypatch):
    import subprocess

    def boom(*args, **kwargs):
        raise subprocess.CalledProcessError(1, "git")

    monkeypatch.setattr(subprocess, "check_output", boom)
    assert run_bench.git_hash() is None


# --------------------------------------------------------------------------- #
# update_log
# --------------------------------------------------------------------------- #
def test_update_log_merges_with_existing_entries(tmp_path):
    run_bench.update_log(tmp_path, "model-a", {"foo": "1"})
    run_bench.update_log(tmp_path, "model-a", {"bar": "2"})
    data = json.loads((tmp_path / "log.json").read_text())
    assert data["model-a"] == {"foo": "1", "bar": "2"}


# --------------------------------------------------------------------------- #
# _error_mask
# --------------------------------------------------------------------------- #
def test_error_mask_returns_all_false_when_no_call_status_column():
    df = pd.DataFrame({"question_id": [1, 2]})
    mask = run_bench._error_mask(df)
    assert list(mask) == [False, False]


# --------------------------------------------------------------------------- #
# run_benchmark
# --------------------------------------------------------------------------- #
def test_run_benchmark_invalid_model_raises(tmp_path):
    ds = write_dataset(tmp_path / "ds.json", [])
    with pytest.raises(ValueError, match="Invalid model"):
        run_bench.run_benchmark(
            "totally-unknown-model",
            language_name="Luganda",
            dataset_path=ds,
            output_dir=tmp_path / "out",
        )


def test_run_benchmark_cache_hit_returns_existing(tmp_path, monkeypatch):
    out = tmp_path / "out"
    out.mkdir()
    cached = pd.DataFrame(
        [
            {
                "question_id": 1,
                "output": "A",
                "call_status": run_bench.CALL_STATUS_SUCCESS,
            }
        ]
    )
    cached.to_csv(out / f"responses_{VALID_MODEL}.csv", index=False)

    def boom(_name):  # get_caller must not be reached on a cache hit
        raise AssertionError("get_caller should not be called on cache hit")

    monkeypatch.setattr(run_bench, "get_caller", boom)
    df = run_bench.run_benchmark(
        VALID_MODEL,
        language_name="Luganda",
        dataset_path=tmp_path / "ds.json",
        output_dir=out,
    )
    assert df.loc[0, "output"] == "A"


def test_run_benchmark_full_run_writes_csv(tmp_path, monkeypatch):
    from conftest import sample_questions

    ds = write_dataset(tmp_path / "ds.json", sample_questions())
    out = tmp_path / "out"
    caller = FakeCaller(make_response("B"))
    monkeypatch.setattr(run_bench, "get_caller", lambda _name: caller)
    monkeypatch.setattr(run_bench, "git_hash", lambda: "abc1234")

    df = run_bench.run_benchmark(
        VALID_MODEL,
        language_name="Luganda",
        dataset_path=ds,
        output_dir=out,
    )

    assert len(df) == 2
    assert list(df["output"]) == ["B", "B"]
    assert list(df["call_status"]) == [run_bench.CALL_STATUS_SUCCESS] * 2
    assert list(df["error_type"]) == ["", ""]
    assert (out / f"responses_{VALID_MODEL}.csv").exists()
    # expected_answers serialized with the pipe join
    assert df.loc[0, "expected_answers"] == "B"
    # fresh run: no retries
    assert list(df["retry_count"]) == [0, 0]
    # log.json written at run dir level (parent of responses/)
    log_path = out.parent / "log.json"
    assert log_path.exists()
    entry = json.loads(log_path.read_text())[VALID_MODEL]
    assert "responses_timestamp-utc" in entry
    assert entry["responses_git-hash"] == "abc1234"
    assert "responses_retry_timestamp-utc" not in entry


def test_run_benchmark_error_branch_records_error_status(tmp_path, monkeypatch):
    from conftest import sample_questions

    ds = write_dataset(tmp_path / "ds.json", sample_questions()[:1])
    caller = FakeCaller(raise_exc=RuntimeError("boom"))
    monkeypatch.setattr(run_bench, "get_caller", lambda _name: caller)

    df = run_bench.run_benchmark(
        VALID_MODEL,
        language_name="Luganda",
        dataset_path=ds,
        output_dir=tmp_path / "out",
    )
    assert df.loc[0, "output"] == ""
    assert pd.isna(df.loc[0, "latency"])
    assert df.loc[0, "call_status"] == run_bench.CALL_STATUS_ERROR
    assert df.loc[0, "error_type"] == "RuntimeError"
    assert df.loc[0, "error_message"] == "boom"


def test_run_benchmark_cache_with_errors_returns_existing(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    pd.DataFrame(
        [
            {
                "question_id": 1,
                "output": "",
                "call_status": run_bench.CALL_STATUS_ERROR,
                "error_type": "RuntimeError",
                "error_message": "boom",
            }
        ]
    ).to_csv(out / f"responses_{VALID_MODEL}.csv", index=False)

    df = run_bench.run_benchmark(
        VALID_MODEL,
        language_name="Luganda",
        dataset_path=tmp_path / "ds.json",
        output_dir=out,
    )
    assert df.loc[0, "call_status"] == run_bench.CALL_STATUS_ERROR


def test_run_benchmark_retry_errors_no_error_rows_returns_cached(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    pd.DataFrame(
        [
            {
                "question_id": 1,
                "output": "A",
                "call_status": run_bench.CALL_STATUS_SUCCESS,
            }
        ]
    ).to_csv(out / f"responses_{VALID_MODEL}.csv", index=False)

    df = run_bench.run_benchmark(
        VALID_MODEL,
        language_name="Luganda",
        dataset_path=tmp_path / "ds.json",
        output_dir=out,
        retry_errors=True,
    )
    assert df.loc[0, "output"] == "A"


def test_run_benchmark_retry_errors_missing_cache_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="cannot use --retry-errors"):
        run_bench.run_benchmark(
            VALID_MODEL,
            language_name="Luganda",
            dataset_path=tmp_path / "ds.json",
            output_dir=tmp_path / "out",
            retry_errors=True,
        )


def test_run_benchmark_retry_errors_replaces_failed_rows(tmp_path, monkeypatch):
    from conftest import sample_questions

    ds = write_dataset(tmp_path / "ds.json", sample_questions()[:2])
    out = tmp_path / "out"
    out.mkdir()
    pd.DataFrame(
        [
            {
                "question_id": 1,
                "domain": "morphology",
                "answer_type": "multiple_choice",
                "expected_answers": "B",
                "output": "",
                "latency": None,
                "total_tokens": None,
                "completion_tokens": None,
                "reasoning_tokens": None,
                "call_status": run_bench.CALL_STATUS_ERROR,
                "error_type": "RuntimeError",
                "error_message": "boom",
            },
            {
                "question_id": 2,
                "domain": "vocabulary",
                "answer_type": "short_form",
                "expected_answers": "kintu",
                "output": "old",
                "latency": 0.1,
                "total_tokens": 10,
                "completion_tokens": 5,
                "reasoning_tokens": 2,
                "call_status": run_bench.CALL_STATUS_SUCCESS,
                "error_type": "",
                "error_message": "",
            },
        ]
    ).to_csv(out / f"responses_{VALID_MODEL}.csv", index=False)
    caller = FakeCaller(make_response("B"))
    monkeypatch.setattr(run_bench, "get_caller", lambda _name: caller)
    monkeypatch.setattr(run_bench, "git_hash", lambda: "retry456")

    df = run_bench.run_benchmark(
        VALID_MODEL,
        language_name="Luganda",
        dataset_path=ds,
        output_dir=out,
        retry_errors=True,
    ).set_index("question_id")

    assert df.loc[1, "output"] == "B"
    assert df.loc[1, "call_status"] == run_bench.CALL_STATUS_SUCCESS
    assert df.loc[1, "retry_count"] == 1  # was an error row, now retried once
    assert df.loc[2, "output"] == "old"
    assert df.loc[2, "retry_count"] == 0  # was already successful, untouched
    assert len(caller.calls) == 1
    # retry run writes responses_retry_* fields, not responses_*
    log_entry = json.loads((out.parent / "log.json").read_text())[VALID_MODEL]
    assert "responses_retry_timestamp-utc" in log_entry
    assert log_entry["responses_retry_git-hash"] == "retry456"
    assert "responses_timestamp-utc" not in log_entry


def test_run_benchmark_caller_not_implemented_raises(tmp_path, monkeypatch):
    from conftest import sample_questions

    ds = write_dataset(tmp_path / "ds.json", sample_questions())

    def raise_ni(_name):
        raise NotImplementedError

    monkeypatch.setattr(run_bench, "get_caller", raise_ni)
    with pytest.raises(ValueError, match="not"):
        run_bench.run_benchmark(
            VALID_MODEL,
            language_name="Luganda",
            dataset_path=ds,
            output_dir=tmp_path / "out",
        )


def test_run_benchmark_default_output_dir(tmp_path, monkeypatch):
    """output_dir=None derives cache/<language>/runs/default/responses."""
    from conftest import sample_questions

    monkeypatch.setattr(shared_utils, "CACHE_BASE_DIR", tmp_path / "cache")
    ds = write_dataset(tmp_path / "ds.json", sample_questions()[:1])
    monkeypatch.setattr(run_bench, "get_caller", lambda _name: FakeCaller())

    run_bench.run_benchmark(
        VALID_MODEL,
        language_name="Luganda",
        dataset_path=ds,
    )
    expected = (
        tmp_path
        / "cache"
        / "luganda"
        / "runs"
        / "default"
        / "responses"
        / f"responses_{VALID_MODEL}.csv"
    )
    assert expected.exists()


def test_run_benchmark_run_id_uses_run_specific_output_dir(tmp_path, monkeypatch):
    from conftest import sample_questions

    monkeypatch.setattr(shared_utils, "CACHE_BASE_DIR", tmp_path / "cache")
    ds = write_dataset(tmp_path / "ds.json", sample_questions()[:1])
    monkeypatch.setattr(run_bench, "get_caller", lambda _name: FakeCaller())

    run_bench.run_benchmark(
        VALID_MODEL,
        language_name="Luganda",
        dataset_path=ds,
        run_id="run-123",
    )
    expected = (
        tmp_path
        / "cache"
        / "luganda"
        / "runs"
        / "run-123"
        / "responses"
        / f"responses_{VALID_MODEL}.csv"
    )
    assert expected.exists()


def test_run_benchmark_rate_limited_sleeps(tmp_path, monkeypatch):
    from conftest import sample_questions

    ds = write_dataset(tmp_path / "ds.json", sample_questions()[:1])
    monkeypatch.setattr(run_bench, "get_caller", lambda _name: FakeCaller())
    monkeypatch.setattr(run_bench, "RATE_LIMITED_MODELS", {VALID_MODEL})
    sleeps: list = []
    monkeypatch.setattr(run_bench.time, "sleep", lambda s: sleeps.append(s))

    run_bench.run_benchmark(
        VALID_MODEL,
        language_name="Luganda",
        dataset_path=ds,
        output_dir=tmp_path / "out",
    )
    assert sleeps == [5]


# --------------------------------------------------------------------------- #
# git_hash / update_log / _error_mask / retry-error edge cases
# --------------------------------------------------------------------------- #
def test_git_hash_returns_none_on_error(monkeypatch, capsys):
    import subprocess

    def boom(*a, **k):
        raise OSError("no git here")

    # git_hash imports subprocess lazily, so patch the real module.
    monkeypatch.setattr(subprocess, "check_output", boom)
    assert run_bench.git_hash() is None
    assert "could not read git hash" in capsys.readouterr().err


def test_update_log_merges_existing_and_skips_none(tmp_path):
    (tmp_path / "log.json").write_text(json.dumps({"m": {"a": "1"}}))
    run_bench.update_log(tmp_path, "m", {"b": "2", "c": None})
    data = json.loads((tmp_path / "log.json").read_text())
    assert data["m"] == {"a": "1", "b": "2"}  # merged; None value skipped


def test_error_mask_without_call_status_column():
    mask = run_bench._error_mask(pd.DataFrame({"question_id": [1, 2]}))
    assert list(mask) == [False, False]


def test_run_benchmark_retry_errors_without_cache_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="cannot use --retry-errors"):
        run_bench.run_benchmark(
            VALID_MODEL,
            language_name="Luganda",
            dataset_path=tmp_path / "ds.json",
            output_dir=tmp_path / "out",
            retry_errors=True,
        )
