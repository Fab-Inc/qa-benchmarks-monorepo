"""Tests for judge_bench: scorers, judging, fallback, summary."""

import json

import pandas as pd
import pytest
from conftest import VALID_MODEL, FakeCaller, make_response, write_dataset
from linguistic_knowledge_benchmark import judge_bench, shared_utils


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def _responses_dir(base, language="kiswahili"):
    d = base / language / "runs" / "default" / "responses"
    d.mkdir(parents=True)
    return d


def _write_responses(responses_dir, rows):
    df = pd.DataFrame(rows)
    df.to_csv(responses_dir / f"responses_{VALID_MODEL}.csv", index=False)
    return df


# --------------------------------------------------------------------------- #
# prompt builder + simple scorers
# --------------------------------------------------------------------------- #
def test_build_judge_system_prompt_includes_language():
    assert "Yoruba" in judge_bench._build_judge_system_prompt("Yoruba")


def test_build_judge_system_prompt_normalizes_kiswahili_to_swahili():
    prompt = judge_bench._build_judge_system_prompt("Kiswahili")
    assert "Swahili" in prompt
    assert "Kiswahili" not in prompt


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("A", "A"),
        ("  C ", "C"),
        ("(D)", "D"),
        ("<think>reasoning here</think>B", "B"),
        ("Some explanation\nB", "B"),
        ("not an answer at all", None),
    ],
)
def test_parse_answer_mcq_patterns(raw, expected):
    assert judge_bench.parse_answer_mcq(raw) == expected


def test_parse_answer_mcq_handles_nan_and_none():
    assert judge_bench.parse_answer_mcq(float("nan")) is None
    assert judge_bench.parse_answer_mcq(None) is None


@pytest.mark.parametrize(
    "output, expected, result",
    [("B", ["B"], True), ("A", ["B"], False)],
)
def test_score_multiple_choice(output, expected, result):
    assert judge_bench.score_multiple_choice(output, expected) == (
        result,
        "",
        output,
    )


def test_score_multiple_choice_unparseable_returns_none():
    assert judge_bench.score_multiple_choice("???", ["B"]) == (None, "", None)


def test_score_with_llm_judge_success():
    caller = FakeCaller(
        make_response(json.dumps({"is_correct": True, "justification": "match"}))
    )
    ok, just = judge_bench.score_with_llm_judge(
        "q", "a", ["a"], "short_form", caller, system_prompt="sys"
    )
    assert ok is True
    assert just == "match"


def test_score_with_llm_judge_failure():
    caller = FakeCaller(raise_exc=RuntimeError("api down"))
    ok, just = judge_bench.score_with_llm_judge(
        "q", "a", ["a"], "short_form", caller, system_prompt="sys"
    )
    assert ok is None
    assert "Judge error" in just


def test_score_with_llm_judges_returns_one_result_per_judge():
    callers = {
        "j1": FakeCaller(
            make_response(json.dumps({"is_correct": True, "justification": "a"}))
        ),
        "j2": FakeCaller(
            make_response(json.dumps({"is_correct": False, "justification": "b"}))
        ),
    }
    result = judge_bench.score_with_llm_judges(
        "q", "out", ["x"], "short_form", callers, system_prompt="sys"
    )
    assert result == {"j1": (True, "a"), "j2": (False, "b")}


@pytest.mark.parametrize(
    "verdicts, expected_verdict, expected_agreement",
    [
        ([True, True, True], True, 1.0),
        ([True, True, False], True, 2 / 3),
        ([False, False, True], False, 2 / 3),
        ([True, False], None, 0.5),  # exact tie -> unscored, not arbitrarily broken
        ([None, None], None, None),  # every judge failed
        ([True, None, None], True, 1.0),  # failed judges don't count against agreement
    ],
)
def test_majority_vote(verdicts, expected_verdict, expected_agreement):
    verdict, agreement = judge_bench.majority_vote(verdicts)
    assert verdict is expected_verdict
    if expected_agreement is None:
        assert agreement is None
    else:
        assert agreement == pytest.approx(expected_agreement)


# --------------------------------------------------------------------------- #
# judge_results
# --------------------------------------------------------------------------- #
def test_judge_results_missing_input_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(shared_utils, "CACHE_BASE_DIR", tmp_path)
    with pytest.raises(FileNotFoundError):
        judge_bench.judge_results(
            VALID_MODEL,
            language_name="Kiswahili",
            dataset_path=tmp_path / "ds.json",
        )


def test_judge_results_cache_hit(tmp_path, monkeypatch):
    monkeypatch.setattr(shared_utils, "CACHE_BASE_DIR", tmp_path)
    responses = _responses_dir(tmp_path)
    _write_responses(
        responses,
        [
            {
                "question_id": 1,
                "answer_type": "multiple_choice",
                "expected_answers": "B",
                "output": "B",
            }
        ],
    )
    judged = tmp_path / "kiswahili" / "runs" / "default" / "responses_judged"
    judged.mkdir(parents=True)
    pd.DataFrame([{"question_id": 1, "is_correct": True}]).to_csv(
        judged / f"responses_judged_{VALID_MODEL}.csv", index=False
    )

    df = judge_bench.judge_results(
        VALID_MODEL, language_name="Kiswahili", dataset_path=tmp_path / "ds.json"
    )
    assert bool(df.loc[0, "is_correct"]) is True


def test_judge_results_full_run_scores_all_types(tmp_path, monkeypatch):
    monkeypatch.setattr(shared_utils, "CACHE_BASE_DIR", tmp_path)
    responses = _responses_dir(tmp_path)
    rows = [
        dict(
            question_id=1,
            answer_type="multiple_choice",
            expected_answers="B",
            output="B",
        ),
        dict(
            question_id=2,
            answer_type="short_form",
            expected_answers="kintu",
            output="kintu",
        ),
        dict(
            question_id=5,
            answer_type="multiple_choice",
            expected_answers="B",
            output="",
        ),
        dict(question_id=6, answer_type="essay", expected_answers="x", output="x"),
        dict(
            question_id=7,
            answer_type="multiple_choice",
            expected_answers="B",
            output="",
            call_status="error",
            error_type="RuntimeError",
            error_message="boom",
        ),
    ]
    _write_responses(responses, rows)
    ds = write_dataset(
        tmp_path / "ds.json",
        [
            {"question_id": r["question_id"], "question": f"q{r['question_id']}"}
            for r in rows
        ],
    )
    caller = FakeCaller(
        make_response(json.dumps({"is_correct": True, "justification": "ok"}))
    )
    monkeypatch.setattr(judge_bench, "get_caller", lambda _n: caller)

    monkeypatch.setattr(judge_bench, "git_hash", lambda: "judge123")

    df = judge_bench.judge_results(
        VALID_MODEL, language_name="Kiswahili", dataset_path=ds
    ).set_index("question_id")

    assert bool(df.loc[1, "is_correct"]) is True  # MCQ regex
    assert df.loc[1, "score_method"] == judge_bench.SCORE_METHOD_REGEX
    assert df.loc[1, "parsed_answer"] == "B"
    assert bool(df.loc[2, "is_correct"]) is True  # short_form judge
    assert df.loc[2, "score_method"] == judge_bench.SCORE_METHOD_LLM_JUDGE
    assert pd.isna(df.loc[5, "is_correct"])  # missing output
    assert df.loc[5, "justification"] == "Missing model output"
    assert df.loc[5, "score_method"] == judge_bench.SCORE_METHOD_UNSCORED
    assert pd.isna(df.loc[6, "is_correct"])  # unknown type
    assert pd.isna(df.loc[7, "is_correct"])  # model call error
    assert "Model call error: RuntimeError: boom" in df.loc[7, "justification"]
    # judged CSV written
    out = (
        tmp_path
        / "kiswahili"
        / "runs"
        / "default"
        / "responses_judged"
        / f"responses_judged_{VALID_MODEL}.csv"
    )
    assert out.exists()
    # log.json written at run dir level
    log_path = tmp_path / "kiswahili" / "runs" / "default" / "log.json"
    assert log_path.exists()
    entry = json.loads(log_path.read_text())[VALID_MODEL]
    assert "responses_judged_timestamp-utc" in entry
    assert entry["responses_judged_git-hash"] == "judge123"
    assert "responses_judged_fallback_timestamp-utc" not in entry


def test_judge_results_ensemble_majority_and_agreement_columns(tmp_path, monkeypatch):
    monkeypatch.setattr(shared_utils, "CACHE_BASE_DIR", tmp_path)
    responses = _responses_dir(tmp_path)
    _write_responses(
        responses,
        [
            {
                "question_id": 1,
                "answer_type": "short_form",
                "expected_answers": "kintu",
                "output": "kintu",
            }
        ],
    )
    ds = write_dataset(tmp_path / "ds.json", [{"question_id": 1, "question": "q"}])

    judges = ["judge-a", "judge-b", "judge-c"]
    callers = {
        "judge-a": FakeCaller(
            make_response(json.dumps({"is_correct": True, "justification": "yes-a"}))
        ),
        "judge-b": FakeCaller(
            make_response(json.dumps({"is_correct": True, "justification": "yes-b"}))
        ),
        "judge-c": FakeCaller(
            make_response(json.dumps({"is_correct": False, "justification": "no-c"}))
        ),
    }
    monkeypatch.setattr(judge_bench, "get_caller", lambda name: callers[name])

    df = judge_bench.judge_results(
        VALID_MODEL, language_name="Kiswahili", dataset_path=ds, judge_models=judges
    ).set_index("question_id")

    assert bool(df.loc[1, "is_correct"]) is True  # 2/3 majority
    assert df.loc[1, "justification"] == "yes-a"  # primary judge's own rationale
    assert df.loc[1, "judge_agreement"] == pytest.approx(2 / 3)
    assert bool(df.loc[1, "judge_disagreement_flag"]) is True
    assert bool(df.loc[1, "is_correct_judge-a"]) is True
    assert bool(df.loc[1, "is_correct_judge-b"]) is True
    assert bool(df.loc[1, "is_correct_judge-c"]) is False
    assert df.loc[1, "justification_judge-c"] == "no-c"

    log_path = tmp_path / "kiswahili" / "runs" / "default" / "log.json"
    assert json.loads(log_path.read_text())[VALID_MODEL]["llm_judges"] == judges


def test_judge_results_ensemble_tie_is_unscored(tmp_path, monkeypatch):
    monkeypatch.setattr(shared_utils, "CACHE_BASE_DIR", tmp_path)
    responses = _responses_dir(tmp_path)
    _write_responses(
        responses,
        [
            {
                "question_id": 1,
                "answer_type": "short_form",
                "expected_answers": "kintu",
                "output": "kintu",
            }
        ],
    )
    ds = write_dataset(tmp_path / "ds.json", [{"question_id": 1, "question": "q"}])

    judges = ["judge-a", "judge-b"]
    callers = {
        "judge-a": FakeCaller(
            make_response(json.dumps({"is_correct": True, "justification": "yes"}))
        ),
        "judge-b": FakeCaller(
            make_response(json.dumps({"is_correct": False, "justification": "no"}))
        ),
    }
    monkeypatch.setattr(judge_bench, "get_caller", lambda name: callers[name])

    df = judge_bench.judge_results(
        VALID_MODEL, language_name="Kiswahili", dataset_path=ds, judge_models=judges
    ).set_index("question_id")

    assert pd.isna(df.loc[1, "is_correct"])
    assert df.loc[1, "judge_agreement"] == 0.5
    assert bool(df.loc[1, "judge_disagreement_flag"]) is True


def test_judge_results_run_id_uses_run_specific_dirs(tmp_path, monkeypatch):
    monkeypatch.setattr(shared_utils, "CACHE_BASE_DIR", tmp_path)
    responses = tmp_path / "kiswahili" / "runs" / "run-123" / "responses"
    responses.mkdir(parents=True)
    _write_responses(
        responses,
        [
            {
                "question_id": 1,
                "answer_type": "multiple_choice",
                "expected_answers": "B",
                "output": "B",
            }
        ],
    )
    ds = write_dataset(tmp_path / "ds.json", [{"question_id": 1, "question": "q"}])

    judge_bench.judge_results(
        VALID_MODEL,
        language_name="Kiswahili",
        dataset_path=ds,
        run_id="run-123",
    )

    assert (
        tmp_path
        / "kiswahili"
        / "runs"
        / "run-123"
        / "responses_judged"
        / f"responses_judged_{VALID_MODEL}.csv"
    ).exists()


def test_judge_results_judge_model_not_registered(tmp_path, monkeypatch):
    monkeypatch.setattr(shared_utils, "CACHE_BASE_DIR", tmp_path)
    responses = _responses_dir(tmp_path)
    _write_responses(
        responses,
        [
            {
                "question_id": 1,
                "answer_type": "short_form",
                "expected_answers": "x",
                "output": "y",
            }
        ],
    )
    ds = write_dataset(tmp_path / "ds.json", [{"question_id": 1, "question": "q"}])

    def raise_ni(_n):
        raise NotImplementedError

    monkeypatch.setattr(judge_bench, "get_caller", raise_ni)
    with pytest.raises(ValueError, match="not registered"):
        judge_bench.judge_results(
            VALID_MODEL, language_name="Kiswahili", dataset_path=ds
        )


# --------------------------------------------------------------------------- #
# judge_results_fallback
# --------------------------------------------------------------------------- #
def _judged_dir(base, language="kiswahili"):
    d = base / language / "runs" / "default" / "responses_judged"
    d.mkdir(parents=True)
    return d


def test_fallback_missing_input_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(shared_utils, "CACHE_BASE_DIR", tmp_path)
    with pytest.raises(FileNotFoundError):
        judge_bench.judge_results_fallback(
            VALID_MODEL, language_name="Kiswahili", dataset_path=tmp_path / "ds.json"
        )


def test_fallback_cache_hit(tmp_path, monkeypatch):
    monkeypatch.setattr(shared_utils, "CACHE_BASE_DIR", tmp_path)
    judged = _judged_dir(tmp_path)
    pd.DataFrame(
        [{"question_id": 1, "is_correct": True, "answer_type": "multiple_choice"}]
    ).to_csv(judged / f"responses_judged_{VALID_MODEL}.csv", index=False)
    fb = tmp_path / "kiswahili" / "runs" / "default" / "responses_judged_fallback"
    fb.mkdir(parents=True)
    pd.DataFrame([{"question_id": 1, "is_correct_fallback": True}]).to_csv(
        fb / f"responses_judged_{VALID_MODEL}.csv", index=False
    )
    df = judge_bench.judge_results_fallback(
        VALID_MODEL, language_name="Kiswahili", dataset_path=tmp_path / "ds.json"
    )
    assert "is_correct_fallback" in df.columns


def test_fallback_no_null_rows_writes_empty_columns(tmp_path, monkeypatch):
    monkeypatch.setattr(shared_utils, "CACHE_BASE_DIR", tmp_path)
    monkeypatch.setattr(judge_bench, "git_hash", lambda: "fb123")
    judged = _judged_dir(tmp_path)
    pd.DataFrame(
        [
            {
                "question_id": 1,
                "is_correct": True,
                "answer_type": "multiple_choice",
                "expected_answers": "B",
                "output": "B",
            }
        ]
    ).to_csv(judged / f"responses_judged_{VALID_MODEL}.csv", index=False)

    df = judge_bench.judge_results_fallback(
        VALID_MODEL, language_name="Kiswahili", dataset_path=tmp_path / "ds.json"
    )
    assert df["is_correct_fallback"].isna().all()
    out = (
        tmp_path
        / "kiswahili"
        / "runs"
        / "default"
        / "responses_judged_fallback"
        / f"responses_judged_{VALID_MODEL}.csv"
    )
    assert out.exists()
    log_path = tmp_path / "kiswahili" / "runs" / "default" / "log.json"
    entry = json.loads(log_path.read_text())[VALID_MODEL]
    assert "responses_judged_fallback_timestamp-utc" in entry
    assert entry["responses_judged_fallback_git-hash"] == "fb123"


def test_fallback_rejudges_null_rows(tmp_path, monkeypatch):
    monkeypatch.setattr(shared_utils, "CACHE_BASE_DIR", tmp_path)
    judged = _judged_dir(tmp_path)
    pd.DataFrame(
        [
            {
                "question_id": 1,
                "is_correct": None,
                "answer_type": "multiple_choice",
                "expected_answers": "B",
                "output": "the answer is B",
            }
        ]
    ).to_csv(judged / f"responses_judged_{VALID_MODEL}.csv", index=False)
    ds = write_dataset(tmp_path / "ds.json", [{"question_id": 1, "question": "q"}])
    caller = FakeCaller(
        make_response(json.dumps({"is_correct": True, "justification": "B it is"}))
    )
    monkeypatch.setattr(judge_bench, "get_caller", lambda _n: caller)

    df = judge_bench.judge_results_fallback(
        VALID_MODEL, language_name="Kiswahili", dataset_path=ds
    )
    assert bool(df.loc[0, "is_correct_fallback"]) is True
    assert df.loc[0, "justification_fallback"] == "B it is"
    assert df.loc[0, "score_method"] == judge_bench.SCORE_METHOD_LLM_FALLBACK


def test_fallback_skips_null_rows_with_no_output(tmp_path, monkeypatch):
    monkeypatch.setattr(shared_utils, "CACHE_BASE_DIR", tmp_path)
    judged = _judged_dir(tmp_path)
    pd.DataFrame(
        [
            {
                "question_id": 1,
                "is_correct": None,
                "answer_type": "multiple_choice",
                "expected_answers": "B",
                "output": None,
                "call_status": "error",
            },
            {
                "question_id": 2,
                "is_correct": None,
                "answer_type": "multiple_choice",
                "expected_answers": "B",
                "output": "the answer is B",
                "call_status": "success",
            },
        ]
    ).to_csv(judged / f"responses_judged_{VALID_MODEL}.csv", index=False)
    ds = write_dataset(
        tmp_path / "ds.json",
        [{"question_id": 1, "question": "q"}, {"question_id": 2, "question": "q"}],
    )
    caller = FakeCaller(
        make_response(json.dumps({"is_correct": True, "justification": "B it is"}))
    )
    monkeypatch.setattr(judge_bench, "get_caller", lambda _n: caller)

    df = judge_bench.judge_results_fallback(
        VALID_MODEL, language_name="Kiswahili", dataset_path=ds
    )
    # no-output row (call error) stays unscored -- not sent to the judge at all
    assert pd.isna(df.loc[0, "is_correct_fallback"])
    # the row with a real output still gets fallback-judged normally
    assert bool(df.loc[1, "is_correct_fallback"]) is True
    assert len(caller.calls) == 1


def test_fallback_ensemble_majority_and_agreement_columns(tmp_path, monkeypatch):
    monkeypatch.setattr(shared_utils, "CACHE_BASE_DIR", tmp_path)
    judged = _judged_dir(tmp_path)
    pd.DataFrame(
        [
            {
                "question_id": 1,
                "is_correct": None,
                "answer_type": "multiple_choice",
                "expected_answers": "B",
                "output": "the answer is B",
            }
        ]
    ).to_csv(judged / f"responses_judged_{VALID_MODEL}.csv", index=False)
    ds = write_dataset(tmp_path / "ds.json", [{"question_id": 1, "question": "q"}])

    judges = ["judge-a", "judge-b", "judge-c"]
    callers = {
        "judge-a": FakeCaller(
            make_response(json.dumps({"is_correct": True, "justification": "yes-a"}))
        ),
        "judge-b": FakeCaller(
            make_response(json.dumps({"is_correct": True, "justification": "yes-b"}))
        ),
        "judge-c": FakeCaller(
            make_response(json.dumps({"is_correct": False, "justification": "no-c"}))
        ),
    }
    monkeypatch.setattr(judge_bench, "get_caller", lambda name: callers[name])

    df = judge_bench.judge_results_fallback(
        VALID_MODEL, language_name="Kiswahili", dataset_path=ds, judge_models=judges
    )
    assert bool(df.loc[0, "is_correct_fallback"]) is True
    assert df.loc[0, "justification_fallback"] == "yes-a"
    assert df.loc[0, "judge_agreement_fallback"] == pytest.approx(2 / 3)
    assert bool(df.loc[0, "judge_disagreement_flag_fallback"]) is True
    assert bool(df.loc[0, "is_correct_fallback_judge-c"]) is False
    assert df.loc[0, "justification_fallback_judge-c"] == "no-c"

    log_path = tmp_path / "kiswahili" / "runs" / "default" / "log.json"
    entry = json.loads(log_path.read_text())[VALID_MODEL]
    assert entry["llm_judges_fallback"] == judges


def test_fallback_judge_not_registered(tmp_path, monkeypatch):
    monkeypatch.setattr(shared_utils, "CACHE_BASE_DIR", tmp_path)
    judged = _judged_dir(tmp_path)
    pd.DataFrame(
        [
            {
                "question_id": 1,
                "is_correct": None,
                "answer_type": "multiple_choice",
                "expected_answers": "B",
                "output": "unclear",
            }
        ]
    ).to_csv(judged / f"responses_judged_{VALID_MODEL}.csv", index=False)
    ds = write_dataset(tmp_path / "ds.json", [{"question_id": 1, "question": "q"}])

    def raise_ni(_n):
        raise NotImplementedError

    monkeypatch.setattr(judge_bench, "get_caller", raise_ni)
    with pytest.raises(ValueError, match="not registered"):
        judge_bench.judge_results_fallback(
            VALID_MODEL, language_name="Kiswahili", dataset_path=ds
        )


# --------------------------------------------------------------------------- #
# compute_summary
# --------------------------------------------------------------------------- #
def test_compute_summary_basic():
    df = pd.DataFrame(
        [
            {"answer_type": "multiple_choice", "is_correct": True},
            {"answer_type": "multiple_choice", "is_correct": False},
            {"answer_type": "short_form", "is_correct": True},
            {"answer_type": "essay", "is_correct": None},
        ]
    )
    summary = judge_bench.compute_summary(df, "m")
    assert summary["model"] == "m"
    assert summary["scored_count"] == 3
    assert summary["unscored_count"] == 1
    assert summary["accuracy"] == round(2 / 3 * 100, 2)
    assert summary["accuracy_mcq"] == 50.0
    assert "accuracy_essay" not in summary


def test_compute_summary_uses_fallback_when_present():
    df = pd.DataFrame(
        [
            {
                "answer_type": "multiple_choice",
                "is_correct": None,
                "is_correct_fallback": True,
            },
            {
                "answer_type": "multiple_choice",
                "is_correct": False,
                "is_correct_fallback": None,
            },
        ]
    )
    summary = judge_bench.compute_summary(df, "m")
    # row0 -> fallback True, row1 -> falls back to is_correct False
    assert summary["scored_count"] == 2
    assert summary["accuracy"] == 50.0


# --------------------------------------------------------------------------- #
# build_detailed
# --------------------------------------------------------------------------- #
def test_build_detailed_columns_and_fallback(tmp_path):
    ds = write_dataset(
        tmp_path / "ds.json",
        [
            {
                "question_id": 2,
                "domain": "D2",
                "grade": "G2",
                "answer_type": "short_form",
                "question": "q2",
                "expected_answers": ["x", "y"],
            },
            {
                "question_id": 1,
                "domain": "D1",
                "grade": "G1",
                "answer_type": "multiple_choice",
                "question": "q1",
                "expected_answers": ["A"],
            },
        ],
    )
    j1 = pd.DataFrame(
        {"question_id": [1, 2], "output": ["A", "z"], "is_correct": [True, False]}
    )
    j2 = pd.DataFrame(
        {
            "question_id": [1, 2],
            "output": ["B", "x"],
            "is_correct": [None, True],
            "is_correct_fallback": [True, None],
        }
    )

    detailed = judge_bench.build_detailed({"m1": j1, "m2": j2}, ds)

    # Rows ordered by question_id; list expected_answers joined with ", ".
    assert list(detailed["question_id"]) == [1, 2]
    assert detailed.loc[0, "expected_answers"] == "A"
    assert detailed.loc[1, "expected_answers"] == "x, y"
    # Per-model output + correctness columns.
    assert detailed.loc[0, "m1_output"] == "A"
    assert bool(detailed.loc[0, "m1_correct"]) is True
    assert bool(detailed.loc[1, "m1_correct"]) is False
    # m2 uses the fallback verdict where present, else the original is_correct.
    assert bool(detailed.loc[0, "m2_correct"]) is True  # fallback True over None
    assert bool(detailed.loc[1, "m2_correct"]) is True  # fallback None -> is_correct


def test_build_detailed_skips_incomplete_frame(tmp_path):
    ds = write_dataset(
        tmp_path / "ds.json",
        [
            {
                "question_id": 1,
                "domain": "D1",
                "grade": "G1",
                "answer_type": "short_form",
                "question": "q",
                "expected_answers": ["a"],
            }
        ],
    )
    detailed = judge_bench.build_detailed({"m1": pd.DataFrame()}, ds)
    assert list(detailed["question_id"]) == [1]
    assert "m1_output" not in detailed.columns


def test_compute_summary_no_judge_agreement_columns_returns_none_and_zero():
    df = pd.DataFrame([{"answer_type": "multiple_choice", "is_correct": True}])
    summary = judge_bench.compute_summary(df, "m")
    assert summary["judge_agreement_mean"] is None
    assert summary["judge_disagreement_count"] == 0


def test_compute_summary_includes_judge_agreement_stats():
    df = pd.DataFrame(
        [
            {
                "answer_type": "short_form",
                "is_correct": True,
                "judge_agreement": 1.0,
                "judge_disagreement_flag": False,
            },
            {
                "answer_type": "short_form",
                "is_correct": True,
                "judge_agreement": 2 / 3,
                "judge_disagreement_flag": True,
            },
            {
                "answer_type": "multiple_choice",
                "is_correct": True,
                "judge_agreement": None,
                "judge_disagreement_flag": False,
            },
        ]
    )
    summary = judge_bench.compute_summary(df, "m")
    assert summary["judge_agreement_mean"] == pytest.approx((1.0 + 2 / 3) / 2, rel=1e-3)
    assert summary["judge_disagreement_count"] == 1


def test_compute_judge_agreement_pairwise():
    df = pd.DataFrame(
        {
            "is_correct_j1": [True, True, False, None],
            "is_correct_j2": [True, False, False, True],
        }
    )
    result = judge_bench.compute_judge_agreement(df, ["j1", "j2"])
    assert result.loc["j1", "j1"] == 1.0
    assert result.loc["j2", "j2"] == 1.0
    # rows 0,1,2 have both non-null; agree on rows 0,2 -> 2/3
    assert result.loc["j1", "j2"] == pytest.approx(2 / 3)
    assert result.loc["j2", "j1"] == pytest.approx(2 / 3)


def test_compute_judge_agreement_fallback_columns():
    df = pd.DataFrame(
        {
            "is_correct_fallback_j1": [True, False],
            "is_correct_fallback_j2": [True, True],
        }
    )
    result = judge_bench.compute_judge_agreement(df, ["j1", "j2"], fallback=True)
    assert result.loc["j1", "j2"] == 0.5


def test_compute_judge_agreement_missing_column_raises():
    df = pd.DataFrame({"is_correct_j1": [True]})
    with pytest.raises(KeyError, match="j2"):
        judge_bench.compute_judge_agreement(df, ["j1", "j2"])
