"""Tests for judge_bench: scorers, judging, fallback, summary."""

import json

import pandas as pd
import pytest
from conftest import VALID_MODEL, FakeCaller, make_response, write_dataset
from linguistic_knowledge_benchmark import judge_bench


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def _answers_dir(base, language="luganda"):
    d = base / f"cache_{language}" / "answers"
    d.mkdir(parents=True)
    return d


def _write_answers(answers_dir, rows):
    df = pd.DataFrame(rows)
    df.to_csv(answers_dir / f"resps_{VALID_MODEL}.csv", index=False)
    return df


# --------------------------------------------------------------------------- #
# prompt builder + simple scorers
# --------------------------------------------------------------------------- #
def test_build_judge_system_prompt_includes_language():
    assert "Yoruba" in judge_bench._build_judge_system_prompt("Yoruba")


@pytest.mark.parametrize(
    "output, expected, result",
    [("B", ["B"], True), ("A", ["B"], False)],
)
def test_score_multiple_choice(output, expected, result):
    assert judge_bench.score_multiple_choice(output, expected) == (result, "")


def test_score_multiple_choice_unparseable_returns_none():
    assert judge_bench.score_multiple_choice("???", ["B"]) == (None, "")


@pytest.mark.parametrize(
    "output, expected, result",
    [("True", ["True"], True), ("false", ["True"], False)],
)
def test_score_true_false_regex_parses(output, expected, result):
    assert judge_bench.score_true_false_regex(output, expected) == (result, "")


@pytest.mark.parametrize("bad", ["", "maybe so", float("nan")])
def test_score_true_false_regex_unparseable(bad):
    assert judge_bench.score_true_false_regex(bad, ["True"]) is None


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


# --------------------------------------------------------------------------- #
# judge_results
# --------------------------------------------------------------------------- #
def test_judge_results_missing_input_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(judge_bench, "CACHE_BASE_DIR", tmp_path)
    with pytest.raises(FileNotFoundError):
        judge_bench.judge_results(
            VALID_MODEL,
            language_name="Luganda",
            dataset_path=tmp_path / "ds.json",
        )


def test_judge_results_cache_hit(tmp_path, monkeypatch):
    monkeypatch.setattr(judge_bench, "CACHE_BASE_DIR", tmp_path)
    answers = _answers_dir(tmp_path)
    _write_answers(
        answers,
        [
            {
                "id": 1,
                "answer_type": "multiple_choice",
                "expected_answers": "B",
                "output": "B",
            }
        ],
    )
    judged = tmp_path / "cache_luganda" / "answers_judged"
    judged.mkdir(parents=True)
    pd.DataFrame([{"id": 1, "is_correct": True}]).to_csv(
        judged / f"resps_{VALID_MODEL}_judged.csv", index=False
    )

    df = judge_bench.judge_results(
        VALID_MODEL, language_name="Luganda", dataset_path=tmp_path / "ds.json"
    )
    assert bool(df.loc[0, "is_correct"]) is True


def test_judge_results_full_run_scores_all_types(tmp_path, monkeypatch):
    monkeypatch.setattr(judge_bench, "CACHE_BASE_DIR", tmp_path)
    answers = _answers_dir(tmp_path)
    rows = [
        dict(id=1, answer_type="multiple_choice", expected_answers="B", output="B"),
        dict(id=2, answer_type="short_form", expected_answers="kintu", output="kintu"),
        dict(id=3, answer_type="true_false", expected_answers="True", output="True"),
        dict(id=4, answer_type="true_false", expected_answers="True", output="maybe"),
        dict(id=5, answer_type="multiple_choice", expected_answers="B", output=""),
        dict(id=6, answer_type="essay", expected_answers="x", output="x"),
    ]
    _write_answers(answers, rows)
    ds = write_dataset(
        tmp_path / "ds.json", [{"id": r["id"], "question": f"q{r['id']}"} for r in rows]
    )
    caller = FakeCaller(
        make_response(json.dumps({"is_correct": True, "justification": "ok"}))
    )
    monkeypatch.setattr(judge_bench, "get_caller", lambda _n: caller)

    df = judge_bench.judge_results(
        VALID_MODEL, language_name="Luganda", dataset_path=ds
    ).set_index("id")

    assert bool(df.loc[1, "is_correct"]) is True  # MC regex
    assert bool(df.loc[2, "is_correct"]) is True  # short_form judge
    assert bool(df.loc[3, "is_correct"]) is True  # TF regex
    assert pd.isna(df.loc[4, "is_correct"])  # TF unparseable
    assert pd.isna(df.loc[5, "is_correct"])  # missing output
    assert df.loc[5, "justification"] == "Missing model output"
    assert pd.isna(df.loc[6, "is_correct"])  # unknown type
    # judged CSV written
    out = (
        tmp_path
        / "cache_luganda"
        / "answers_judged"
        / f"resps_{VALID_MODEL}_judged.csv"
    )
    assert out.exists()


def test_judge_results_judge_model_not_registered(tmp_path, monkeypatch):
    monkeypatch.setattr(judge_bench, "CACHE_BASE_DIR", tmp_path)
    answers = _answers_dir(tmp_path)
    _write_answers(
        answers,
        [
            {
                "id": 1,
                "answer_type": "short_form",
                "expected_answers": "x",
                "output": "y",
            }
        ],
    )
    ds = write_dataset(tmp_path / "ds.json", [{"id": 1, "question": "q"}])

    def raise_ni(_n):
        raise NotImplementedError

    monkeypatch.setattr(judge_bench, "get_caller", raise_ni)
    with pytest.raises(ValueError, match="not registered"):
        judge_bench.judge_results(VALID_MODEL, language_name="Luganda", dataset_path=ds)


# --------------------------------------------------------------------------- #
# judge_results_fallback
# --------------------------------------------------------------------------- #
def _judged_dir(base, language="luganda"):
    d = base / f"cache_{language}" / "answers_judged"
    d.mkdir(parents=True)
    return d


def test_fallback_missing_input_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(judge_bench, "CACHE_BASE_DIR", tmp_path)
    with pytest.raises(FileNotFoundError):
        judge_bench.judge_results_fallback(
            VALID_MODEL, language_name="Luganda", dataset_path=tmp_path / "ds.json"
        )


def test_fallback_cache_hit(tmp_path, monkeypatch):
    monkeypatch.setattr(judge_bench, "CACHE_BASE_DIR", tmp_path)
    judged = _judged_dir(tmp_path)
    pd.DataFrame(
        [{"id": 1, "is_correct": True, "answer_type": "multiple_choice"}]
    ).to_csv(judged / f"resps_{VALID_MODEL}_judged.csv", index=False)
    fb = tmp_path / "cache_luganda" / "answers_judged_fallback"
    fb.mkdir(parents=True)
    pd.DataFrame([{"id": 1, "is_correct_fallback": True}]).to_csv(
        fb / f"resps_{VALID_MODEL}_judged.csv", index=False
    )
    df = judge_bench.judge_results_fallback(
        VALID_MODEL, language_name="Luganda", dataset_path=tmp_path / "ds.json"
    )
    assert "is_correct_fallback" in df.columns


def test_fallback_no_null_rows_writes_empty_columns(tmp_path, monkeypatch):
    monkeypatch.setattr(judge_bench, "CACHE_BASE_DIR", tmp_path)
    judged = _judged_dir(tmp_path)
    pd.DataFrame(
        [
            {
                "id": 1,
                "is_correct": True,
                "answer_type": "multiple_choice",
                "expected_answers": "B",
                "output": "B",
            }
        ]
    ).to_csv(judged / f"resps_{VALID_MODEL}_judged.csv", index=False)

    df = judge_bench.judge_results_fallback(
        VALID_MODEL, language_name="Luganda", dataset_path=tmp_path / "ds.json"
    )
    assert df["is_correct_fallback"].isna().all()
    out = (
        tmp_path
        / "cache_luganda"
        / "answers_judged_fallback"
        / f"resps_{VALID_MODEL}_judged.csv"
    )
    assert out.exists()


def test_fallback_rejudges_null_rows(tmp_path, monkeypatch):
    monkeypatch.setattr(judge_bench, "CACHE_BASE_DIR", tmp_path)
    judged = _judged_dir(tmp_path)
    pd.DataFrame(
        [
            {
                "id": 1,
                "is_correct": None,
                "answer_type": "multiple_choice",
                "expected_answers": "B",
                "output": "the answer is B",
            }
        ]
    ).to_csv(judged / f"resps_{VALID_MODEL}_judged.csv", index=False)
    ds = write_dataset(tmp_path / "ds.json", [{"id": 1, "question": "q"}])
    caller = FakeCaller(
        make_response(json.dumps({"is_correct": True, "justification": "B it is"}))
    )
    monkeypatch.setattr(judge_bench, "get_caller", lambda _n: caller)

    df = judge_bench.judge_results_fallback(
        VALID_MODEL, language_name="Luganda", dataset_path=ds
    )
    assert bool(df.loc[0, "is_correct_fallback"]) is True
    assert df.loc[0, "justification_fallback"] == "B it is"


def test_fallback_judge_not_registered(tmp_path, monkeypatch):
    monkeypatch.setattr(judge_bench, "CACHE_BASE_DIR", tmp_path)
    judged = _judged_dir(tmp_path)
    pd.DataFrame(
        [
            {
                "id": 1,
                "is_correct": None,
                "answer_type": "true_false",
                "expected_answers": "True",
                "output": "unclear",
            }
        ]
    ).to_csv(judged / f"resps_{VALID_MODEL}_judged.csv", index=False)
    ds = write_dataset(tmp_path / "ds.json", [{"id": 1, "question": "q"}])

    def raise_ni(_n):
        raise NotImplementedError

    monkeypatch.setattr(judge_bench, "get_caller", raise_ni)
    with pytest.raises(ValueError, match="not registered"):
        judge_bench.judge_results_fallback(
            VALID_MODEL, language_name="Luganda", dataset_path=ds
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
            {"answer_type": "true_false", "is_correct": None},
        ]
    )
    summary = judge_bench.compute_summary(df, "m")
    assert summary["model"] == "m"
    assert summary["scored_count"] == 3
    assert summary["unscored_count"] == 1
    assert summary["accuracy"] == round(2 / 3 * 100, 2)
    assert summary["accuracy_mc"] == 50.0
    assert summary["accuracy_true_false"] == 0.0  # all unscored


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
