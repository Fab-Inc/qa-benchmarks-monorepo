import hashlib
import json
from pathlib import Path

import pytest
from reading_comprehension_question_generation.pairs import (
    ExtractionError,
    extract_run,
)

THREE = json.dumps([{"question": f"Q{i}?", "answer": f"A{i}"} for i in (1, 2, 3)])
TWO = json.dumps([{"question": f"Q{i}?", "answer": f"A{i}"} for i in (1, 2)])


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_ok_responses_become_linked_pairs_and_a_per_model_summary(
    tmp_path, write_run_dir
) -> None:
    items = tmp_path / "items.jsonl"
    items.write_text('{"benchmark_item_id": "i-1"}\n', encoding="utf-8")
    sha = hashlib.sha256(items.read_bytes()).hexdigest()
    run_dir = write_run_dir(
        tmp_path,
        {
            "m-a": [
                {"benchmark_item_id": "i-1", "output": THREE},
                {"benchmark_item_id": "i-2", "output": THREE},
            ],
            "m-b": [{"benchmark_item_id": "i-1", "output": THREE}],
        },
        items_path=items,
        items_sha256=sha,
    )

    summary = extract_run(run_dir)

    pairs = _read_jsonl(run_dir / "pairs.jsonl")
    assert len(pairs) == 9
    for pair in pairs:
        assert set(pair) == {
            "run_id",
            "model",
            "benchmark_item_id",
            "text_id",
            "pair_index",
            "question",
            "answer",
            "parse_status",
            "requested_pairs",
            "extracted_pairs",
        }
        assert pair["text_id"] == pair["benchmark_item_id"]
        assert pair["parse_status"] == "ok"
        assert (pair["requested_pairs"], pair["extracted_pairs"]) == (3, 3)
    first = [p for p in pairs if (p["model"], p["benchmark_item_id"]) == ("m-a", "i-1")]
    assert [p["pair_index"] for p in first] == [1, 2, 3]
    assert [p["question"] for p in first] == ["Q1?", "Q2?", "Q3?"]
    assert summary["models"]["m-a"] == {
        "responses": 2,
        "total_pairs": 6,
        "status_counts": {"ok": 2},
    }
    assert summary["models"]["m-b"]["total_pairs"] == 3
    assert summary["items_file"]["match"] is True
    assert json.loads((run_dir / "extraction_summary.json").read_text()) == summary


def test_unparseable_response_is_reported_and_counted_per_model(
    tmp_path, write_run_dir
) -> None:
    run_dir = write_run_dir(
        tmp_path,
        {
            "good": [{"benchmark_item_id": "i-1", "output": THREE}],
            "bad": [{"benchmark_item_id": "i-1", "output": "Sorry, no JSON here."}],
        },
    )

    summary = extract_run(run_dir)

    statuses = {r["model"]: r for r in _read_jsonl(run_dir / "response_status.jsonl")}
    assert statuses["bad"]["parse_status"] == "parse_failed"
    assert statuses["bad"]["extracted_pairs"] == 0
    assert summary["models"]["bad"]["status_counts"] == {"parse_failed": 1}
    assert summary["models"]["good"]["status_counts"] == {"ok": 1}
    assert not [p for p in _read_jsonl(run_dir / "pairs.jsonl") if p["model"] == "bad"]


def test_array_without_a_valid_pair_is_reported_as_no_pairs(
    tmp_path, write_run_dir
) -> None:
    run_dir = write_run_dir(
        tmp_path,
        {"bad": [{"benchmark_item_id": "i-1", "output": '[{"question": "Q?"}]'}]},
    )

    summary = extract_run(run_dir)

    assert summary["models"]["bad"]["status_counts"] == {"no_pairs": 1}
    assert summary["models"]["bad"]["total_pairs"] == 0
    assert _read_jsonl(run_dir / "pairs.jsonl") == []


def test_count_mismatch_keeps_the_extracted_pairs_and_flags_each(
    tmp_path, write_run_dir
) -> None:
    run_dir = write_run_dir(
        tmp_path, {"short": [{"benchmark_item_id": "i-1", "output": TWO}]}
    )

    summary = extract_run(run_dir)

    pairs = _read_jsonl(run_dir / "pairs.jsonl")
    assert [p["parse_status"] for p in pairs] == ["count_mismatch"] * 2
    assert {(p["requested_pairs"], p["extracted_pairs"]) for p in pairs} == {(3, 2)}
    assert summary["models"]["short"]["status_counts"] == {"count_mismatch": 1}
    assert summary["models"]["short"]["total_pairs"] == 2


def test_a_failed_call_is_reported_as_call_error_and_not_parsed(
    tmp_path, write_run_dir
) -> None:
    run_dir = write_run_dir(
        tmp_path,
        {
            "down": [
                {"benchmark_item_id": "i-1", "call_status": "error", "output": THREE}
            ]
        },
    )

    summary = extract_run(run_dir)

    (status,) = _read_jsonl(run_dir / "response_status.jsonl")
    assert (status["call_status"], status["parse_status"]) == ("error", "call_error")
    assert summary["models"]["down"]["status_counts"] == {"call_error": 1}
    assert _read_jsonl(run_dir / "pairs.jsonl") == []


def test_a_changed_items_file_is_flagged_in_the_summary(
    tmp_path, write_run_dir
) -> None:
    """The recorded items hash no longer matches after the items file changes."""
    items = tmp_path / "items.jsonl"
    items.write_text('{"benchmark_item_id": "i-1"}\n', encoding="utf-8")
    sha = hashlib.sha256(items.read_bytes()).hexdigest()
    run_dir = write_run_dir(
        tmp_path,
        {"m": [{"benchmark_item_id": "i-1", "output": THREE}]},
        items_path=items,
        items_sha256=sha,
    )
    items.write_text('{"benchmark_item_id": "i-1", "edited": true}\n', encoding="utf-8")

    summary = extract_run(run_dir)

    assert summary["items_file"]["match"] is False
    assert summary["items_file"]["recorded_sha256"] == sha


def test_a_run_without_response_files_is_an_error(tmp_path) -> None:
    (tmp_path / "empty-run" / "responses").mkdir(parents=True)

    with pytest.raises(ExtractionError, match="No response files found"):
        extract_run(tmp_path / "empty-run")


def test_a_malformed_response_line_names_the_file_and_line(
    tmp_path, write_run_dir
) -> None:
    run_dir = write_run_dir(
        tmp_path, {"m": [{"benchmark_item_id": "i-1", "output": THREE}]}
    )
    response_file = run_dir / "responses" / "responses_m.jsonl"
    response_file.write_text(
        response_file.read_text(encoding="utf-8") + "{broken\n", encoding="utf-8"
    )

    with pytest.raises(ExtractionError, match=r"responses_m\.jsonl:2: invalid JSON"):
        extract_run(run_dir)
