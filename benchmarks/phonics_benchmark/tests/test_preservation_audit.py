"""Deliberately corrupt the evidence used by the historical preservation audit."""

import csv
import importlib
import io
import json
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
audit = importlib.import_module("audit_unified_preservation")


def response(**changes):
    return {
        "language": "en",
        "model": "model",
        "benchmark_item_id": "item",
        "model_call_id": "call",
        "run_id": "original",
        "result_id": "result",
        "prompt": "Unmodified prompt",
        "call_status": "success",
        "timestamp_utc": "20260922T100000",
        "output": '["cat"]',
        "max_tokens": 30000,
        **changes,
    }


def write_rows(path, records):
    path.write_text("".join(json.dumps(row) + "\n" for row in records))


def test_numeric_serialization_is_the_only_content_normalization():
    original = response()
    assert audit.content_digest(original) == audit.content_digest(
        response(max_tokens=30000.0)
    )
    for changed in [
        response(output='["dog"]'),
        response(prompt="Other prompt"),
        response(run_id="other"),
        response(max_tokens=29999),
        response(extra=None),
        response(output='[ "cat" ]'),
    ]:
        assert audit.content_digest(original) != audit.content_digest(changed)
    with pytest.raises(ValueError, match="Non-finite"):
        audit.content_digest(response(max_tokens=float("nan")))


@pytest.mark.parametrize("mutation", ["missing", "unexpected", "changed"])
def test_response_reconciliation_rejects_lost_inserted_and_changed_rows(mutation):
    expected = {"call": audit.Response.from_row(response())}
    actual = dict(expected)
    if mutation == "missing":
        actual.clear()
    elif mutation == "unexpected":
        actual["other"] = audit.Response.from_row(response(model_call_id="other"))
    else:
        actual["call"] = audit.Response.from_row(response(output='["dog"]'))
    with pytest.raises(ValueError, match="call IDs|values changed"):
        audit.verify_response_set(expected, actual, "Unified")


@pytest.mark.parametrize(
    "changes, message",
    [
        ({}, "Duplicate call"),
        ({"model_call_id": "other"}, "Duplicate model/item"),
    ],
)
def test_duplicate_output_records_cannot_hide_in_an_index(tmp_path, changes, message):
    path = tmp_path / "responses.jsonl"
    write_rows(path, [response(), response(**changes)])
    with pytest.raises(ValueError, match=message):
        audit.response_index([path])


def evidence():
    # A copied success and a later failure: source b wins the exact timestamp tie.
    candidates = {
        ("sources/runs/a/responses/a.jsonl", 1): audit.Response.from_row(response()),
        ("sources/runs/b/responses/b.jsonl", 2): audit.Response.from_row(response()),
        ("sources/runs/c/responses/c.jsonl", 3): audit.Response.from_row(
            response(call_status="error", timestamp_utc="20260923T100000", output="")
        ),
    }
    sources = [{"file": key[0], "line": key[1]} for key in candidates]
    provenance = {"selected_source": sources[1], "all_sources": sources}
    return candidates, provenance, audit.Response.from_row(response())


def test_duplicate_evidence_is_checked_against_source_occurrences():
    candidates, provenance, merged = evidence()
    assert audit.verify_winner(candidates, provenance, merged) == (
        "sources/runs/b/responses/b.jsonl",
        2,
    )


@pytest.mark.parametrize(
    "mutation, message",
    [
        ("missing_source", "Incomplete source occurrences"),
        ("duplicated_source", "Incomplete source occurrences"),
        ("wrong_winner", "Unexpected duplicate winner"),
        ("changed_input", "Conflicting source call identity"),
        ("changed_merged", "Merged response changed"),
    ],
)
def test_corrupted_provenance_or_response_is_rejected(mutation, message):
    candidates, provenance, merged = evidence()
    if mutation == "missing_source":
        provenance["all_sources"] = provenance["all_sources"][:2]
    elif mutation == "duplicated_source":
        provenance["all_sources"].append(provenance["all_sources"][0])
    elif mutation == "wrong_winner":
        provenance["selected_source"] = provenance["all_sources"][-1]
    elif mutation == "changed_input":
        candidates[next(iter(candidates))] = audit.Response.from_row(
            response(prompt="Other")
        )
    else:
        merged = audit.Response.from_row(response(output='["dog"]'))
    with pytest.raises(ValueError, match=message):
        audit.verify_winner(candidates, provenance, merged)


def test_hash_verification_rejects_modified_archived_files(tmp_path):
    path = tmp_path / "responses.jsonl"
    write_rows(path, [response()])
    expected = audit.digest(path)
    checker = audit.Audit(tmp_path, csv.DictWriter(io.StringIO(), fieldnames=[]))
    checker.check_hash(path, expected)
    write_rows(path, [response(output='["dog"]')])
    with pytest.raises(ValueError, match="Hash mismatch"):
        checker.check_hash(path, expected)


def test_metric_report_fails_and_keeps_diagnostics_for_wrong_expectation(
    tmp_path, monkeypatch
):
    examples = importlib.import_module("build_metric_examples")
    name, prediction, counts, _, explanation = examples.CASES[0]
    changed = (name, prediction, counts, (0, 1, 1, 100), explanation)
    monkeypatch.setattr(examples, "CASES", [changed, *examples.CASES[1:]])
    output = tmp_path / "examples.html"
    assert examples.build(output) is False
    html = output.read_text()
    assert "<strong>FAIL</strong>" in html
    assert "Perfect: four metrics" in html
    assert "data:image/png;base64," in html
